from typing import Annotated
from uuid import uuid4

from pydantic import Field, ValidationError

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import (
    EventKind,
    ExperimentRequest,
    Identifier,
    Model,
    NonNegativeInt,
    Provenance,
    TraceBundle,
)
from gpu_os_copilot.worker.protocol import Worker
from gpu_os_copilot.workloads.launch_test import SCENARIO_SHAPES, WORKLOAD_VERSION


class SummaryArgs(Model):
    run_id: Identifier


class TimelineArgs(SummaryArgs):
    event_types: list[EventKind] = Field(min_length=1, max_length=6)
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 20


class EmptyArgs(Model):
    pass


class AffinityArgs(Model):
    workload_id: str = Field(pattern=r"^launch_test$")
    candidate_cpu_set: list[NonNegativeInt] = Field(min_length=1, max_length=1)
    repeats: Annotated[int, Field(strict=True, ge=2, le=5)] = 5


class CompareArgs(Model):
    baseline_run_ids: list[Identifier] = Field(min_length=2, max_length=5)
    treatment_run_ids: list[Identifier] = Field(min_length=2, max_length=5)


ARGUMENTS = {
    "get_performance_summary": (SummaryArgs, "Read deterministic metrics and missing-data flags."),
    "inspect_timeline": (TimelineArgs, "Read a bounded sorted timeline with event IDs and units."),
    "list_available_experiments": (EmptyArgs, "List available strategies and approval state."),
    "run_affinity_ab": (
        AffinityArgs,
        "Request bounded project-child A/B; dry run without approval.",
    ),
    "compare_benchmark_results": (CompareArgs, "Compare actual trial IDs from this agent session."),
}


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Schema v1.0. {description}",
                "parameters": args.model_json_schema(),
            },
        }
        for name, (args, description) in ARGUMENTS.items()
    ]


class ToolExecutor:
    def __init__(
        self,
        worker: Worker,
        trace: TraceBundle,
        *,
        owner_approved: bool = False,
        max_experiments: int = 2,
    ):
        self.worker, self.trace = worker, trace
        self.owner_approved = owner_approved
        self.max_experiments = max_experiments
        self.experiment_count = 0
        self.results = []

    def workload_bound(self) -> bool:
        config = self.trace.configuration
        scenario = config.get("scenario")
        if not isinstance(scenario, str) or scenario not in SCENARIO_SHAPES:
            return False
        expected = {
            "workload_version": WORKLOAD_VERSION,
            "shape": list(SCENARIO_SHAPES[scenario]),
            "seed": 42,
            "dtype": "float32",
            "warmup": 10,
            "operations": 200,
            "synchronized": True,
        }
        return (
            self.trace.workload_id == "launch_test"
            and self.trace.provenance.confidence == "measured"
            and self.trace.trace_completeness == "complete"
            and all(
                type(config.get(key)) is type(value) and config[key] == value
                for key, value in expected.items()
            )
            and all(
                isinstance(config.get(key), str) and bool(config[key])
                for key in ("gpu", "cuda", "torch")
            )
        )

    def local_configuration_matches(self) -> bool:
        caps = self.worker.capabilities()
        config = self.trace.configuration
        return (
            caps.cuda_available is True
            and bool(caps.gpus)
            and config.get("gpu") == caps.gpus[0].get("model")
            and config.get("cuda") == caps.cuda_version
            and config.get("torch") == caps.torch_version
        )

    def execute(self, name: str, arguments: str) -> dict:
        if name not in ARGUMENTS:
            return {"schema_version": "1.0", "error_code": "TOOL_NOT_ALLOWLISTED"}
        if len(arguments) > 8192:
            return {"schema_version": "1.0", "error_code": "ARGUMENT_BUDGET_EXCEEDED"}
        try:
            args = ARGUMENTS[name][0].model_validate_json(arguments)
            return {"schema_version": "1.0", "result": self._dispatch(name, args)}
        except (ValidationError, ValueError, KeyError):
            return {"schema_version": "1.0", "error_code": "INVALID_TOOL_ARGUMENTS"}
        except (OSError, RuntimeError):
            return {"schema_version": "1.0", "error_code": "WORKER_FAILED"}

    def _dispatch(self, name, args):
        if isinstance(args, SummaryArgs) and args.run_id != self.trace.run_id:
            raise ValueError("Run ID not available")
        if name == "get_performance_summary":
            return self.compact_summary()
        if name == "inspect_timeline":
            selected = sorted(
                (e for e in self.window_events() if e.kind in args.event_types),
                key=lambda e: (e.start_ns, e.end_ns, e.id),
            )[: args.limit]
            return {
                "provenance": self.trace.provenance.model_dump(),
                "timebase": self.trace.timebase,
                "events": [e.model_dump(exclude={"attributes", "pid", "tid"}) for e in selected],
            }
        if name == "list_available_experiments":
            caps = self.worker.capabilities()
            if len(caps.cpu_affinity) < 2:
                return {"experiments": [], "reason": "Two CPUs required for controlled contention"}
            return {
                "experiments": [
                    {
                        "name": "affinity_ab",
                        "workload_id": "launch_test",
                        "baseline_cpu": caps.cpu_affinity[0],
                        "candidate_cpu_sets": [caps.cpu_affinity[1:2]],
                        "can_execute": self.local_configuration_matches()
                        and self.owner_approved
                        and self.workload_bound(),
                        "scenario": self.trace.configuration.get("scenario"),
                        "workload_bound": self.workload_bound(),
                        "dry_run": not self.owner_approved,
                        "limitations": caps.limitations,
                    }
                ]
            }
        if name == "run_affinity_ab":
            if self.owner_approved and not self.workload_bound():
                return {"status": "blocked", "error_code": "TRACE_WORKLOAD_UNBOUND"}
            if self.owner_approved and not self.local_configuration_matches():
                return {"status": "blocked", "error_code": "TRACE_ENVIRONMENT_MISMATCH"}
            if self.experiment_count >= self.max_experiments:
                return {"status": "blocked", "error_code": "EXPERIMENT_BUDGET_EXCEEDED"}
            self.experiment_count += 1
            request = ExperimentRequest(
                experiment_id=f"agent-{uuid4().hex[:16]}",
                workload_id=args.workload_id,
                scenario=self.trace.configuration.get("scenario", "cpu_contention")
                if self.owner_approved
                else "cpu_contention",
                candidate_cpu_set=args.candidate_cpu_set,
                repeats=args.repeats,
                owner_approved=self.owner_approved,
                provenance=Provenance(
                    source="benchmark",
                    confidence="derived",
                    description="Agent-proposed bounded test",
                ),
            )
            result = self.worker.run_experiment(request)
            self.results.append(result)
            return result.model_dump(mode="json")
        if name == "compare_benchmark_results":
            lookup = {
                trial.run_id: trial
                for result in self.results
                if result.status == "ok"
                for trial in result.trials
            }
            baseline = [lookup[run_id] for run_id in args.baseline_run_ids]
            treatment = [lookup[run_id] for run_id in args.treatment_run_ids]
            if any(t.condition != "baseline" for t in baseline) or any(
                t.condition != "treatment" for t in treatment
            ):
                raise ValueError("Condition mismatch")
            return compare_trials(baseline + treatment)
        raise ValueError("Unsupported tool")

    def has_measured_comparison(self) -> bool:
        if not self.workload_bound():
            return False
        required = {"scenario", "seed", "shape", "dtype", "warmup", "operations", "gpu", "cuda"}
        if not required.issubset(self.trace.configuration):
            return False
        return any(
            r.status == "ok"
            and r.provenance.confidence == "measured"
            and r.comparison.get("status") == "ok"
            and bool(r.trials)
            and all(
                self.trace.configuration.get(key) == value
                for key, value in r.trials[0].configuration.items()
                if key not in {"host", "profiled"}
            )
            for r in self.results
        )

    def compact_summary(self) -> dict:
        summary = summarize(self.trace).model_dump(mode="json")
        summary["gpu_idle_interval_count"] = len(summary["gpu_idle_intervals_ns"])
        summary["gpu_idle_intervals_ns"] = summary["gpu_idle_intervals_ns"][:100]
        summary["missing_data"] = [text[:500] for text in summary["missing_data"][:20]]
        summary["provenance"]["limitations"] = [
            text[:500] for text in summary["provenance"]["limitations"][:20]
        ]
        return summary

    def evidence(self) -> dict:
        return {
            "run_id": self.trace.run_id,
            "input_confidence": self.trace.provenance.confidence,
            "summary": self.compact_summary(),
            "event_ids": [e.id for e in self.window_events()[:100]],
            "workload": {
                "id": self.trace.workload_id,
                "bound": self.workload_bound(),
                "owner_approved": self.owner_approved,
                "configuration": {
                    key: self.trace.configuration.get(key)
                    for key in (
                        "scenario",
                        "workload_version",
                        "shape",
                        "seed",
                        "dtype",
                        "operations",
                        "warmup",
                        "gpu",
                        "cuda",
                        "torch",
                        "profiled",
                    )
                },
            },
        }

    def window_events(self):
        window = self.trace.window
        return sorted(
            (
                event
                for event in self.trace.events
                if event.end_ns > window.start_ns
                and event.start_ns < window.end_ns
                and (event.device_id is None or event.device_id == window.device_id)
            ),
            key=lambda event: (event.start_ns, event.end_ns, event.id),
        )
