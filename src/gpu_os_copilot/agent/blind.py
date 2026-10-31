"""Blinded, read-only evaluation on recorded measurements; never launches an experiment."""

import json

from gpu_os_copilot.agent.tools import EmptyArgs, SummaryArgs, TimelineArgs
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import ExperimentResult, TraceBundle


class BlindExecutor:
    owner_approved = False

    def __init__(
        self,
        trace: TraceBundle,
        recorded: ExperimentResult,
        *,
        case_id: str,
        repeated: ExperimentResult | None = None,
        prior_decision: dict | None = None,
    ):
        if trace.provenance.confidence != "measured":
            raise ValueError("Blind hardware evaluation requires measured traces")
        if repeated is not None and repeated.experiment_id == recorded.experiment_id:
            raise ValueError("Later evidence must be a distinct batch")
        self.recorded, self.repeated = recorded, repeated
        for result in [recorded, *([repeated] if repeated is not None else [])]:
            if result.status != "ok" or result.provenance.confidence != "measured":
                raise ValueError("Recorded evidence must be successful and measured")
            if compare_trials(result.trials).get("status") != "ok":
                raise ValueError("Recorded trial comparison is invalid")
            keys = (
                "scenario",
                "workload_version",
                "seed",
                "shape",
                "dtype",
                "warmup",
                "operations",
            )
            if any(
                result.trials[0].configuration.get(k) != trace.configuration.get(k) for k in keys
            ):
                raise ValueError("Recorded experiment does not match the trace workload")
        data = trace.model_dump()
        data.update(run_id=case_id, experiment_id=case_id, workload_id="anonymous")
        data["events"] = [dict(e, id=f"event-{i:05d}") for i, e in enumerate(data["events"])]
        self.trace = TraceBundle.model_validate(data)
        self.prior_decision = prior_decision
        self.reads = []

    def tool_schemas(self):
        declarations = [
            ("get_performance_summary", SummaryArgs, "Read measured metrics and missing flags."),
            ("inspect_timeline", TimelineArgs, "Inspect bounded anonymous measured events."),
            (
                "list_available_experiments",
                EmptyArgs,
                "Read the same available experiment for every case.",
            ),
            (
                "read_recorded_comparison",
                EmptyArgs,
                "Reveal the recorded affinity A/B result if useful; no new experiment is executed.",
            ),
        ]
        if self.repeated is not None:
            declarations.append(
                (
                    "read_repeated_comparisons",
                    EmptyArgs,
                    "Read a later batch to reassess the prior decision and reproducibility.",
                )
            )
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model, description in declarations
        ]

    def compact_summary(self):
        summary = summarize(self.trace)
        fields = [
            "window",
            "trace_completeness",
            "gpu_idle_fraction",
            "gpu_busy_ms",
            "kernel_duration_ms",
            "cuda_api_duration_ms",
            "transfer_duration_ms",
            "transfer_busy_fraction",
            "cpu_runnable_delay_ms",
            "sched_idle_overlap_ms",
        ]
        result = summary.model_dump(mode="json", include=set(fields))
        result.update(
            run_id=self.trace.run_id,
            source="measured",
            clock_aligned=self.trace.clock_aligned,
            missing_metrics=[name for name in fields if result[name] is None],
        )
        return result

    def evidence(self):
        result = {
            "run_id": self.trace.run_id,
            "input_confidence": "measured",
            "summary": self.compact_summary(),
            "evaluation": "Anonymous recorded-evidence evaluation, no new GPU execution.",
        }
        if self.prior_decision is not None:
            result["prior_decision"] = self.prior_decision
            result["prior_measured_comparison"] = compare_trials(self.recorded.trials)
            result["new_evidence"] = "A later measured batch is available; read it and reassess."
        return result

    def execute(self, name: str, arguments: str):
        try:
            if len(arguments) > 8192:
                raise ValueError("Arguments too large")
            if name in {"get_performance_summary", "inspect_timeline"}:
                model = TimelineArgs if name == "inspect_timeline" else SummaryArgs
                args = model.model_validate_json(arguments)
                if args.run_id != self.trace.run_id:
                    raise ValueError("Unknown run")
                if name == "get_performance_summary":
                    value = self.compact_summary()
                else:
                    window = self.trace.window
                    events = [
                        e
                        for e in self.trace.events
                        if e.kind in args.event_types
                        and e.end_ns > window.start_ns
                        and e.start_ns < window.end_ns
                        and e.device_id in {None, window.device_id}
                    ]
                    value = {
                        "timebase": self.trace.timebase,
                        "events": [
                            e.model_dump(include={"id", "kind", "start_ns", "end_ns", "device_id"})
                            for e in sorted(events, key=lambda e: e.start_ns)[: args.limit]
                        ],
                    }
            elif name == "list_available_experiments":
                EmptyArgs.model_validate_json(arguments)
                value = {
                    "experiments": [
                        {
                            "name": "affinity_ab",
                            "execution": "recorded replay",
                            "description": "Separate launcher/contender CPUs.",
                            "new_execution_allowed": False,
                            "recorded_results_available": True,
                        }
                    ]
                }
            elif name == "read_recorded_comparison" or (
                name == "read_repeated_comparisons" and self.repeated is not None
            ):
                EmptyArgs.model_validate_json(arguments)
                result = self.recorded if name == "read_recorded_comparison" else self.repeated
                self.reads.append(name)
                value = {
                    "source": "measured",
                    "execution": "recorded replay; no new experiment",
                    "batch": "initial" if name == "read_recorded_comparison" else "later",
                    "comparison": compare_trials(result.trials),
                }
            else:
                return {"error_code": "TOOL_NOT_ALLOWLISTED"}
            return {"schema_version": "1.0", "result": value}
        except (ValueError, TypeError, KeyError):
            return {"error_code": "INVALID_TOOL_ARGUMENTS"}

    def has_measured_comparison(self):
        return bool(self.reads)

    def comparison_evidence(self):
        # Prior-decision sessions expose the initial comparison in the opening evidence.
        # Repeated reads do not become independent batches. Never reveal unread outcomes.
        results = []
        if self.prior_decision is not None or "read_recorded_comparison" in self.reads:
            results.append(self.recorded)
        if self.repeated is not None and "read_repeated_comparisons" in self.reads:
            results.append(self.repeated)
        return [result.model_copy(deep=True) for result in results]

    def unread_comparison_count(self):
        return int(
            self.prior_decision is None and "read_recorded_comparison" not in self.reads
        ) + int(self.repeated is not None and "read_repeated_comparisons" not in self.reads)


def audit_usage(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    responses = [row for row in rows if row["kind"] == "response"]
    return {
        "responses": len(responses),
        "provider_tokens": sum(
            (row["response"].get("usage") or {}).get("total_tokens", 0) for row in responses
        ),
        "elapsed_ms": rows[-1]["elapsed_ms"],
        "provider_response_ids": [row["response"].get("response_id") for row in responses],
    }
