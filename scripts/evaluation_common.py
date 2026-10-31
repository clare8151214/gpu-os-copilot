"""Shared replay environment for isolated old/new agents; never runs hardware work."""

import copy
import hashlib
import json
import time

from gpu_os_copilot.agent.tools import EmptyArgs, SummaryArgs, TimelineArgs
from gpu_os_copilot.config import CopilotError
from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import ExperimentResult, MetricSummary, TraceBundle


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


LIMITS = {
    "max_steps": 6,
    "max_tool_calls": 20,
    "max_experiments": 0,
    "max_completion_tokens": 4096,
    "max_total_completion_tokens": 24576,
    "max_request_bytes": 131072,
    "max_session_request_bytes": 786432,
    "reasoning_effort": "low",
}


class ReplayExecutor:
    owner_approved = False

    def __init__(self, case):
        self.trace = TraceBundle.model_validate(case["trace"])
        self.results = [ExperimentResult.model_validate(result) for result in case["batches"]]
        self.opening = copy.deepcopy(case["opening"])
        self.prior_decision = self.opening.get("prior_decision")
        self.reads = []
        if not 2 <= len(self.results) <= 3:
            raise ValueError("Two or three recorded batches required")
        if len({r.experiment_id for r in self.results}) != len(self.results):
            raise ValueError("Duplicate batch identity")
        for result in self.results:
            if result.status != "ok" or result.provenance.confidence != "measured":
                raise ValueError("Unverified recorded result")
            if compare_trials(result.trials).get("status") != "ok":
                raise ValueError("Invalid repeated comparison")
            keys = (
                "scenario",
                "workload_version",
                "seed",
                "shape",
                "dtype",
                "warmup",
                "operations",
                "gpu",
                "cuda",
                "torch",
            )
            if any(
                result.trials[0].configuration.get(k) != self.trace.configuration.get(k)
                for k in keys
            ):
                raise ValueError("Recorded workload does not match trace")

    def evidence(self):
        return copy.deepcopy(self.opening)

    def summary_for_rules(self):
        """Feed unchanged historical rules the same screened metrics given to both models."""
        metadata = self.trace.model_dump(
            include={
                "schema_version",
                "experiment_id",
                "worker_id",
                "workload_id",
                "created_at",
                "provenance",
            }
        )
        values = {
            name: copy.deepcopy(value)
            for name, value in self.opening["summary"].items()
            if name in MetricSummary.model_fields
        }
        # Missing metric names are safe, anonymous facts, unlike raw provenance notes.
        values["missing_data"] = list(self.opening["summary"]["missing_metrics"])
        return MetricSummary(**metadata, **values)

    def tool_schemas(self):
        declarations = [
            (
                "get_performance_summary",
                SummaryArgs,
                "Read measured metrics and explicit missing data.",
            ),
            (
                "inspect_timeline",
                TimelineArgs,
                "Read anonymous events from the observation window.",
            ),
            ("list_available_experiments", EmptyArgs, "Read the shared recorded-test menu."),
            (
                "read_recorded_comparison",
                EmptyArgs,
                "Reveal the first recorded affinity test; no new execution.",
            ),
            (
                "read_repeated_comparisons",
                EmptyArgs,
                "Reveal all later recorded batches to reassess reproducibility.",
            ),
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": args.model_json_schema(),
                },
            }
            for name, args, description in declarations
        ]

    def execute(self, name, arguments):
        try:
            if len(arguments) > 8192:
                raise ValueError("Arguments too large")
            if name in {"get_performance_summary", "inspect_timeline"}:
                args = (
                    TimelineArgs if name == "inspect_timeline" else SummaryArgs
                ).model_validate_json(arguments)
                if args.run_id != self.trace.run_id:
                    raise ValueError("Unknown run")
                if name == "get_performance_summary":
                    value = copy.deepcopy(self.opening["summary"])
                else:
                    w = self.trace.window
                    events = sorted(
                        (
                            e
                            for e in self.trace.events
                            if e.kind in args.event_types
                            and e.end_ns > w.start_ns
                            and e.start_ns < w.end_ns
                            and e.device_id in {None, w.device_id}
                        ),
                        key=lambda e: (e.start_ns, e.end_ns, e.id),
                    )[: args.limit]
                    value = {
                        "timebase": self.trace.timebase,
                        "events": [
                            e.model_dump(include={"id", "kind", "start_ns", "end_ns", "device_id"})
                            for e in events
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
                            "recorded_batches": len(self.results),
                        }
                    ]
                }
            elif name in {"read_recorded_comparison", "read_repeated_comparisons"}:
                EmptyArgs.model_validate_json(arguments)
                self.reads.append(name)
                group = self.results[:1] if name == "read_recorded_comparison" else self.results[1:]
                value = {
                    "source": "measured",
                    "execution": "recorded replay; no new experiment",
                    "batch": "initial" if name == "read_recorded_comparison" else "later",
                    "comparison": compare_trials(group[0].trials),
                    "additional_comparisons": [compare_trials(r.trials) for r in group[1:]],
                }
            else:
                return {"error_code": "TOOL_NOT_ALLOWLISTED"}
            return {"schema_version": "1.0", "result": value}
        except (ValueError, TypeError, KeyError):
            return {"error_code": "INVALID_TOOL_ARGUMENTS"}

    def comparison_evidence(self):
        results = (
            self.results[:1]
            if self.prior_decision is not None or "read_recorded_comparison" in self.reads
            else []
        )
        if "read_repeated_comparisons" in self.reads:
            results += self.results[1:]
        return [result.model_copy(deep=True) for result in results]

    def unread_comparison_count(self):
        return int(
            self.prior_decision is None and "read_recorded_comparison" not in self.reads
        ) + int("read_repeated_comparisons" not in self.reads)

    def has_measured_comparison(self):
        return bool(self.comparison_evidence())


class GuardedClient:
    """Identical external budgets and input checks apply even to the old agent."""

    def __init__(self, delegate, opening, tools, checkpoint=None):
        self.delegate, self.mode, self.config = delegate, delegate.mode, delegate.config
        self.opening_hash = digest(opening)
        self.common_tools = {t["function"]["name"]: t for t in tools}
        self.calls = 0
        self.input_bytes = 0
        self.usage = []
        self.checkpoint = checkpoint
        self.started = time.monotonic()

    def save_usage(self):
        if self.checkpoint is not None:
            self.checkpoint.write_text(json.dumps(self.report(), indent=2) + "\n")

    def complete(self, messages, tools):
        if digest(json.loads(messages[1]["content"])) != self.opening_hash:
            raise CopilotError("EVALUATION_INPUT_DRIFT")
        supplied = {t["function"]["name"]: t for t in tools}
        final_response = (
            (
                messages[-1].get("role") == "user"
                and json.loads(messages[-1]["content"])
                .get("controller_state", {})
                .get("final_response_required")
                is True
            )
            if not tools
            else False
        )
        if not final_response and any(
            supplied.get(name) != tool for name, tool in self.common_tools.items()
        ):
            raise CopilotError("EVALUATION_TOOL_DRIFT")
        if set(supplied) - set(self.common_tools) - {"plan_next_action"}:
            raise CopilotError("EVALUATION_UNEXPECTED_TOOL")
        size = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode())
        if (
            self.calls >= LIMITS["max_steps"]
            or size > LIMITS["max_request_bytes"]
            or self.input_bytes + size > LIMITS["max_session_request_bytes"]
        ):
            raise CopilotError("EVALUATION_REQUEST_BUDGET")
        self.calls += 1
        self.input_bytes += size
        self.save_usage()
        started = time.monotonic()
        response = self.delegate.complete(messages, tools)
        usage = response.get("usage")
        self.usage.append(
            {
                "response_id": response.get("response_id"),
                "usage": usage,
                "response_latency_ms": (time.monotonic() - started) * 1000,
            }
        )
        self.save_usage()
        if usage is not None and (
            not isinstance(usage, dict)
            or type(usage.get("completion_tokens")) is not int
            or usage["completion_tokens"] < 0
            or usage["completion_tokens"] > LIMITS["max_completion_tokens"]
        ):
            raise CopilotError("EVALUATION_INVALID_USAGE")
        return response

    def report(self):
        totals = [
            row["usage"].get("total_tokens") if isinstance(row["usage"], dict) else None
            for row in self.usage
        ]
        return {
            "requests": self.calls,
            "serialized_input_bytes": self.input_bytes,
            "completion_capacity_reserved": self.calls * LIMITS["max_completion_tokens"],
            "reported_total_tokens": sum(n for n in totals if type(n) is int and n >= 0),
            "usage_complete": len(totals) == self.calls
            and bool(totals)
            and all(type(n) is int and n >= 0 for n in totals),
            "provider_responses": self.usage,
            "cost_usd": None,
            "elapsed_ms": (time.monotonic() - self.started) * 1000,
        }
