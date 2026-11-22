"""Bounded removal experiments: measure how much throughput removing one factor recovers.

The pattern workloads expose one switch per designed factor (``launch_test --ablate
--factor``). One experiment is one batch of interleaved base/removed pairs in project child
processes. Evaluation labels come from separate, larger ablation runs; results offered here
are never the label runs.
"""

from statistics import median
from typing import Literal

from pydantic import ValidationError

from gpu_os_copilot.schemas import Model

TOOL = "run_removal_experiment"
FACTOR_CATEGORY = {"sync": "synchronization", "launch": "launch_overhead", "copy": "data_transfer"}
CATEGORY_FACTOR = {category: factor for factor, category in FACTOR_CATEGORY.items()}
MAX_EXPERIMENTS = 2
PAIRS = 5
SCOPE = "Separate unprofiled runs of the same workload, not the input trace window."
METRICS = {
    "measured_removal_gain_{}": "Throughput gain from removing this factor: median removed "
    "throughput / median base throughput - 1, over interleaved pairs.",
    "measured_removal_share_{}": "Window share recovered by removing this factor, "
    "gain / (1 + gain); same scale as removal_estimate_{}_fraction and measured, not estimated.",
}
INSTRUCTION = (
    "run_removal_experiment removes one designed factor from the workload and measures "
    "throughput in interleaved base/removed pairs. Its measured_removal_share_<category> is a "
    "measurement of the quantity the ranking uses, on the removal_estimate scale; the "
    "estimates are first-order approximations that can mis-rank close or interacting factors. "
    "A measured share supersedes the estimate for that category; an unmeasured category keeps "
    "its estimate; apply the gpu_compute rule to the revised shares. Experiments cost GPU "
    "time and the budget is limited: choose "
    "the ones whose result could change your ranking. Invoke it directly; it needs no "
    "plan_next_action. Revise the ranking after each result."
)


class RemovalArgs(Model):
    category: Literal["synchronization", "launch_overhead", "data_transfer"]


def measured_effect(result: dict) -> dict:
    """Effect of one batch; pairs are the consecutive base/removed runs of each trial."""
    if result.get("status") != "ok":
        return {"status": "failed", "error": str(result.get("error", "unknown"))[:200]}
    trials = result["trials"]
    pairs = [
        {t["condition"]: t["throughput_ops_s"] for t in trials[i : i + 2]}
        for i in range(0, len(trials), 2)
    ]
    if not pairs or any(set(pair) != {"base", "ablated"} for pair in pairs):
        raise ValueError("Removal batch is not made of base/removed pairs")
    base = median(pair["base"] for pair in pairs)
    removed = median(pair["ablated"] for pair in pairs)
    gain = removed / base - 1
    pair_gains = [pair["ablated"] / pair["base"] - 1 for pair in pairs]
    return {
        "status": "ok",
        "pairs": len(pairs),
        "base_median_throughput_ops_s": base,
        "removed_median_throughput_ops_s": removed,
        "throughput_gain_fraction": gain,
        "pair_gain_range": [min(pair_gains), max(pair_gains)],
        "recovered_window_share": gain / (1 + gain),
    }


class RemovalExperiments:
    """Offered categories, budget and consumed results for one agent session.

    ``run`` maps a category to a raw batch result (recorded replay or a live run).
    """

    def __init__(self, offered, run, *, execution, max_experiments=MAX_EXPERIMENTS):
        self.offered = sorted(offered)
        if not self.offered or set(self.offered) - set(CATEGORY_FACTOR):
            raise ValueError("Unknown removal category")
        self.run, self.execution, self.max_experiments = run, execution, max_experiments
        self.results = {}

    def tool_schema(self):
        parameters = RemovalArgs.model_json_schema()
        parameters["properties"]["category"]["enum"] = self.offered
        return {
            "type": "function",
            "function": {
                "name": TOOL,
                "description": "Remove one designed factor from the workload and measure "
                f"throughput in {PAIRS} interleaved base/removed pairs ({self.execution}). "
                f"At most {self.max_experiments} per session, about 12 s of GPU time each. "
                "Returns the measured throughput gain and the recovered window share.",
                "parameters": parameters,
            },
        }

    def metrics(self):
        rows = {}
        for category in self.offered:
            effect = self.results.get(category, {})
            values = (
                effect.get("throughput_gain_fraction"),
                effect.get("recovered_window_share"),
            )
            for (name, definition), value in zip(METRICS.items(), values, strict=True):
                rows[name.format(category)] = {
                    "value": value,
                    "unit": "fraction",
                    "available": value is not None,
                    "availability_state": "available"
                    if value is not None
                    else "failed"
                    if category in self.results
                    else "not_run",
                    "scope": SCOPE,
                    "definition": definition.format(category),
                    "experiment_tool": TOOL,
                }
        return rows

    def execute(self, arguments: str) -> dict:
        if len(arguments) > 8192:
            return {"schema_version": "1.0", "error_code": "ARGUMENT_BUDGET_EXCEEDED"}
        try:
            category = RemovalArgs.model_validate_json(arguments).category
        except (ValidationError, ValueError):
            return {"schema_version": "1.0", "error_code": "INVALID_TOOL_ARGUMENTS"}
        if category not in self.offered:
            return {"schema_version": "1.0", "error_code": "REMOVAL_NOT_OFFERED"}
        if category in self.results:
            return {"schema_version": "1.0", "error_code": "EXPERIMENT_ALREADY_RUN"}
        if len(self.results) >= self.max_experiments:
            return {"schema_version": "1.0", "error_code": "EXPERIMENT_BUDGET_EXCEEDED"}
        effect = measured_effect(self.run(category))
        self.results[category] = effect
        return {
            "schema_version": "1.0",
            "result": {
                "category": category,
                **effect,
                "comparable_estimate_ref": f"removal_estimate_{category}_fraction",
                "metric_refs": [name.format(category) for name in METRICS],
                "execution": self.execution,
                "scope": SCOPE,
                "experiments_remaining": self.max_experiments - len(self.results),
            },
        }

    def report(self):
        return {
            "version": "removal-experiments-v1",
            "offered": self.offered,
            "run_order": list(self.results),
            "results": self.results,
            "max_experiments": self.max_experiments,
            "execution": self.execution,
        }
