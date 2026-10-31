"""Batch-aware summaries preserve variability instead of treating trials as independent batches."""

from statistics import median

from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import ExperimentResult


def summarize_batches(results: list[ExperimentResult]) -> dict:
    if len(results) < 2:
        raise ValueError("At least two independent experiment batches required")
    if len({r.experiment_id for r in results}) != len(results):
        raise ValueError("Duplicate batch IDs")
    records = []
    reference = None
    run_ids = set()
    for result in results:
        if result.status != "ok" or result.provenance.confidence != "measured":
            raise ValueError("Only successful measured batches may establish a result")
        comparison = compare_trials(result.trials)
        if comparison.get("status") != "ok":
            raise ValueError("Invalid within-batch comparison")
        configurations = result.trials[0].configuration
        placement = {
            condition: sorted({tuple(t.cpu_set) for t in result.trials if t.condition == condition})
            for condition in ("baseline", "treatment")
        }
        signature = (result.workload_id, configurations, placement)
        if reference is not None and signature != reference:
            raise ValueError("Batch workloads, environments or placements differ")
        reference = signature
        if any(t.run_id in run_ids for t in result.trials):
            raise ValueError("Trial reused across batches")
        run_ids.update(t.run_id for t in result.trials)
        records.append({"experiment_id": result.experiment_id, "comparison": comparison})
    changes = [r["comparison"]["median_throughput_change_fraction"] for r in records]
    return {
        "batch_count": len(records),
        "trial_count": sum(len(r.trials) for r in results),
        "batches": records,
        "batch_throughput_change_fraction": {
            "median": median(changes),
            "min": min(changes),
            "max": max(changes),
        },
        "positive_in_every_batch": all(change > 0 for change in changes),
        "interpretation": (
            "Descriptive same-host batch repetitions; no statistical significance or production "
            "generalization claimed. Trials are not counted as independent experiment batches."
        ),
    }
