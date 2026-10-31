import pytest

from gpu_os_copilot.experiments.study import summarize_batches
from gpu_os_copilot.schemas import BenchmarkTrial, ExperimentResult, Provenance


def batch(number, baseline, treatment):
    # Synthetic values exercise the measured-result contract; not hardware evidence.
    return ExperimentResult(
        experiment_id=f"batch-{number}",
        workload_id="launch_test",
        status="ok",
        provenance=Provenance(source="benchmark", confidence="measured", description="Unit test"),
        trials=[
            BenchmarkTrial(
                run_id=f"b{number}-{condition}-{i}",
                condition=condition,
                wall_time_ms=value,
                operation_count=100,
                throughput_ops_s=100000 / value,
                cpu_set=[0] if condition == "baseline" else [2],
                configuration={"seed": 42},
            )
            for condition, times in [("baseline", baseline), ("treatment", treatment)]
            for i, value in enumerate(times)
        ],
    )


def test_batch_variation_is_preserved_instead_of_claiming_uniform_benefit():
    results = [batch(1, [10, 10], [5, 5]), batch(2, [10, 10], [20, 20])]
    summary = summarize_batches(results)
    assert summary["batch_count"] == 2 and summary["trial_count"] == 8
    assert summary["batch_throughput_change_fraction"] == {"median": 0.25, "min": -0.5, "max": 1.0}
    assert not summary["positive_in_every_batch"]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "reused_trial", "workload", "placement", "partial"]
)
def test_invalid_batches_cannot_establish_reproducibility(mutation):
    results = [batch(1, [10, 12], [5, 6]), batch(2, [10, 12], [5, 6])]
    if mutation == "duplicate":
        results[1].experiment_id = results[0].experiment_id
    elif mutation == "reused_trial":
        results[1].trials[0].run_id = results[0].trials[0].run_id
    elif mutation == "workload":
        for trial in results[1].trials:
            trial.configuration = {"seed": 13}
    elif mutation == "placement":
        for trial in results[1].trials:
            if trial.condition == "treatment":
                trial.cpu_set = [3]
    else:
        results[1].status = "partial"
    with pytest.raises(ValueError):
        summarize_batches(results)
