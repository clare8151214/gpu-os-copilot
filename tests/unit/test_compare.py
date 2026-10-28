from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import BenchmarkTrial


def trials():
    return [
        BenchmarkTrial(
            run_id=f"trial-{i}",
            condition="baseline" if i < 3 else "treatment",
            wall_time_ms=ms,
            operation_count=100,
            throughput_ops_s=100000 / ms,
            cpu_set=[0] if i < 3 else [1],
            configuration={"seed": 42},
        )
        for i, ms in enumerate([10, 12, 14, 5, 6, 7])
    ]


def test_repeated_summary():
    summary = compare_trials(trials())
    assert summary["baseline"]["median_wall_time_ms"] == 12
    assert summary["treatment"]["median_wall_time_ms"] == 6
    assert summary["median_throughput_change_fraction"] == 1
    assert summary["baseline"]["stddev_wall_time_ms"] > 0


def test_failed_missing_mismatched_and_duplicate_trials():
    assert compare_trials([])["status"] == "inconclusive"
    data = trials()
    assert compare_trials(data[:-1])["status"] == "inconclusive"
    data[0].configuration = {"seed": 99}
    assert compare_trials(data)["status"] == "inconclusive"
    data = trials()
    data[0].run_id = data[1].run_id
    assert compare_trials(data)["status"] == "inconclusive"
