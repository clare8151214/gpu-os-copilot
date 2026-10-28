from statistics import median, pstdev

from gpu_os_copilot.schemas import BenchmarkTrial


def compare_trials(trials: list[BenchmarkTrial]) -> dict:
    groups = {
        name: [t for t in trials if t.condition == name] for name in ("baseline", "treatment")
    }
    if any(len(values) < 2 for values in groups.values()):
        return {"status": "inconclusive", "reason": "At least two trials per condition required"}
    if len(groups["baseline"]) != len(groups["treatment"]):
        return {"status": "inconclusive", "reason": "Unequal trial counts"}
    if len({t.run_id for t in trials}) != len(trials):
        return {"status": "inconclusive", "reason": "Duplicate trial IDs"}
    reference = trials[0]
    if any(
        t.operation_count != reference.operation_count or t.configuration != reference.configuration
        for t in trials
    ):
        return {
            "status": "inconclusive",
            "reason": "Workloads or measurement configurations differ",
        }
    stats = {}
    for name, group in groups.items():
        times = [t.wall_time_ms for t in group]
        rates = [t.throughput_ops_s for t in group]
        stats[name] = {
            "n": len(group),
            "median_wall_time_ms": median(times),
            "min_wall_time_ms": min(times),
            "max_wall_time_ms": max(times),
            "stddev_wall_time_ms": pstdev(times),
            "median_throughput_ops_s": median(rates),
        }
    base = stats["baseline"]["median_throughput_ops_s"]
    treated = stats["treatment"]["median_throughput_ops_s"]
    return {
        "status": "ok",
        **stats,
        "median_throughput_change_fraction": treated / base - 1,
        "interpretation": "Descriptive repeated comparison; not proof of a sole cause.",
        "missing_data": ["Timed trials have no GPU idle or CPU scheduling trace"],
    }
