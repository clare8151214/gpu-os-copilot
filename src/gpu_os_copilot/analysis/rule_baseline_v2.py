"""Stronger deterministic baseline using API-class and same-clock GPU overlap evidence.

Principle: rank the factor whose removal would most shorten the window. Activity that runs
concurrently with kernels recovers little; exposed transfers and per-kernel host waits do.
Written from the dev split only (new-bottleneck-protocol-v1); frozen before holdout.
"""

from gpu_os_copilot.schemas import Hypothesis, MetricSummary, TraceBundle

DENSE_KERNELS = 0.8
EXPOSED_TRANSFER = 0.3
IDLE_FLOOR = 0.1
SYNC_PER_KERNEL = 0.5


def diagnose_v2(trace: TraceBundle, summary: MetricSummary) -> list[Hypothesis]:
    idle, kernels = summary.gpu_idle_fraction, summary.kernel_busy_fraction
    window_ms = (trace.window.end_ns - trace.window.start_ns) / 1e6
    category, statement = "unknown", "Insufficient complete GPU evidence to rank a cause."
    if idle is not None and kernels is not None:
        transfer_ms = (summary.transfer_busy_fraction or 0) * window_ms
        exposed = max(0.0, transfer_ms - (summary.kernel_transfer_overlap_ms or 0)) / window_ms
        kernel_count = summary.kernel_duration_ms.get("count", 0)
        syncs = summary.cuda_api_class_counts.get("sync")
        if kernels >= DENSE_KERNELS:
            category, statement = "gpu_compute", "Kernels run for most of the window."
        elif exposed >= EXPOSED_TRANSFER:
            category, statement = "data_transfer", "Transfers not hidden by kernels are exposed."
        elif idle >= IDLE_FLOOR and syncs is not None and syncs >= SYNC_PER_KERNEL * kernel_count:
            category, statement = "synchronization", "Host waits recur at kernel frequency."
        elif idle >= IDLE_FLOOR and (summary.sched_idle_overlap_ms or 0) > 0:
            category, statement = "cpu_scheduling", "Runnable CPU intervals overlap GPU gaps."
        elif idle >= IDLE_FLOOR and summary.cuda_api_duration_ms is not None:
            category, statement = "launch_overhead", "GPU gaps coexist with launch submissions."
        else:
            category, statement = "gpu_compute", "Little idle time and no exposed transfer."
    return [
        Hypothesis(
            category=category,
            statement=statement + " Deterministic symptom ranking; not causal proof.",
            confidence="low",
            missing_evidence=[*summary.missing_data[:19], "Controlled single-factor ablation"],
        )
    ]
