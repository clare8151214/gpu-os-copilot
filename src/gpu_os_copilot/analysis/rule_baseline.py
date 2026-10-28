"""Symptom candidates, never a causal diagnosis without an intervention."""

from gpu_os_copilot.schemas import Hypothesis, MetricSummary, TraceBundle


def diagnose(trace: TraceBundle, summary: MetricSummary) -> list[Hypothesis]:
    category, statement, confidence = "unknown", "Insufficient evidence to rank a cause.", "low"
    kinds = set()
    if summary.gpu_idle_fraction is not None:
        if (summary.transfer_busy_fraction or 0) >= 0.5:
            category, kinds = "data_transfer", {"memcpy"}
            statement = "Transfers occupy much of the observed window; affinity is not established."
            confidence = "medium"
        elif summary.gpu_idle_fraction <= 0.1:
            category, kinds = "gpu_compute", {"gpu_kernel"}
            statement = "Captured GPU activity is dense; CPU scheduler starvation is unsupported."
            confidence = "medium"
        elif summary.gpu_idle_fraction >= 0.3:
            if (summary.sched_idle_overlap_ms or 0) > 0:
                category, kinds = "cpu_scheduling", {"cpu_sched", "cuda_api", "gpu_kernel"}
                statement = (
                    "Runnable CPU intervals overlap observed GPU gaps. A controlled affinity "
                    "experiment could discriminate this candidate; overlap alone is not causation."
                )
            elif summary.cuda_api_duration_ms is not None:
                category, kinds = "launch_overhead", {"cuda_api", "gpu_kernel"}
                statement = (
                    "GPU gaps and CUDA submissions coexist; scheduling causality is unknown."
                )
    evidence = [
        e.id
        for e in trace.events
        if e.kind in kinds
        and e.end_ns > trace.window.start_ns
        and e.start_ns < trace.window.end_ns
        and (e.device_id is None or e.device_id == trace.window.device_id)
    ][:100]
    return [
        Hypothesis(
            category=category,
            statement=statement,
            confidence=confidence,
            evidence_event_ids=evidence,
            missing_evidence=[*summary.missing_data[:19], "Controlled repeated intervention"],
        )
    ]
