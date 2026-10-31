"""Compare two validated, identically instrumented placement captures."""

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.observability import observability
from gpu_os_copilot.schemas import TraceBundle


def compare_profiled_traces(baseline: TraceBundle, treatment: TraceBundle) -> dict:
    summaries = []
    reports = []
    instrumentation = []
    for trace in (baseline, treatment):
        scheduler = trace.configuration.get("scheduler_capture", {})
        alignment = trace.configuration.get("scheduler_alignment", {})
        instrumentation.append(
            (
                scheduler.get("clock"),
                scheduler.get("buffer_kb_per_cpu"),
                alignment.get("method"),
                len(alignment.get("anchors", [])),
                scheduler.get("enrollment_scope", "main_thread"),
            )
        )
        uncertainty = trace.configuration.get("scheduler_alignment", {}).get("uncertainty_ns")
        if type(uncertainty) is not int or not 0 <= uncertainty <= 100000:
            raise ValueError("Validated clock uncertainty unavailable or outside bounds")
        if (
            trace.provenance.confidence != "measured"
            or trace.trace_completeness != "complete"
            or not trace.clock_aligned
            or scheduler.get("coverage_validated") is not True
            or scheduler.get("target_events_verified") is not True
            or scheduler.get("loss_stats_verified") is not True
            or type(scheduler.get("lost_events")) is not int
            or scheduler.get("lost_events") != 0
        ):
            raise ValueError(
                "Both captures require complete measured GPU and aligned scheduler evidence"
            )
        summary = summarize(trace)
        required = (
            summary.gpu_idle_fraction,
            summary.cpu_runnable_delay_ms,
        )
        if any(value is None for value in required):
            raise ValueError("Required before/after metrics unavailable")
        report = observability(trace)
        if not report["api_gpu_correlation"]["conditional_timing_eligible"]:
            raise ValueError("API/GPU temporal order contradicted or incompletely checked")
        if report["conditional_covered_thread_idle_overlap_bounds"] is None:
            raise ValueError("Verified scheduler coverage of CUDA submission threads unavailable")
        summaries.append(summary)
        reports.append(report)
    if instrumentation[0] != instrumentation[1]:
        raise ValueError("Scheduler instrumentation or clock calibration differs")
    excluded = {"capture", "scheduler_capture", "scheduler_alignment"}
    configurations = [
        {k: v for k, v in trace.configuration.items() if k not in excluded}
        for trace in (baseline, treatment)
    ]
    if (
        configurations[0] != configurations[1]
        or baseline.workload_id != treatment.workload_id
        or baseline.operation_count != treatment.operation_count
        or baseline.window.device_id != treatment.window.device_id
    ):
        raise ValueError("Workload, environment, operation count or GPU device differs")
    captures = [trace.configuration.get("capture", {}) for trace in (baseline, treatment)]
    if {k: v for k, v in captures[0].items() if k not in {"cpu_set", "sqlite_sha256"}} != {
        k: v for k, v in captures[1].items() if k not in {"cpu_set", "sqlite_sha256"}
    }:
        raise ValueError("Capture instrumentation or contender differs")
    contender = captures[0].get("contender_cpu")
    if (
        contender is None
        or captures[0].get("cpu_set") != [contender]
        or len(captures[1].get("cpu_set", [])) != 1
        or contender in captures[1]["cpu_set"]
    ):
        raise ValueError("Expected a controlled baseline and separate treatment CPU")
    fields = {
        "gpu_idle_fraction",
        "gpu_busy_ms",
        "cpu_runnable_delay_ms",
        "sched_idle_overlap_ms",
        "wall_time_ms",
        "window",
    }
    before, after = summaries
    conditional = [report["conditional_covered_thread_idle_overlap_bounds"] for report in reports]
    return {
        "source": "measured",
        "kind": "two_profiled_placement_captures",
        "baseline": before.model_dump(mode="json", include=fields),
        "treatment": after.model_dump(mode="json", include=fields),
        "changes": {
            "gpu_idle_percentage_points": 100
            * (after.gpu_idle_fraction - before.gpu_idle_fraction),
            "cpu_runnable_delay_ms": after.cpu_runnable_delay_ms - before.cpu_runnable_delay_ms,
            "sched_idle_overlap_ms": after.sched_idle_overlap_ms - before.sched_idle_overlap_ms
            if after.sched_idle_overlap_ms is not None and before.sched_idle_overlap_ms is not None
            else None,
        },
        "conditional_changes": {
            "sched_idle_overlap_ms": conditional[1]["nominal_ms"] - conditional[0]["nominal_ms"]
            if all(conditional)
            else None,
            "meaning": "Assumes correct GPU/CPU conversion in both captures; not verified overlap.",
        },
        "alignment_uncertainty_ns": [
            trace.configuration.get("scheduler_alignment", {}).get("uncertainty_ns")
            for trace in (baseline, treatment)
        ],
        "observability": reports,
        "limitations": [
            "Two captures with profiler overhead; unprofiled throughput requires separate trials.",
            "Scheduler evidence covers validated enrolled threads; see per-capture coverage.",
            "NVTX calibration does not independently bound GPU/CPU conversion error.",
            "CPU delay response does not establish GPU idle causation; "
            "conditional overlaps exclude unknown GPU/CPU conversion error.",
        ],
    }
