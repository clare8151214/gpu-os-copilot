"""Report coverage and conditional timing bounds without inventing missing telemetry."""

from gpu_os_copilot.analysis.correlate import correlation_audit
from gpu_os_copilot.analysis.metrics import duration, summarize
from gpu_os_copilot.schemas import TraceBundle


def observability(trace: TraceBundle) -> dict:
    window = trace.window
    events = [
        e
        for e in trace.events
        if e.end_ns > window.start_ns
        and e.start_ns < window.end_ns
        and e.device_id in {None, window.device_id}
    ]
    api = [e for e in events if e.kind == "cuda_api"]
    threads = {(e.pid, e.tid) for e in api if e.pid is not None and e.tid is not None}
    unknown_thread_events = sum(e.pid is None or e.tid is None for e in api)
    capture = trace.configuration.get("scheduler_capture", {})
    alignment = trace.configuration.get("scheduler_alignment", {})
    uncertainty = alignment.get("uncertainty_ns")
    validated = (
        trace.clock_aligned
        and capture.get("coverage_validated") is True
        and capture.get("target_events_verified") is True
        and capture.get("loss_stats_verified") is True
        and type(capture.get("lost_events")) is int
        and capture.get("lost_events") == 0
        and type(uncertainty) is int
        and 0 <= uncertainty <= 100_000
    )
    enrolled = capture.get("validated_threads")
    identities = (
        {(row.get("pid"), row.get("tid")) for row in enrolled}
        if isinstance(enrolled, list)
        and capture.get("thread_lifetimes_verified") is True
        and all(isinstance(row, dict) for row in enrolled)
        else {(capture.get("pid"), capture.get("tid"))}
        if "thread_mappings" not in capture
        else set()
    )
    covered = identities & threads if validated else set()
    missing = threads - covered

    correlation = correlation_audit(trace)
    summary = summarize(trace)
    bounds = None
    if (
        validated
        and covered
        and summary.gpu_idle_fraction is not None
        and correlation["cross_layer_timing_eligible"]
    ):
        runnable = [
            event
            for event in events
            if event.kind == "cpu_sched"
            and (event.pid, event.tid) in covered
            and event.attributes.get("state") == "runnable"
            and event.attributes.get("alignment_validated") is True
        ]

        def overlap(delta):
            intervals = []
            for event in runnable:
                left = max(window.start_ns, event.start_ns - delta)
                right = min(window.end_ns, event.end_ns + delta)
                for a, b in summary.gpu_idle_intervals_ns:
                    if min(right, b) > max(left, a):
                        intervals.append((max(left, a), min(right, b)))
            return duration(intervals) / 1e6

        bounds = {
            "nominal_ms": overlap(0),
            "lower_ms": overlap(-uncertainty),
            "upper_ms": overlap(uncertainty),
            "timestamp_uncertainty_ns": uncertainty,
            "interval_duration_uncertainty_bound_ns": 2 * uncertainty,
            "meaning": "Conditional bounds on covered-thread overlap; not a confidence interval.",
        }
    return {
        "version": "observability-v1",
        "input_confidence": trace.provenance.confidence,
        "gpu_trace_completeness": trace.trace_completeness,
        "scheduling": {
            "collection_validated": validated,
            "observed_cuda_api_threads": len(threads),
            "api_events_with_unknown_thread": unknown_thread_events,
            "covered_cuda_api_threads": len(covered),
            "uncovered_or_unverified_cuda_api_threads": len(missing),
            "all_observed_api_threads_covered": bool(threads)
            and not missing
            and not unknown_thread_events,
            "scope": "Validated enrolled threads; CUDA API threads bound the submission scope.",
        },
        "api_gpu_correlation": correlation,
        "covered_thread_idle_overlap_bounds": bounds,
        "limitations": [
            "Overlap bounds assume GPU timestamps and an affine scheduler/NVTX clock mapping.",
            "NVTX anchors do not independently bound GPU/CPU conversion error.",
            "Bounds exclude unobserved threads, lost telemetry and profiler perturbation.",
            "Aligned overlap and submission links do not establish a sole scheduling cause.",
        ],
    }
