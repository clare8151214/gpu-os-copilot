"""Use only unambiguous, process-scoped CUPTI links on an aligned trace timebase."""

from statistics import median

from gpu_os_copilot.schemas import GPU_ACTIVITY_KINDS, TraceBundle


def correlate(trace: TraceBundle) -> list[dict]:
    api = {}
    for event in trace.events:
        if event.kind == "cuda_api" and event.pid is not None and event.correlation_id is not None:
            api.setdefault(
                (event.attributes.get("vm_id"), event.pid, event.correlation_id), []
            ).append(event)
    links = []
    for event in trace.events:
        if event.kind not in GPU_ACTIVITY_KINDS or event.pid is None:
            continue
        candidates = api.get((event.attributes.get("vm_id"), event.pid, event.correlation_id), [])
        if len(candidates) == 1 and candidates[0].start_ns <= event.start_ns:
            source = candidates[0]
            if source.source != event.source and not trace.clock_aligned:
                continue
            links.append(
                {
                    "api_event_id": source.id,
                    "gpu_event_id": event.id,
                    "launch_to_activity_ns": event.start_ns - source.start_ns,
                    "meaning": "Correlated submission delay; not CPU scheduling latency",
                }
            )
    return links


def correlation_audit(trace: TraceBundle) -> dict:
    """Check identity and temporal order; never estimate a clock offset from violations."""
    window = trace.window
    api = {}
    for event in trace.events:
        if event.kind == "cuda_api" and event.pid is not None and event.correlation_id is not None:
            api.setdefault(
                (event.attributes.get("vm_id"), event.pid, event.correlation_id), []
            ).append(event)
    gpu = [
        e
        for e in trace.events
        if e.kind in GPU_ACTIVITY_KINDS
        and e.device_id == window.device_id
        and e.end_ns > window.start_ns
        and e.start_ns < window.end_ns
    ]
    reasons = dict.fromkeys(
        (
            "missing_identity_or_correlation",
            "no_matching_api",
            "ambiguous_api",
            "unaligned_source",
            "gpu_precedes_api",
        ),
        0,
    )
    inversions = []
    linked = 0
    for event in gpu:
        candidates = api.get((event.attributes.get("vm_id"), event.pid, event.correlation_id), [])
        if event.pid is None or event.correlation_id is None:
            reason = "missing_identity_or_correlation"
        elif not candidates:
            reason = "no_matching_api"
        elif len(candidates) != 1:
            reason = "ambiguous_api"
        elif candidates[0].source != event.source and not trace.clock_aligned:
            reason = "unaligned_source"
        elif candidates[0].start_ns > event.start_ns:
            reason = "gpu_precedes_api"
            inversions.append(candidates[0].start_ns - event.start_ns)
        else:
            linked += 1
            continue
        reasons[reason] += 1
    eligible = bool(gpu) and linked == len(gpu)
    return {
        "observed_gpu_activities": len(gpu),
        "unambiguously_linked_activities": linked,
        "linked_fraction": linked / len(gpu) if gpu else None,
        "unlinked_reasons": reasons,
        "timestamp_order_status": "contradicted"
        if inversions
        else "no_violations_observed"
        if eligible
        else "incomplete_check",
        "observed_reversal_ns": {
            "min": min(inversions),
            "median": median(inversions),
            "max": max(inversions),
        }
        if inversions
        else None,
        "cross_layer_timing_eligible": eligible,
        "gpu_cpu_clock_error_bound_ns": None,
        "scope": "GPU activities in the window; their preceding API may start earlier.",
        "meaning": (
            "Temporal-order screening, not independent GPU/CPU clock calibration. "
            "Observed reversal sizes are not upper bounds on clock error."
        ),
    }
