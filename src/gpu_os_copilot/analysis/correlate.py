"""Use only unambiguous, process-scoped CUPTI links on an aligned trace timebase."""

from gpu_os_copilot.schemas import TraceBundle


def correlate(trace: TraceBundle) -> list[dict]:
    api = {}
    for event in trace.events:
        if event.kind == "cuda_api" and event.pid is not None and event.correlation_id is not None:
            api.setdefault((event.pid, event.correlation_id), []).append(event)
    links = []
    for event in trace.events:
        if event.kind not in {"gpu_kernel", "memcpy"} or event.pid is None:
            continue
        candidates = api.get((event.pid, event.correlation_id), [])
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
