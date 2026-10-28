"""Interval unions across streams; missing telemetry never becomes a measurement."""

from statistics import median

from gpu_os_copilot.schemas import MetricSummary, TraceBundle


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def summarize(trace: TraceBundle) -> MetricSummary:
    window = trace.window
    start, end = window.start_ns, window.end_ns
    events = [e for e in trace.events if e.end_ns > start and e.start_ns < end]

    def interval(event):
        return max(start, event.start_ns), min(end, event.end_ns)

    gpu = [
        e for e in events if e.kind in {"gpu_kernel", "memcpy"} and e.device_id == window.device_id
    ]
    kernels = [e for e in gpu if e.kind == "gpu_kernel"]
    transfers = [e for e in gpu if e.kind == "memcpy"]
    busy = merge_intervals([interval(e) for e in gpu])
    gaps = []
    cursor = start
    for left, right in busy:
        if left > cursor:
            gaps.append((cursor, left))
        cursor = right
    if cursor < end:
        gaps.append((cursor, end))
    reliable_gpu = bool(gpu) and trace.trace_completeness == "complete"
    missing = list(trace.provenance.limitations)
    if not gpu:
        missing.append("GPU events unavailable in selected device/window")
    if trace.trace_completeness != "complete":
        missing.append("Incomplete capture: idle fraction is unavailable")
    api = [e for e in events if e.kind == "cuda_api"]
    if not api:
        missing.append("CUDA API events unavailable")
    runnable = (
        [
            e
            for e in events
            if e.kind == "cpu_sched"
            and e.attributes.get("state") == "runnable"
            and e.attributes.get("alignment_validated") is True
            and e.tid is not None
        ]
        if trace.clock_aligned
        else []
    )
    if not runnable:
        missing.append("Validated CPU runnable-to-running events unavailable")
    per_thread = {}
    for e in runnable:
        per_thread.setdefault((e.pid, e.tid), []).append(interval(e))
    launch_threads = {(e.pid, e.tid) for e in api if e.pid is not None and e.tid is not None}
    matched_runnable = [e for e in runnable if (e.pid, e.tid) in launch_threads]
    overlaps = [
        (max(a, c), min(b, d))
        for e in matched_runnable
        for a, b in [interval(e)]
        for c, d in gaps
        if min(b, d) > max(a, c)
    ]
    kernel_ms = [(b - a) / 1e6 for e in kernels for a, b in [interval(e)]]
    transfer_ms = {}
    for e in transfers:
        direction = e.attributes.get("direction", "unknown")
        if direction not in {"H2D", "D2H", "D2D"}:
            direction = "unknown"
        a, b = interval(e)
        transfer_ms[direction] = transfer_ms.get(direction, 0.0) + (b - a) / 1e6
    return MetricSummary(
        **trace.model_dump(
            include={
                "schema_version",
                "experiment_id",
                "worker_id",
                "workload_id",
                "created_at",
                "provenance",
            }
        ),
        run_id=trace.run_id,
        window=window,
        trace_completeness=trace.trace_completeness,
        gpu_idle_fraction=1 - duration(busy) / (end - start) if reliable_gpu else None,
        gpu_busy_ms=duration(busy) / 1e6 if gpu else None,
        gpu_idle_intervals_ns=gaps if reliable_gpu else [],
        kernel_duration_ms={
            "min": min(kernel_ms),
            "median": median(kernel_ms),
            "max": max(kernel_ms),
            "sum": sum(kernel_ms),
            "count": len(kernel_ms),
        }
        if kernel_ms
        else {},
        cuda_api_duration_ms=sum(b - a for e in api for a, b in [interval(e)]) / 1e6
        if api
        else None,
        transfer_duration_ms=transfer_ms,
        transfer_busy_fraction=duration([interval(e) for e in transfers]) / (end - start)
        if gpu
        else None,
        cpu_runnable_delay_ms=sum(duration(v) for v in per_thread.values()) / 1e6
        if runnable
        else None,
        sched_idle_overlap_ms=duration(overlaps) / 1e6
        if matched_runnable and reliable_gpu
        else None,
        wall_time_ms=trace.wall_time_ms,
        throughput_ops_s=trace.operation_count * 1000 / trace.wall_time_ms
        if trace.wall_time_ms is not None
        else None,
        missing_data=missing,
    )
