"""Interval unions across streams; missing telemetry never becomes a measurement."""

from bisect import bisect_left
from statistics import median

from gpu_os_copilot.analysis.api_classes import API_CLASSES, classify_api, normalize_api_name
from gpu_os_copilot.analysis.correlate import correlation_audit
from gpu_os_copilot.schemas import GPU_ACTIVITY_KINDS, MetricSummary, TraceBundle


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


def intersection(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    """Overlap of two merged interval lists."""
    total, i, j = 0, 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


SUBMITTING_OR_SYNC = frozenset({"launch", "memcpy", "memset", "sync"})


def host_call_order(trace: TraceBundle) -> dict | None:
    """Top-level CUDA calls per thread in host order; None if any API name is unknown.

    Runtime calls nest driver calls, so each correlated call maps to its top-level call, kept
    as (start, end, name, class). For each top-level call, next_class is the class of the next
    top-level call on the thread that submits GPU work or synchronizes.
    """
    threads: dict[tuple, list[tuple[int, int, str, str]]] = {}
    current_end: dict[tuple, int] = {}
    position: dict[tuple, tuple] = {}
    for e in sorted((e for e in trace.events if e.kind == "cuda_api"), key=lambda e: e.start_ns):
        name = normalize_api_name(e.attributes.get("api_name"))
        if name is None:
            return None
        thread = (e.pid, e.tid)
        calls = threads.setdefault(thread, [])
        if e.end_ns > current_end.get(thread, -1):
            current_end[thread] = e.end_ns
            calls.append((e.start_ns, e.end_ns, name, classify_api(name)))
        if e.correlation_id is not None:
            position[(e.pid, e.correlation_id)] = (thread, len(calls) - 1)
    next_class = {}
    for thread, calls in threads.items():
        following, upcoming = [None] * len(calls), None
        for index in range(len(calls) - 1, -1, -1):
            following[index] = upcoming
            if calls[index][3] in SUBMITTING_OR_SYNC:
                upcoming = calls[index][3]
        next_class[thread] = following
    return {
        "calls": threads,
        "position": position,
        "next_class": next_class,
        "sync_starts": {
            thread: [start for start, _, _, kind in calls if kind == "sync"]
            for thread, calls in threads.items()
        },
    }


HOST_COST_QUANTILE = 0.25
REMOVABLE = ("synchronization", "data_transfer", "launch_overhead")


def host_submission_ns(order: dict, start: int, end: int) -> dict[str, int]:
    """Longest per-thread host submission time in the window, as is and after each removal.

    Async calls block while the device queue is full, so a call costs the lower quartile of
    its API's durations in the window; host time between calls is kept. Synchronize calls cost
    nothing here because their waits are device time. Removing a factor drops its calls with
    the host time leading to each; cheaper submission keeps one launch per back-to-back run of
    launches. Host clock only.
    """
    inside = {
        thread: [
            (index, call) for index, call in enumerate(calls) if start <= call[0] and call[1] <= end
        ]
        for thread, calls in order["calls"].items()
    }
    durations: dict[str, list[int]] = {}
    for calls in inside.values():
        for _, (a, b, name, _) in calls:
            durations.setdefault(name, []).append(b - a)
    cost = {name: sorted(v)[int(HOST_COST_QUANTILE * len(v))] for name, v in durations.items()}

    def longest(drop=lambda kind, following: False, batch=False):
        result = 0
        for thread, calls in inside.items():
            following, busy, previous = order["next_class"][thread], 0, None
            for index, (a, b, name, kind) in calls:
                gap = 0 if previous is None else max(0, a - previous[1])
                repeated = (
                    batch and kind == "launch" and previous is not None and previous[3] == "launch"
                )
                previous = (a, b, name, kind)
                if not (repeated or drop(kind, following[index])):
                    busy += gap + (0 if kind == "sync" else cost[name])
            result = max(result, busy)
        return result

    return {
        "base": longest(),
        "synchronization": longest(
            lambda kind, following: (
                kind == "sync" or (kind in {"memcpy", "memset"} and following == "sync")
            )
        ),
        "data_transfer": longest(
            lambda kind, following: kind in {"memcpy", "memset"} and following != "sync"
        ),
        "launch_overhead": longest(batch=True),
    }


def removal_estimates(trace: TraceBundle, gpu, interval) -> dict | None:
    """Partition GPU-device window time by the factor whose removal would first recover it.

    Busy time: kernels alone, kernel/transfer overlap, and exposed transfers (memcpy or
    memset with no kernel running). A transfer the host waits on before submitting more work
    is part of a synchronize round trip. Interior idle gaps go to the work that ends them:
    behind a synchronize when the next work's submitting thread entered one after the previous
    work was submitted, otherwise to the next work's kind. Gaps before the first or after the
    last work stay unattributed. Correlation IDs link work to calls; host order uses host
    timestamps only and lengths the GPU clock only. None when any work is unlinked.

    The recoverable time per factor is its partition time, capped so the window cannot shrink
    below the host's own submission time after that removal: when the host sets the pace,
    device time freed by removing work stays idle.
    """
    order = host_call_order(trace)
    if order is None:
        return None
    position, next_class = order["position"], order["next_class"]
    if any(e.correlation_id is None or (e.pid, e.correlation_id) not in position for e in gpu):
        return None

    def awaited(e):
        thread, index = position[(e.pid, e.correlation_id)]
        return e.kind != "gpu_kernel" and next_class[thread][index] == "sync"

    def host_start(e):
        thread, index = position[(e.pid, e.correlation_id)]
        return thread, order["calls"][thread][index][0]

    kernels = merge_intervals([interval(e) for e in gpu if e.kind == "gpu_kernel"])
    moves = merge_intervals([interval(e) for e in gpu if e.kind != "gpu_kernel" and not awaited(e)])
    waits = merge_intervals([interval(e) for e in gpu if awaited(e)])
    transfers = merge_intervals(moves + waits)
    parts = {
        "kernel_only": duration(kernels) - intersection(kernels, transfers),
        "overlap": intersection(kernels, transfers),
        "synchronization": duration(waits) - intersection(waits, merge_intervals(kernels + moves)),
        "data_transfer": duration(moves) - intersection(moves, kernels),
        "launch_overhead": 0,
    }
    runs = []  # Merged busy runs: [start, end, first work, work that ends the run].
    for e in sorted(gpu, key=lambda e: (e.start_ns, e.end_ns)):
        a, b = interval(e)
        if a == b:
            continue
        if runs and a <= runs[-1][1]:
            if b > runs[-1][1]:
                runs[-1][1], runs[-1][3] = b, e
        else:
            runs.append([a, b, e, e])
    for previous, following in zip(runs, runs[1:], strict=False):
        _, before = host_start(previous[3])
        thread, after = host_start(following[2])
        starts = order["sync_starts"].get(thread, [])
        index = bisect_left(starts, before)
        behind = index < len(starts) and starts[index] < after
        factor = (
            "synchronization"
            if behind or awaited(following[2])
            else "launch_overhead"
            if following[2].kind == "gpu_kernel"
            else "data_transfer"
        )
        parts[factor] += following[0] - previous[1]
    window = trace.window
    span = window.end_ns - window.start_ns
    host = host_submission_ns(order, window.start_ns, window.end_ns)
    return {
        "partition": parts,
        "recoverable": {f: max(0, min(parts[f], span - host[f])) for f in REMOVABLE},
        "host_submission": host["base"],
    }


def summarize(trace: TraceBundle) -> MetricSummary:
    window = trace.window
    start, end = window.start_ns, window.end_ns
    events = [e for e in trace.events if e.end_ns > start and e.start_ns < end]

    def interval(event):
        return max(start, event.start_ns), min(end, event.end_ns)

    gpu = [e for e in events if e.kind in GPU_ACTIVITY_KINDS and e.device_id == window.device_id]
    kernels = [e for e in gpu if e.kind == "gpu_kernel"]
    transfers = [e for e in gpu if e.kind == "memcpy"]
    busy = merge_intervals([interval(e) for e in gpu])
    kernel_busy = merge_intervals([interval(e) for e in kernels])
    transfer_busy = merge_intervals([interval(e) for e in transfers])
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
    api_names = [normalize_api_name(e.attributes.get("api_name")) for e in api]
    api_classes = {}
    if api and None not in api_names:
        api_classes = {name: [] for name in API_CLASSES}
        # Count only top-level calls per thread: runtime calls nest driver calls.
        current_end = {}
        for e, name in sorted(zip(api, api_names, strict=True), key=lambda p: p[0].start_ns):
            thread = (e.pid, e.tid)
            if e.end_ns <= current_end.get(thread, -1):
                continue
            current_end[thread] = e.end_ns
            api_classes[classify_api(name)].append(interval(e))
    elif api:
        missing.append("CUDA API names unavailable for some events; API class time unavailable")
    removal = removal_estimates(trace, gpu, interval) if reliable_gpu and api_classes else None
    if reliable_gpu and removal is None:
        missing.append(
            "GPU work not fully linked to named CUDA API calls; removal estimates unavailable"
        )
    share = {
        key: value / (end - start) for key, value in (removal or {}).get("partition", {}).items()
    }
    recoverable = {
        key: value / (end - start) for key, value in (removal or {}).get("recoverable", {}).items()
    }
    timing = correlation_audit(trace)
    if not timing["conditional_timing_eligible"]:
        missing.append(
            "API/GPU temporal order is contradicted or incompletely checked; "
            "CPU/GPU overlap is unavailable"
        )
    elif not timing["cross_layer_timing_eligible"]:
        missing.append(
            "Independent GPU/CPU clock error bound unavailable; verified CPU/GPU overlap "
            "is unavailable even though no API/GPU order violation was observed"
        )
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
    sched_capture = trace.configuration.get("scheduler_capture", {})
    scheduler_available = bool(runnable) or (
        trace.clock_aligned
        and sched_capture.get("coverage_validated") is True
        and sched_capture.get("target_events_verified") is True
    )
    if not scheduler_available:
        missing.append("Validated CPU runnable-to-running events unavailable")
    per_thread = {}
    for e in runnable:
        per_thread.setdefault((e.pid, e.tid), []).append(interval(e))
    launch_threads = {(e.pid, e.tid) for e in api if e.pid is not None and e.tid is not None}
    scheduler_matches_launcher = scheduler_available and (
        any(
            (row.get("pid"), row.get("tid")) in launch_threads
            for row in sched_capture.get("validated_threads", [])
            if isinstance(row, dict)
        )
        if "thread_mappings" in sched_capture
        and sched_capture.get("thread_lifetimes_verified") is True
        else (sched_capture.get("pid"), sched_capture.get("tid")) in launch_threads
        if "thread_mappings" not in sched_capture
        else False
    )
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
        cuda_api_class_ms={name: duration(v) / 1e6 for name, v in api_classes.items()},
        cuda_api_class_counts={name: len(v) for name, v in api_classes.items()},
        transfer_duration_ms=transfer_ms,
        transfer_busy_fraction=duration([interval(e) for e in transfers]) / (end - start)
        if gpu
        else None,
        kernel_busy_fraction=duration(kernel_busy) / (end - start) if reliable_gpu else None,
        kernel_transfer_overlap_ms=intersection(kernel_busy, transfer_busy) / 1e6
        if reliable_gpu
        else None,
        removal_estimate_synchronization_fraction=recoverable.get("synchronization"),
        removal_estimate_data_transfer_fraction=recoverable.get("data_transfer"),
        removal_estimate_launch_overhead_fraction=recoverable.get("launch_overhead"),
        kernel_only_busy_fraction=share.get("kernel_only"),
        unattributed_idle_fraction=max(0.0, 1 - sum(share.values())) if share else None,
        host_submission_fraction=min(1.0, removal["host_submission"] / (end - start))
        if removal
        else None,
        cpu_runnable_delay_ms=sum(duration(v) for v in per_thread.values()) / 1e6
        if scheduler_available
        else None,
        sched_idle_overlap_ms=duration(overlaps) / 1e6
        if (matched_runnable or scheduler_matches_launcher)
        and reliable_gpu
        and timing["cross_layer_timing_eligible"]
        else None,
        wall_time_ms=trace.wall_time_ms,
        throughput_ops_s=trace.operation_count * 1000 / trace.wall_time_ms
        if trace.wall_time_ms is not None
        else None,
        missing_data=missing,
    )
