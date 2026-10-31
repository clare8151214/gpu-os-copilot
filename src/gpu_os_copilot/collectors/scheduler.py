"""Validate a monotonic-clock bridge and reconstruct measured runnable intervals."""

import re
import sqlite3
from decimal import Decimal
from fractions import Fraction
from math import ceil
from pathlib import Path

from gpu_os_copilot.schemas import TraceBundle, TraceEvent

LINE = re.compile(r"\s(\d+\.\d+):\s+(sched_switch|sched_wakeup_new|sched_wakeup):\s+(.*)$")


def align_clocks(database: Path, anchors: list[dict], window, *, clock: str = "mono") -> dict:
    if clock not in {"mono", "mono_raw"} or any(
        anchor.get("clock", "mono") != clock for anchor in anchors
    ):
        raise ValueError("Clock anchors and scheduler use different clock domains")
    if len(anchors) not in {2, 5} or len({a["name"] for a in anchors}) != len(anchors):
        raise ValueError("Two or five distinct bracketing clock anchors required")
    bounds, timestamps = [], []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        for anchor in anchors:
            rows = db.execute(
                "SELECT start FROM NVTX_EVENTS WHERE text=? LIMIT 2", (anchor["name"],)
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("Clock anchor missing or ambiguous")
            before, after = anchor["before_ns"], anchor["after_ns"]
            if type(before) is not int or type(after) is not int or not 0 <= before <= after:
                raise ValueError("Invalid monotonic clock bracket")
            if after - before > 100000:
                raise ValueError("Clock anchor uncertainty exceeds 100 microseconds")
            timestamp = rows[0][0]
            timestamps.append(timestamp)
            bounds.append((timestamp - after, timestamp - before))
    if not timestamps[0] <= window.start_ns < window.end_ns <= timestamps[-1]:
        raise ValueError("Clock anchors must bracket the measured window")
    if len(anchors) == 5:
        return affine_alignment(anchors, timestamps, clock)
    low, high = max(b[0] for b in bounds), min(b[1] for b in bounds)
    if high < low:
        raise ValueError("Clock offset inconsistent between anchors")
    return {
        "clock": clock,
        "offset_ns": (low + high) // 2,
        "uncertainty_ns": (high - low + 1) // 2 + 1000,
        "anchors": anchors,
        "nsys_anchor_ns": timestamps,
        "method": "Intersection of two NVTX call brackets; ftrace text precision bounded by 1 us",
    }


def affine_alignment(anchors: list[dict], timestamps: list[int], clock: str) -> dict:
    """Bound all positive affine maps consistent with five NVTX call brackets."""
    # A bounded clock-rate calibration, never a fit to just two endpoint samples.
    lower, upper = Fraction(95, 100), Fraction(105, 100)
    for i, first in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            last = anchors[j]
            if first["after_ns"] >= last["before_ns"] or timestamps[j] <= timestamps[i]:
                raise ValueError("Clock anchors must be strictly ordered")
            delta = timestamps[j] - timestamps[i]
            lower = max(lower, Fraction(delta, last["after_ns"] - first["before_ns"]))
            upper = min(upper, Fraction(delta, last["before_ns"] - first["after_ns"]))
    if lower > upper:
        raise ValueError("No consistent bounded affine clock mapping across all five anchors")
    slope = (lower + upper) / 2
    reference = (anchors[0]["before_ns"] + anchors[0]["after_ns"]) // 2
    lows = [
        t - slope * (a["after_ns"] - reference) for a, t in zip(anchors, timestamps, strict=True)
    ]
    highs = [
        t - slope * (a["before_ns"] - reference) for a, t in zip(anchors, timestamps, strict=True)
    ]
    if max(lows) > min(highs):
        raise ValueError("Affine clock intercept intervals do not intersect")
    target = int((max(lows) + min(highs)) / 2)
    # Bound any feasible rate/intercept over the whole calibration span, plus
    # ftrace's microsecond text precision and integer conversion rounding.
    first_radius = max(anchors[0]["after_ns"] - reference, reference - anchors[0]["before_ns"])
    span = anchors[-1]["after_ns"] - anchors[0]["before_ns"]
    uncertainty = (
        ceil(
            abs(target - timestamps[0])
            + upper * first_radius
            + (upper - lower) / 2 * span
            + upper * 1000
        )
        + 1
    )
    if uncertainty > 100001:
        raise ValueError("Affine clock uncertainty exceeds 100 microseconds")
    return {
        "clock": clock,
        "method": "bounded_affine_nvtx_brackets",
        "reference_clock_ns": reference,
        "reference_nsys_ns": target,
        "slope_numerator": slope.numerator,
        "slope_denominator": slope.denominator,
        "rate_ratio": float(slope),
        "rate_ratio_bounds": [float(lower), float(upper)],
        "uncertainty_ns": uncertainty,
        "anchors": anchors,
        "nsys_anchor_ns": timestamps,
        "limitation": "Assumes an affine clock relation between the five verified samples.",
    }


def map_clock_timestamp(timestamp: int, alignment: dict) -> int:
    if alignment.get("method") == "bounded_affine_nvtx_brackets":
        return alignment["reference_nsys_ns"] + (
            (timestamp - alignment["reference_clock_ns"])
            * alignment["slope_numerator"]
            // alignment["slope_denominator"]
        )
    return timestamp + alignment["offset_ns"]


def scheduler_records(text: str, offset_ns: int, alignment: dict | None = None) -> list[tuple]:
    if "LOST" in text or "EVENTS DROPPED" in text:
        raise ValueError("Scheduler events lost")
    records = []
    for line in text.splitlines():
        match = LINE.search(line)
        if match:
            stamp, kind, fields = match.groups()
            raw_ns = int(Decimal(stamp) * 1_000_000_000)
            ns = map_clock_timestamp(raw_ns, alignment) if alignment else raw_ns + offset_ns
            records.append((ns, kind, dict(re.findall(r"(\w+)=([^\s]+)", fields))))
        elif "sched_switch:" in line or "sched_wakeup:" in line or "sched_wakeup_new:" in line:
            raise ValueError("Unrecognized scheduler event format")
    return sorted(records, key=lambda r: r[0])


def verify_target_events(records: list[tuple], kernel_tid: int) -> dict:
    counts = {"switch_in": 0, "switch_out": 0, "wakeup": 0}
    for _, kind, fields in records:
        if kind in {"sched_wakeup", "sched_wakeup_new"}:
            if int(fields["pid"]) != kernel_tid:
                raise ValueError("Scheduler event outside the verified thread filter")
            counts["wakeup"] += 1
        else:
            previous, following = int(fields["prev_pid"]), int(fields["next_pid"])
            if kernel_tid not in (previous, following):
                raise ValueError("Scheduler event outside the verified thread filter")
            counts["switch_in"] += following == kernel_tid
            counts["switch_out"] += previous == kernel_tid
    if not all(counts.values()):
        raise ValueError("Positive scheduler collection evidence unavailable")
    return counts


def verify_enrolled_events(records: list[tuple], kernel_tids: set[int]) -> dict:
    """Every event must match the enrolled set; each thread retains its own evidence counts."""
    counts = {tid: {"switch_in": 0, "switch_out": 0, "wakeup": 0} for tid in kernel_tids}
    for _, kind, fields in records:
        if kind in {"sched_wakeup", "sched_wakeup_new"}:
            target = int(fields["pid"])
            if target not in counts:
                raise ValueError("Scheduler event outside the enrolled thread filters")
            counts[target]["wakeup"] += 1
        else:
            previous, following = int(fields["prev_pid"]), int(fields["next_pid"])
            if not {previous, following} & kernel_tids:
                raise ValueError("Scheduler event outside the enrolled thread filters")
            if previous in counts:
                counts[previous]["switch_out"] += 1
            if following in counts:
                counts[following]["switch_in"] += 1
    return counts


def runnable_intervals(
    text: str,
    *,
    pid: int,
    tid: int,
    offset_ns: int,
    kernel_tid: int | None = None,
    alignment: dict | None = None,
) -> list[TraceEvent]:
    records = scheduler_records(text, offset_ns, alignment)
    target = tid if kernel_tid is None else kernel_tid
    pending, result = None, []
    for ns, kind, fields in records:
        if kind in {"sched_wakeup", "sched_wakeup_new"}:
            if int(fields["pid"]) == target and pending is None:
                pending = ns
        else:
            previous, next_tid = int(fields["prev_pid"]), int(fields["next_pid"])
            if next_tid == target and pending is not None:
                if pending < ns and pending >= 0:
                    result.append(
                        TraceEvent(
                            id=f"sched-{pid}-{tid}-{len(result):05d}",
                            source="linux_trace",
                            kind="cpu_sched",
                            start_ns=pending,
                            end_ns=ns,
                            pid=pid,
                            tid=tid,
                            attributes={
                                "state": "runnable",
                                "alignment_validated": True,
                                "kernel_tid": target,
                            },
                            confidence="measured",
                        )
                    )
                pending = None
            if previous == target and next_tid != target:
                # R/R+ means runnable/preempted. Sleeping time is never counted as runnable.
                pending = ns if fields["prev_state"] in {"R", "R+"} else None
    return result


def merge_scheduler(
    trace: TraceBundle, database: Path, text: str, metadata: dict, anchors: list[dict]
) -> TraceBundle:
    if (
        metadata.get("clock") not in {"mono", "mono_raw"}
        or type(metadata.get("lost_events")) is not int
        or metadata.get("lost_events") != 0
        or metadata.get("loss_stats_verified") is not True
    ):
        raise ValueError("Scheduler clock/collection completeness is not validated")
    pid, tid = metadata["pid"], metadata["tid"]
    if type(pid) is not int or pid <= 0 or tid != pid:
        raise ValueError("Only the verified main launch thread is supported")
    if not any(e.kind == "cuda_api" and (e.pid, e.tid) == (pid, tid) for e in trace.events):
        raise ValueError("Scheduler PID/TID does not match CUDA launch thread")
    kernel_tid = metadata.get("kernel_tid")
    mapping = metadata.get("pid_mapping", {})
    if (
        type(kernel_tid) is not int
        or kernel_tid <= 1
        or mapping.get("namespace_pid") != pid
        or mapping.get("kernel_tid") != kernel_tid
        or mapping.get("method") != "unique_task_comm_in_sched_debug"
    ):
        raise ValueError("Verified namespace-to-kernel PID mapping unavailable")
    alignment = align_clocks(database, anchors, trace.window, clock=metadata["clock"])
    start = metadata.get("capture_start_monotonic_ns")
    end = metadata.get("capture_end_monotonic_ns")
    if (
        type(start) is not int
        or type(end) is not int
        or start > anchors[0]["before_ns"]
        or end < anchors[-1]["after_ns"]
    ):
        raise ValueError("Scheduler capture does not cover the full measurement window")
    records = scheduler_records(text, 0, alignment)
    mappings = metadata.get("thread_mappings")
    extra = {}
    if mappings is None:
        counts = verify_target_events(records, kernel_tid)
        targets = [(tid, kernel_tid)]
    else:
        from gpu_os_copilot.collectors.tracefs_helper import thread_filters

        if (
            not isinstance(mappings, list)
            or not 1 <= len(mappings) <= 32
            or metadata.get("thread_lifetimes_verified") is not True
            or any(
                not isinstance(row, dict)
                or row.get("namespace_pid") != pid
                or type(row.get("namespace_tid")) is not int
                or row["namespace_tid"] <= 1
                or type(row.get("kernel_tid")) is not int
                or row["kernel_tid"] <= 1
                or row.get("method") != "unique_task_comm_in_sched_debug"
                for row in mappings
            )
        ):
            raise ValueError("Verified bounded thread mappings unavailable")
        mapped = {row["namespace_tid"]: row["kernel_tid"] for row in mappings}
        kernel_tids = set(mapped.values())
        if len(mapped) != len(mappings) or len(kernel_tids) != len(mappings):
            raise ValueError("Duplicate thread mappings")
        if mapped.get(pid) != kernel_tid:
            raise ValueError("Main thread mapping differs")
        expected = thread_filters(list(kernel_tids))
        if metadata.get("filters") != expected or metadata.get("event_settings") != {
            name: {"filter": expression, "enable": "1"} for name, expression in expected.items()
        }:
            raise ValueError("Enrolled thread filter readback differs")
        all_counts = verify_enrolled_events(records, kernel_tids)
        counts = all_counts[kernel_tid]
        if not all(counts.values()):
            raise ValueError("Positive scheduler collection evidence unavailable for main thread")
        launch_tids = {
            e.tid
            for e in trace.events
            if e.kind == "cuda_api"
            and e.pid == pid
            and e.tid is not None
            and e.end_ns > trace.window.start_ns
            and e.start_ns < trace.window.end_ns
        }
        targets = [
            (thread, kernel)
            for thread, kernel in mapped.items()
            if thread in launch_tids and all(all_counts[kernel].values())
        ]
        if not targets:
            raise ValueError("No observed CUDA API thread has validated scheduling evidence")
        extra = {
            "validated_threads": [{"pid": pid, "tid": thread} for thread, _ in targets],
            "thread_event_counts": {str(thread): all_counts[k] for thread, k in mapped.items()},
            "uncovered_cuda_api_tids": sorted(launch_tids - {thread for thread, _ in targets}),
        }
    events = [
        event
        for thread, kernel in targets
        for event in runnable_intervals(
            text, pid=pid, tid=thread, offset_ns=0, kernel_tid=kernel, alignment=alignment
        )
    ]
    # Clock calibration applies to the measured range; retain raw events separately.
    events = [
        e for e in events if e.end_ns > trace.window.start_ns and e.start_ns < trace.window.end_ns
    ]
    data = trace.model_dump()
    data["events"].extend(e.model_dump() for e in events)
    data["clock_aligned"] = True
    data["configuration"]["scheduler_alignment"] = alignment
    data["configuration"]["scheduler_capture"] = {
        **metadata,
        "coverage_validated": True,
        "target_events_verified": True,
        "target_event_counts": counts,
        **extra,
    }
    data["provenance"]["limitations"] = [
        s
        for s in data["provenance"]["limitations"]
        if "CPU scheduling normalization/alignment" not in s
    ]
    data["provenance"]["limitations"].append(
        "Scheduling covers validated enrolled CUDA API threads; boundary-open intervals omitted."
        if mappings is not None
        else "Scheduling covers the verified main thread only; boundary-open intervals omitted."
    )
    if "limitation" in alignment:
        data["provenance"]["limitations"].append(alignment["limitation"])
    return TraceBundle.model_validate(data)
