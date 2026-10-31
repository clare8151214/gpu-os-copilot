"""Read paired profiled CPU scheduling evidence without asserting GPU causality."""

import json
import re
from copy import deepcopy
from pathlib import Path
from statistics import median, stdev

from gpu_os_copilot.analysis.correlate import correlation_audit
from gpu_os_copilot.analysis.metrics import duration
from gpu_os_copilot.collectors.scheduler import affine_alignment
from gpu_os_copilot.schemas import TraceBundle
from gpu_os_copilot.workloads.specs import SCENARIO_SHAPES, expected_configuration

READ_TOOL = "read_profiled_scheduler_comparisons"
MAX_PAIRS = 12
METRICS = {
    "profiled_scheduling_pair_count": "pairs",
    "profiled_scheduling_batch_count": "batches",
    "profiled_cpu_delay_change_median_ms": "ms",
    "profiled_throughput_change_median_fraction": "fraction",
}
SCOPE = "Separate recorded profiled pairs; not the input trace or unprofiled throughput trials."


def workload_key(trace):
    config = trace.configuration
    scenario = config.get("scenario")
    if not isinstance(scenario, str) or scenario not in SCENARIO_SHAPES:
        raise ValueError("Scheduling study requires a known workload")
    expected = expected_configuration(scenario)
    if (
        trace.workload_id != "launch_test"
        or any(
            type(config.get(k)) is not type(v) or config.get(k) != v for k, v in expected.items()
        )
        or any(
            not isinstance(config.get(k), str) or not config[k] for k in ("gpu", "cuda", "torch")
        )
    ):
        raise ValueError("Scheduling study requires a bound workload/environment")
    return deepcopy(
        {
            key: value
            for key, value in config.items()
            if key not in {"capture", "scheduler_capture", "scheduler_alignment"}
        }
    )


def cpu_measurement(trace):
    """Verify canonical CPU evidence and recompute clipped wait, independent of GPU order."""
    capture = trace.configuration.get("scheduler_capture", {})
    alignment = trace.configuration.get("scheduler_alignment", {})
    window = trace.window
    if (
        trace.provenance.confidence != "measured"
        or trace.provenance.source != "nsys_sqlite"
        or trace.trace_completeness != "complete"
        or not trace.clock_aligned
        or trace.configuration.get("profiled") is not True
        or trace.configuration.get("synchronized") is not True
        or trace.wall_time_ms is None
        or trace.operation_count != trace.configuration.get("operations")
        or any(
            capture.get(key) is not True
            for key in (
                "coverage_validated",
                "target_events_verified",
                "loss_stats_verified",
                "thread_lifetimes_verified",
            )
        )
        or type(capture.get("lost_events")) is not int
        or capture["lost_events"] != 0
        or type(capture.get("buffer_kb_per_cpu")) is not int
        or capture["buffer_kb_per_cpu"] <= 0
        or capture.get("enrollment_scope") != "owned_threads_after_warmup"
    ):
        raise ValueError("Complete measured CPU collection evidence required")
    anchors, stamps = alignment.get("anchors", []), alignment.get("nsys_anchor_ns", [])
    if (
        len(anchors) != 5
        or len(stamps) != 5
        or any(type(stamp) is not int or stamp < 0 for stamp in stamps)
        or len({a["name"] for a in anchors}) != 5
        or capture.get("clock") != alignment.get("clock")
        or alignment.get("clock") not in {"mono", "mono_raw"}
        or not stamps[0] <= window.start_ns < window.end_ns <= stamps[-1]
        or any(
            a.get("clock") != alignment["clock"]
            or type(a.get("before_ns")) is not int
            or type(a.get("after_ns")) is not int
            or not 0 <= a["before_ns"] <= a["after_ns"] <= a["before_ns"] + 100000
            for a in anchors
        )
        or capture.get("capture_start_monotonic_ns", anchors[0]["before_ns"] + 1)
        > anchors[0]["before_ns"]
        or capture.get("capture_end_monotonic_ns", anchors[-1]["after_ns"] - 1)
        < anchors[-1]["after_ns"]
    ):
        raise ValueError("Five bracketing CPU anchors required")
    recalculated = affine_alignment(anchors, stamps, alignment["clock"])
    if any(alignment.get(key) != value for key, value in recalculated.items()) or not (
        type(alignment.get("uncertainty_ns")) is int and 0 <= alignment["uncertainty_ns"] <= 100000
    ):
        raise ValueError("CPU clock calibration does not reproduce")
    events = [
        e
        for e in trace.events
        if e.end_ns > window.start_ns
        and e.start_ns < window.end_ns
        and e.device_id in {None, window.device_id}
    ]
    api = [e for e in events if e.kind == "cuda_api"]
    if not api or any(e.pid is None or e.tid is None or e.confidence != "measured" for e in api):
        raise ValueError("Observed CUDA API thread identities required")
    threads = {(e.pid, e.tid) for e in api}
    validated = {(row["pid"], row["tid"]) for row in capture.get("validated_threads", [])}
    mappings = capture.get("thread_mappings", [])
    mapped = {(row["namespace_pid"], row["namespace_tid"]): row for row in mappings}
    if not threads <= validated or len(mapped) != len(mappings) or not 1 <= len(mapped) <= 32:
        raise ValueError("Full observed CUDA API thread coverage required")
    if len({row["kernel_tid"] for row in mappings}) != len(mappings):
        raise ValueError("Duplicate kernel thread mapping")
    for pid, tid in threads:
        row = mapped.get((pid, tid), {})
        counts = capture.get("thread_event_counts", {}).get(str(tid), {})
        if row.get("method") != "unique_task_comm_in_sched_debug" or any(
            type(counts.get(key)) is not int or counts[key] <= 0
            for key in ("switch_in", "switch_out", "wakeup")
        ):
            raise ValueError("Thread mapping or positive collection evidence missing")
    runnable = [e for e in events if e.kind == "cpu_sched"]
    per_thread = {}
    for event in runnable:
        if (
            (event.pid, event.tid) not in threads
            or event.source != "linux_trace"
            or event.confidence != "measured"
            or event.attributes.get("state") != "runnable"
            or event.attributes.get("alignment_validated") is not True
            or event.attributes.get("kernel_tid") != mapped[(event.pid, event.tid)]["kernel_tid"]
        ):
            raise ValueError("Unverified scheduler interval")
        per_thread.setdefault((event.pid, event.tid), []).append(
            (max(window.start_ns, event.start_ns), min(window.end_ns, event.end_ns))
        )
    wait_ms = sum(duration(intervals) for intervals in per_thread.values()) / 1e6
    error_ms = 2 * alignment["uncertainty_ns"] * len(runnable) / 1e6
    order = correlation_audit(trace)
    return {
        "cpu_runnable_delay_ms": wait_ms,
        "conditional_endpoint_error_ms": error_ms,
        "profiled_wall_time_ms": trace.wall_time_ms,
        "profiled_throughput_ops_s": trace.operation_count * 1000 / trace.wall_time_ms,
        "operation_count": trace.operation_count,
        "covered_cuda_api_threads": len(threads),
        "runnable_intervals": len(runnable),
        "cpu_clock_uncertainty_ns": alignment["uncertainty_ns"],
        "gpu_api_timestamp_order": order["timestamp_order_status"],
        "gpu_api_reversal_count": order["unlinked_reasons"]["gpu_precedes_api"],
        "gpu_cpu_clock_error_bound_ns": None,
    }


class SchedulingStudy:
    """Owner-supplied immutable snapshots; outcomes are exposed only by an explicit read."""

    def __init__(
        self,
        reference: TraceBundle,
        pairs: list[tuple[str, TraceBundle | None, TraceBundle | None]],
    ):
        if not 1 <= len(pairs) <= MAX_PAIRS:
            raise ValueError("Scheduling study requires 1 to 12 pairs")
        if reference.provenance.confidence != "measured":
            raise ValueError("Scheduling study requires a measured reference")
        self._reference = workload_key(reference)
        self._pairs = []
        for batch, before, after in pairs:
            if not isinstance(batch, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", batch):
                raise ValueError("Invalid batch identifier")
            self._pairs.append(
                (
                    batch,
                    *(
                        TraceBundle.model_validate(trace.model_dump())
                        if trace is not None
                        else None
                        for trace in (before, after)
                    ),
                )
            )
        self._assessment = None

    def matches(self, trace):
        return workload_key(trace) == self._reference

    def assess(self):
        if self._assessment is not None:
            return deepcopy(self._assessment)
        rows, seen_ids, seen_hashes, batches = [], set(), set(), {}
        invariant, placements = None, None
        for index, (batch, before, after) in enumerate(self._pairs, 1):
            batches.setdefault(batch, len(batches) + 1)
            row = {
                "pair_ref": f"pair-{index}",
                "batch_ref": f"batch-{batches[batch]}",
                "status": "unavailable",
            }
            try:
                if before is None or after is None:
                    raise ValueError("CAPTURE_UNAVAILABLE")
                values = []
                for trace in (before, after):
                    if workload_key(trace) != self._reference:
                        raise ValueError("WORKLOAD_OR_ENVIRONMENT_MISMATCH")
                    cap = trace.configuration["capture"]
                    digest = cap.get("sqlite_sha256", "")
                    if not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("CAPTURE_DIGEST_MISSING")
                    if trace.run_id in seen_ids or digest in seen_hashes:
                        raise ValueError("REUSED_CAPTURE")
                    seen_ids.add(trace.run_id)
                    seen_hashes.add(digest)
                    scheduler = trace.configuration["scheduler_capture"]
                    settings = {
                        "capture": {
                            k: v for k, v in cap.items() if k not in {"cpu_set", "sqlite_sha256"}
                        },
                        "cpu_clock": scheduler.get("clock"),
                        "buffer_kb_per_cpu": scheduler.get("buffer_kb_per_cpu"),
                        "enrollment_scope": scheduler.get("enrollment_scope"),
                        "device": trace.window.device_id,
                        "timebase": trace.timebase,
                    }
                    if invariant is None:
                        invariant = settings
                    elif settings != invariant:
                        raise ValueError("INSTRUMENTATION_MISMATCH")
                    values.append(cpu_measurement(trace))
                a, b = (trace.configuration["capture"] for trace in (before, after))
                if (
                    type(a.get("contender_cpu")) is not int
                    or a.get("cpu_set") != [a["contender_cpu"]]
                    or len(b.get("cpu_set", [])) != 1
                    or b["cpu_set"] == a["cpu_set"]
                ):
                    raise ValueError("UNCONTROLLED_PLACEMENT")
                current_placements = (a["cpu_set"], b["cpu_set"])
                if placements is None:
                    placements = current_placements
                elif placements != current_placements:
                    raise ValueError("PLACEMENT_DIFFERS_BETWEEN_PAIRS")
                if values[0]["covered_cuda_api_threads"] != values[1]["covered_cuda_api_threads"]:
                    raise ValueError("OBSERVED_THREAD_SCOPE_DIFFERS")
                delta = values[1]["cpu_runnable_delay_ms"] - values[0]["cpu_runnable_delay_ms"]
                error = sum(value["conditional_endpoint_error_ms"] for value in values)
                row.update(
                    status="ok",
                    baseline=values[0],
                    treatment=values[1],
                    cpu_runnable_delay_change_ms=delta,
                    conditional_delta_bounds_ms=[delta - error, delta + error],
                    profiled_throughput_change_fraction=before.wall_time_ms / after.wall_time_ms
                    - 1,
                )
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                # Never pass arbitrary source strings/private labels to a model as errors.
                code = str(exc)
                row["reason_code"] = (
                    code if re.fullmatch(r"[A-Z_]{1,64}", code) else "INVALID_CPU_EVIDENCE"
                )
            rows.append(row)
        valid = [row for row in rows if row["status"] == "ok"]
        complete = len(valid) == len(rows)
        deltas = [row["cpu_runnable_delay_change_ms"] for row in valid] if complete else []
        effects = [row["profiled_throughput_change_fraction"] for row in valid] if complete else []
        repeated = complete and len(valid) >= 5 and len(batches) >= 3
        direction = "inconclusive"
        if repeated and all(row["conditional_delta_bounds_ms"][1] < 0 for row in valid):
            direction = "consistent_decrease"
        elif repeated and all(row["conditional_delta_bounds_ms"][0] > 0 for row in valid):
            direction = "consistent_increase"
        self._assessment = {
            "version": "profiled-scheduling-study-v1",
            "status": "complete" if complete else "incomplete",
            "scope": SCOPE,
            "pairs": rows,
            "pair_count": len(rows),
            "batch_count": len(batches),
            "valid_pair_count": len(valid),
            "cpu_delay_change_median_ms": median(deltas) if deltas else None,
            "cpu_delay_change_range_ms": [min(deltas), max(deltas)] if deltas else None,
            "cpu_delay_change_sample_sd_ms": stdev(deltas) if len(deltas) > 1 else None,
            "profiled_throughput_change_median_fraction": median(effects) if effects else None,
            "profiled_throughput_change_range_fraction": [min(effects), max(effects)]
            if effects
            else None,
            "repeated_evidence_threshold_met": repeated,
            "cpu_delay_direction": direction,
            "policy": {"minimum_pairs": 5, "minimum_batches": 3, "require_all_pairs_valid": True},
            "statistical_support": "not_evaluated",
            "gpu_idle_causality": "not_established",
            "limitations": [
                "Measured collector JSON is the input contract; retain raw provenance separately.",
                "CPU wait sums covered CUDA API threads; boundary-open intervals are omitted.",
                "Endpoint bounds assume the verified affine CPU clock map, "
                "not a confidence interval.",
                "GPU temporal ordering does not independently calibrate GPU/CPU clocks.",
                "Profiled wall time includes instrumentation overhead; "
                "do not pool with unprofiled trials.",
                "Matching recorded software/GPU configuration "
                "does not prove identical host conditions.",
                "Batch labels do not establish statistical independence or host generalization.",
                "Co-improvement cannot isolate scheduling mediation "
                "or establish a sole GPU stall cause.",
            ],
        }
        return deepcopy(self._assessment)

    def metrics(self, *, consumed: bool):
        report = self.assess() if consumed else None
        values = (
            dict(
                zip(
                    METRICS,
                    (
                        report["pair_count"],
                        report["batch_count"],
                        report["cpu_delay_change_median_ms"],
                        report["profiled_throughput_change_median_fraction"],
                    ),
                    strict=True,
                )
            )
            if report is not None
            else dict.fromkeys(METRICS)
        )
        return {
            name: {
                "value": values[name],
                "available": values[name] is not None,
                "unit": unit,
                "scope": SCOPE,
            }
            for name, unit in METRICS.items()
        }


def load_scheduling_study(path: Path, reference: TraceBundle) -> SchedulingStudy:
    """Owner-selected manifest; model tools never accept filesystem paths."""
    if path.stat().st_size > 65536:
        raise ValueError("Scheduling manifest exceeds 64 KiB")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"pairs"} or not isinstance(data["pairs"], list):
        raise ValueError("Expected a pairs manifest")
    if not 1 <= len(data["pairs"]) <= MAX_PAIRS:
        raise ValueError("Scheduling study requires 1 to 12 pairs")
    root, total, pairs = path.resolve().parent, 0, []
    for item in data["pairs"]:
        if not isinstance(item, dict) or set(item) != {"batch", "baseline", "treatment"}:
            raise ValueError("Invalid pair manifest entry")
        traces = []
        for arm in ("baseline", "treatment"):
            if item[arm] is None:
                traces.append(None)
                continue
            if not isinstance(item[arm], str) or Path(item[arm]).is_absolute():
                raise ValueError("Capture paths must be relative to the manifest")
            source = (root / item[arm]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Capture path escapes the manifest directory")
            size = source.stat().st_size
            total += size
            if size > 32 * 1024**2 or total > 128 * 1024**2:
                raise ValueError("Scheduling trace size budget exceeded")
            traces.append(TraceBundle.model_validate_json(source.read_text()))
        pairs.append((item["batch"], *traces))
    return SchedulingStudy(reference, pairs)
