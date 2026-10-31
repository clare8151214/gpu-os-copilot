import sqlite3

import pytest

from gpu_os_copilot.collectors.scheduler import align_clocks, merge_scheduler, runnable_intervals
from gpu_os_copilot.schemas import ObservationWindow


def capture_metadata():
    return {
        "pid": 7,
        "tid": 7,
        "kernel_tid": 107,
        "pid_mapping": {
            "method": "unique_task_comm_in_sched_debug",
            "namespace_pid": 7,
            "kernel_tid": 107,
        },
        "clock": "mono",
        "lost_events": 0,
        "loss_stats_verified": True,
        "capture_start_monotonic_ns": 0,
        "capture_end_monotonic_ns": 20000,
    }


def test_preemption_wakeup_and_sleep_are_distinguished():
    text = """
 process-7 [000] d..2. 10.000000: sched_switch: prev_pid=7 prev_state=R+ ==> next_pid=8
 process-8 [000] d..2. 10.003000: sched_switch: prev_pid=8 prev_state=R ==> next_pid=7
 process-7 [000] d..2. 10.004000: sched_switch: prev_pid=7 prev_state=S ==> next_pid=8
 process-8 [001] d..2. 10.009000: sched_wakeup: comm=process pid=7 prio=120 target_cpu=0
 process-8 [000] d..2. 10.010000: sched_switch: prev_pid=8 prev_state=R ==> next_pid=7
 """
    events = runnable_intervals(text, pid=7, tid=7, offset_ns=-10_000_000_000)
    assert [(e.start_ns, e.end_ns) for e in events] == [(0, 3000000), (9000000, 10000000)]
    assert all(e.attributes["alignment_validated"] for e in events)


def test_lost_and_unknown_scheduler_formats_are_rejected():
    for text in ["CPU 0 LOST 10 EVENTS", "invalid sched_switch: format"]:
        with pytest.raises(ValueError):
            runnable_intervals(text, pid=7, tid=7, offset_ns=0)


def anchors_database(path):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, text TEXT)")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?)", [(19000, "before"), (31000, "after")]
        )
    return [
        dict(name="before", before_ns=1000, after_ns=1200),
        dict(name="after", before_ns=13000, after_ns=13200),
    ]


def test_clock_alignment_uses_brackets_and_rejects_drift(tmp_path):
    path = tmp_path / "clock.sqlite"
    anchors = anchors_database(path)
    window = ObservationWindow(start_ns=20000, end_ns=30000)
    alignment = align_clocks(path, anchors, window)
    assert alignment["offset_ns"] == 17900 and alignment["uncertainty_ns"] == 1100
    anchors[1].update(before_ns=14000, after_ns=14200)
    with pytest.raises(ValueError, match="inconsistent"):
        align_clocks(path, anchors, window)


def test_raw_clock_anchors_require_matching_tracefs_domain(tmp_path):
    path = tmp_path / "clock.sqlite"
    anchors = anchors_database(path)
    for anchor in anchors:
        anchor["clock"] = "mono_raw"
    window = ObservationWindow(start_ns=20000, end_ns=30000)
    with pytest.raises(ValueError, match="different clock domains"):
        align_clocks(path, anchors, window, clock="mono")
    assert align_clocks(path, anchors, window, clock="mono_raw")["offset_ns"] == 17900
    anchors[1]["clock"] = "mono"
    with pytest.raises(ValueError, match="different clock domains"):
        align_clocks(path, anchors, window, clock="mono_raw")


def test_five_anchors_calibrate_rate_and_map_runnable_intervals(tmp_path):
    from gpu_os_copilot.collectors.scheduler import map_clock_timestamp

    path = tmp_path / "affine.sqlite"
    origin = 1_000_000_000_000_000
    anchors, timestamps = [], []
    for i in range(5):
        x = origin + i * 25_000_000
        anchors.append(dict(name=f"a{i}", clock="mono_raw", before_ns=x - 200, after_ns=x + 200))
        timestamps.append(1_000_000 + i * 24_687_500)  # 0.9875 clock rate.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, text TEXT)")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?)",
            zip(timestamps, [a["name"] for a in anchors], strict=True),
        )
    window = ObservationWindow(start_ns=timestamps[0] + 1000, end_ns=timestamps[-1] - 1000)
    result = align_clocks(path, anchors, window, clock="mono_raw")
    assert result["rate_ratio"] == pytest.approx(0.9875, abs=1e-7)
    for anchor, target in zip(anchors, timestamps, strict=True):
        mid = (anchor["before_ns"] + anchor["after_ns"]) // 2
        assert abs(map_clock_timestamp(mid, result) - target) <= result["uncertainty_ns"]
    text = (
        " t-7 [000] d..2. 1000000.010000: sched_switch: prev_pid=7 prev_state=R ==> next_pid=8\n"
        " t-8 [000] d..2. 1000000.013000: sched_switch: prev_pid=8 prev_state=R ==> next_pid=7\n"
    )
    event = runnable_intervals(text, pid=7, tid=7, offset_ns=0, alignment=result)[0]
    assert event.end_ns - event.start_ns == pytest.approx(2_962_500, abs=2)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE NVTX_EVENTS SET start=start+50000 WHERE text='a2'")
    with pytest.raises(ValueError, match="No consistent bounded affine"):
        align_clocks(path, anchors, window, clock="mono_raw")


def test_affine_mapping_rejects_excessive_rate_and_insufficient_samples(tmp_path):
    from gpu_os_copilot.collectors.scheduler import affine_alignment

    anchors = [
        dict(name=f"a{i}", before_ns=i * 1000000, after_ns=i * 1000000 + 10) for i in range(5)
    ]
    with pytest.raises(ValueError, match="No consistent bounded affine"):
        affine_alignment(anchors, [i * 2000000 for i in range(5)], "mono_raw")
    with pytest.raises(ValueError, match="Two or five"):
        align_clocks(
            tmp_path / "unused.sqlite", anchors[:3], ObservationWindow(start_ns=1, end_ns=2)
        )


def test_matching_pid_clock_and_loss_are_required(load_fixture, tmp_path):
    trace = load_fixture()
    path = tmp_path / "clock.sqlite"
    anchors = anchors_database(path)
    metadata = {
        "pid": 999,
        "tid": 999,
        "clock": "mono",
        "lost_events": 0,
        "loss_stats_verified": True,
    }
    with pytest.raises(ValueError, match="PID/TID"):
        merge_scheduler(trace, path, "", metadata, anchors)
    metadata["lost_events"] = 1
    with pytest.raises(ValueError, match="completeness"):
        merge_scheduler(trace, path, "", metadata, anchors)


def test_aligned_scheduler_events_feed_workload_idle_overlap(load_fixture, tmp_path):
    from gpu_os_copilot.analysis.metrics import summarize

    trace = load_fixture()
    trace.window = ObservationWindow(start_ns=20000, end_ns=30000)
    api = next(e for e in trace.events if e.kind == "cuda_api").model_copy(
        update={"pid": 7, "tid": 7, "start_ns": 20000, "end_ns": 21000}
    )
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel").model_copy(
        update={"pid": 7, "start_ns": 26000, "end_ns": 27000}
    )
    trace.events = [api, gpu]
    path = tmp_path / "aligned.sqlite"
    anchors = anchors_database(path)
    text = (
        " t-8 [000] d..2. 0.000000: sched_wakeup: pid=107\n"
        " t-8 [000] d..2. 0.000001: sched_switch: prev_pid=8 prev_state=R ==> next_pid=107\n"
        " t-107 [000] d..2. 0.000005: sched_switch: prev_pid=107 prev_state=R ==> next_pid=8\n"
        " t-8 [000] d..2. 0.000008: sched_switch: prev_pid=8 prev_state=R ==> next_pid=107\n"
    )
    merged = merge_scheduler(
        trace,
        path,
        text,
        capture_metadata(),
        anchors,
    )
    assert merged.clock_aligned
    summary = summarize(merged)
    assert summary.cpu_runnable_delay_ms == pytest.approx(0.003)
    assert summary.sched_idle_overlap_ms == pytest.approx(0.003)
    assert all((e.pid, e.tid) == (7, 7) for e in merged.events if e.kind == "cpu_sched")
    assert merged.configuration["scheduler_capture"]["target_event_counts"] == {
        "switch_in": 2,
        "switch_out": 1,
        "wakeup": 1,
    }
    with pytest.raises(ValueError, match="Positive scheduler"):
        merge_scheduler(trace, path, "# entries-in-buffer: 0", capture_metadata(), anchors)
    with pytest.raises(ValueError, match="outside the verified thread"):
        merge_scheduler(trace, path, text.replace("107", "999"), capture_metadata(), anchors)
    mismatched = capture_metadata()
    mismatched["pid_mapping"]["namespace_pid"] = 8
    with pytest.raises(ValueError, match="PID mapping"):
        merge_scheduler(trace, path, text, mismatched, anchors)
    # An actual sleep transition produces no runnable delay, with positive evidence.
    merged = merge_scheduler(
        trace,
        path,
        text.replace("prev_pid=107 prev_state=R", "prev_pid=107 prev_state=S"),
        capture_metadata(),
        anchors,
    )
    assert summarize(merged).cpu_runnable_delay_ms == 0
    assert summarize(merged).sched_idle_overlap_ms == 0
    # Previously emitted coverage-only metadata cannot turn absent events into zero.
    merged.configuration["scheduler_capture"].pop("target_events_verified")
    assert summarize(merged).cpu_runnable_delay_ms is None
    assert summarize(merged).sched_idle_overlap_ms is None


def test_kernel_identity_uses_unique_comm_not_namespace_pid():
    from gpu_os_copilot.collectors.tracefs_helper import kernel_tid_for_comm

    text = ">Rcopilot1234abcd 1173633 123.000 E 9.000\n S other 7 4.000 E 9.000\n"
    assert kernel_tid_for_comm(text, "copilot1234abcd") == 1173633
    with pytest.raises(ValueError, match="missing or ambiguous"):
        kernel_tid_for_comm(text + text, "copilot1234abcd")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        kernel_tid_for_comm(text, "copilot00000000")
    with pytest.raises(ValueError, match="unique scheduler task name"):
        kernel_tid_for_comm(text, "python")


def test_missing_loss_statistics_cannot_be_reported_as_zero(tmp_path):
    from gpu_os_copilot.collectors.tracefs_helper import read_loss_stats

    with pytest.raises(ValueError, match="unavailable"):
        read_loss_stats(tmp_path)
    cpu = tmp_path / "per_cpu/cpu0"
    cpu.mkdir(parents=True)
    (cpu / "stats").write_text("overrun: 0\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_loss_stats(tmp_path)
    (cpu / "stats").write_text("overrun: 0\ncommit overrun: 0\ndropped events: 0\n")
    assert read_loss_stats(tmp_path) == 0
    (cpu / "stats").write_text("overrun: 2\ncommit overrun: 1\ndropped events: 3\n")
    assert read_loss_stats(tmp_path) == 6
