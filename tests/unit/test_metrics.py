import random

import pytest

from gpu_os_copilot.analysis.metrics import merge_intervals, summarize
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.schemas import TraceBundle


@pytest.mark.parametrize(
    "scenario,idle,category",
    [
        ("cpu_starvation", 0.81, "cpu_scheduling"),
        ("transfer_bound", 0.06, "data_transfer"),
        ("compute_bound", 0.02, "gpu_compute"),
    ],
)
def test_expected_scenarios_and_negative_controls(load_fixture, scenario, idle, category):
    trace = load_fixture(scenario)
    summary = summarize(trace)
    assert summary.gpu_idle_fraction == pytest.approx(idle)
    assert summary.throughput_ops_s == 2000
    assert diagnose(trace, summary)[0].category == category
    assert summary.provenance.confidence == "simulated"


def test_missing_gpu_is_unknown_not_fully_idle(load_fixture):
    data = load_fixture().model_dump()
    data["events"] = [e for e in data["events"] if e["kind"] != "gpu_kernel"]
    trace = TraceBundle.model_validate(data)
    summary = summarize(trace)
    assert summary.gpu_idle_fraction is None
    assert summary.gpu_busy_ms is None
    assert diagnose(trace, summary)[0].category == "unknown"


def test_incomplete_capture_cannot_establish_idle(load_fixture):
    trace = load_fixture()
    trace.trace_completeness = "partial"
    assert summarize(trace).gpu_idle_fraction is None


def test_overlaps_across_streams_and_other_devices(load_fixture):
    data = load_fixture().model_dump()
    base = data["events"][1]
    data["events"] = [
        dict(base, id="a", start_ns=0, end_ns=6000000, stream_id=1),
        dict(base, id="b", start_ns=4000000, end_ns=8000000, stream_id=2),
        dict(base, id="c", start_ns=0, end_ns=10000000, device_id=1),
    ]
    summary = summarize(TraceBundle.model_validate(data))
    assert summary.gpu_busy_ms == 8
    assert summary.gpu_idle_fraction == pytest.approx(0.2)
    assert summary.kernel_duration_ms["sum"] == 10
    assert summary.gpu_idle_intervals_ns == [(8000000, 10000000)]


def test_window_clipping_and_shuffled_events(load_fixture):
    trace = load_fixture()
    expected = summarize(trace).model_dump()
    random.Random(42).shuffle(trace.events)
    assert summarize(trace).model_dump() == expected
    trace.window.end_ns = 8000000
    assert summarize(trace).gpu_busy_ms == 1.8


@pytest.mark.parametrize("alignment", [False, True])
def test_no_valid_scheduler_events_no_scheduler_claim(load_fixture, alignment):
    trace = load_fixture()
    trace.clock_aligned = alignment
    trace.events = [e for e in trace.events if e.kind != "cpu_sched"]
    summary = summarize(trace)
    assert summary.cpu_runnable_delay_ms is None
    assert diagnose(trace, summary)[0].category == "launch_overhead"


def test_unaligned_scheduler_events_not_used(load_fixture):
    trace = load_fixture()
    trace.clock_aligned = False
    assert summarize(trace).cpu_runnable_delay_ms is None


def test_runnable_delay_and_overlap(load_fixture):
    summary = summarize(load_fixture())
    assert summary.cpu_runnable_delay_ms == 6
    assert summary.sched_idle_overlap_ms == 6


def test_unrelated_thread_cannot_support_launch_starvation(load_fixture):
    trace = load_fixture()
    trace.events[2].tid = 999
    summary = summarize(trace)
    assert summary.sched_idle_overlap_ms is None
    assert diagnose(trace, summary)[0].category == "launch_overhead"


def test_interval_union():
    assert merge_intervals([(4, 8), (1, 5), (8, 9), (3, 3)]) == [(1, 9)]
    assert merge_intervals([]) == []
    with pytest.raises(ValueError):
        merge_intervals([(9, 1)])
