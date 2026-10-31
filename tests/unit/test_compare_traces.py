import pytest

from gpu_os_copilot.analysis.compare_traces import compare_profiled_traces
from gpu_os_copilot.schemas import Provenance


def paired_test_traces(load_fixture):
    # Synthetic software test objects, exercising the measured-evidence contract.
    before = load_fixture()
    for event in before.events:
        event.source = "nsys_sqlite"
        event.confidence = "measured"
    before.provenance = Provenance(
        source="nsys_sqlite", confidence="measured", description="Unit test"
    )
    before.configuration = {
        "scenario": "test",
        "capture": {"cpu_set": [0], "contender_cpu": 0, "cuda_trace_mode": "cuda-sw"},
        "scheduler_capture": {
            "coverage_validated": True,
            "target_events_verified": True,
            "loss_stats_verified": True,
            "lost_events": 0,
        },
        "scheduler_alignment": {"uncertainty_ns": 1000},
    }
    after = before.model_copy(deep=True)
    after.configuration["capture"]["cpu_set"] = [2]
    for event in after.events:
        if event.kind == "cpu_sched":
            event.end_ns = event.start_ns + 1000000
    return before, after


def test_profiled_comparison_preserves_units_and_limits(load_fixture):
    before, after = paired_test_traces(load_fixture)
    result = compare_profiled_traces(before, after)
    assert result["changes"]["cpu_runnable_delay_ms"] == -5
    assert result["changes"]["gpu_idle_percentage_points"] == 0
    assert "profiler overhead" in result["limitations"][0]


@pytest.mark.parametrize(
    "mismatch",
    [
        "workload",
        "instrumentation",
        "coverage",
        "placement",
        "no_events",
        "clock",
        "gpu_clock_order",
        "boolean_loss",
        "enrollment_scope",
    ],
)
def test_incomparable_captures_do_not_establish_before_after(load_fixture, mismatch):
    before, after = paired_test_traces(load_fixture)
    if mismatch == "workload":
        after.configuration["scenario"] = "other"
    elif mismatch == "instrumentation":
        after.configuration["capture"]["cuda_trace_mode"] = "cuda"
    elif mismatch == "coverage":
        after.configuration["scheduler_capture"]["coverage_validated"] = False
    elif mismatch == "no_events":
        after.configuration["scheduler_capture"].pop("target_events_verified")
    elif mismatch == "clock":
        after.configuration["scheduler_alignment"]["method"] = "different_calibration"
    elif mismatch == "gpu_clock_order":
        api = next(e for e in after.events if e.kind == "cuda_api")
        api.end_ns = 200_000
        api.start_ns = 150_000
    elif mismatch == "boolean_loss":
        after.configuration["scheduler_capture"]["lost_events"] = False
    elif mismatch == "enrollment_scope":
        after.configuration["scheduler_capture"]["enrollment_scope"] = "owned_threads_after_warmup"
    else:
        after.configuration["capture"]["cpu_set"] = [0]
    with pytest.raises(ValueError):
        compare_profiled_traces(before, after)
