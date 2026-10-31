"""Synthetic counterexamples: an ordered trace is not a calibrated GPU clock."""

import pytest

from gpu_os_copilot.agent.investigator import evidence_catalog, validate_investigation
from gpu_os_copilot.analysis.compare_traces import compare_profiled_traces
from gpu_os_copilot.analysis.correlate import correlate, correlation_audit
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.observability import observability
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.schemas import AgentDecision, Hypothesis, TraceBundle


def measured_contract(load_fixture):
    # Exercise the measured-input contract with synthetic data, never real acquisition.
    data = load_fixture().model_dump()
    data["provenance"].update(source="nsys_sqlite", confidence="measured")
    for event in data["events"]:
        event.update(source="nsys_sqlite", confidence="measured")
    trace = TraceBundle.model_validate(data)
    api = next(e for e in trace.events if e.kind == "cuda_api")
    trace.configuration["scheduler_capture"] = {
        "pid": api.pid,
        "tid": api.tid,
        "coverage_validated": True,
        "target_events_verified": True,
        "loss_stats_verified": True,
        "lost_events": 0,
    }
    trace.configuration["scheduler_alignment"] = {"uncertainty_ns": 1000}
    return trace


def test_ordered_measured_trace_keeps_cpu_gpu_facts_but_not_validated_overlap(load_fixture):
    trace = measured_contract(load_fixture)
    report = observability(trace)
    timing = report["api_gpu_correlation"]
    assert timing["timestamp_order_status"] == "no_violations_observed"
    assert timing["linked_fraction"] == 1
    assert timing["conditional_timing_eligible"]
    assert not timing["cross_layer_timing_eligible"]
    assert timing["gpu_cpu_clock_error_bound_ns"] is None
    assert timing["clock_basis"] == "unverified_gpu_cpu_conversion"
    assert report["scheduling"]["all_observed_api_threads_covered"]
    assert report["covered_thread_idle_overlap_bounds"] is None
    conditional = report["conditional_covered_thread_idle_overlap_bounds"]
    assert conditional["lower_ms"] <= conditional["nominal_ms"] <= conditional["upper_ms"]
    assert conditional["nominal_ms"] == 6
    assert conditional["gpu_cpu_clock_error_bound_ns"] is None
    summary = summarize(trace)
    assert summary.sched_idle_overlap_ms is None
    assert summary.cpu_runnable_delay_ms == 6
    assert summary.gpu_idle_fraction == pytest.approx(0.81)
    assert summary.throughput_ops_s == 2000
    assert any("clock error bound unavailable" in value for value in summary.missing_data)
    assert diagnose(trace, summary)[0].category == "launch_overhead"


@pytest.mark.parametrize("clock_flag", [0, 1, True, "validated", {"error_bound_ns": 0}])
def test_unverified_metadata_cannot_authorize_gpu_clock_claim(load_fixture, clock_flag):
    trace = measured_contract(load_fixture)
    trace.configuration["gpu_cpu_clock_error_bound_ns"] = clock_flag
    trace.configuration["gpu_cpu_alignment"] = clock_flag
    trace.configuration["scheduler_alignment"]["gpu_cpu_clock_error_bound_ns"] = clock_flag
    assert trace.clock_aligned  # Existing flag describes declared/CPU alignment only.
    assert not correlation_audit(trace)["cross_layer_timing_eligible"]
    assert summarize(trace).sched_idle_overlap_ms is None


def test_identity_links_survive_without_promoting_nominal_latency(load_fixture):
    trace = measured_contract(load_fixture)
    links = correlate(trace)
    assert len(links) == 2
    assert all(link["launch_to_activity_ns"] is None for link in links)
    assert all(link["timing_status"] == "conditional" for link in links)
    events = {event.id: event for event in trace.events}
    for link in links:
        assert link["nominal_launch_to_activity_ns"] == (
            events[link["gpu_event_id"]].start_ns - events[link["api_event_id"]].start_ns
        )


def test_fixture_clock_is_explicitly_simulated_not_measured_calibration(load_fixture):
    trace = load_fixture()
    timing = correlation_audit(trace)
    assert timing["clock_basis"] == "simulated_shared_timebase"
    assert timing["gpu_cpu_clock_error_bound_ns"] == 0
    assert timing["cross_layer_timing_eligible"]
    assert summarize(trace).sched_idle_overlap_ms == 6
    assert all(link["timing_status"] == "simulated" for link in correlate(trace))


def test_simulated_bundle_with_nonfixture_clock_cannot_get_zero_error(load_fixture):
    trace = load_fixture()
    next(e for e in trace.events if e.kind == "gpu_kernel").source = "nsys_sqlite"
    assert correlation_audit(trace)["gpu_cpu_clock_error_bound_ns"] is None
    assert summarize(trace).sched_idle_overlap_ms is None


@pytest.mark.parametrize("damage", ["reversal", "ambiguous", "loss", "partial", "unknown_cpu"])
def test_invalid_evidence_withholds_even_conditional_overlap(load_fixture, damage):
    trace = measured_contract(load_fixture)
    api = next(e for e in trace.events if e.kind == "cuda_api")
    if damage == "reversal":
        gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
        api.end_ns = gpu.start_ns + 2
        api.start_ns = gpu.start_ns + 1
    elif damage == "ambiguous":
        trace.events.append(api.model_copy(update={"id": "ambiguous-api"}))
    elif damage == "loss":
        trace.configuration["scheduler_capture"]["lost_events"] = 1
    elif damage == "partial":
        trace.trace_completeness = "partial"
    else:
        trace.configuration["scheduler_capture"]["pid"] = 999
    report = observability(trace)
    assert report["covered_thread_idle_overlap_bounds"] is None
    assert report["conditional_covered_thread_idle_overlap_bounds"] is None


def test_conditional_overlap_cannot_support_a_high_confidence_investigation(load_fixture):
    trace = measured_contract(load_fixture)
    catalog = evidence_catalog(trace)
    first = Hypothesis(
        category="cpu_scheduling",
        statement="Scheduling is a candidate explanation.",
        supporting_metric_refs=["cpu_runnable_delay_ms"],
        missing_metric_refs=["sched_idle_overlap_ms"],
        confidence="high",
        counterevidence_note="GPU/CPU conversion error is unknown.",
        testable_prediction="A scheduling intervention could reduce CPU wait.",
    )
    alternative = first.model_copy(update={"category": "launch_overhead", "confidence": "low"})
    decision = AgentDecision(
        hypotheses=[first, alternative], decision="inconclusive", rationale="Candidates only."
    )
    assert (
        "HIGH_CONFIDENCE_WITH_INCOMPLETE_EVIDENCE"
        in validate_investigation(decision, catalog)["error_codes"]
    )
    first.confidence = "low"
    assert validate_investigation(decision, catalog)["status"] == "valid"
    first.missing_metric_refs = []
    first.supporting_metric_refs = ["sched_idle_overlap_ms"]
    assert (
        "MISSING_METRIC_USED_AS_OBSERVATION"
        in validate_investigation(decision, catalog)["error_codes"]
    )


def test_profiled_pair_separates_cpu_change_from_conditional_overlap(load_fixture):
    before = measured_contract(load_fixture)
    before.configuration["capture"] = {"cpu_set": [0], "contender_cpu": 0}
    after = before.model_copy(deep=True)
    after.configuration["capture"]["cpu_set"] = [2]
    for event in after.events:
        if event.kind == "cpu_sched":
            event.end_ns = event.start_ns + 1_000_000
    report = compare_profiled_traces(before, after)
    assert report["changes"]["cpu_runnable_delay_ms"] == -5
    assert report["changes"]["sched_idle_overlap_ms"] is None
    assert report["conditional_changes"]["sched_idle_overlap_ms"] == -5
    assert all(
        report[condition]["sched_idle_overlap_ms"] is None
        for condition in ("baseline", "treatment")
    )


@pytest.mark.parametrize("identity", [None, 999])
def test_cpu_comparison_still_requires_verified_submission_thread_scope(load_fixture, identity):
    before = measured_contract(load_fixture)
    before.configuration["capture"] = {"cpu_set": [0], "contender_cpu": 0}
    after = before.model_copy(deep=True)
    after.configuration["capture"]["cpu_set"] = [2]
    after.configuration["scheduler_capture"].update(pid=identity, tid=identity)
    with pytest.raises(ValueError, match="scheduler coverage"):
        compare_profiled_traces(before, after)
