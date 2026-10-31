import pytest

from gpu_os_copilot.analysis.observability import observability


def validated_metadata(trace):
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


def test_untraced_additional_api_thread_is_explicit(load_fixture):
    trace = load_fixture()
    validated_metadata(trace)
    api = next(e for e in trace.events if e.kind == "cuda_api")
    extra = api.model_copy(update={"id": "another-thread", "tid": api.tid + 1})
    trace.events.append(extra)
    report = observability(trace)
    scope = report["scheduling"]
    assert scope["observed_cuda_api_threads"] == 2
    assert scope["covered_cuda_api_threads"] == 1
    assert scope["uncovered_or_unverified_cuda_api_threads"] == 1
    assert not scope["all_observed_api_threads_covered"]


@pytest.mark.parametrize(
    "missing",
    ["coverage_validated", "target_events_verified", "loss_stats_verified", "lost_events"],
)
def test_missing_collection_proof_does_not_produce_zero_overlap(load_fixture, missing):
    trace = load_fixture()
    validated_metadata(trace)
    trace.configuration["scheduler_capture"].pop(missing)
    report = observability(trace)
    assert not report["scheduling"]["collection_validated"]
    assert report["covered_thread_idle_overlap_bounds"] is None


def test_clock_uncertainty_propagates_to_overlap_bounds(load_fixture):
    trace = load_fixture()
    validated_metadata(trace)
    report = observability(trace)
    bounds = report["covered_thread_idle_overlap_bounds"]
    assert bounds["lower_ms"] <= bounds["nominal_ms"] <= bounds["upper_ms"]
    assert bounds["interval_duration_uncertainty_bound_ns"] == 2000
    trace.configuration["scheduler_alignment"]["uncertainty_ns"] = 100_001
    assert observability(trace)["covered_thread_idle_overlap_bounds"] is None


def test_partial_gpu_coverage_suppresses_overlap_bounds(load_fixture):
    trace = load_fixture()
    validated_metadata(trace)
    trace.trace_completeness = "partial"
    assert observability(trace)["covered_thread_idle_overlap_bounds"] is None


def test_correlation_reasons_account_for_every_selected_gpu_activity(load_fixture):
    trace = load_fixture()
    report = observability(trace)["api_gpu_correlation"]
    assert report["unambiguously_linked_activities"] > 0
    assert (
        report["unambiguously_linked_activities"] + sum(report["unlinked_reasons"].values())
        == report["observed_gpu_activities"]
    )
    api = next(e for e in trace.events if e.kind == "cuda_api")
    trace.events.append(api.model_copy(update={"id": "ambiguous-api"}))
    report = observability(trace)["api_gpu_correlation"]
    assert report["unlinked_reasons"]["ambiguous_api"] > 0
    for e in trace.events:
        if e.kind == "gpu_kernel":
            e.correlation_id = None
    assert (
        observability(trace)["api_gpu_correlation"]["unlinked_reasons"][
            "missing_identity_or_correlation"
        ]
        > 0
    )


def test_no_gpu_is_unknown_correlation_fraction_not_zero(load_fixture):
    trace = load_fixture()
    trace.events = [e for e in trace.events if e.kind not in {"gpu_kernel", "memcpy", "memset"}]
    report = observability(trace)["api_gpu_correlation"]
    assert report["observed_gpu_activities"] == 0 and report["linked_fraction"] is None


def test_unknown_api_thread_prevents_full_scope_claim(load_fixture):
    trace = load_fixture()
    validated_metadata(trace)
    next(e for e in trace.events if e.kind == "cuda_api").tid = None
    report = observability(trace)["scheduling"]
    assert report["api_events_with_unknown_thread"] == 1
    assert not report["all_observed_api_threads_covered"]


@pytest.mark.parametrize("bad", ["foreign_thread", "boolean_loss"])
def test_unmatched_or_invalid_metadata_cannot_report_zero_delay(load_fixture, bad):
    trace = load_fixture()
    validated_metadata(trace)
    if bad == "foreign_thread":
        trace.configuration["scheduler_capture"].update(pid=999, tid=999)
    else:
        trace.configuration["scheduler_capture"]["lost_events"] = False
    report = observability(trace)
    assert report["covered_thread_idle_overlap_bounds"] is None
    assert not report["scheduling"]["all_observed_api_threads_covered"]


def test_gpu_api_time_reversal_disables_cross_layer_overlap_only(load_fixture):
    from gpu_os_copilot.analysis.metrics import summarize

    trace = load_fixture()
    validated_metadata(trace)
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    api = next(
        e for e in trace.events if e.kind == "cuda_api" and e.correlation_id == gpu.correlation_id
    )
    api.end_ns = gpu.start_ns + 27_549
    api.start_ns = gpu.start_ns + 22_549
    report = observability(trace)
    check = report["api_gpu_correlation"]
    assert check["timestamp_order_status"] == "contradicted"
    assert check["unlinked_reasons"]["gpu_precedes_api"] == 1
    assert check["observed_reversal_ns"] == {"min": 22_549, "median": 22_549, "max": 22_549}
    assert check["gpu_cpu_clock_error_bound_ns"] is None
    assert report["covered_thread_idle_overlap_bounds"] is None
    assert report["scheduling"]["collection_validated"]
    summary = summarize(trace)
    assert summary.sched_idle_overlap_ms is None
    assert summary.cpu_runnable_delay_ms == 6
    assert summary.gpu_idle_fraction == pytest.approx(0.81)


def test_late_api_return_is_not_a_start_time_reversal(load_fixture):
    trace = load_fixture()
    api = next(e for e in trace.events if e.kind == "cuda_api")
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    api.end_ns = gpu.end_ns + 1
    check = observability(trace)["api_gpu_correlation"]
    assert check["timestamp_order_status"] == "no_violations_observed"
    assert check["cross_layer_timing_eligible"]


def test_outside_device_or_window_reversals_do_not_poison_selected_window(load_fixture):
    trace = load_fixture()
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    trace.events += [
        gpu.model_copy(update={"id": "other-device", "device_id": 1, "start_ns": 0}),
        gpu.model_copy(update={"id": "out-of-window", "start_ns": 0, "end_ns": 1}),
    ]
    trace.window.start_ns = 1
    check = observability(trace)["api_gpu_correlation"]
    assert check["timestamp_order_status"] == "no_violations_observed"
    assert check["observed_gpu_activities"] == 2


@pytest.mark.parametrize("kind", ["memcpy", "memset"])
def test_memory_operations_also_screened_for_time_reversal(load_fixture, kind):
    trace = load_fixture()
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    api = next(e for e in trace.events if e.kind == "cuda_api")
    gpu.kind = kind
    api.start_ns = 1
    gpu.start_ns = api.start_ns - 1
    assert observability(trace)["api_gpu_correlation"]["unlinked_reasons"]["gpu_precedes_api"] == 1


def test_unmapped_sources_are_not_mislabeled_as_measured_reversals(load_fixture):
    trace = load_fixture()
    trace.clock_aligned = False
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    gpu.source = "nsys_sqlite"
    check = observability(trace)["api_gpu_correlation"]
    assert check["unlinked_reasons"]["unaligned_source"] == 1
    assert check["observed_reversal_ns"] is None
    assert check["timestamp_order_status"] == "incomplete_check"
