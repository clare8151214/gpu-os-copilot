from gpu_os_copilot.analysis.rule_baseline_v4 import experiment_plan, rank_v4

ESTIMATES = {"synchronization": 0.06, "data_transfer": 0.09, "launch_overhead": 0.37}


def test_busy_kernels_without_a_dominant_removal_rank_gpu_compute():
    assert rank_v4(ESTIMATES, 0.5) == "gpu_compute"
    assert rank_v4(ESTIMATES, 0.49) == "launch_overhead"
    assert rank_v4({**ESTIMATES, "launch_overhead": 0.5}, 0.9) == "launch_overhead"


def test_measured_shares_replace_estimates():
    measured = {"data_transfer": 0.42}
    assert rank_v4(ESTIMATES, 0.2, measured) == "data_transfer"
    assert rank_v4(ESTIMATES, 0.2, {"launch_overhead": float("-inf")}) == "data_transfer"


def test_missing_evidence_or_ties_are_unknown():
    assert rank_v4(ESTIMATES, None) == "unknown"
    assert rank_v4({**ESTIMATES, "synchronization": None}, 0.2) == "unknown"
    assert rank_v4({**ESTIMATES, "data_transfer": 0.37}, 0.2) == "unknown"


def test_experiment_plan_takes_the_largest_offered_estimates():
    offered = {"synchronization", "launch_overhead"}
    assert experiment_plan(ESTIMATES, offered) == ["launch_overhead", "synchronization"]
    assert experiment_plan(ESTIMATES, offered, limit=1) == ["launch_overhead"]
    assert experiment_plan(ESTIMATES, set()) == []


def summary_with(trace, **values):
    from gpu_os_copilot.analysis.metrics import summarize

    fields = {f"removal_estimate_{c}_fraction": v for c, v in ESTIMATES.items()}
    return summarize(trace).model_copy(update={**fields, **values})


def test_rules_mode_falls_back_to_v1_where_v4_cannot_rank(load_fixture):
    from gpu_os_copilot.analysis.metrics import summarize
    from gpu_os_copilot.analysis.rule_baseline import diagnose
    from gpu_os_copilot.analysis.rule_baseline_v4 import diagnose_v4, ranking_rule

    for scenario, reason in (
        ("cpu_starvation", "cpu_scheduling_candidate"),
        ("transfer_bound", "removal_estimates_unavailable"),
    ):
        trace = load_fixture(scenario)
        summary = summarize(trace)
        assert ranking_rule(trace, summary) == f"rules_v1_fallback:{reason}"
        assert diagnose_v4(trace, summary) == diagnose(trace, summary)
    trace = load_fixture("compute_bound")
    tied = summary_with(
        trace, kernel_busy_fraction=0.2, removal_estimate_data_transfer_fraction=0.37
    )
    assert ranking_rule(trace, tied) == "rules_v1_fallback:tied_estimates"


def test_rules_mode_ranks_removal_estimates_with_predictions(load_fixture):
    from gpu_os_copilot.analysis.rule_baseline_v4 import diagnose_v4, ranking_rule

    trace = load_fixture("compute_bound")
    summary = summary_with(trace, kernel_busy_fraction=0.2)
    assert ranking_rule(trace, summary) == "rules_v4"
    ranked = diagnose_v4(trace, summary)
    assert [h.category for h in ranked] == ["launch_overhead", "data_transfer", "synchronization"]
    assert [h.confidence for h in ranked] == ["medium", "low", "low"]
    assert "37% of the window, the largest" in ranked[0].statement
    assert "roughly 59%" in ranked[0].testable_prediction
    busy = diagnose_v4(trace, summary_with(trace, kernel_busy_fraction=0.7))
    assert [h.category for h in busy][:2] == ["gpu_compute", "launch_overhead"]
    assert "busy for 70%" in busy[0].statement
    assert all(h.evidence_event_ids for h in busy[:1])
