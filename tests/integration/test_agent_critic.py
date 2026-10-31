"""Synthetic measurements test decision integrity; they are not hardware evidence."""

import json
from pathlib import Path

import pytest

from gpu_os_copilot.agent.blind import BlindExecutor
from gpu_os_copilot.agent.critic import assess_batches, verify_decision
from gpu_os_copilot.agent.fake_client import tool_call
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision, BenchmarkTrial, ExperimentResult, Provenance
from gpu_os_copilot.workloads.specs import expected_configuration


@pytest.fixture
def measured_trace(load_fixture):
    trace = load_fixture()
    for event in trace.events:
        event.source = "nsys_sqlite"
        event.confidence = "measured"
    trace.provenance = Provenance(
        source="nsys_sqlite", confidence="measured", description="Synthetic contract test only"
    )
    trace.workload_id = "launch_test"
    trace.trace_completeness = "complete"
    trace.configuration = {
        **expected_configuration("cpu_contention"),
        "gpu": "test GPU",
        "cuda": "test CUDA",
        "torch": "test torch",
        "profiled": True,
    }
    return trace


def batch(trace, number, baseline=None, treatment=None):
    baseline = [10] * 5 if baseline is None else baseline
    treatment = [5] * 5 if treatment is None else treatment
    return ExperimentResult(
        experiment_id=f"private-batch-{number}",
        workload_id="launch_test",
        status="ok",
        provenance=Provenance(
            source="benchmark", confidence="measured", description="Synthetic contract test only"
        ),
        # Deliberately wrong saved summary: the critic must recompute from trials.
        comparison={"status": "ok", "median_throughput_change_fraction": 9999},
        trials=[
            BenchmarkTrial(
                run_id=f"private-{number}-{condition}-{i}",
                condition=condition,
                wall_time_ms=value,
                operation_count=200,
                throughput_ops_s=200000 / value,
                cpu_set=[0] if condition == "baseline" else [2],
                configuration={**trace.configuration, "profiled": False, "host": {"os": "test"}},
            )
            for condition, values in (("baseline", baseline), ("treatment", treatment))
            for i, value in enumerate(values)
        ],
    )


def proposal(decision="supported", scope="affinity_throughput", ids=None):
    return AgentDecision(
        hypotheses=[
            {
                "category": "unknown",
                "statement": "Candidate",
                "evidence_event_ids": ids or [],
                "missing_evidence": ["Causal attribution"],
                "counterevidence_note": "Cause remains unknown despite throughput measurements.",
                "testable_prediction": "Additional attributed telemetry could distinguish causes.",
            }
        ],
        claim_scope=scope,
        decision=decision,
        rationale="Model claims an invented 9999-fold gain.",
    )


def test_single_batch_cannot_support_optimization_but_consistent_batches_can(measured_trace):
    results = [batch(measured_trace, 1)]
    proposed = proposal()
    effective, check = verify_decision(proposed, measured_trace, results)
    assert effective.decision == "inconclusive"
    assert "INSUFFICIENT_DISTINCT_BATCHES" in check["reason_codes"]
    results.append(batch(measured_trace, 2))
    effective, check = verify_decision(proposed, measured_trace, results)
    assert effective.decision == "supported"
    assert check["throughput_change_fractions"] == [1, 1]
    assert check["policy"]["statistical_support"] == "not_evaluated"
    assert "9999" not in effective.rationale
    assert "9999" in proposed.rationale
    assert "private-" not in json.dumps(check)


@pytest.mark.parametrize("later_treatment", [9.6, 20])
def test_smaller_or_reversed_effect_forces_revision(measured_trace, later_treatment):
    results = [batch(measured_trace, 1), batch(measured_trace, 2, treatment=[later_treatment] * 5)]
    effective, check = verify_decision(proposal(), measured_trace, results)
    assert effective.decision == "inconclusive" and check["proposal_downgraded"]
    assert "SMALL_OR_CONFLICTING_EFFECTS" in check["reason_codes"]
    assert len(check["batches"]) == 2


def test_large_positive_but_collapsed_effect_is_unstable(measured_trace):
    results = [batch(measured_trace, 1), batch(measured_trace, 2, treatment=[8] * 5)]
    check = assess_batches(results)
    assert check["eligible_decision"] == "inconclusive"
    assert "EFFECT_SIZE_UNSTABLE" in check["reason_codes"]


@pytest.mark.parametrize("decision", ["supported", "rejected"])
def test_small_positive_or_null_effect_does_not_prove_equivalence(measured_trace, decision):
    results = [batch(measured_trace, i, treatment=[9.99] * 5) for i in (1, 2)]
    effective, check = verify_decision(proposal(decision), measured_trace, results)
    assert effective.decision == "inconclusive"
    assert check["policy"]["statistical_support"] == "not_evaluated"


def test_consistent_regression_rejects_only_scoped_throughput_improvement(measured_trace):
    results = [batch(measured_trace, i, treatment=[20] * 5) for i in (1, 2)]
    effective, _ = verify_decision(proposal("rejected"), measured_trace, results)
    assert effective.decision == "rejected"
    assert "regression" in effective.rationale
    effective, check = verify_decision(proposal("rejected", None), measured_trace, results)
    assert effective.decision == "inconclusive"
    assert "UNSCOPED_PERFORMANCE_CLAIM" in check["reason_codes"]


def test_range_overlap_and_low_repetition_are_reported(measured_trace):
    results = [batch(measured_trace, i, [10, 10, 10, 30], [5, 5, 5, 20]) for i in (1, 2)]
    check = assess_batches(results)
    assert check["eligible_decision"] == "inconclusive"
    assert {"OBSERVED_RANGES_OVERLAP", "INSUFFICIENT_TRIALS_PER_BATCH"} <= set(
        check["reason_codes"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "batch_id",
        "trial_id",
        "environment",
        "placement",
        "mixed_placement",
        "no_intervention",
        "operation_count",
        "rate",
        "partial",
        "provenance",
    ],
)
def test_invalid_evidence_cannot_be_cherry_picked(measured_trace, mutation):
    results = [batch(measured_trace, i) for i in (1, 2)]
    second = results[1]
    if mutation == "batch_id":
        second.experiment_id = results[0].experiment_id
    elif mutation == "trial_id":
        second.trials[0].run_id = results[0].trials[0].run_id
    elif mutation == "environment":
        for trial in second.trials:
            trial.configuration["host"] = {"os": "different host"}
    elif mutation == "placement":
        for trial in second.trials:
            if trial.condition == "treatment":
                trial.cpu_set = [3]
    elif mutation == "mixed_placement":
        second.trials[-1].cpu_set = [3]
    elif mutation == "no_intervention":
        for trial in second.trials:
            trial.cpu_set = [0]
    elif mutation == "operation_count":
        second.trials[0].operation_count = 201
    elif mutation == "rate":
        second.trials[0].throughput_ops_s = 123456
    elif mutation == "partial":
        second.status = "partial"
    elif mutation == "provenance":
        second.provenance.confidence = "derived"
    check = assess_batches(results)
    assert check["eligible_decision"] == "inconclusive"
    assert "INVALID_OR_INCOMPARABLE_BATCHES" in check["reason_codes"]
    assert check["batches"] == []


@pytest.mark.parametrize(
    "mutation", ["partial_trace", "different_gpu", "missing_host", "other_workload"]
)
def test_successful_unrelated_measurements_do_not_ground_claim(measured_trace, mutation):
    results = [batch(measured_trace, i) for i in (1, 2)]
    if mutation == "partial_trace":
        measured_trace.trace_completeness = "partial"
    elif mutation == "different_gpu":
        measured_trace.configuration["gpu"] = "other GPU"
    elif mutation == "other_workload":
        for result in results:
            result.workload_id = "other_workload"
    else:
        for result in results:
            for trial in result.trials:
                del trial.configuration["host"]
    effective, check = verify_decision(proposal(), measured_trace, results)
    assert effective.decision == "inconclusive"
    assert check["proposal_downgraded"]


def test_consistent_but_mislabeled_operation_counts_are_invalid(measured_trace):
    results = [batch(measured_trace, i) for i in (1, 2)]
    for result in results:
        for trial in result.trials:
            trial.operation_count = 100
            trial.throughput_ops_s = 100000 / trial.wall_time_ms
    check = assess_batches(results)
    assert "INVALID_OR_INCOMPARABLE_BATCHES" in check["reason_codes"]


def test_critic_never_promotes_a_cautious_proposal(measured_trace):
    results = [batch(measured_trace, i) for i in (1, 2)]
    effective, check = verify_decision(proposal("inconclusive"), measured_trace, results)
    assert check["eligible_decision"] == "supported"
    assert effective.decision == "inconclusive" and not check["proposal_downgraded"]


def test_unread_later_data_cannot_be_ignored_and_duplicate_reads_are_one_batch(measured_trace):
    ex = BlindExecutor(
        measured_trace,
        batch(measured_trace, 1),
        case_id="case-test",
        repeated=batch(measured_trace, 2, treatment=[9.6] * 5),
        prior_decision=proposal().model_dump(),
    )
    for _ in range(2):
        ex.execute("read_recorded_comparison", "{}")
    assert len(ex.comparison_evidence()) == 1
    check = assess_batches(
        ex.comparison_evidence(), unread_comparisons=ex.unread_comparison_count()
    )
    assert "UNREAD_AVAILABLE_COMPARISONS" in check["reason_codes"]
    assert check["throughput_change_fractions"] == [1]
    ex.execute("read_repeated_comparisons", "{}")
    assert len(ex.comparison_evidence()) == 2 and ex.unread_comparison_count() == 0


class OfflineScript:
    # Exercise the production gate without a provider call; tests block network globally.
    mode = "meta"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.messages = []

    def complete(self, messages, tools):
        self.calls += 1
        self.messages = list(messages)
        return {"message": next(self.responses)}


def test_feedback_and_final_downgrade_fit_original_budget(measured_trace, tmp_path):
    ex = BlindExecutor(
        measured_trace,
        batch(measured_trace, 1),
        case_id="case-test",
        repeated=batch(measured_trace, 2, treatment=[9.6] * 5),
    )
    client = OfflineScript(
        [
            {"tool_calls": [tool_call("first", "read_recorded_comparison", {})]},
            {"tool_calls": [tool_call("later", "read_repeated_comparisons", {})]},
            {"content": proposal().model_dump_json()},
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=3, max_tool_calls=2), tmp_path)
    assert client.calls == 3 and report["tool_calls"] == 2
    assert report["proposed_decision"]["decision"] == "supported"
    assert report["decision"]["decision"] == "inconclusive"
    feedback = next(
        json.loads(message["content"])["verification"]
        for message in reversed(client.messages)
        if message["role"] == "tool"
    )
    assert "EFFECT_SIZE_UNSTABLE" in feedback["reason_codes"]
    rows = [json.loads(line) for line in Path(report["audit_path"]).read_text().splitlines()]
    assert [row["kind"] for row in rows[-3:]] == ["verification", "numeric_grounding", "decision"]
    assert report["numeric_grounding"]["model_rationale_replaced_by_critic"]
    assert "9999" not in report["decision"]["rationale"]
    assert "private-batch" not in json.dumps(client.messages)


@pytest.mark.parametrize("kind", ["unread", "outside_window", "other_device"])
def test_existing_but_unobserved_event_cannot_be_cited(measured_trace, tmp_path, kind):
    ex = BlindExecutor(measured_trace, batch(measured_trace, 1), case_id="case-test")
    event = next(e for e in ex.trace.events if e.kind == "gpu_kernel")
    if kind == "unread":
        event.kind = "other"  # Not part of the bounded initial event samples.
    elif kind == "outside_window":
        event.end_ns = ex.trace.window.end_ns + 10
        event.start_ns = ex.trace.window.end_ns + 1
    elif kind == "other_device":
        event.device_id = ex.trace.window.device_id + 1
    client = OfflineScript(
        [{"content": proposal("inconclusive", ids=[event.id]).model_dump_json()}]
    )
    with pytest.raises(CopilotError, match="UNOBSERVED_EVIDENCE"):
        run_agent(client, ex, AgentConfig(), tmp_path)


def test_timeline_read_makes_window_event_citable(measured_trace, tmp_path):
    ex = BlindExecutor(measured_trace, batch(measured_trace, 1), case_id="case-test")
    event = min((e for e in ex.trace.events if e.kind == "gpu_kernel"), key=lambda e: e.start_ns)
    event.kind = "other"  # Only the requested timeline response reveals this event.
    conclusion = proposal("inconclusive", ids=[event.id])
    conclusion.rationale = "The inspected event is available; causality remains unknown."
    client = OfflineScript(
        [
            {
                "tool_calls": [
                    tool_call(
                        "timeline",
                        "inspect_timeline",
                        {
                            "run_id": "case-test",
                            "event_types": ["other"],
                            "limit": 1,
                        },
                    )
                ]
            },
            {"content": conclusion.model_dump_json()},
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=2), tmp_path)
    assert report["decision"]["hypotheses"][0]["evidence_event_ids"] == [event.id]
