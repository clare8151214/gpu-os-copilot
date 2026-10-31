import json
from pathlib import Path

import pytest

from gpu_os_copilot.agent.investigator import evidence_catalog, validate_investigation
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision, Hypothesis
from gpu_os_copilot.worker.local import LocalWorker


def candidates():
    return AgentDecision(
        hypotheses=[
            Hypothesis(
                category="launch_overhead",
                statement="Launch delays are a candidate.",
                supporting_metric_refs=["gpu_idle_fraction", "cuda_api_duration_ms"],
                counterevidence_note="No unique attribution; overlap alone cannot select a cause.",
                testable_prediction="Changing the launch path may change gaps and wall time.",
            ),
            Hypothesis(
                category="gpu_compute",
                statement="Kernel duration is a competing candidate.",
                supporting_metric_refs=["kernel_total_ms"],
                contradicting_metric_refs=["gpu_idle_fraction"],
                counterevidence_note="Large idle gaps argue against continuous GPU computation.",
                testable_prediction="A compute intervention would change kernel time if relevant.",
            ),
        ],
        decision="inconclusive",
        rationale="Candidate explanations, no causal conclusion.",
    )


def unknown():
    return AgentDecision(
        hypotheses=[
            Hypothesis(
                category="unknown",
                statement="Available observations do not identify a cause.",
                missing_evidence=["Controlled causal evidence"],
                counterevidence_note="No specific cause is claimed; evidence is insufficient.",
                testable_prediction="Additional controlled measurements could distinguish causes.",
            )
        ],
        decision="inconclusive",
        rationale="Explicit abstention.",
    )


def test_catalog_has_measured_units_bounded_events_and_no_private_labels(load_fixture):
    trace = load_fixture()
    trace.configuration["secret_label"] = "PRIVATE-SCENARIO"
    trace.provenance.description = "PRIVATE-DESCRIPTION"
    trace.events[0].attributes["path"] = "/private/raw.sqlite"
    catalog = evidence_catalog(trace)
    assert len(catalog["event_samples"]) <= 15
    assert catalog["metrics"]["gpu_idle_fraction"]["unit"] == "fraction"
    for private in ("PRIVATE-", "/private/", '"pid"', '"tid"', '"attributes"', '"configuration"'):
        assert private not in json.dumps(catalog)


def test_partial_or_missing_telemetry_does_not_become_zero(load_fixture):
    trace = load_fixture()
    trace.events = [e for e in trace.events if e.kind != "cpu_sched"]
    trace.trace_completeness = "partial"
    catalog = evidence_catalog(trace)
    for ref in ("gpu_idle_fraction", "cpu_runnable_delay_ms", "sched_idle_overlap_ms"):
        assert catalog["metrics"][ref]["value"] is None
        assert not catalog["metrics"][ref]["available"]
        assert ref in catalog["missing_metric_refs"]


def test_zero_kernels_in_complete_transfer_trace_is_observation(load_fixture):
    trace = load_fixture("transfer_bound")
    trace.events = [event for event in trace.events if event.kind != "gpu_kernel"]
    catalog = evidence_catalog(trace)
    assert catalog["metrics"]["kernel_count"]["value"] == 0
    assert catalog["metrics"]["kernel_count"]["available"]
    assert catalog["metrics"]["kernel_median_ms"]["value"] is None


def test_ranked_diagnosis_links_both_sides_to_application_computed_facts(load_fixture):
    catalog = evidence_catalog(load_fixture())
    check = validate_investigation(candidates(), catalog)
    assert check["status"] == "valid"
    assert [h["rank"] for h in check["ranked_hypotheses"]] == [1, 2]
    assert (
        check["ranked_hypotheses"][1]["contradicting_metrics"]["gpu_idle_fraction"]
        == catalog["metrics"]["gpu_idle_fraction"]
    )
    assert all(h["status"] == "candidate" for h in check["ranked_hypotheses"])
    assert validate_investigation(unknown(), catalog)["status"] == "valid"


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("unknown_metric", "UNKNOWN_METRIC_REFERENCE"),
        ("missing_as_observed", "MISSING_METRIC_USED_AS_OBSERVATION"),
        ("available_as_missing", "AVAILABLE_METRIC_MARKED_MISSING"),
        ("same_fact", "SAME_FACT_USED_AS_SUPPORT_AND_CONTRADICTION"),
        ("no_counter", "COUNTEREVIDENCE_ASSESSMENT_REQUIRED"),
        ("no_prediction", "TESTABLE_PREDICTION_REQUIRED"),
        ("duplicate", "DUPLICATE_HYPOTHESIS_CATEGORY"),
        ("high_confidence", "HIGH_CONFIDENCE_WITH_INCOMPLETE_EVIDENCE"),
        ("single", "RANK_TWO_OR_THREE_COMPETING_HYPOTHESES_OR_EXPLICIT_UNKNOWN"),
    ],
)
def test_bad_investigations_get_specific_feedback(load_fixture, mutation, code):
    catalog = evidence_catalog(load_fixture())
    decision = candidates()
    first = decision.hypotheses[0]
    if mutation == "unknown_metric":
        first.supporting_metric_refs = ["invented_speedup"]
    elif mutation == "missing_as_observed":
        first.supporting_metric_refs = ["wall_time_ms"]
        catalog["metrics"]["wall_time_ms"] = {"value": None, "available": False, "unit": "ms"}
    elif mutation == "available_as_missing":
        first.missing_metric_refs = ["gpu_idle_fraction"]
    elif mutation == "same_fact":
        first.contradicting_metric_refs = ["gpu_idle_fraction"]
    elif mutation == "no_counter":
        first.counterevidence_note = ""
    elif mutation == "no_prediction":
        first.testable_prediction = ""
    elif mutation == "duplicate":
        decision.hypotheses[1].category = first.category
    elif mutation == "high_confidence":
        first.confidence = "high"  # Simulated fixture cannot substantiate this claim.
    elif mutation == "single":
        decision.hypotheses = [first]
    assert code in validate_investigation(decision, catalog)["error_codes"]


class Responses:
    mode = "mock"

    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0
        self.messages = []

    def complete(self, messages, tools):
        self.calls += 1
        self.messages = messages.copy()
        return {"message": {"content": next(self.values).model_dump_json()}}


def test_investigation_repair_is_bounded_and_preserves_failed_proposal(load_fixture, tmp_path):
    bad = candidates()
    bad.hypotheses[0].supporting_metric_refs = ["invented_speedup"]
    client = Responses([bad, candidates()])
    ex = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    result = run_agent(client, ex, AgentConfig(max_steps=2), tmp_path / "audit")
    assert client.calls == 2 and ex.experiment_count == 0
    assert result["investigation"]["status"] == "valid"
    assert any(
        "UNKNOWN_METRIC_REFERENCE" in message.get("content", "")
        for message in client.messages
        if message["role"] == "user"
    )
    rows = [json.loads(line) for line in Path(result["audit_path"]).read_text().splitlines()]
    assert len([row for row in rows if row["kind"] == "investigation_feedback"]) == 1
    assert any("invented_speedup" in json.dumps(row) for row in rows if row["kind"] == "response")


def test_invalid_diagnosis_cannot_increase_response_budget(load_fixture, tmp_path):
    bad = candidates()
    bad.hypotheses[0].counterevidence_note = ""
    client = Responses([bad, bad])
    ex = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    with pytest.raises(CopilotError, match="AGENT_INVESTIGATION_INVALID"):
        run_agent(client, ex, AgentConfig(max_steps=2), tmp_path / "audit")
    assert client.calls == 2 and ex.experiment_count == 0


def test_fabricated_counterevent_is_rejected_before_repair(load_fixture, tmp_path):
    bad = candidates()
    bad.hypotheses[0].contradicting_event_ids = ["invented-counterevent"]
    client = Responses([bad])
    ex = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    with pytest.raises(CopilotError, match="AGENT_CITED_UNKNOWN_EVIDENCE"):
        run_agent(client, ex, AgentConfig(), tmp_path / "audit")
    assert client.calls == 1
