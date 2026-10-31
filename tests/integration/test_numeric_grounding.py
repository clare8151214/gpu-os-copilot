import json
from copy import deepcopy
from pathlib import Path

import pytest

from gpu_os_copilot.agent.investigator import evidence_catalog
from gpu_os_copilot.agent.numeric_grounding import ground_narrative
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision
from gpu_os_copilot.worker.local import LocalWorker


def proposal(text="No numerical observation is asserted."):
    return AgentDecision(
        hypotheses=[
            {
                "category": "unknown",
                "statement": "The causal explanation remains unknown.",
                "missing_evidence": ["Controlled causal evidence"],
                "counterevidence_note": text,
                "testable_prediction": "A controlled comparison may distinguish candidates.",
            }
        ],
        decision="inconclusive",
        rationale="Candidate only.",
    )


@pytest.fixture
def catalog(load_fixture):
    result = evidence_catalog(load_fixture())
    for name, value in (("contradictory_pairs", 5), ("total_pairs", 6)):
        result["metrics"][name] = {
            "value": value,
            "unit": "pairs",
            "scope": "Separate CPU study, not the input trace.",
            "available": True,
            "availability_state": "available",
        }
    return result


def test_exact_count_ratio_and_units_are_rendered_from_evidence(catalog):
    decision = proposal("Order contradictions in {{ratio:contradictory_pairs:total_pairs}}.")
    original = decision.model_dump()
    rendered, check = ground_narrative(decision, catalog)
    assert decision.model_dump() == original
    assert rendered.hypotheses[0].counterevidence_note == "Order contradictions in 5/6 pairs."
    assert check["status"] == "references_resolved"
    assert check["references"][0]["facts"][0]["scope"].startswith("Separate CPU")
    assert check["narrative_semantics"] == "not_verified"


def test_literals_including_recorded_count_error_are_not_accepted(catalog):
    for text in ("4/6 pairs contradicted", "gain 9999%", "gain １.２%", "gain ½"):
        rendered, check = ground_narrative(proposal(text), catalog)
        assert rendered is None
        assert check["errors"] == [
            {
                "field": "hypotheses.0.counterevidence_note",
                "code": "NUMERIC_LITERAL_REQUIRES_REFERENCE",
            }
        ]


@pytest.mark.parametrize(
    "text,code",
    [
        ("{{metric:invented}}", "UNKNOWN_NUMERIC_METRIC"),
        ("{{metric:total_pairs:contradictory_pairs}}", "MALFORMED_NUMERIC_REFERENCE"),
        ("{{total_pairs}}", "MALFORMED_NUMERIC_REFERENCE"),
        ("{{metric:total_pairs}", "MALFORMED_NUMERIC_REFERENCE"),
        ("{{percent:total_pairs}}", "INVALID_NUMERIC_UNIT"),
        ("{{ratio:total_pairs:contradictory_pairs}}", "INVALID_NUMERIC_RATIO"),
    ],
)
def test_unknown_malformed_and_dimensionally_invalid_references(catalog, text, code):
    rendered, check = ground_narrative(proposal(text), catalog)
    assert rendered is None
    assert code in [error["code"] for error in check["errors"]]


@pytest.mark.parametrize("state", ["unread", "unavailable"])
def test_reference_cannot_expose_unread_or_missing_measurement(catalog, state):
    catalog["metrics"]["total_pairs"]["availability_state"] = state
    rendered, check = ground_narrative(proposal("{{metric:total_pairs}}"), catalog)
    assert rendered is None
    assert check["references"] == []
    assert check["errors"][0]["code"] == "UNAVAILABLE_NUMERIC_METRIC"


@pytest.mark.parametrize("mutation", ["scope", "unit", "zero", "noninteger"])
def test_ratios_cannot_mix_scope_units_or_invalid_denominators(catalog, mutation):
    denominator = catalog["metrics"]["total_pairs"]
    if mutation == "scope":
        denominator["scope"] = "An unrelated study"
    elif mutation == "unit":
        denominator["unit"] = "events"
    elif mutation == "zero":
        denominator["value"] = 0
    else:
        denominator["value"] = 6.5
    rendered, check = ground_narrative(
        proposal("{{ratio:contradictory_pairs:total_pairs}}"), catalog
    )
    assert rendered is None
    assert check["errors"][0]["code"] == "INVALID_NUMERIC_RATIO"


def test_zero_negative_values_and_percent_conversion_are_not_reinterpreted(catalog):
    catalog["metrics"]["zero"] = {"value": 0, "unit": "events", "available": True}
    catalog["metrics"]["change"] = {"value": -0.125, "unit": "fraction", "available": True}
    catalog["metrics"]["delay"] = {"value": -183.461748, "unit": "ms", "available": True}
    text = "Observed {{metric:zero}}, {{percent:change}}, {{metric:delay}}."
    rendered, check = ground_narrative(proposal(text), catalog)
    assert (
        rendered.hypotheses[0].counterevidence_note == "Observed 0 events, -12.5%, -183.461748 ms."
    )
    assert len(check["references"]) == 3


def test_known_event_identifier_is_not_a_measurement_literal(catalog):
    text = "Event event-0123 is a supplied observation."
    rendered, _ = ground_narrative(proposal(text), catalog, visible_ids={"event-0123"})
    assert rendered.hypotheses[0].counterevidence_note == text
    assert ground_narrative(proposal(text), catalog)[0] is None
    assert ground_narrative(proposal("event-01234"), catalog, visible_ids={"event-0123"})[0] is None


def test_no_numeric_references_does_not_mean_prose_is_verified(catalog):
    _, check = ground_narrative(proposal("Four pairs have contradictions."), catalog)
    assert check["status"] == "no_references"
    assert check["narrative_semantics"] == "not_verified"
    assert any("Spelled-out" in limit for limit in check["limitations"])


class Responses:
    mode = "mock"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        self.messages = deepcopy(messages)
        return {"message": {"content": next(self.decisions).model_dump_json()}}


def test_invalid_number_feedback_repairs_with_consumed_fact_and_preserves_raw(
    load_fixture, tmp_path
):
    client = Responses(
        [proposal("Observed idle is 9999%."), proposal("Idle: {{percent:gpu_idle_fraction}}")]
    )
    executor = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    result = run_agent(client, executor, AgentConfig(max_steps=2), tmp_path / "audit")
    assert client.calls == 2 and executor.experiment_count == 0
    assert result["numeric_grounding"]["status"] == "references_resolved"
    assert "{{percent:" in result["proposed_decision"]["hypotheses"][0]["counterevidence_note"]
    expected = evidence_catalog(executor.trace)["metrics"]["gpu_idle_fraction"]["value"] * 100
    assert result["decision"]["hypotheses"][0]["counterevidence_note"] == f"Idle: {expected:.12g}%"
    rows = [json.loads(line) for line in Path(result["audit_path"]).read_text().splitlines()]
    checks = [row for row in rows if row["kind"] == "numeric_grounding"]
    assert [row["assessment"]["status"] for row in checks] == ["invalid", "references_resolved"]
    assert "9999" in json.dumps(checks[0]["proposed_decision"])
    assert "numeric_grounding_feedback" in json.dumps(client.messages)


def test_invalid_numeric_answer_stops_at_original_response_cap(load_fixture, tmp_path):
    client = Responses([proposal("Wrong 4/6"), proposal("Wrong 4/6")])
    executor = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    with pytest.raises(CopilotError, match="AGENT_NUMERIC_GROUNDING_INVALID"):
        run_agent(client, executor, AgentConfig(max_steps=2), tmp_path / "audit")
    assert client.calls == 2 and executor.experiment_count == 0
