"""Evidence-access bookkeeping checks, not model-quality evaluation."""

import pytest
from pydantic import ValidationError

from gpu_os_copilot.agent.evidence_access import assess_evidence_access, unread_opportunities
from gpu_os_copilot.analysis.scheduling_study import READ_TOOL, SCOPE
from gpu_os_copilot.schemas import AgentDecision, EvidenceDeferral


def catalog(state="unread"):
    return {
        "metrics": {
            "profiled_cpu_delay_change_median_ms": {
                "value": None,
                "available": False,
                "availability_state": state,
                "read_tool": READ_TOOL,
                "scope": SCOPE,
            }
        }
    }


def deferral(reason="existing_evidence_sufficient"):
    return EvidenceDeferral(
        tool_name=READ_TOOL,
        reason=reason,
        rationale="The current question is limited to separately measured throughput.",
    )


def test_unread_is_not_unavailable_and_unoffered_tools_are_not_requested():
    assert READ_TOOL in unread_opportunities(catalog(), {READ_TOOL})
    assert unread_opportunities(catalog(), set()) == {}
    assert unread_opportunities(catalog("unavailable"), {READ_TOOL}) == {}
    assert unread_opportunities(catalog("available"), {READ_TOOL}) == {}
    legacy = catalog()
    legacy["metrics"]["profiled_cpu_delay_change_median_ms"].pop("availability_state")
    assert unread_opportunities(legacy, {READ_TOOL}) == {}


def test_deferral_accounts_for_evidence_without_fabricating_consumption():
    before = catalog()
    result = assess_evidence_access([deferral()], before, {READ_TOOL}, can_read_and_finish=True)
    assert result["status"] == "accounted"
    assert result["unread_opportunities"][READ_TOOL]["status"] == "unread"
    assert result["model_deferrals"][0]["reason"] == "existing_evidence_sufficient"
    assert before == catalog()


@pytest.mark.parametrize("available", [True, False])
def test_budget_deferral_uses_actual_feedback_capacity(available):
    result = assess_evidence_access(
        [deferral("insufficient_budget")],
        catalog(),
        {READ_TOOL},
        can_read_and_finish=available,
    )
    assert ("EVIDENCE_READ_STILL_FITS_BUDGET" in result["error_codes"]) == available
    assert (result["status"] == "accounted") == (not available)


def test_duplicate_stale_and_unoffered_deferrals_do_not_clear_feedback():
    duplicate = assess_evidence_access(
        [deferral(), deferral()], catalog(), {READ_TOOL}, can_read_and_finish=True
    )
    assert "DUPLICATE_EVIDENCE_DEFERRAL" in duplicate["error_codes"]
    for observed, offered in [(catalog("unavailable"), {READ_TOOL}), (catalog(), set())]:
        result = assess_evidence_access([deferral()], observed, offered, can_read_and_finish=False)
        assert "DEFERRAL_NOT_AN_OFFERED_UNREAD_TOOL" in result["error_codes"]


@pytest.mark.parametrize("rationale", ["", " ", "\n\t"])
def test_empty_deferral_is_not_an_explanation(rationale):
    with pytest.raises(ValidationError):
        EvidenceDeferral(tool_name=READ_TOOL, reason="not_relevant", rationale=rationale)


def test_old_decisions_remain_readable_without_inventing_a_deferral():
    result = AgentDecision.model_validate(
        {
            "hypotheses": [{"category": "unknown", "statement": "Historical uncertainty."}],
            "decision": "inconclusive",
            "rationale": "Historical record.",
        }
    )
    assert result.deferred_evidence == []
    assert result.hypotheses[0].category == "unknown"
