import json
from pathlib import Path

import pytest

from gpu_os_copilot.agent.budget import ModelBudget
from gpu_os_copilot.agent.fake_client import FakeClient
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.worker.local import LocalWorker


def test_reservations_do_not_depend_on_missing_or_optimistic_provider_usage():
    budget = ModelBudget(AgentConfig(max_completion_tokens=256, max_total_completion_tokens=512))
    budget.reserve([], [])
    budget.observe(None)
    budget.reserve([], [])
    budget.observe({"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21})
    with pytest.raises(CopilotError, match="COMPLETION_BUDGET"):
        budget.reserve([], [])
    report = budget.report()
    assert report["completion_tokens_reserved"] == 512
    assert report["reported_provider_tokens"]["completion_tokens"] == 1
    assert report["responses_with_complete_usage"] == 1
    assert not report["provider_usage_complete"] and report["cost_usd"] is None


def test_session_input_limit_counts_repeated_context_bytes():
    budget = ModelBudget(AgentConfig(max_request_bytes=1024, max_session_request_bytes=1024))
    messages = [{"role": "user", "content": "x" * 600}]
    budget.reserve(messages, [])
    with pytest.raises(CopilotError, match="SESSION_BYTE_BUDGET"):
        budget.reserve(messages, [])
    assert budget.requests == 1


@pytest.mark.parametrize(
    "usage,error",
    [
        ({"completion_tokens": 257}, "PROVIDER_COMPLETION_LIMIT"),
        ({"total_tokens": -1}, "INVALID_USAGE"),
        ({"total_tokens": True}, "INVALID_USAGE"),
        ({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 10}, "INVALID_USAGE"),
        ([], "INVALID_USAGE"),
    ],
)
def test_invalid_usage_is_not_accepted_as_a_cost_measurement(usage, error):
    budget = ModelBudget(AgentConfig(max_completion_tokens=256))
    with pytest.raises(CopilotError, match=error):
        budget.observe(usage)


@pytest.mark.parametrize(
    "limit,expected_calls,error",
    [
        ({"max_request_bytes": 1024}, 0, "REQUEST_BYTE_BUDGET"),
    ],
)
def test_runtime_stops_before_unbudgeted_provider_request(
    load_fixture, tmp_path, limit, expected_calls, error
):
    trace = load_fixture()
    fake = FakeClient(trace)
    ex = ToolExecutor(LocalWorker(tmp_path / "worker"), trace)
    with pytest.raises(CopilotError, match=error):
        run_agent(fake, ex, AgentConfig(**limit), tmp_path / "audit")
    assert fake.turn == expected_calls and ex.experiment_count == 0
    rows = [
        json.loads(line)
        for line in next((tmp_path / "audit").glob("*.jsonl")).read_text().splitlines()
    ]
    assert rows[-1]["model_budget"]["requests_reserved"] == expected_calls


@pytest.mark.parametrize(
    "limits,expected_requests",
    [
        ({"max_steps": 1}, 1),
        ({"max_total_completion_tokens": 4096}, 1),
        ({"max_request_bytes": 65536, "max_session_request_bytes": 65536}, 1),
        ({"max_tool_calls": 2}, 2),
    ],
)
def test_last_available_response_is_reserved_for_validated_conclusion(
    load_fixture, tmp_path, limits, expected_requests
):
    trace = load_fixture()
    ex = ToolExecutor(LocalWorker(tmp_path / "worker"), trace)
    result = run_agent(FakeClient(trace), ex, AgentConfig(**limits), tmp_path / "audit")
    assert result["model_budget"]["requests_reserved"] == expected_requests
    assert result["decision"]["decision"] == "inconclusive"
    assert ex.experiment_count == 0
    rows = [json.loads(line) for line in Path(result["audit_path"]).read_text().splitlines()]
    reservations = [r for r in rows if r["kind"] == "budget_reservation"]
    assert reservations[-1]["controller_state"]["final_response_required"]


def test_runtime_completion_limit_matches_client_configuration(load_fixture, tmp_path):
    trace = load_fixture()
    fake = FakeClient(trace)
    fake.config = AgentConfig(max_completion_tokens=8192)
    with pytest.raises(CopilotError, match="CLIENT_BUDGET_MISMATCH"):
        run_agent(
            fake, ToolExecutor(LocalWorker(tmp_path), trace), AgentConfig(), tmp_path / "audit"
        )
    assert fake.turn == 0


def test_budget_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("META_MAX_TOTAL_COMPLETION_TOKENS", "8192")
    monkeypatch.setenv("META_MAX_REQUEST_BYTES", "65536")
    monkeypatch.setenv("META_MAX_SESSION_REQUEST_BYTES", "262144")
    config = AgentConfig.from_environment()
    assert config.max_total_completion_tokens == 8192
    assert config.max_request_bytes == 65536
    assert config.max_session_request_bytes == 262144
