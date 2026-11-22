"""Provider failures: classified without bodies; transient ones re-sent within the budget."""

import json

import httpx
import openai
import pytest

from gpu_os_copilot.agent.meta_client import MetaClient
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision, Hypothesis
from gpu_os_copilot.worker.local import LocalWorker


def decision():
    return AgentDecision(
        hypotheses=[
            Hypothesis(
                category="launch_overhead",
                statement="Launch delays are a candidate.",
                supporting_metric_refs=["gpu_idle_fraction"],
                counterevidence_note="Overlap alone cannot select a cause.",
                testable_prediction="Changing the launch path may change gaps.",
            ),
            Hypothesis(
                category="gpu_compute",
                statement="Kernel duration is a competing candidate.",
                supporting_metric_refs=["kernel_total_ms"],
                counterevidence_note="Idle gaps argue against continuous computation.",
                testable_prediction="A compute change would alter kernel time if relevant.",
            ),
        ],
        decision="inconclusive",
        rationale="Candidate explanations, no causal conclusion.",
    ).model_dump_json()


class Flaky:
    mode = "meta"

    def __init__(self, failures):
        self.failures = list(failures)
        self.sent = []

    def complete(self, messages, tools):
        self.sent.append(json.loads(json.dumps(messages)))
        if self.failures:
            raise CopilotError(self.failures.pop(0))
        return {"message": {"content": decision()}}


def audit_rows(path):
    return [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]


def run(client, load_fixture, tmp_path, steps=5):
    executor = ToolExecutor(LocalWorker(tmp_path / "worker"), load_fixture())
    return run_agent(client, executor, AgentConfig(max_steps=steps), tmp_path / "audit")


@pytest.mark.parametrize(
    "code", ["META_API_TIMEOUT", "META_API_CONNECTION_FAILED", "META_API_SERVER_ERROR"]
)
def test_transient_failure_is_resent_as_a_new_reserved_request(load_fixture, tmp_path, code):
    client = Flaky([f"{code}: detail"])
    report = run(client, load_fixture, tmp_path)
    assert report["decision"]["decision"] == "inconclusive"
    assert report["model_budget"]["requests_reserved"] == 2
    # Same conversation, one controller message, refreshed for the remaining capacity.
    assert len(client.sent[0]) == len(client.sent[1])
    assert client.sent[0][:-1] == client.sent[1][:-1]
    states = [json.loads(sent[-1]["content"])["controller_state"] for sent in client.sent]
    assert [s["remaining_responses"] for s in states] == [5, 4]
    rows = audit_rows(tmp_path / "audit")
    assert [(r["error_code"], r["retry"]) for r in rows if r["kind"] == "provider_retry"] == [
        (code, 1)
    ]


@pytest.mark.parametrize(
    "failures,steps",
    [
        (["META_API_FAILED: HTTP 400"], 5),
        (["META_QUOTA_OR_RATE_LIMIT"], 5),
        (["META_API_TIMEOUT"] * 3, 5),
        (["META_API_TIMEOUT"], 1),
    ],
)
def test_other_or_repeated_provider_failures_stop_the_session(
    load_fixture, tmp_path, failures, steps
):
    client = Flaky(failures)
    with pytest.raises(CopilotError, match=failures[-1].split(":")[0]):
        run(client, load_fixture, tmp_path, steps)
    assert len(client.sent) == min(len(failures), steps, 3)
    assert audit_rows(tmp_path / "audit")[-1]["kind"] == "failure"


def provider_error(kind):
    request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    if kind == "timeout":
        return openai.APITimeoutError(request=request)
    if kind == "connection":
        return openai.APIConnectionError(request=request)
    response = httpx.Response(int(kind), request=request)
    status = openai.InternalServerError if int(kind) >= 500 else openai.BadRequestError
    return status("provider secret body", response=response, body={"error": "secret body"})


@pytest.mark.parametrize(
    "kind,code",
    [
        ("timeout", "META_API_TIMEOUT"),
        ("connection", "META_API_CONNECTION_FAILED"),
        ("502", "META_API_SERVER_ERROR: HTTP 502"),
        ("400", "META_API_FAILED: HTTP 400"),
    ],
)
def test_meta_client_classifies_provider_errors_without_bodies(monkeypatch, kind, code):
    class Completions:
        def create(self, **kwargs):
            raise provider_error(kind)

    class SDK:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(openai, "OpenAI", SDK)
    monkeypatch.setenv("MODEL_API_KEY", "test-only-secret")
    client = MetaClient(AgentConfig(), allow_paid_api=True)
    with pytest.raises(CopilotError) as error:
        client.complete([{"role": "user", "content": "x"}], [])
    assert str(error.value).startswith(code)
    assert "secret" not in str(error.value)
