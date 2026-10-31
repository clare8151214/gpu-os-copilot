import json
from pathlib import Path

import pytest

from gpu_os_copilot.agent.fake_client import FakeClient, tool_call
from gpu_os_copilot.agent.meta_client import MetaClient
from gpu_os_copilot.agent.runner import run_agent, sanitize
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import WorkerCapabilities
from gpu_os_copilot.worker.local import LocalWorker


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1})
    result = LocalWorker(tmp_path)
    result._capabilities = WorkerCapabilities(
        os="test",
        python_version="test",
        cpu_affinity=[0, 1],
        cuda_available=False,
    )
    return result


@pytest.mark.parametrize(
    "scenario,experiments",
    [
        ("cpu_starvation", 1),
        ("transfer_bound", 0),
        ("compute_bound", 0),
    ],
)
def test_full_mock_round_trip(load_fixture, worker, tmp_path, scenario, experiments):
    trace = load_fixture(scenario)
    executor = ToolExecutor(worker, trace)
    result = run_agent(FakeClient(trace), executor, AgentConfig(), tmp_path)
    assert result["mode"] == "mock"
    assert result["decision"]["decision"] == "inconclusive"
    assert executor.experiment_count == experiments
    rows = [json.loads(row) for row in Path(result["audit_path"]).read_text().splitlines()]
    assert rows[0]["kind"] == "start"
    assert rows[-1]["kind"] == "decision"
    tool_ids = {row["tool_call_id"] for row in rows if row["kind"] == "tool"}
    assert {"summary", "available", "timeline"}.issubset(tool_ids)


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("execute_shell", {"command": "unused"}, "TOOL_NOT_ALLOWLISTED"),
        ("get_performance_summary", {"run_id": "not-found"}, "INVALID_TOOL_ARGUMENTS"),
        (
            "inspect_timeline",
            {"run_id": "cpu_starvation", "event_types": ["gpu_kernel"], "limit": 101},
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            "run_affinity_ab",
            {"workload_id": "launch_test", "candidate_cpu_set": [1], "owner_approved": True},
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            "run_affinity_ab",
            {"workload_id": "launch_test", "candidate_cpu_set": [1], "repeats": 50},
            "INVALID_TOOL_ARGUMENTS",
        ),
        (
            "run_affinity_ab",
            {"workload_id": "launch_test", "candidate_cpu_set": [100]},
            "INVALID_TOOL_ARGUMENTS",
        ),
    ],
)
def test_tool_allowlist_and_strict_args(load_fixture, worker, name, args, code):
    executor = ToolExecutor(worker, load_fixture())
    assert executor.execute(name, json.dumps(args))["error_code"] == code


def test_bad_json_and_experiment_budget(load_fixture, worker):
    executor = ToolExecutor(worker, load_fixture(), max_experiments=1)
    assert (
        executor.execute("get_performance_summary", "{")["error_code"] == "INVALID_TOOL_ARGUMENTS"
    )
    args = json.dumps({"workload_id": "launch_test", "candidate_cpu_set": [1]})
    assert executor.execute("run_affinity_ab", args)["result"]["status"] == "dry_run"
    assert (
        executor.execute("run_affinity_ab", args)["result"]["error_code"]
        == "EXPERIMENT_BUDGET_EXCEEDED"
    )


def test_paid_call_requires_opt_in_and_key():
    with pytest.raises(CopilotError, match="NOT_APPROVED"):
        MetaClient(AgentConfig())
    with pytest.raises(CopilotError, match="KEY_MISSING"):
        MetaClient(AgentConfig(), allow_paid_api=True)


def test_meta_sdk_round_trip_without_network(monkeypatch, load_fixture, worker, tmp_path):
    import openai
    from openai.types.chat import ChatCompletion

    trace = load_fixture()
    fake = FakeClient(trace)
    seen = []

    class Completion:
        def create(self, **kwargs):
            seen.append(kwargs)
            assert kwargs["model"] == "muse-spark-1.3"
            if len(seen) < 3:
                assert kwargs["tools"] and kwargs["tool_choice"] == "auto"
            else:
                assert "tools" not in kwargs and "tool_choice" not in kwargs
            assert kwargs["max_completion_tokens"] == 3072
            assert kwargs["reasoning_effort"] == "low"
            response = fake.complete(kwargs["messages"], kwargs.get("tools", []))
            if len(seen) > 1:
                messages = kwargs["messages"]
                for index, message in enumerate(messages):
                    if message["role"] == "tool":
                        assert any(
                            call["id"] == message["tool_call_id"]
                            for prior in messages[:index]
                            for call in prior.get("tool_calls", [])
                        )
            return ChatCompletion.model_validate(
                {
                    "id": response["response_id"],
                    "created": 0,
                    "model": kwargs["model"],
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": response["message"],
                        }
                    ],
                }
            )

    class SDK:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "https://api.meta.ai/v1"
            assert kwargs["max_retries"] == 0
            self.chat = type("Chat", (), {"completions": Completion()})()

    monkeypatch.setattr(openai, "OpenAI", SDK)
    monkeypatch.setenv("MODEL_API_KEY", "test-only-secret")
    config = AgentConfig(max_steps=3, max_completion_tokens=3072, reasoning_effort="low")
    client = MetaClient(config, allow_paid_api=True)
    result = run_agent(client, ToolExecutor(worker, trace), config, tmp_path)
    assert len(seen) == 3
    assert result["tool_calls"] == 4
    assert "test-only-secret" not in Path(result["audit_path"]).read_text()


def test_redaction(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "fake-credential")
    assert sanitize({"api_key": "x", "text": "key=fake-credential"}) == {
        "api_key": "[REDACTED]",
        "text": "key=[REDACTED]",
    }


class ScriptedClient:
    mode = "mock"

    def __init__(self, message):
        self.message = message

    def complete(self, messages, tools):
        return {"message": self.message}


def test_truncated_response_never_executes_tools(load_fixture, worker, tmp_path):
    class TruncatedClient:
        mode = "meta"

        def complete(self, messages, tools):
            return {
                "finish_reason": "length",
                "message": {
                    "tool_calls": [
                        tool_call(
                            "truncated",
                            "run_affinity_ab",
                            {"workload_id": "launch_test", "candidate_cpu_set": [1], "repeats": 2},
                        )
                    ]
                },
            }

    executor = ToolExecutor(worker, load_fixture())
    with pytest.raises(CopilotError, match="AGENT_RESPONSE_TRUNCATED"):
        run_agent(TruncatedClient(), executor, AgentConfig(), tmp_path)
    assert executor.experiment_count == 0
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert rows[-1]["error_code"] == "AGENT_RESPONSE_TRUNCATED"
    assert not any(row["kind"] == "tool" for row in rows)


def test_response_budget_configuration(monkeypatch):
    monkeypatch.setenv("META_MAX_COMPLETION_TOKENS", "3072")
    monkeypatch.setenv("META_REASONING_EFFORT", "low")
    config = AgentConfig.from_environment()
    assert config.max_completion_tokens == 3072
    assert config.reasoning_effort == "low"
    monkeypatch.setenv("META_MAX_COMPLETION_TOKENS", "999999")
    with pytest.raises(ValueError):
        AgentConfig.from_environment()


def test_unbound_trace_cannot_start_live_experiment(load_fixture, worker, monkeypatch):
    executor = ToolExecutor(worker, load_fixture(), owner_approved=True)
    monkeypatch.setattr(worker, "run_experiment", lambda request: pytest.fail("Unbound experiment"))
    result = executor.execute(
        "run_affinity_ab",
        json.dumps({"workload_id": "launch_test", "candidate_cpu_set": [1], "repeats": 2}),
    )
    assert result["result"]["error_code"] == "TRACE_WORKLOAD_UNBOUND"
    assert executor.experiment_count == 0


def test_agent_timeline_excludes_warmup_and_shutdown(load_fixture, worker):
    trace = load_fixture()
    trace.window.start_ns = 1_000_000
    trace.window.end_ns = 8_000_000
    executor = ToolExecutor(worker, trace)
    result = executor.execute(
        "inspect_timeline",
        json.dumps(
            {"run_id": trace.run_id, "event_types": ["gpu_kernel", "cpu_sched"], "limit": 100}
        ),
    )["result"]
    assert result["events"]
    assert all(
        event["start_ns"] < 8_000_000 and event["end_ns"] > 1_000_000 for event in result["events"]
    )
    ids = set(executor.evidence()["event_ids"])
    assert all(
        event.end_ns > 1_000_000 and event.start_ns < 8_000_000
        for event in trace.events
        if event.id in ids
    )


@pytest.mark.parametrize(
    "change,error",
    [
        ({}, None),
        ({"seed": 7}, "TRACE_WORKLOAD_UNBOUND"),
        ({"dtype": "float16"}, "TRACE_WORKLOAD_UNBOUND"),
        ({"operations": 20}, "TRACE_WORKLOAD_UNBOUND"),
        ({"warmup": 0}, "TRACE_WORKLOAD_UNBOUND"),
        ({"synchronized": 1}, "TRACE_WORKLOAD_UNBOUND"),
        ({"scenario": []}, "TRACE_WORKLOAD_UNBOUND"),
        ({"gpu": "another GPU"}, "TRACE_ENVIRONMENT_MISMATCH"),
        ({"cuda": "different runtime"}, "TRACE_ENVIRONMENT_MISMATCH"),
        ({"torch": "different version"}, "TRACE_ENVIRONMENT_MISMATCH"),
    ],
)
def test_live_experiment_requires_exact_workload_and_environment(
    load_fixture, worker, monkeypatch, change, error
):
    trace = load_fixture()
    trace.workload_id = "launch_test"
    # Test-only metadata: no GPU execution or measured performance claim.
    for event in trace.events:
        event.source = "nsys_sqlite"
        event.confidence = "measured"
    trace.provenance.source = "nsys_sqlite"
    trace.provenance.confidence = "measured"
    trace.trace_completeness = "complete"
    trace.configuration = {
        "scenario": "cpu_contention",
        "workload_version": 2,
        "shape": [256, 256],
        "seed": 42,
        "dtype": "float32",
        "warmup": 10,
        "operations": 200,
        "synchronized": True,
        "gpu": "test GPU",
        "cuda": "test CUDA",
        "torch": "test torch",
        **change,
    }
    worker._capabilities.cuda_available = True
    worker._capabilities.gpus = [{"model": "test GPU"}]
    worker._capabilities.cuda_version = "test CUDA"
    worker._capabilities.torch_version = "test torch"
    executor = ToolExecutor(worker, trace, owner_approved=True)
    calls = []

    def fake_experiment(request):
        calls.append(request)
        raise RuntimeError("Test stops before any GPU execution")

    monkeypatch.setattr(worker, "run_experiment", fake_experiment)
    available = executor.execute("list_available_experiments", "{}")
    assert available["result"]["experiments"][0]["can_execute"] is (error is None)
    result = executor.execute(
        "run_affinity_ab",
        json.dumps({"workload_id": "launch_test", "candidate_cpu_set": [1]}),
    )
    if error:
        assert result["result"]["error_code"] == error
        assert calls == [] and executor.experiment_count == 0
    else:
        assert len(calls) == 1 and calls[0].scenario == "cpu_contention"


def test_step_and_tool_budgets(load_fixture, worker, tmp_path):
    executor = ToolExecutor(worker, load_fixture())
    call = tool_call("one", "list_available_experiments", {})
    with pytest.raises(CopilotError, match="FINAL_RESPONSE_REQUIRED"):
        run_agent(
            ScriptedClient({"tool_calls": [call]}), executor, AgentConfig(max_steps=1), tmp_path
        )
    with pytest.raises(CopilotError, match="TOOL_BUDGET"):
        run_agent(ScriptedClient({"tool_calls": [call] * 5}), executor, AgentConfig(), tmp_path)


@pytest.mark.parametrize(
    "decision,ids,error",
    [
        ("supported", [], "CLAIM_WITHOUT"),
        ("inconclusive", ["invented-event"], "UNKNOWN_EVIDENCE"),
    ],
)
def test_reject_ungrounded_model_decision(load_fixture, worker, tmp_path, decision, ids, error):
    message = {
        "content": json.dumps(
            {
                "hypotheses": [
                    {"category": "unknown", "statement": "Candidate", "evidence_event_ids": ids}
                ],
                "decision": decision,
                "rationale": "Unverified response",
            }
        )
    }
    with pytest.raises(CopilotError, match=error):
        run_agent(
            ScriptedClient(message), ToolExecutor(worker, load_fixture()), AgentConfig(), tmp_path
        )
