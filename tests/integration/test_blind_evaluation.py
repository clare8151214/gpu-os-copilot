import json

import pytest

from gpu_os_copilot.agent.blind import BlindExecutor
from gpu_os_copilot.schemas import BenchmarkTrial, ExperimentResult, Provenance


def evidence(load_fixture):
    trace = load_fixture()
    # Synthetic unit-test objects, deliberately marked measured to test the gate.
    for event in trace.events:
        event.source = "nsys_sqlite"
        event.confidence = "measured"
    trace.provenance = Provenance(
        source="nsys_sqlite", confidence="measured", description="HIDDEN_CASE_LABEL"
    )
    trace.configuration = {"scenario": "HIDDEN_CASE_LABEL"}
    result = ExperimentResult(
        experiment_id="hidden-experiment",
        workload_id=trace.workload_id,
        status="ok",
        provenance=trace.provenance,
        trials=[
            BenchmarkTrial(
                run_id=f"secret-{i}",
                condition="baseline" if i < 2 else "treatment",
                wall_time_ms=10,
                operation_count=200,
                throughput_ops_s=20000,
                cpu_set=[0 if i < 2 else 2],
                configuration=trace.configuration,
            )
            for i in range(4)
        ],
    )
    return trace, result


def test_all_model_visible_paths_are_blinded_and_read_only(load_fixture):
    trace, result = evidence(load_fixture)
    ex = BlindExecutor(trace, result, case_id="case-test")
    visible = [ex.evidence(), ex.tool_schemas()]
    for name, args in [
        ("get_performance_summary", {"run_id": "case-test"}),
        (
            "inspect_timeline",
            {"run_id": "case-test", "event_types": ["gpu_kernel", "cpu_sched"], "limit": 100},
        ),
        ("list_available_experiments", {}),
        ("read_recorded_comparison", {}),
    ]:
        visible.append(ex.execute(name, json.dumps(args)))
    text = json.dumps(visible)
    for hidden in [
        "HIDDEN_CASE_LABEL",
        "cpu_starvation",
        "hidden-experiment",
        "secret-",
        "configuration",
    ]:
        assert hidden not in text
    assert ex.execute("run_affinity_ab", "{}")["error_code"] == "TOOL_NOT_ALLOWLISTED"
    assert ex.has_measured_comparison()


def test_revision_exposes_later_results_only_when_available(load_fixture):
    trace, result = evidence(load_fixture)
    ex = BlindExecutor(trace, result, case_id="case-test")
    assert not ex.has_measured_comparison()
    assert ex.execute("read_repeated_comparisons", "{}")["error_code"] == "TOOL_NOT_ALLOWLISTED"
    ex = BlindExecutor(
        trace,
        result,
        case_id="case-test",
        repeated=result.model_copy(update={"experiment_id": "later-batch"}),
        prior_decision={"decision": "supported"},
    )
    assert ex.evidence()["prior_decision"]["decision"] == "supported"
    assert ex.execute("read_repeated_comparisons", "{}")["result"]["batch"] == "later"
    assert "read_repeated_comparisons" in ex.reads


def test_mismatched_recordings_rejected(load_fixture):
    trace, result = evidence(load_fixture)
    for trial in result.trials:
        trial.configuration = {"scenario": "other"}
    with pytest.raises(ValueError, match="does not match"):
        BlindExecutor(trace, result, case_id="case-test")


def test_offline_blinded_tool_roundtrip_before_live_api(load_fixture, tmp_path):
    from gpu_os_copilot.agent.runner import run_agent
    from gpu_os_copilot.config import AgentConfig

    trace, recorded = evidence(load_fixture)
    ex = BlindExecutor(trace, recorded, case_id="case-test")

    class OfflineClient:
        mode = "mock"
        count = 0

        def complete(self, messages, tools):
            assert "HIDDEN_CASE_LABEL" not in json.dumps(messages)
            assert "run_affinity_ab" not in json.dumps(tools)
            self.count += 1
            if self.count == 1:
                return {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "offline-call",
                                "type": "function",
                                "function": {"name": "read_recorded_comparison", "arguments": "{}"},
                            }
                        ]
                    }
                }
            assert any(
                message.get("tool_call_id") == "offline-call"
                for message in messages
                if message["role"] == "tool"
            )
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "category": "unknown",
                                    "statement": "Synthetic software test",
                                    "missing_evidence": ["Real hardware evidence"],
                                    "counterevidence_note": "No causal comparison in mock mode.",
                                    "testable_prediction": "Measurements may resolve uncertainty.",
                                }
                            ],
                            "decision": "inconclusive",
                            "rationale": "Offline test only",
                        }
                    )
                }
            }

    result = run_agent(OfflineClient(), ex, AgentConfig(), tmp_path)
    assert result["mode"] == "mock" and result["tool_calls"] == 1
