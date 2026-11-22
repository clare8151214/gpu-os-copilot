"""Synthetic contracts for the removal-experiment tool; no hardware or model evidence."""

import importlib
import json
from pathlib import Path

import pytest

from gpu_os_copilot.agent.fake_client import tool_call
from gpu_os_copilot.agent.removal import TOOL, RemovalExperiments, measured_effect
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.config import AgentConfig
from gpu_os_copilot.schemas import BenchmarkTrial, ExperimentResult, Provenance
from gpu_os_copilot.workloads.specs import expected_configuration


def raw_batch(base, removed, scenario="mixed_private"):
    trials = []
    for index, (b, r) in enumerate(zip(base, removed, strict=True)):
        order = [("base", b), ("ablated", r)]
        for condition, value in order if index % 2 == 0 else order[::-1]:
            trials.append(
                {"batch": 0, "condition": condition, "throughput_ops_s": value, "wall_time_ms": 1}
            )
    return {"scenario": scenario, "factor": "private", "status": "ok", "trials": trials}


def test_measured_effect_uses_pair_medians_and_window_share():
    effect = measured_effect(raw_batch([100, 110, 90, 100, 105], [200, 210, 190, 200, 220]))
    assert effect["status"] == "ok" and effect["pairs"] == 5
    assert effect["throughput_gain_fraction"] == pytest.approx(1.0)
    assert effect["recovered_window_share"] == pytest.approx(0.5)
    assert effect["pair_gain_range"][0] < 1 < effect["pair_gain_range"][1]
    assert measured_effect({"status": "failed", "error": "x"})["status"] == "failed"
    broken = raw_batch([100, 100], [120, 120])
    broken["trials"][1]["condition"] = "base"
    with pytest.raises(ValueError):
        measured_effect(broken)


def test_budget_duplicates_and_unoffered_categories_do_not_reveal_results():
    recorded = {
        "data_transfer": raw_batch([100] * 5, [300] * 5),
        "synchronization": raw_batch([100] * 5, [110] * 5),
        "launch_overhead": raw_batch([100] * 5, [150] * 5),
    }
    runs = []
    experiments = RemovalExperiments(
        ["data_transfer", "synchronization", "launch_overhead"],
        lambda category: runs.append(category) or recorded[category],
        execution="test replay",
    )
    enum = experiments.tool_schema()["function"]["parameters"]["properties"]["category"]["enum"]
    assert enum == ["data_transfer", "launch_overhead", "synchronization"]
    assert all(m["availability_state"] == "not_run" for m in experiments.metrics().values())
    first = experiments.execute('{"category":"data_transfer"}')["result"]
    assert first["recovered_window_share"] == pytest.approx(2 / 3)
    assert first["experiments_remaining"] == 1
    again = experiments.execute('{"category":"data_transfer"}')
    assert again["error_code"] == "EXPERIMENT_ALREADY_RUN" and "result" not in again
    assert experiments.execute('{"category":"gpu_compute"}')["error_code"] == (
        "INVALID_TOOL_ARGUMENTS"
    )
    experiments.execute('{"category":"synchronization"}')
    third = experiments.execute('{"category":"launch_overhead"}')
    assert third["error_code"] == "EXPERIMENT_BUDGET_EXCEEDED"
    assert runs == ["data_transfer", "synchronization"]
    metrics = experiments.metrics()
    assert metrics["measured_removal_share_data_transfer"]["available"] is True
    assert metrics["measured_removal_share_launch_overhead"]["availability_state"] == "not_run"
    assert experiments.report()["run_order"] == ["data_transfer", "synchronization"]
    limited = RemovalExperiments(["synchronization"], recorded.get, execution="test replay")
    assert limited.execute('{"category":"data_transfer"}')["error_code"] == "REMOVAL_NOT_OFFERED"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return [importlib.import_module(n) for n in ("evaluation_common", "prepare_evaluation")]


@pytest.fixture
def item(load_fixture, tmp_path):
    trace = load_fixture()
    for event in trace.events:
        event.source, event.confidence = "nsys_sqlite", "measured"
    trace.provenance = Provenance(
        source="nsys_sqlite", confidence="measured", description="TEST-ONLY-PRIVATE"
    )
    trace.configuration = {
        **expected_configuration("cpu_contention"),
        "gpu": "test GPU",
        "cuda": "test CUDA",
        "torch": "test torch",
    }
    (tmp_path / "trace.json").write_text(trace.model_dump_json())
    batches = []
    for index in range(2):
        result = ExperimentResult(
            experiment_id=f"private-batch-{index}",
            workload_id="launch_test",
            status="ok",
            provenance=Provenance(source="benchmark", confidence="measured", description="Test"),
            trials=[
                BenchmarkTrial(
                    run_id=f"private-{index}-{i}",
                    condition="baseline" if i < 5 else "treatment",
                    wall_time_ms=10,
                    operation_count=200,
                    throughput_ops_s=20000,
                    cpu_set=[0] if i < 5 else [2],
                    configuration=trace.configuration,
                )
                for i in range(10)
            ],
        )
        batches.append(tmp_path / f"batch-{index}.json")
        batches[-1].write_text(result.model_dump_json())
    removal = {}
    for factor, removed in (("copy", 300), ("sync", 110)):
        removal[factor] = tmp_path / f"private-{factor}.json"
        removal[factor].write_text(json.dumps(raw_batch([100] * 5, [removed] * 5)))
    return {
        "trace": str(tmp_path / "trace.json"),
        "batches": [str(p) for p in batches],
        "removal_experiments": {factor: str(path) for factor, path in removal.items()},
        "family": "private family",
        "role": "diagnosis",
        "rubric": {"top_categories": ["data_transfer"], "allowed_decisions": ["inconclusive"]},
        "runtime_evidence": True,
    }


def test_prepared_case_offers_only_present_factors_without_private_names(modules, item):
    common, prepare = modules
    case, key = prepare.prepare_case(item, "case-test")
    assert sorted(case["removal_experiments"]) == ["data_transfer", "synchronization"]
    assert all(path in key["sources_sha256"] for path in item["removal_experiments"].values())
    ex = common.ReplayExecutor(case)
    visible = [ex.evidence(), ex.tool_schemas(), ex.execute(TOOL, '{"category":"data_transfer"}')]
    text = json.dumps(visible) + json.dumps(case["removal_experiments"])
    for secret in ("mixed_private", "private-", '"factor"', "TEST-ONLY-PRIVATE"):
        assert secret not in text
    metrics = ex.additional_metrics()
    assert metrics["measured_removal_share_data_transfer"]["value"] == pytest.approx(2 / 3)
    assert "measured_removal_share_launch_overhead" not in metrics
    without = dict(case)
    without.pop("removal_experiments")
    plain = common.ReplayExecutor(without)
    names = {t["function"]["name"] for t in plain.tool_schemas()}
    assert TOOL not in names
    # The version-specific tool is hashed per method; the common tools stay matched.
    assert ex.common_tool_schemas() == plain.tool_schemas() == plain.common_tool_schemas()
    no_runtime = {**case, "runtime_evidence": False}
    with pytest.raises(ValueError, match="runtime evidence"):
        common.ReplayExecutor(no_runtime)


def hypothesis(category, support, statement):
    return {
        "category": category,
        "statement": statement,
        "evidence_event_ids": [],
        "supporting_metric_refs": support,
        "contradicting_metric_refs": [],
        "missing_metric_refs": [],
        "missing_evidence": [],
        "counterevidence_note": "Single bounded batch; no sole cause is established.",
        "testable_prediction": "Removing this factor again should recover a similar share.",
        "confidence": "medium",
    }


def test_agent_runs_experiments_cites_measurements_and_reports_them(modules, item, tmp_path):
    common, prepare = modules
    case, _ = prepare.prepare_case(item, "case-test")
    ex = common.ReplayExecutor(case)
    seen = []

    class Client:
        mode = "meta"  # Test double; network is blocked.

        def __init__(self):
            self.turn = 0

        def complete(self, messages, tools):
            seen.append((messages.copy(), tools))
            self.turn += 1
            if self.turn == 1:
                return {
                    "message": {
                        "tool_calls": [
                            tool_call("r-1", TOOL, {"category": "data_transfer"}),
                            tool_call("r-2", TOOL, {"category": "synchronization"}),
                        ]
                    }
                }
            decision = {
                "hypotheses": [
                    hypothesis(
                        "data_transfer",
                        ["measured_removal_share_data_transfer"],
                        "Removing copies recovered "
                        "{{percent:measured_removal_share_data_transfer}}.",
                    ),
                    hypothesis(
                        "synchronization",
                        ["measured_removal_share_synchronization"],
                        "Removing synchronizes recovered less.",
                    ),
                ],
                "decision": "inconclusive",
                "rationale": "Measured removal shares rank copies first.",
            }
            return {"message": {"content": json.dumps(decision)}}

    report = run_agent(Client(), ex, AgentConfig(max_steps=3), tmp_path / "audit")
    assert [h["category"] for h in report["decision"]["hypotheses"]][0] == "data_transfer"
    assert "66.66" in report["decision"]["hypotheses"][0]["statement"]
    assert report["removal_experiments"]["run_order"] == ["data_transfer", "synchronization"]
    system = seen[0][0][0]["content"]
    assert "run_removal_experiment removes one designed factor" in system
    tool_results = [m for m in seen[1][0] if m["role"] == "tool"]
    assert json.loads(tool_results[0]["content"])["result"]["category"] == "data_transfer"


def test_prompt_is_unchanged_when_no_removal_tool_is_offered(modules, item, tmp_path):
    common, prepare = modules
    plain = {key: value for key, value in item.items() if key != "removal_experiments"}
    case, _ = prepare.prepare_case(plain, "case-test")
    captured = []

    class Client:
        mode = "meta"

        def complete(self, messages, tools):
            captured.append(messages[0]["content"])
            raise RuntimeError("stop")

    with pytest.raises(Exception):  # noqa: B017 - only the first prompt matters here.
        run_agent(Client(), common.ReplayExecutor(case), AgentConfig(max_steps=2), tmp_path)
    assert "run_removal_experiment" not in captured[0]


class FakeWorker:
    """Records removal requests; never starts processes."""

    def __init__(self):
        self.requests = []

    def capabilities(self):
        from gpu_os_copilot.schemas import WorkerCapabilities

        return WorkerCapabilities(
            os="test",
            python_version="3",
            cpu_affinity=[3, 1],
            gpus=[{"model": "test GPU"}],
            cuda_available=True,
            cuda_version="test CUDA",
            torch_version="test torch",
        )

    def run_removal(self, request):
        self.requests.append(request)
        return raw_batch([100] * 5, [150] * 5)


def bound_trace(load_fixture, scenario="mixed_r1"):
    trace = load_fixture()
    for event in trace.events:
        event.source, event.confidence = "nsys_sqlite", "measured"
    trace.provenance = Provenance(source="nsys_sqlite", confidence="measured", description="T")
    trace.workload_id = "launch_test"
    trace.configuration = {
        **expected_configuration(scenario),
        "gpu": "test GPU",
        "cuda": "test CUDA",
        "torch": "test torch",
    }
    return trace


def test_live_tool_needs_approval_binding_and_matching_hardware(load_fixture):
    from gpu_os_copilot.agent.tools import ToolExecutor

    worker = FakeWorker()
    executor = ToolExecutor(worker, bound_trace(load_fixture), owner_approved=True)
    names = {t["function"]["name"] for t in executor.tool_schemas()}
    assert TOOL in names
    enum = executor.removal.tool_schema()["function"]["parameters"]["properties"]["category"]
    assert enum["enum"] == ["data_transfer", "launch_overhead"]  # mixed_r1: burst + copy.
    result = executor.execute(TOOL, '{"category":"data_transfer"}')["result"]
    assert result["execution"] == "new project child processes"
    request = worker.requests[0]
    assert (request.scenario, request.factor, request.cpu) == ("mixed_r1", "copy", 1)
    assert request.owner_approved is True and request.pairs == 5
    assert executor.additional_metrics()["measured_removal_share_data_transfer"]["available"]
    assert ToolExecutor(worker, bound_trace(load_fixture)).removal is None
    unbound = bound_trace(load_fixture)
    unbound.configuration["gpu"] = "other GPU"
    assert ToolExecutor(worker, unbound, owner_approved=True).removal is None
    assert ToolExecutor(worker, load_fixture(), owner_approved=True).removal is None
    single = ToolExecutor(worker, bound_trace(load_fixture, "sync_device_a"), owner_approved=True)
    assert single.removable_factors() == {"synchronization": None}


def test_local_worker_removal_requires_approval_and_one_bounded_batch(monkeypatch):
    from gpu_os_copilot.schemas import RemovalRequest
    from gpu_os_copilot.worker.local import LocalWorker

    calls = []
    monkeypatch.setattr(
        "gpu_os_copilot.experiments.ablation.run_ablation",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "ok"},
    )
    provenance = Provenance(source="benchmark", confidence="derived", description="Test")
    request = RemovalRequest(
        experiment_id="r", scenario="mixed_r1", factor="copy", cpu=1, provenance=provenance
    )
    assert LocalWorker().run_removal(request)["error"] == "OWNER_APPROVAL_REQUIRED"
    assert calls == []
    with pytest.raises(ValueError, match="mismatch"):
        LocalWorker().run_removal(request.model_copy(update={"worker_id": "other"}))
    LocalWorker().run_removal(request.model_copy(update={"owner_approved": True}))
    assert calls == [
        (("mixed_r1", 1), {"factor": "copy", "batches": 1, "trials": 5, "timeout_s": 60})
    ]
    with pytest.raises(ValueError):
        RemovalRequest(
            experiment_id="r", scenario="../x", factor="copy", cpu=1, provenance=provenance
        )
