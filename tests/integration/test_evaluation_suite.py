"""Synthetic contracts and offline transport only; no benchmark/model accuracy claim."""

import importlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import BenchmarkTrial, ExperimentResult, Provenance
from gpu_os_copilot.workloads.specs import expected_configuration


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return [
        importlib.import_module(name)
        for name in ("evaluation_common", "prepare_evaluation", "run_evaluation")
    ]


def test_invalidated_plan_blocks_before_source_or_provider_access(modules, tmp_path):
    (tmp_path / "invalidated.json").write_text('{"reason":"later measurement audit"}')
    with pytest.raises(ValueError, match="invalidated"):
        modules[2].validate_suite(tmp_path)


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
    source = tmp_path / "private-trace.json"
    source.write_text(trace.model_dump_json())
    paths = []
    for index in range(3):
        result = ExperimentResult(
            experiment_id=f"private-batch-{index}",
            workload_id="launch_test",
            status="ok",
            provenance=Provenance(
                source="benchmark", confidence="measured", description="Synthetic test contract"
            ),
            trials=[
                BenchmarkTrial(
                    run_id=f"private-{index}-{i}",
                    condition="baseline" if i < 5 else "treatment",
                    wall_time_ms=10 if i < 5 else 5,
                    operation_count=200,
                    throughput_ops_s=20000 if i < 5 else 40000,
                    cpu_set=[0] if i < 5 else [2],
                    configuration=trace.configuration,
                )
                for i in range(10)
            ],
        )
        path = tmp_path / f"private-batch-{index}.json"
        path.write_text(result.model_dump_json())
        paths.append(str(path))
    return {
        "trace": str(source),
        "batches": paths,
        "family": "private family",
        "role": "diagnosis",
        "rubric": {
            "top_categories": ["launch_overhead", "cpu_scheduling"],
            "comparison_read_informative": True,
            "allowed_decisions": ["inconclusive"],
        },
    }


def test_blinded_views_never_expose_private_labels_paths_or_configuration(modules, item):
    common, prepare, _ = modules
    case, _ = prepare.prepare_case(item, "case-test")
    ex = common.ReplayExecutor(case)
    visible = [ex.evidence(), ex.tool_schemas()]
    for name, arguments in [
        ("list_available_experiments", {}),
        ("read_recorded_comparison", {}),
        ("read_repeated_comparisons", {}),
        ("inspect_timeline", {"run_id": "case-test", "event_types": ["gpu_kernel"], "limit": 10}),
    ]:
        visible.append(ex.execute(name, json.dumps(arguments)))
    text = json.dumps(visible)
    for secret in (
        "TEST-ONLY-PRIVATE",
        "private family",
        "cpu_contention",
        '"configuration"',
        '"pid"',
        '"tid"',
        "private-batch-",
    ):
        assert secret not in text
    assert ex.execute("run_affinity_ab", "{}")["error_code"] == "TOOL_NOT_ALLOWLISTED"


def test_three_batches_are_exposed_only_when_read_and_duplicate_reads_do_not_add_batches(
    modules, item
):
    common, prepare, _ = modules
    case, _ = prepare.prepare_case(item, "case-test")
    ex = common.ReplayExecutor(case)
    assert ex.comparison_evidence() == [] and ex.unread_comparison_count() == 2
    ex.execute("read_recorded_comparison", "{}")
    ex.execute("read_recorded_comparison", "{}")
    assert len(ex.comparison_evidence()) == 1
    value = ex.execute("read_repeated_comparisons", "{}")["result"]
    assert len(value["additional_comparisons"]) == 1
    assert len(ex.comparison_evidence()) == 3 and ex.unread_comparison_count() == 0


def test_derived_incomplete_view_is_labeled_and_never_changes_original(modules, item):
    _, prepare, _ = modules
    original = Path(item["trace"]).read_bytes()
    case, key = prepare.prepare_case({**item, "derivation": "partial_gpu_view"}, "case-test")
    assert Path(item["trace"]).read_bytes() == original
    assert key["data_kind"] == "derived_incomplete_view"
    assert case["opening"]["input_confidence"] == "derived"
    assert case["opening"]["summary"]["gpu_idle_fraction"] is None
    assert case["trace"]["trace_completeness"] == "partial"
    assert all(event["confidence"] == "measured" for event in case["trace"]["events"])


def test_historical_rules_receive_screened_summary_not_legacy_recomputation(modules, item):
    from gpu_os_copilot.analysis.rule_baseline import diagnose

    common, prepare, _ = modules
    case, _ = prepare.prepare_case(item, "case-test")
    case["opening"]["summary"]["sched_idle_overlap_ms"] = None
    case["opening"]["summary"]["missing_metrics"].append("sched_idle_overlap_ms")
    ex = common.ReplayExecutor(case)
    # Raw fixture retains overlapping runnable events. The historical rule must
    # honor the frozen missing value, rather than recreating an unsafe overlap.
    summary = ex.summary_for_rules()
    assert summary.sched_idle_overlap_ms is None
    assert diagnose(ex.trace, summary)[0].category == "launch_overhead"
    assert summary.gpu_idle_fraction == case["opening"]["summary"]["gpu_idle_fraction"]
    assert "sched_idle_overlap_ms" in summary.missing_data


def test_temporal_contradiction_is_visible_and_overlap_withheld_in_prepared_case(modules, item):
    from gpu_os_copilot.schemas import TraceBundle

    common, prepare, _ = modules
    p = Path(item["trace"])
    trace = TraceBundle.model_validate_json(p.read_text())
    api = next(e for e in trace.events if e.kind == "cuda_api")
    api.end_ns = 150_000
    api.start_ns = 120_000
    p.write_text(trace.model_dump_json())
    original = p.read_bytes()
    case, _ = prepare.prepare_case(item, "case-test")
    assert p.read_bytes() == original
    check = case["opening"]["evidence_catalog"]["observability"]["api_gpu_correlation"]
    assert check["timestamp_order_status"] == "contradicted"
    assert case["opening"]["summary"]["sched_idle_overlap_ms"] is None
    assert common.ReplayExecutor(case).summary_for_rules().sched_idle_overlap_ms is None


def test_uncertain_replay_value_is_unscored_not_an_unnecessary_intervention(modules):
    runner = modules[2]
    key = {
        "case_id": "partial",
        "family": "test",
        "role": "robustness",
        "rubric": {
            "comparison_read_informative": None,
            "top_categories": [],
            "allowed_decisions": ["inconclusive"],
        },
    }
    row = runner.score(
        {
            "status": "ok",
            "method": "rules",
            "comparison_reads": ["read_recorded_comparison"],
            "agent": {"decision": {"decision": "inconclusive", "hypotheses": []}},
        },
        key,
    )
    assert row["recorded_test_selection_matches"] is None
    assert row["abstention_matches"] is None
    assert row["low_value_recorded_reads"] is None
    assert row["repeated_recorded_reads"] == 0
    group = next(
        g for g in runner.aggregate([row]) if g["method"] == "rules" and g["role"] == "robustness"
    )
    assert group["low_value_read_scored_cases"] == 0
    assert group["low_value_recorded_reads"] is None


@pytest.mark.parametrize("mutation", ["different_gpu", "partial_result", "duplicate_batch"])
def test_invalid_recorded_comparisons_rejected_before_model_calls(modules, item, mutation):
    common, prepare, _ = modules
    case, _ = prepare.prepare_case(item, "case-test")
    if mutation == "different_gpu":
        for trial in case["batches"][1]["trials"]:
            trial["configuration"]["gpu"] = "other GPU"
    elif mutation == "partial_result":
        case["batches"][1]["status"] = "partial"
    else:
        case["batches"][1]["experiment_id"] = case["batches"][0]["experiment_id"]
    with pytest.raises(ValueError):
        common.ReplayExecutor(case)


class Delegate:
    mode = "mock"
    config = AgentConfig()
    calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return {"message": {"content": "{}"}, "usage": None}


@pytest.mark.parametrize("drift", ["opening", "tool", "extra_tool"])
def test_input_or_tool_drift_blocks_before_delegate_call(modules, item, drift):
    common, prepare, _ = modules
    ex = common.ReplayExecutor(prepare.prepare_case(item, "case-test")[0])
    delegate = Delegate()
    guard = common.GuardedClient(delegate, ex.evidence(), ex.tool_schemas())
    opening, tools = ex.evidence(), ex.tool_schemas()
    if drift == "opening":
        opening["input_confidence"] = "changed"
    elif drift == "tool":
        tools[0]["function"]["description"] = "changed"
    else:
        tools.append({"function": {"name": "execute_shell"}})
    with pytest.raises(CopilotError, match="EVALUATION_"):
        guard.complete([{}, {"content": json.dumps(opening)}], tools)
    assert delegate.calls == 0


def test_shared_call_budget_also_protects_legacy_runner(modules, item, tmp_path):
    common, prepare, _ = modules
    ex = common.ReplayExecutor(prepare.prepare_case(item, "case-test")[0])
    delegate = Delegate()
    checkpoint = tmp_path / "usage.json"
    guard = common.GuardedClient(delegate, ex.evidence(), ex.tool_schemas(), checkpoint)
    for _ in range(common.LIMITS["max_steps"]):
        guard.complete([{}, {"content": json.dumps(ex.evidence())}], ex.tool_schemas())
    with pytest.raises(CopilotError, match="REQUEST_BUDGET"):
        guard.complete([{}, {"content": json.dumps(ex.evidence())}], ex.tool_schemas())
    assert delegate.calls == common.LIMITS["max_steps"]
    assert not json.loads(checkpoint.read_text())["usage_complete"]


def test_guard_allows_only_explicit_final_tool_withdrawal_without_extra_budget(modules, item):
    common, prepare, _ = modules
    ex = common.ReplayExecutor(prepare.prepare_case(item, "case-test")[0])
    delegate = Delegate()
    guard = common.GuardedClient(delegate, ex.evidence(), ex.tool_schemas())
    messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": json.dumps(ex.evidence())},
    ]
    with pytest.raises(CopilotError, match="TOOL_DRIFT"):
        guard.complete(messages, [])
    assert delegate.calls == 0
    for _ in range(common.LIMITS["max_steps"] - 1):
        guard.complete(messages, ex.tool_schemas())
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "controller_state": {"final_response_required": True},
                }
            ),
        }
    )
    with pytest.raises(CopilotError, match="TOOL_DRIFT"):
        guard.complete(messages, ex.tool_schemas()[:-1])
    guard.complete(messages, [])
    assert delegate.calls == common.LIMITS["max_steps"]
    with pytest.raises(CopilotError, match="REQUEST_BUDGET"):
        guard.complete(messages, [])


def test_scoring_uses_consumed_actions_and_keeps_failed_sessions(modules, item):
    _, _, run = modules
    key = {
        "case_id": "case-test",
        "family": "control",
        "role": "diagnosis",
        "rubric": {
            "top_categories": ["gpu_compute"],
            "comparison_read_informative": False,
            "allowed_decisions": ["inconclusive"],
        },
    }
    result = {
        "method": "legacy",
        "status": "ok",
        "comparison_reads": [],
        "agent": {
            "decision": {
                "hypotheses": [{"category": "gpu_compute"}],
                "decision": "inconclusive",
                "selected_experiment": {"name": "affinity_ab"},
            },
        },
    }
    assert run.score(result, key)["abstention_matches"]
    result["comparison_reads"] = ["read_recorded_comparison"]
    assert run.score(result, key)["low_value_recorded_reads"] == 1
    result["status"] = "failed"
    assert not run.score(result, key)["diagnosis_matches_symptom_rubric"]
    assert run.score(result, key)["actual_unnecessary_hardware_interventions"] is None


def test_archive_cannot_escape_source_directory(modules, tmp_path):
    _, prepare, _ = modules
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.py")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        prepare.safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "escape.py").exists()


def test_runtime_metrics_are_recomputed_only_with_explicit_opt_in(modules, item):
    common, prepare, _ = modules
    case, _ = prepare.prepare_case(item, "case-test")
    case["opening"]["summary"]["gpu_idle_fraction"] = 999
    assert common.ReplayExecutor(case).evidence()["summary"]["gpu_idle_fraction"] == 999
    case["runtime_evidence"] = True
    ex = common.ReplayExecutor(case)
    assert 0 <= ex.evidence()["summary"]["gpu_idle_fraction"] <= 1
    assert '"configuration"' not in json.dumps(ex.evidence())
    assert "private-" not in json.dumps(ex.evidence())


def test_per_method_input_hashes_are_strict_and_keep_legacy_fallback(modules):
    run = modules[2]
    key = {"opening_sha256": "common", "common_tools_sha256": "tools"}
    result = {"opening_sha256": "common", "common_tools_sha256": "tools"}
    run.validate_result_inputs(result, key, "legacy")
    key["opening_sha256_by_method"] = {"legacy": "old", "enhanced": "new"}
    with pytest.raises(ValueError):
        run.validate_result_inputs(result, key, "legacy")
    result["opening_sha256"] = "old"
    run.validate_result_inputs(result, key, "legacy")
    with pytest.raises(ValueError):
        run.validate_result_inputs(result, key, "enhanced")
    with pytest.raises(KeyError):
        run.validate_result_inputs(result, key, "unknown")


@pytest.mark.parametrize("mutation", ["bytes", "escape", "symlink"])
def test_frozen_scheduler_data_is_verified_before_execution(modules, tmp_path, mutation):
    common, prepare, run = modules
    answer = tmp_path / "private-answer-key.json"
    answer.write_text("[]")
    data = tmp_path / "study.json"
    data.write_text("{}")
    name, expected = data.name, prepare.sha(data)
    if mutation == "bytes":
        data.write_text('{"changed": true}')
    else:
        outside = tmp_path.parent / (tmp_path.name + "-outside.json")
        outside.write_text("{}")
        if mutation == "escape":
            name = "../" + outside.name
        else:
            data.unlink()
            data.symlink_to(outside)
    plan = {
        "answer_key_sha256": prepare.sha(answer),
        "harness_sha256": {},
        "source_sha256": {},
        "cases": [],
        "scheduler_data_sha256": {name: expected},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="scheduling|Scheduling"):
        run.validate_suite(tmp_path)
