"""Synthetic contract tests, not real hardware/model evidence."""

import hashlib
import json

import pytest

from gpu_os_copilot.agent.fake_client import tool_call
from gpu_os_copilot.agent.investigator import evidence_catalog
from gpu_os_copilot.agent.planner import ExperimentPlanner, PlanArgs
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.analysis.scheduling_study import (
    METRICS,
    READ_TOOL,
    SchedulingStudy,
    load_scheduling_study,
)
from gpu_os_copilot.cli import main
from gpu_os_copilot.collectors.scheduler import affine_alignment
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision, Hypothesis, Provenance
from gpu_os_copilot.workloads.specs import expected_configuration


@pytest.fixture
def pairs(load_fixture):
    pairs = []
    for index in range(6):
        captures = []
        for arm in ("baseline", "treatment"):
            trace = load_fixture()
            trace.run_id = trace.experiment_id = f"private-{index}-{arm}"
            trace.workload_id = "launch_test"
            for event in trace.events:
                event.source = "nsys_sqlite"
                event.confidence = "measured"
            trace.provenance = Provenance(
                source="nsys_sqlite",
                confidence="measured",
                description="Synthetic test contract, not measured hardware",
            )
            trace.operation_count = 200
            trace.wall_time_ms = 10 if arm == "baseline" else 5
            anchors = [
                {
                    "name": f"anchor-{i}",
                    "clock": "mono_raw",
                    "before_ns": 1000000 + i * 2500000 - 100,
                    "after_ns": 1000000 + i * 2500000 + 100,
                }
                for i in range(5)
            ]
            alignment = affine_alignment(anchors, [i * 2500000 for i in range(5)], "mono_raw")
            trace.configuration = {
                **expected_configuration("cnn_training"),
                "gpu": "test GPU",
                "torch": "test torch",
                "cuda": "test CUDA",
                "profiled": True,
                "capture": {
                    "cpu_set": [0] if arm == "baseline" else [2],
                    "contender_cpu": 0,
                    "cuda_trace_mode": "cuda-sw",
                    "cpu_context_switches_requested": False,
                    "sqlite_sha256": hashlib.sha256(trace.run_id.encode()).hexdigest(),
                },
                "scheduler_alignment": alignment,
                "scheduler_capture": {
                    "clock": "mono_raw",
                    "buffer_kb_per_cpu": 512,
                    "enrollment_scope": "owned_threads_after_warmup",
                    "coverage_validated": True,
                    "target_events_verified": True,
                    "loss_stats_verified": True,
                    "lost_events": 0,
                    "thread_lifetimes_verified": True,
                    "capture_start_monotonic_ns": 0,
                    "capture_end_monotonic_ns": 12000000,
                    "validated_threads": [{"pid": 100, "tid": 101}],
                    "thread_mappings": [
                        {
                            "namespace_pid": 100,
                            "namespace_tid": 101,
                            "kernel_tid": 201,
                            "method": "unique_task_comm_in_sched_debug",
                        }
                    ],
                    "thread_event_counts": {"101": {"switch_in": 3, "switch_out": 3, "wakeup": 2}},
                },
            }
            for event in trace.events:
                event.confidence = "measured"
                event.source = "linux_trace" if event.kind == "cpu_sched" else "nsys_sqlite"
                if event.kind == "cpu_sched":
                    event.attributes["kernel_tid"] = 201
                    if arm == "treatment":
                        event.end_ns = 2000000
            captures.append(trace)
        pairs.append((f"private-batch-{index // 2}", *captures))
    return pairs


def test_recomputed_cpu_changes_and_variation_are_separate_from_gpu_order(pairs):
    for _, _, after in pairs:
        api = next(e for e in after.events if e.kind == "cuda_api")
        api.end_ns = 200000
        api.start_ns = 150000
    study = SchedulingStudy(pairs[0][1], pairs)
    assert all(not value["available"] for value in study.metrics(consumed=False).values())
    report = study.assess()
    assert report["status"] == "complete"
    assert report["cpu_delay_direction"] == "consistent_decrease"
    assert report["cpu_delay_change_median_ms"] == -5
    assert report["cpu_delay_change_sample_sd_ms"] == 0
    assert report["profiled_throughput_change_median_fraction"] == 1
    assert all(
        row["treatment"]["gpu_api_timestamp_order"] == "contradicted" for row in report["pairs"]
    )
    assert report["gpu_idle_causality"] == "not_established"
    assert report["statistical_support"] == "not_evaluated"
    assert "private-" not in json.dumps(report)
    assert '"pid"' not in json.dumps(report)
    pairs[0][2].wall_time_ms = 99999
    report["pairs"].clear()
    assert len(study.assess()["pairs"]) == 6
    assert study.assess()["profiled_throughput_change_median_fraction"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "lost_events",
        "boolean_loss",
        "partial",
        "missing_mapping",
        "unknown_api_thread",
        "uncovered_thread",
        "no_positive_events",
        "clock_forgery",
        "anchor_bracket",
        "clock_scope",
        "kernel_identity",
        "simulated_event",
        "workload",
        "environment",
        "instrumentation",
        "missing_instrumentation",
        "timebase",
        "placement",
        "placement_between_pairs",
        "duplicate_id",
        "duplicate_digest",
        "unprofiled",
        "operation_count",
        "missing_capture",
    ],
)
def test_invalid_pair_cannot_be_removed_to_manufacture_consistency(pairs, mutation):
    after = pairs[-1][2]
    scheduler = after.configuration["scheduler_capture"]
    if mutation == "lost_events":
        scheduler["lost_events"] = 1
    elif mutation == "boolean_loss":
        scheduler["lost_events"] = False
    elif mutation == "partial":
        after.trace_completeness = "partial"
    elif mutation == "missing_mapping":
        scheduler["thread_mappings"] = []
    elif mutation == "unknown_api_thread":
        after.events[0].tid = None
    elif mutation == "uncovered_thread":
        scheduler["validated_threads"] = []
    elif mutation == "no_positive_events":
        scheduler["thread_event_counts"]["101"]["wakeup"] = 0
    elif mutation == "clock_forgery":
        after.configuration["scheduler_alignment"]["uncertainty_ns"] = 0
    elif mutation == "anchor_bracket":
        after.configuration["scheduler_alignment"]["anchors"][1]["after_ns"] += 100001
    elif mutation == "clock_scope":
        scheduler["capture_end_monotonic_ns"] = 1
    elif mutation == "kernel_identity":
        after.events[2].attributes["kernel_tid"] = 999
    elif mutation == "simulated_event":
        after.events[2].confidence = "derived"
    elif mutation == "workload":
        after.configuration["shape"] = [1]
    elif mutation == "environment":
        after.configuration["gpu"] = "other GPU"
    elif mutation == "instrumentation":
        after.configuration["capture"]["cuda_trace_mode"] = "cuda"
    elif mutation == "missing_instrumentation":
        scheduler.pop("buffer_kb_per_cpu")
    elif mutation == "timebase":
        after.timebase = "different_clock"
    elif mutation == "placement":
        after.configuration["capture"]["cpu_set"] = [0]
    elif mutation == "placement_between_pairs":
        after.configuration["capture"]["cpu_set"] = [3]
    elif mutation == "duplicate_id":
        after.run_id = pairs[0][2].run_id
    elif mutation == "duplicate_digest":
        after.configuration["capture"]["sqlite_sha256"] = pairs[0][2].configuration["capture"][
            "sqlite_sha256"
        ]
    elif mutation == "unprofiled":
        after.configuration["profiled"] = False
    elif mutation == "operation_count":
        after.operation_count = 201
    elif mutation == "missing_capture":
        pairs[-1] = (pairs[-1][0], pairs[-1][1], None)
    report = SchedulingStudy(pairs[0][1], pairs).assess()
    assert report["status"] == "incomplete" and len(report["pairs"]) == 6
    assert report["valid_pair_count"] == 5
    assert report["cpu_delay_change_median_ms"] is None
    assert report["cpu_delay_direction"] == "inconclusive"
    assert report["pairs"][-1]["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["few_pairs", "one_batch", "reversal", "zero_delay"])
def test_weak_or_conflicting_data_remains_visible_without_consistency_claim(pairs, kind):
    if kind == "few_pairs":
        pairs = pairs[:2]
    elif kind == "one_batch":
        pairs = [("same", a, b) for _, a, b in pairs]
    elif kind == "reversal":
        pairs[-1][2].events[2].end_ns = 8000000
    elif kind == "zero_delay":
        for _, a, b in pairs:
            a.events = [e for e in a.events if e.kind != "cpu_sched"]
            b.events = [e for e in b.events if e.kind != "cpu_sched"]
    report = SchedulingStudy(pairs[0][1], pairs).assess()
    assert report["status"] == "complete"
    assert report["cpu_delay_direction"] == "inconclusive"
    if kind == "reversal":
        assert report["pairs"][-1]["cpu_runnable_delay_change_ms"] == 1
    if kind == "zero_delay":
        assert report["cpu_delay_change_median_ms"] == 0


class NoWorker:
    def capabilities(self):
        raise AssertionError("Recorded scheduling reads must not inspect or run hardware")

    def run_experiment(self, request):
        raise AssertionError("Recorded read must never launch a workload")


def decision(*, observed=True):
    return AgentDecision(
        hypotheses=[
            Hypothesis(
                category="cpu_scheduling",
                statement="Placement response is a candidate.",
                supporting_metric_refs=["profiled_cpu_delay_change_median_ms"]
                if observed
                else ["cpu_runnable_delay_ms"],
                missing_evidence=["GPU clock validity and causal mediation"],
                counterevidence_note="Other placement mechanisms remain possible.",
                testable_prediction="An independent mechanism test might separate explanations.",
            ),
            Hypothesis(
                category="launch_overhead",
                statement="Submission overhead is another candidate.",
                supporting_metric_refs=["cuda_api_duration_ms"],
                counterevidence_note="API duration does not isolate host launch overhead.",
                testable_prediction="A launch-path intervention might change submission timing.",
            ),
        ],
        decision="inconclusive",
        rationale="Separate profiled observations do not establish GPU causality.",
    )


class Client:
    mode = "mock"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, tools):
        self.messages = messages.copy()
        value = next(self.responses)
        return {
            "message": {"content": value.model_dump_json()}
            if isinstance(value, AgentDecision)
            else {"content": None, "tool_calls": value}
        }


def test_agent_can_read_then_cite_new_cpu_measurements_without_affinity_claim(pairs, tmp_path):
    ex = ToolExecutor(NoWorker(), pairs[0][1], scheduling_study=SchedulingStudy(pairs[0][1], pairs))
    client = Client([[tool_call("scheduling", READ_TOOL, {})], decision()])
    result = run_agent(client, ex, AgentConfig(max_steps=2), tmp_path)
    assert result["scheduling_evidence"]["cpu_delay_change_median_ms"] == -5
    assert (
        result["investigation"]["ranked_hypotheses"][0]["supporting_metrics"][
            "profiled_cpu_delay_change_median_ms"
        ]["value"]
        == -5
    )
    assert result["verification"]["batch_count"] == 0 and ex.comparison_evidence() == []
    assert ex.experiment_count == 0 and result["planning"]["live_attempts"] == 0
    tool = json.loads(next(m["content"] for m in client.messages if m["role"] == "tool"))
    assert tool["catalog_metrics"]["profiled_cpu_delay_change_median_ms"]["value"] == -5
    assert "unprofiled" in tool["result"]["scope"]


def test_unread_cpu_results_never_enter_model_or_final_report(pairs, tmp_path):
    ex = ToolExecutor(NoWorker(), pairs[0][1], scheduling_study=SchedulingStudy(pairs[0][1], pairs))
    client = Client([decision(observed=False)])
    result = run_agent(client, ex, AgentConfig(max_steps=1), tmp_path)
    assert result["scheduling_evidence"]["status"] == "unread"
    opening = json.loads(client.messages[1]["content"])["evidence_catalog"]
    assert all(opening["metrics"][name]["value"] is None for name in METRICS)
    with pytest.raises(CopilotError, match="AGENT_INVESTIGATION_INVALID"):
        run_agent(Client([decision()]), ex, AgentConfig(max_steps=1), tmp_path)


def test_new_read_is_bounded_and_never_satisfies_unprofiled_claim(pairs, tmp_path):
    study = SchedulingStudy(pairs[0][1], pairs)
    ex = ToolExecutor(NoWorker(), pairs[0][1], scheduling_study=study)
    catalog = evidence_catalog(ex.trace)
    catalog["metrics"]["gpu_idle_fraction"]["value"] = 0
    planner = ExperimentPlanner(ex, AgentConfig(), catalog, {READ_TOOL})
    assert (
        planner.before_action(READ_TOOL, "{}", remaining_responses=0)["error_code"]
        == "INSUFFICIENT_FEEDBACK_BUDGET"
    )
    assert planner.before_action(READ_TOOL, "{}", remaining_responses=1) is None
    assert ex.execute(READ_TOOL, '{"path":"/private"}')["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert not ex.scheduling_read
    value = ex.execute(READ_TOOL, "{}")
    planner.after_action(READ_TOOL, value)
    assert (
        planner.before_action(READ_TOOL, "{}", remaining_responses=1)["error_code"]
        == "COMPARISON_ALREADY_CONSUMED"
    )
    assert ex.execute(READ_TOOL, "{}")["error_code"] == "COMPARISON_ALREADY_CONSUMED"
    bad = decision()
    bad.decision = "supported"
    bad.claim_scope = "affinity_throughput"
    client = Client([bad])
    client.mode = "meta"  # Protocol double; no provider request.
    with pytest.raises(CopilotError, match="AGENT_CLAIM_WITHOUT_MEASURED_COMPARISON"):
        run_agent(client, ex, AgentConfig(max_steps=1), tmp_path)


def test_optional_plan_requires_final_capacity_and_is_available_only_if_supplied(pairs):
    ex = ToolExecutor(NoWorker(), pairs[0][1], scheduling_study=SchedulingStudy(pairs[0][1], pairs))
    catalog = evidence_catalog(ex.trace)
    planner = ExperimentPlanner(ex, AgentConfig(), catalog, {READ_TOOL})
    args = PlanArgs(
        action=READ_TOOL,
        hypotheses=[h.model_dump() for h in decision(observed=False).hypotheses],
        expected_information_gain="high",
        expected_outcomes=["Delay decreases", "Delay does not decrease"],
        justification="Test a measured CPU delay response.",
    ).model_dump_json()
    result = planner.propose(
        args,
        visible_ids=set(),
        remaining_tool_calls=2,
        remaining_responses=1,
        following_call={"name": READ_TOOL, "arguments": "{}"},
    )
    assert result["result"]["status"] == "approved"
    assert result["result"]["cost_upper_bound"]["new_gpu_experiments"] == 0
    default = ToolExecutor(NoWorker(), pairs[0][1])
    assert READ_TOOL not in {s["function"]["name"] for s in default.tool_schemas()}
    assert default.execute(READ_TOOL, "{}")["error_code"] == "TOOL_NOT_ALLOWLISTED"


def write_manifest(pairs, tmp_path):
    entries = []
    for i, (batch, before, after) in enumerate(pairs):
        row = {"batch": batch}
        for arm, trace in (("baseline", before), ("treatment", after)):
            name = f"{i}-{arm}.json"
            (tmp_path / name).write_text(trace.model_dump_json())
            row[arm] = name
        entries.append(row)
    manifest = tmp_path / "study.json"
    manifest.write_text(json.dumps({"pairs": entries}))
    return manifest


def test_read_only_cli_exposes_study_without_meta_or_hardware(pairs, tmp_path, capsys, monkeypatch):
    manifest = write_manifest(pairs, tmp_path)
    monkeypatch.setattr("gpu_os_copilot.cli.LocalWorker", NoWorker)
    assert (
        main(
            [
                "diagnose",
                "--trace",
                str(tmp_path / "0-baseline.json"),
                "--scheduler-study",
                str(manifest),
                "--mode",
                "rules",
                "--json",
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["scheduling_evidence"]["cpu_delay_change_median_ms"] == -5
    assert "agent" not in value


@pytest.mark.parametrize(
    "attack", ["escape", "symlink", "absolute", "large_file", "many_pairs", "unknown_field"]
)
def test_manifest_paths_and_size_are_bounded(pairs, tmp_path, attack):
    manifest = write_manifest(pairs, tmp_path)
    data = json.loads(manifest.read_text())
    if attack == "escape":
        data["pairs"][0]["baseline"] = "../outside.json"
    elif attack == "absolute":
        data["pairs"][0]["baseline"] = "/outside.json"
    elif attack == "symlink":
        (tmp_path / "escape.json").symlink_to(tmp_path.parent / "outside.json")
        data["pairs"][0]["baseline"] = "escape.json"
    elif attack == "large_file":
        with (tmp_path / "0-baseline.json").open("wb") as f:
            f.truncate(33 * 1024**2)
    elif attack == "many_pairs":
        data["pairs"] *= 3
    elif attack == "unknown_field":
        data["shell"] = "echo forbidden"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_scheduling_study(manifest, pairs[0][1])
