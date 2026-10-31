"""Offline plan/execution boundaries; the executor below never launches a workload."""

import json

import pytest

from gpu_os_copilot.agent.fake_client import tool_call
from gpu_os_copilot.agent.investigator import evidence_catalog
from gpu_os_copilot.agent.planner import ExperimentPlanner, PlanArgs, plan_schema
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import (
    AgentDecision,
    BenchmarkTrial,
    ExperimentResult,
    Hypothesis,
    Provenance,
)


def hypotheses():
    return [
        Hypothesis(
            category="cpu_scheduling",
            statement="Placement may contribute to launch gaps.",
            supporting_metric_refs=["gpu_idle_fraction"],
            missing_evidence=["Causal placement attribution"],
            counterevidence_note="Gaps could arise from launch cost instead.",
            testable_prediction="Placement could reduce gaps if this candidate matters.",
        ),
        Hypothesis(
            category="gpu_compute",
            statement="Kernel cost may instead dominate.",
            supporting_metric_refs=["kernel_total_ms"],
            contradicting_metric_refs=["gpu_idle_fraction"],
            counterevidence_note="GPU idle gaps weaken this candidate.",
            testable_prediction="An irrelevant CPU change would not reduce GPU kernel cost.",
        ),
    ]


def arguments(action="run_affinity_ab", **changes):
    return (
        PlanArgs(
            action=action,
            hypotheses=[h.model_dump() for h in hypotheses()],
            expected_information_gain="high",
            expected_outcomes=[
                "Improvement favors a placement contribution.",
                "No improvement weakens placement as an explanation.",
            ],
            justification="Distinguish placement response from fixed kernel cost.",
            experiment_arguments={
                "workload_id": "launch_test",
                "candidate_cpu_set": [2],
                "repeats": 5,
            }
            if action == "run_affinity_ab"
            else None,
        )
        .model_copy(update=changes)
        .model_dump_json()
    )


class Executor:
    owner_approved = True

    def __init__(self, trace):
        self.trace = trace
        self.executions = 0
        self.reads = []
        self.environment_matches = True

    def evidence(self):
        return {"run_id": self.trace.run_id}

    def workload_bound(self):
        return True

    def local_configuration_matches(self):
        return self.environment_matches

    def comparison_evidence(self):
        return []

    def unread_comparison_count(self):
        return 0

    def execute(self, name, args):
        if name == "list_available_experiments":
            return {
                "result": {
                    "experiments": [
                        {"name": "affinity_ab", "can_execute": True, "candidate_cpu_sets": [[2]]}
                    ]
                }
            }
        if name == "run_affinity_ab":
            self.executions += 1
            return {"result": {"status": "ok"}}  # Protocol double, no measured trials.
        if name.startswith("read_"):
            self.reads.append(name)
        return {"result": {"source": "measured", "execution": "recorded replay"}}


def planner(executor, **config):
    return ExperimentPlanner(
        executor,
        AgentConfig(**config),
        evidence_catalog(executor.trace),
        {"run_affinity_ab", "read_recorded_comparison"},
    )


def propose(planner, args=None, **budgets):
    return planner.propose(
        args or arguments(),
        visible_ids=set(),
        remaining_tool_calls=budgets.get("tools", 4),
        remaining_responses=budgets.get("responses", 3),
    )["result"]


def test_plan_estimates_cost_and_is_not_execution(load_fixture):
    ex = Executor(load_fixture())
    plans = planner(ex)
    result = propose(plans)
    assert result["status"] == "approved" and ex.executions == 0
    assert result["cost_upper_bound"]["new_gpu_experiments"] == 1
    assert result["cost_upper_bound"]["max_wall_time_s"] == 60
    assert result["information_gain_source"] == "model_prediction_not_measurement"


@pytest.mark.parametrize("action", ["read_recorded_comparison", "run_affinity_ab"])
@pytest.mark.parametrize("responses", [0, 1, 2])
def test_separate_plan_reserves_action_and_final_responses(load_fixture, action, responses):
    ex = Executor(load_fixture())
    plans = planner(ex)
    result = propose(plans, arguments(action), responses=responses)
    assert (result["status"] == "approved") == (responses >= 2)
    assert result["cost_upper_bound"]["additional_model_responses_including_final"] == 2
    if responses < 2:
        assert "INSUFFICIENT_FEEDBACK_BUDGET" in result["reason_codes"]
        assert plans.pending is None
    assert ex.executions == 0


@pytest.mark.parametrize("queued", ["matching", "mismatch", "malformed", "different_action"])
def test_one_remaining_response_requires_matching_action_already_queued(load_fixture, queued):
    ex = Executor(load_fixture())
    plans = planner(ex)
    actual = json.loads(arguments())["experiment_arguments"]
    if queued == "mismatch":
        actual["repeats"] = 2
    following = {
        "name": "run_affinity_ab" if queued != "different_action" else "read_recorded_comparison",
        "arguments": "not JSON" if queued == "malformed" else json.dumps(actual),
    }
    result = plans.propose(
        arguments(),
        visible_ids=set(),
        remaining_tool_calls=1,
        remaining_responses=1,
        following_call=following,
    )["result"]
    assert (result["status"] == "approved") == (queued == "matching")
    assert ex.executions == 0


def test_fifth_turn_plan_is_blocked_before_reserved_sixth_turn(load_fixture, tmp_path):
    ex = Executor(load_fixture())
    messages = [
        {"tool_calls": [tool_call(f"summary-{i}", "get_performance_summary", {})]} for i in range(4)
    ]
    messages += [
        {"tool_calls": [tool_call("late-plan", "plan_next_action", json.loads(arguments()))]},
        final_response(),
    ]
    client = Responses(messages)
    result = run_agent(
        client,
        ex,
        AgentConfig(
            max_steps=6, max_total_completion_tokens=24576, max_session_request_bytes=786432
        ),
        tmp_path,
    )
    plan = result["planning"]["plans"][0]
    assert plan["remaining_responses"] == 1
    assert plan["status"] == "blocked"
    assert "INSUFFICIENT_FEEDBACK_BUDGET" in plan["reason_codes"]
    assert result["planning"]["pending_action"] is None
    assert result["model_budget"]["requests_reserved"] == 6
    assert ex.executions == 0


def test_batched_plan_and_live_action_keep_final_response(load_fixture, tmp_path):
    ex = Executor(load_fixture())
    plan = json.loads(arguments())
    client = Responses(
        [
            {
                "tool_calls": [
                    tool_call("plan", "plan_next_action", plan),
                    tool_call("run", "run_affinity_ab", plan["experiment_arguments"]),
                ]
            },
            final_response(),
        ]
    )
    result = run_agent(client, ex, AgentConfig(max_steps=2, max_experiments=1), tmp_path)
    assert ex.executions == 1
    assert result["planning"]["plans"][0]["status"] == "approved"
    assert result["model_budget"]["requests_reserved"] == 2


@pytest.mark.parametrize("reason", ["budget", "response", "tool", "environment", "cpu", "low_gain"])
def test_plan_cannot_override_observable_limits(load_fixture, reason):
    ex = Executor(load_fixture())
    plans = planner(ex, max_experiments=0 if reason == "budget" else 2)
    data = json.loads(arguments())
    if reason == "environment":
        ex.environment_matches = False
    elif reason == "cpu":
        data["experiment_arguments"]["candidate_cpu_set"] = [99]
    elif reason == "low_gain":
        data["expected_information_gain"] = "low"
    result = propose(
        plans,
        json.dumps(data),
        responses=0 if reason == "response" else 3,
        tools=0 if reason == "tool" else 4,
    )
    assert result["status"] == "blocked" and result["reason_codes"]
    assert ex.executions == 0 and plans.pending is None


def test_dense_gpu_control_is_low_value_for_first_live_affinity_test(load_fixture):
    ex = Executor(load_fixture("compute_bound"))
    result = propose(planner(ex))
    assert "LOW_AFFINITY_INFORMATION_VALUE" in result["reason_codes"]
    assert result["status"] == "blocked"


@pytest.mark.parametrize("scenario", ["compute_bound", "transfer_bound"])
def test_low_value_replay_cannot_bypass_policy_with_high_gain_or_direct_read(
    load_fixture, scenario
):
    ex = Executor(load_fixture(scenario))
    plans = planner(ex)
    result = propose(plans, arguments("read_recorded_comparison"))
    assert result["status"] == "blocked"
    assert "LOW_AFFINITY_INFORMATION_VALUE" in result["reason_codes"]
    for action in ("read_recorded_comparison", "read_repeated_comparisons"):
        denied = plans.before_action(action, "{}", remaining_responses=2)
        assert denied["error_code"] == "LOW_AFFINITY_INFORMATION_VALUE"
    assert not ex.reads and not plans.report()["abstained"]


@pytest.mark.parametrize("prior", ["supported", "rejected"])
def test_dense_gpu_does_not_prevent_reassessment_of_prior_claim(load_fixture, prior):
    ex = Executor(load_fixture("compute_bound"))
    ex.prior_decision = {"decision": prior}
    plans = planner(ex)
    plans.tool_names.add("read_repeated_comparisons")
    result = propose(plans, arguments("read_repeated_comparisons"))
    assert result["status"] == "approved"
    assert plans.recorded_read_policy()["prior_claim_needs_reassessment"]
    assert plans.before_action("read_repeated_comparisons", "{}", remaining_responses=1) is None
    assert (
        plans.before_action("read_recorded_comparison", "{}", remaining_responses=1)["error_code"]
        == "COMPARISON_ALREADY_CONSUMED"
    )


def comparison_batch(treatment_time):
    """Synthetic contract values, never hardware evidence."""
    return ExperimentResult(
        experiment_id="test-observed-batch",
        workload_id="launch_test",
        status="ok",
        provenance=Provenance(
            source="benchmark", confidence="measured", description="Synthetic test contract only"
        ),
        trials=[
            BenchmarkTrial(
                run_id=f"test-{condition}-{i}",
                condition=condition,
                wall_time_ms=value,
                operation_count=200,
                throughput_ops_s=200000 / value,
                cpu_set=[0] if condition == "baseline" else [2],
                configuration={"operations": 200},
            )
            for condition, value in (("baseline", 10), ("treatment", treatment_time))
            for i in range(5)
        ],
    )


@pytest.mark.parametrize("time,allowed", [(5, True), (20, True), (10, False)])
def test_replay_value_uses_consumed_effect_not_hidden_later_results(load_fixture, time, allowed):
    class VisibleOnly(Executor):
        @property
        def results(self):
            raise AssertionError("Planner must not inspect unread batches")

        def comparison_evidence(self):
            return [comparison_batch(time)]

    ex = VisibleOnly(load_fixture("compute_bound"))
    plans = planner(ex)
    policy = plans.recorded_read_policy()
    assert (policy["status"] == "available") == allowed
    assert policy["consumed_effect_needs_reassessment"] == allowed
    denied = plans.before_action("read_repeated_comparisons", "{}", remaining_responses=1)
    assert (denied is None) == allowed


def test_missing_gpu_coverage_is_not_a_low_value_observation(load_fixture):
    trace = load_fixture("compute_bound")
    trace.trace_completeness = "partial"
    ex = Executor(trace)
    plans = planner(ex)
    assert plans.catalog["metrics"]["gpu_idle_fraction"]["value"] is None
    assert plans.recorded_read_policy()["status"] == "available"
    assert plans.before_action("read_recorded_comparison", "{}", remaining_responses=1) is None


@pytest.mark.parametrize("scenario", ["compute_bound", "transfer_bound"])
def test_runtime_blocks_low_value_reads_without_counting_model_abstention(
    load_fixture, tmp_path, scenario
):
    ex = Executor(load_fixture(scenario))
    ex.unread_comparison_count = lambda: 2 - len(ex.reads)
    client = Responses(
        [
            {
                "tool_calls": [
                    tool_call("first", "read_recorded_comparison", {}),
                    tool_call("later", "read_repeated_comparisons", {}),
                ]
            },
            final_response(),
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=2), tmp_path)
    assert not ex.reads and ex.executions == 0
    assert not report["planning"]["abstained"]
    assert {a["error_code"] for a in report["planning"]["action_history"]} == {
        "LOW_AFFINITY_INFORMATION_VALUE"
    }
    assert report["verification"]["unread_comparisons"] == 2
    assert report["decision"]["decision"] == "inconclusive"


def test_explicit_abstention_withdraws_tools_on_next_response(load_fixture, tmp_path):
    ex = Executor(load_fixture())

    class StopClient(Responses):
        turns = 0

        def complete(self, messages, tools):
            self.turns += 1
            if self.turns == 2:
                state = json.loads(messages[-1]["content"])["controller_state"]
                assert state["planner_abstained"] and state["final_response_required"]
                assert tools == []
                assert state["remaining_responses"] > 1
            return super().complete(messages, tools)

    client = StopClient(
        [
            {
                "tool_calls": [
                    tool_call(
                        "stop",
                        "plan_next_action",
                        json.loads(
                            arguments(
                                "abstain", expected_information_gain="low", expected_outcomes=[]
                            )
                        ),
                    )
                ]
            },
            final_response(),
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=5), tmp_path)
    assert report["planning"]["abstained"] and client.turns == 2
    assert not ex.reads and ex.executions == 0


def test_replay_available_without_live_authorization_and_second_read_stops(load_fixture):
    ex = Executor(load_fixture())
    ex.owner_approved = False
    plans = planner(ex)
    result = propose(plans, arguments("read_recorded_comparison"))
    assert result["status"] == "approved"
    assert result["cost_upper_bound"]["new_gpu_experiments"] == 0
    assert plans.before_action("read_recorded_comparison", "{}", remaining_responses=2) is None
    plans.after_action("read_recorded_comparison", ex.execute("read_recorded_comparison", "{}"))
    assert (
        plans.before_action("read_recorded_comparison", "{}", remaining_responses=2)["error_code"]
        == "COMPARISON_ALREADY_CONSUMED"
    )
    assert propose(plans, arguments("read_recorded_comparison"))["status"] == "blocked"


def test_abstention_stops_new_actions(load_fixture):
    ex = Executor(load_fixture())
    plans = planner(ex)
    result = propose(plans, arguments("abstain", expected_information_gain="low"))
    assert result["status"] == "abstained"
    assert (
        plans.before_action("run_affinity_ab", "{}", remaining_responses=4)["error_code"]
        == "PLANNER_STOPPED"
    )
    assert plans.report()["abstained"] and ex.executions == 0


class Responses:
    mode = "meta"  # Test double for the authorized runtime path; network is blocked.

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, tools):
        assert not tools or any(tool["function"]["name"] == "plan_next_action" for tool in tools)
        self.messages = messages.copy()
        return {"message": next(self.responses)}


def final_response():
    return {
        "content": AgentDecision(
            hypotheses=hypotheses(),
            decision="inconclusive",
            rationale="Offline protocol test; no measurements.",
        ).model_dump_json()
    }


@pytest.mark.parametrize("case", ["valid", "no_plan", "changed_args", "zero_budget"])
def test_runtime_intercepts_execution_before_worker(load_fixture, tmp_path, case):
    ex = Executor(load_fixture())
    plan = json.loads(arguments())
    actual = dict(plan["experiment_arguments"])
    if case == "changed_args":
        actual["repeats"] = 2
    calls = (
        [] if case == "no_plan" else [{"tool_calls": [tool_call("plan", "plan_next_action", plan)]}]
    )
    calls.extend([{"tool_calls": [tool_call("run", "run_affinity_ab", actual)]}, final_response()])
    client = Responses(calls)
    result = run_agent(
        client,
        ex,
        AgentConfig(max_steps=3, max_experiments=0 if case == "zero_budget" else 1),
        tmp_path,
    )
    assert ex.executions == (1 if case == "valid" else 0)
    assert result["planning"]["live_attempts"] == (1 if case == "valid" else 0)
    assert result["planning"]["action_history"][0]["status"] == (
        "consumed" if case == "valid" else "not_executed_or_failed"
    )


def test_plan_is_single_use_and_final_response_cannot_start_execution(load_fixture):
    ex = Executor(load_fixture())
    plans = planner(ex)
    propose(plans)
    args = json.dumps(json.loads(arguments())["experiment_arguments"])
    assert (
        plans.before_action("run_affinity_ab", args, remaining_responses=0)["error_code"]
        == "INSUFFICIENT_FEEDBACK_BUDGET"
    )
    assert plans.before_action("run_affinity_ab", args, remaining_responses=2) is None
    assert (
        plans.before_action("run_affinity_ab", args, remaining_responses=2)["error_code"]
        == "APPROVED_PLAN_REQUIRED"
    )
    assert plans.live_attempts == 1


def test_no_experiment_when_remaining_steps_have_no_completion_capacity(load_fixture, tmp_path):
    ex = Executor(load_fixture())
    plan = json.loads(arguments())
    client = Responses(
        [
            {
                "tool_calls": [
                    tool_call("plan", "plan_next_action", plan),
                    tool_call("run", "run_affinity_ab", plan["experiment_arguments"]),
                ]
            }
        ]
    )
    with pytest.raises(CopilotError, match="FINAL_RESPONSE_REQUIRED"):
        run_agent(client, ex, AgentConfig(max_steps=3, max_total_completion_tokens=4096), tmp_path)
    assert ex.executions == 0


def test_controller_can_revise_hypotheses_and_plan_a_second_batch(load_fixture, tmp_path):
    ex = Executor(load_fixture())
    first = json.loads(arguments())
    revised = json.loads(arguments())
    revised["hypotheses"][0]["category"] = "launch_overhead"
    revised["hypotheses"][0]["statement"] = "Fixed launch cost remains a competing explanation."
    revised["justification"] = "The first batch alone cannot establish a reproducible response."
    client = Responses(
        [
            {"tool_calls": [tool_call("plan1", "plan_next_action", first)]},
            {"tool_calls": [tool_call("run1", "run_affinity_ab", first["experiment_arguments"])]},
            {"tool_calls": [tool_call("plan2", "plan_next_action", revised)]},
            {"tool_calls": [tool_call("run2", "run_affinity_ab", revised["experiment_arguments"])]},
            final_response(),
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=5, max_experiments=2), tmp_path)
    assert ex.executions == 2 and report["planning"]["remaining_experiment_budget"] == 0
    assert report["planning"]["plans"][1]["hypotheses"][0]["category"] == "launch_overhead"
    assert (
        len([m for m in client.messages if m["role"] == "tool" and "verification" in m["content"]])
        == 2
    )


def test_replay_null_argument_error_can_be_repaired_within_original_budget(load_fixture, tmp_path):
    ex = Executor(load_fixture())
    ex.owner_approved = False
    ex.tool_schemas = lambda: [
        {
            "type": "function",
            "function": {
                "name": "read_recorded_comparison",
                "description": "Read a recorded comparison.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    good = json.loads(arguments("read_recorded_comparison"))
    good.pop("experiment_arguments")
    bad = {**good, "experiment_arguments": "null"}

    class RepairClient(Responses):
        def complete(self, messages, tools):
            if tools:
                schema = next(
                    t["function"]["parameters"]
                    for t in tools
                    if t["function"]["name"] == "plan_next_action"
                )
                assert "experiment_arguments" not in schema["properties"]
                assert "run_affinity_ab" not in schema["properties"]["action"]["enum"]
            errors = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            if errors:
                assert errors[0]["validation_errors"] == [
                    {"field": ["experiment_arguments"], "type": "model_type"}
                ]
                assert "Budgets are unchanged" in errors[0]["repair_instruction"]
            return super().complete(messages, tools)

    client = RepairClient(
        [
            {"tool_calls": [tool_call("bad", "plan_next_action", bad)]},
            {
                "tool_calls": [
                    tool_call("fixed", "plan_next_action", good),
                    tool_call("read", "read_recorded_comparison", {}),
                ]
            },
            final_response(),
        ]
    )
    report = run_agent(client, ex, AgentConfig(max_steps=3, max_experiments=0), tmp_path)
    assert report["model_budget"]["requests_reserved"] == 3
    assert report["planning"]["plans"][0]["status"] == "approved"
    assert report["planning"]["action_history"][0]["status"] == "consumed"
    assert ex.executions == report["planning"]["live_attempts"] == 0


def test_live_plan_requires_object_and_does_not_coerce_serialized_arguments(load_fixture):
    schema = plan_schema({"run_affinity_ab"})["function"]["parameters"]
    assert schema["properties"]["experiment_arguments"]["$ref"] == "#/$defs/AffinityArgs"
    assert "anyOf" not in schema["properties"]["experiment_arguments"]
    ex = Executor(load_fixture())
    plans = planner(ex)
    data = json.loads(arguments())
    data["experiment_arguments"] = json.dumps(data["experiment_arguments"])
    result = plans.propose(
        json.dumps(data), visible_ids=set(), remaining_tool_calls=4, remaining_responses=2
    )
    assert result["error_code"] == "INVALID_PLAN_ARGUMENTS"
    assert result["validation_errors"][0]["field"] == ["experiment_arguments"]
    assert ex.executions == 0 and plans.pending is None
    assert propose(plans)["status"] == "approved"


def test_plan_validation_feedback_is_bounded_and_does_not_echo_input(load_fixture):
    ex = Executor(load_fixture())
    data = json.loads(arguments())
    data.update({f"private-marker-{i}": "private-value" for i in range(8)})
    result = planner(ex).propose(
        json.dumps(data), visible_ids=set(), remaining_tool_calls=4, remaining_responses=2
    )
    assert result["error_code"] == "INVALID_PLAN_ARGUMENTS"
    assert len(result["validation_errors"]) == 5
    assert "private-marker" not in json.dumps(result) and "private-value" not in json.dumps(result)
    assert ex.executions == 0


@pytest.mark.parametrize(
    "field",
    [
        "evidence_event_ids",
        "contradicting_event_ids",
        "supporting_metric_refs",
        "contradicting_metric_refs",
        "missing_metric_refs",
        "missing_evidence",
        "counterevidence_note",
        "testable_prediction",
    ],
)
def test_planning_schema_requires_fields_also_required_by_parser(load_fixture, field):
    schema = plan_schema({"read_recorded_comparison"})["function"]["parameters"]
    assert field in schema["$defs"]["PlannedHypothesis"]["required"]
    data = json.loads(arguments("read_recorded_comparison"))
    del data["hypotheses"][0][field]
    ex = Executor(load_fixture())
    plans = planner(ex)
    result = plans.propose(
        json.dumps(data), visible_ids=set(), remaining_tool_calls=4, remaining_responses=3
    )
    assert result["error_code"] == "INVALID_PLAN_ARGUMENTS"
    assert {"field": ["hypotheses", 0, field], "type": "missing"} in result["validation_errors"]
    assert ex.executions == 0 and plans.pending is None


@pytest.mark.parametrize("field", ["counterevidence_note", "testable_prediction"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_required_hypothesis_text_is_rejected(load_fixture, field, value):
    data = json.loads(arguments("read_recorded_comparison"))
    data["hypotheses"][0][field] = value
    result = planner(Executor(load_fixture())).propose(
        json.dumps(data),
        visible_ids=set(),
        remaining_tool_calls=4,
        remaining_responses=3,
    )
    assert result["error_code"] == "INVALID_PLAN_ARGUMENTS"


def test_missing_evidence_abstention_needs_no_invented_support_or_outcomes(load_fixture):
    data = json.loads(arguments("abstain", expected_information_gain="low", expected_outcomes=[]))
    h = data["hypotheses"][0]
    h.update(
        category="unknown",
        supporting_metric_refs=[],
        evidence_event_ids=[],
        missing_evidence=["No measured causal attribution"],
    )
    data["hypotheses"] = [h]
    result = propose(planner(Executor(load_fixture())), json.dumps(data), responses=1, tools=0)
    assert result["status"] == "abstained"


def test_invalid_replacement_plan_cannot_reuse_previous_live_approval(load_fixture):
    ex = Executor(load_fixture())
    plans = planner(ex)
    assert propose(plans)["status"] == "approved"
    result = plans.propose("{}", visible_ids=set(), remaining_tool_calls=3, remaining_responses=2)
    assert result["error_code"] == "INVALID_PLAN_ARGUMENTS" and plans.pending is None
    actual = json.dumps(json.loads(arguments())["experiment_arguments"])
    assert (
        plans.before_action("run_affinity_ab", actual, remaining_responses=1)["error_code"]
        == "APPROVED_PLAN_REQUIRED"
    )
    assert ex.executions == 0


@pytest.mark.parametrize("case", ["contradictory_clock", "cpu_contention"])
def test_observed_repair_loops_leave_sixth_response_for_decision(load_fixture, tmp_path, case):
    """Reproduce the tool ordering of both failures, without model or hardware calls."""
    ex = Executor(load_fixture())
    ex.owner_approved = False
    good = json.loads(arguments("read_recorded_comparison"))
    good.pop("experiment_arguments")
    malformed = {**good, "experiment_arguments": "null"}
    incomplete = {**good, "expected_outcomes": []}
    incomplete["hypotheses"] = [{"category": "cpu_scheduling", "statement": "Incomplete repair"}]
    first = [("get_performance_summary", {})]
    timeline = [("inspect_timeline", {}), ("list_available_experiments", {})]
    sequence = (
        [first, timeline, [("plan_next_action", malformed)], [("plan_next_action", incomplete)]]
        if case == "contradictory_clock"
        else [
            first + timeline,
            [("plan_next_action", malformed)],
            [("plan_next_action", incomplete)],
            [("plan_next_action", good)],
        ]
    ) + [[("read_recorded_comparison", {})]]

    class RepairLoop:
        mode = "meta"
        calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            state = json.loads(messages[-1]["content"])["controller_state"]
            assert state["remaining_responses"] == 7 - self.calls
            if self.calls < 6:
                assert tools and not state["final_response_required"]
                return {
                    "message": {
                        "tool_calls": [
                            tool_call(f"turn-{self.calls}-{i}", name, args)
                            for i, (name, args) in enumerate(sequence[self.calls - 1])
                        ]
                    }
                }
            assert tools == [] and state["final_response_required"]
            outputs = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            assert any(o.get("error_code") == "INVALID_PLAN_ARGUMENTS" for o in outputs)
            incomplete_errors = [
                o
                for o in outputs
                if any(e["field"][:1] == ["hypotheses"] for e in o.get("validation_errors", []))
            ]
            assert (
                incomplete_errors
                and "Keep the hypotheses" in incomplete_errors[0]["repair_instruction"]
            )
            assert incomplete_errors[0]["remaining_responses"] > 0
            return {"message": final_response()}

    client = RepairLoop()
    report = run_agent(
        client,
        ex,
        AgentConfig(
            max_steps=6,
            max_tool_calls=20,
            max_experiments=0,
            max_total_completion_tokens=24576,
            max_session_request_bytes=786432,
        ),
        tmp_path,
    )
    assert client.calls == report["model_budget"]["requests_reserved"] == 6
    assert report["decision"]["decision"] == "inconclusive"
    assert [a["action"] for a in report["planning"]["action_history"]] == [
        "read_recorded_comparison"
    ]
    assert ex.executions == 0


@pytest.mark.parametrize(
    "bad_final,error",
    [
        ({"content": "invalid JSON"}, "DECISION_INVALID"),
        (
            {"tool_calls": [tool_call("late", "read_recorded_comparison", {})]},
            "FINAL_RESPONSE_REQUIRED",
        ),
        (
            {
                "content": AgentDecision(
                    hypotheses=[
                        Hypothesis(category="cpu_scheduling", statement="No supporting evidence")
                    ],
                    decision="inconclusive",
                    rationale="Incomplete investigation",
                ).model_dump_json()
            },
            "INVESTIGATION_INVALID",
        ),
    ],
)
def test_final_response_errors_stay_failures_without_retry_or_execution(
    load_fixture, tmp_path, bad_final, error
):
    ex = Executor(load_fixture())
    client = Responses([bad_final])
    with pytest.raises(CopilotError, match=error):
        run_agent(client, ex, AgentConfig(max_steps=1), tmp_path)
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert sum(row["kind"] == "response" for row in rows) == 1
    assert not any(row["kind"] in {"tool", "decision"} for row in rows)
    assert ex.executions == 0
