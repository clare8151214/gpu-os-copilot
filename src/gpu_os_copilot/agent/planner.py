"""Model-proposed experiment plans with deterministic cost, evidence and safety gates."""

from typing import Annotated, Literal

from pydantic import Field, ValidationError

from gpu_os_copilot.agent.critic import assess_batches
from gpu_os_copilot.agent.evidence_access import assess_evidence_access
from gpu_os_copilot.agent.investigator import validate_investigation
from gpu_os_copilot.agent.tools import AffinityArgs, EmptyArgs
from gpu_os_copilot.analysis.scheduling_study import READ_TOOL
from gpu_os_copilot.schemas import AgentDecision, EvidenceDeferral, Hypothesis, Identifier, Model

AFFINITY_READS = {"read_recorded_comparison", "read_repeated_comparisons"}
ACTIONS = AFFINITY_READS | {"run_affinity_ab", READ_TOOL}


class PlannedHypothesis(Hypothesis):
    """Explicit planning input; historical/final hypothesis documents stay compatible."""

    evidence_event_ids: list[Identifier] = Field(
        max_length=100, description="Observed supporting IDs, or []."
    )
    contradicting_event_ids: list[Identifier] = Field(
        max_length=30, description="Observed counterevidence IDs, or []."
    )
    supporting_metric_refs: list[str] = Field(
        max_length=12, description="Available catalog metrics supporting this hypothesis, or []."
    )
    contradicting_metric_refs: list[str] = Field(
        max_length=12,
        description="Available counterevidence metrics, or []; do not reuse supporting facts.",
    )
    missing_metric_refs: list[str] = Field(
        max_length=12, description="Unavailable catalog metrics, or []. Unknown is not zero."
    )
    missing_evidence: list[str] = Field(
        max_length=20,
        description="Explicit evidence limitations, or []. "
        "Provide support or an explicit limitation.",
    )
    counterevidence_note: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"\S",
        description="Assess counterevidence, or explain why none is available.",
    )
    testable_prediction: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"\S",
        description="State an observable outcome that would test this hypothesis.",
    )


class PlanArgs(Model):
    action: Literal[
        "read_recorded_comparison",
        "read_repeated_comparisons",
        "run_affinity_ab",
        "abstain",
        "read_profiled_scheduler_comparisons",
    ]
    hypotheses: list[PlannedHypothesis] = Field(
        min_length=1,
        max_length=3,
        description="Two or three competing categories, "
        "or one unknown with explicit missing evidence.",
    )
    expected_information_gain: Literal["low", "medium", "high"]
    expected_outcomes: list[Annotated[str, Field(min_length=1, max_length=500, pattern=r"\S")]] = (
        Field(
            max_length=3,
            description="At least two distinct outcomes for an action; use [] when abstaining.",
        )
    )
    justification: str = Field(min_length=1, max_length=1000)
    experiment_arguments: AffinityArgs | None = None
    deferred_evidence: list[EvidenceDeferral] = Field(
        default_factory=list,
        max_length=8,
        description="When abstaining, explicitly explain each offered unread evidence tool "
        "you skip. Stopping an action does not discard your ranked symptom hypotheses.",
    )


def plan_schema(tool_names):
    parameters = PlanArgs.model_json_schema()
    parameters["properties"]["action"]["enum"] = [
        action
        for action in parameters["properties"]["action"]["enum"]
        if action == "abstain" or action in tool_names
    ]
    if "run_affinity_ab" in tool_names:
        # Optional by omission: an object/nullable union elicited the string "null"
        # in the live replay evaluation. Keep live argument validation unchanged.
        parameters["properties"]["experiment_arguments"] = {
            "$ref": "#/$defs/AffinityArgs",
            "description": "Required for run_affinity_ab; omit for recorded reads or abstention. "
            "Supply an object, never a JSON-encoded string.",
        }
    else:
        parameters["properties"].pop("experiment_arguments")
        parameters["$defs"].pop("AffinityArgs")
    return {
        "type": "function",
        "function": {
            "name": "plan_next_action",
            "description": (
                "Propose a recorded read, new bounded affinity batch, or abstention. "
                "Information gain is your qualitative prediction, not a measured number. Include "
                "two distinct possible outcomes and competing hypotheses. This plans only; "
                "invoke the approved action separately. Reserve an action response and a final "
                "response, or include the matching action immediately after this plan in the "
                "same response and reserve a final response. Every live execution requires a plan."
            ),
            "parameters": parameters,
        },
    }


class ExperimentPlanner:
    def __init__(self, executor, config, catalog, tool_names):
        self.executor, self.config, self.catalog = executor, config, catalog
        self.tool_names = tool_names
        self.pending = None
        self.history = []
        self.actions = []
        self.live_attempts = 0
        self.stopped = False
        self.deferred_evidence = []

    def recorded_read_policy(self):
        """Triage affinity replay using only visible metrics and already-consumed batches.

        This bounded heuristic does not establish equivalence or causal irrelevance.
        Missing GPU coverage cannot be used as evidence that a read has low value.
        """
        comparison = assess_batches(self.executor.comparison_evidence())
        prior = getattr(self.executor, "prior_decision", None)
        prior_claim = isinstance(prior, dict) and prior.get("decision") in {"supported", "rejected"}
        observed_effect = any(
            abs(effect) >= 0.05 for effect in comparison["throughput_change_fractions"]
        )
        idle = self.catalog["metrics"]["gpu_idle_fraction"]["value"]
        low_value = (
            idle is not None
            and idle <= 0.1
            and not prior_claim
            and not observed_effect
            and "INVALID_OR_INCOMPARABLE_BATCHES" not in comparison["reason_codes"]
        )
        return {
            "status": "low_value" if low_value else "available",
            "reason_codes": ["LOW_AFFINITY_INFORMATION_VALUE"] if low_value else [],
            "prior_claim_needs_reassessment": prior_claim,
            "consumed_effect_needs_reassessment": observed_effect,
            "scope": "Affinity recorded reads only; visible GPU activity and consumed batches.",
            "instruction": (
                "Stop affinity reads and report inconclusive with unread evidence disclosed. "
                "Dense captured GPU activity without observed affinity counterevidence makes "
                "these reads low priority; this does not prove no affinity effect."
                if low_value
                else "Read only if it can change the decision or reassess a prior claim. "
                "Availability alone does not require a read."
            ),
        }

    def propose(
        self,
        arguments,
        *,
        visible_ids,
        remaining_tool_calls,
        remaining_responses,
        following_call=None,
    ):
        # A malformed replacement must not leave an older live approval usable.
        self.pending = None
        try:
            if len(arguments) > 16384:
                raise ValueError("Too large")
            args = PlanArgs.model_validate_json(arguments)
        except ValidationError as error:
            known = (
                set(PlanArgs.model_fields)
                | set(Hypothesis.model_fields)
                | set(AffinityArgs.model_fields)
                | set(EvidenceDeferral.model_fields)
            )
            return {
                "error_code": "INVALID_PLAN_ARGUMENTS",
                "validation_errors": [
                    {
                        "field": [
                            part if isinstance(part, int) or part in known else "unknown_field"
                            for part in issue["loc"]
                        ],
                        "type": issue["type"],
                    }
                    for issue in error.errors(include_input=False, include_context=False)[:5]
                ],
                "repair_instruction": "Keep the hypotheses, evidence, counterevidence, predictions "
                "and distinct outcomes when repairing the indicated fields. Omit "
                "experiment_arguments for recorded reads or abstention; for run_affinity_ab "
                "use an object, never a JSON-encoded string. Budgets are unchanged.",
                "remaining_responses": remaining_responses,
                "remaining_tool_calls": remaining_tool_calls,
            }
        except (ValueError, TypeError):
            return {"error_code": "INVALID_PLAN_ARGUMENTS"}
        diagnosis = AgentDecision(
            hypotheses=args.hypotheses, decision="inconclusive", rationale=args.justification
        )
        issues = validate_investigation(diagnosis, self.catalog)["error_codes"]
        if args.action == "abstain":
            access = assess_evidence_access(
                args.deferred_evidence,
                self.catalog,
                self.tool_names,
                can_read_and_finish=remaining_responses >= 2 and remaining_tool_calls >= 1,
            )
            issues.extend(access["error_codes"])
        elif args.deferred_evidence:
            issues.append("DEFERRALS_REQUIRE_ABSTENTION_PLAN")
        cited = {
            ref for h in args.hypotheses for ref in h.evidence_event_ids + h.contradicting_event_ids
        }
        if cited - visible_ids:
            issues.append("PLAN_CITED_UNOBSERVED_EVIDENCE")
        if self.stopped:
            issues.append("PLANNER_STOPPED")
        # Only an immediately following, validated matching action avoids another model turn.
        # A plan alone still needs both the action request and the final feedback response.
        action_in_response = False
        if following_call and following_call.get("name") == args.action:
            try:
                action_model = AffinityArgs if args.action == "run_affinity_ab" else EmptyArgs
                actual = action_model.model_validate_json(following_call["arguments"]).model_dump()
                expected = (
                    args.experiment_arguments.model_dump() if args.experiment_arguments else {}
                )
                action_in_response = actual == expected
            except (ValueError, TypeError, KeyError):
                pass
        required_responses = 1 if action_in_response or args.action == "abstain" else 2
        cost = {"new_gpu_experiments": 0, "max_wall_time_s": 0, "additional_tool_calls": 0}
        cost["additional_model_responses_including_final"] = required_responses
        if args.action != "abstain":
            cost["additional_tool_calls"] = 1
            if len({outcome.strip() for outcome in args.expected_outcomes if outcome.strip()}) < 2:
                issues.append("DISCRIMINATING_OUTCOMES_REQUIRED")
            if args.expected_information_gain == "low":
                issues.append("LOW_EXPECTED_INFORMATION_GAIN")
            if remaining_tool_calls < 1 or remaining_responses < required_responses:
                issues.append("INSUFFICIENT_FEEDBACK_BUDGET")
            if args.action not in self.tool_names:
                issues.append("ACTION_UNAVAILABLE")
        if args.action == "run_affinity_ab":
            cost.update(
                new_gpu_experiments=1, max_wall_time_s=60, max_gpu_memory_mb=256, max_cpu_cores=2
            )
            if args.experiment_arguments is None:
                issues.append("EXPERIMENT_ARGUMENTS_REQUIRED")
            if self.live_attempts >= self.config.max_experiments:
                issues.append("EXPERIMENT_BUDGET_EXCEEDED")
            if not self.executor.owner_approved:
                issues.append("NEW_EXECUTION_NOT_APPROVED")
            if hasattr(self.executor, "workload_bound") and not self.executor.workload_bound():
                issues.append("TRACE_WORKLOAD_UNBOUND")
            if (
                hasattr(self.executor, "local_configuration_matches")
                and not self.executor.local_configuration_matches()
            ):
                issues.append("TRACE_ENVIRONMENT_MISMATCH")
            available = self.executor.execute("list_available_experiments", "{}")
            options = available.get("result", {}).get("experiments", [])
            allowed = [
                option
                for option in options
                if option.get("name") == "affinity_ab" and option.get("can_execute") is True
            ]
            if not allowed:
                issues.append("NO_EXECUTABLE_AFFINITY_OPTION")
            elif (
                args.experiment_arguments
                and args.experiment_arguments.candidate_cpu_set
                not in allowed[0].get("candidate_cpu_sets", [])
            ):
                issues.append("CPU_SET_NOT_OFFERED")
            comparison = assess_batches(self.executor.comparison_evidence())
            if comparison["eligible_decision"] != "inconclusive":
                issues.append("EVIDENCE_ALREADY_CONSISTENT")
            idle = self.catalog["metrics"]["gpu_idle_fraction"]["value"]
            if idle is None:
                issues.append("INCOMPLETE_GPU_EVIDENCE")
            elif idle <= 0.1 and not any(
                abs(effect) >= 0.05 for effect in comparison["throughput_change_fractions"]
            ):
                issues.append("LOW_AFFINITY_INFORMATION_VALUE")
            if not any(
                h.category in {"cpu_scheduling", "launch_overhead"} for h in args.hypotheses
            ):
                issues.append("NO_AFFINITY_TARGET_HYPOTHESIS")
        elif args.experiment_arguments is not None:
            issues.append("UNUSED_EXPERIMENT_ARGUMENTS")
        if args.action in AFFINITY_READS:
            issues.extend(self.recorded_read_policy()["reason_codes"])
        if args.action.startswith("read_") and any(
            a["action"] == args.action and a["status"] == "consumed" for a in self.actions
        ):
            issues.append("COMPARISON_ALREADY_CONSUMED")
        if (
            args.action == "read_recorded_comparison"
            and getattr(self.executor, "prior_decision", None) is not None
        ):
            issues.append("COMPARISON_ALREADY_CONSUMED")

        status = "blocked" if issues else "abstained" if args.action == "abstain" else "approved"
        row = {
            "plan_number": len(self.history) + 1,
            "action": args.action,
            "status": status,
            "expected_information_gain": args.expected_information_gain,
            "information_gain_source": "model_prediction_not_measurement",
            "expected_outcomes": args.expected_outcomes,
            "experiment_arguments": args.experiment_arguments.model_dump()
            if args.experiment_arguments
            else None,
            "justification": args.justification,
            "hypotheses": [h.model_dump() for h in args.hypotheses],
            "deferred_evidence": [item.model_dump() for item in args.deferred_evidence],
            "cost_upper_bound": cost,
            "remaining_experiment_budget": max(0, self.config.max_experiments - self.live_attempts),
            "remaining_responses": remaining_responses,
            "remaining_tool_calls": remaining_tool_calls,
            "reason_codes": list(dict.fromkeys(issues)),
        }
        if issues:
            row["repair_instruction"] = (
                "Address the reason codes while retaining all hypotheses, measured references, "
                "counterevidence, predictions and distinct outcomes. A blocked plan authorizes "
                "no live execution. If no justified action fits the remaining budget, return "
                "the final decision with missing evidence and uncertainty. Budgets are unchanged."
            )
        self.history.append(row)
        if status == "approved":
            self.pending = (
                args.action,
                args.experiment_arguments.model_dump() if args.experiment_arguments else {},
            )
        if status == "abstained":
            self.stopped = True
            self.deferred_evidence = args.deferred_evidence
        return {"schema_version": "1.0", "result": row}

    def before_action(self, name, arguments, *, remaining_responses):
        if name not in ACTIONS:
            return None
        if self.stopped:
            return {"error_code": "PLANNER_STOPPED"}
        if remaining_responses < 1:
            return {"error_code": "INSUFFICIENT_FEEDBACK_BUDGET"}
        if name in AFFINITY_READS:
            read_policy = self.recorded_read_policy()
            if read_policy["status"] == "low_value":
                return {"error_code": read_policy["reason_codes"][0], "read_policy": read_policy}
        if name.startswith("read_") and (
            any(
                action["action"] == name and action["status"] == "consumed"
                for action in self.actions
            )
            or name == "read_recorded_comparison"
            and getattr(self.executor, "prior_decision", None) is not None
        ):
            return {"error_code": "COMPARISON_ALREADY_CONSUMED"}
        if name == "run_affinity_ab" and self.executor.owner_approved:
            if self.live_attempts >= self.config.max_experiments:
                return {"error_code": "EXPERIMENT_BUDGET_EXCEEDED"}
            try:
                normalized = AffinityArgs.model_validate_json(arguments).model_dump()
            except (ValueError, TypeError):
                return {"error_code": "INVALID_TOOL_ARGUMENTS"}
            if self.pending != (name, normalized):
                return {"error_code": "APPROVED_PLAN_REQUIRED"}
            self.pending = None
            self.live_attempts += 1
        elif self.pending is not None and self.pending[0] == name:
            self.pending = None
        return None

    def after_action(self, name, result):
        if name not in ACTIONS:
            return
        value = result.get("result", {})
        consumed = (
            not result.get("error_code")
            and not value.get("error_code")
            and (name.startswith("read_") or value.get("status") in {"ok", "partial"})
        )
        self.actions.append(
            {
                "action": name,
                "status": "consumed" if consumed else "not_executed_or_failed",
                "execution_kind": "recorded_read" if name.startswith("read_") else "live_request",
                "error_code": result.get("error_code") or value.get("error_code"),
            }
        )

    def report(self):
        return {
            "version": "experiment-planner-v1",
            "plans": self.history,
            "action_history": self.actions,
            "live_attempts": self.live_attempts,
            "remaining_experiment_budget": max(0, self.config.max_experiments - self.live_attempts),
            "abstained": self.stopped,
            "pending_action": self.pending[0] if self.pending else None,
            "recorded_read_policy": self.recorded_read_policy(),
            "limitations": [
                "Information gain is a qualitative prediction, not calibrated entropy.",
                "Only bounded affinity batches and available recorded comparisons are supported.",
            ],
        }
