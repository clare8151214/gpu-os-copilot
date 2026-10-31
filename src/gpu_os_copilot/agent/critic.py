"""Deterministic checks over consumed measurements, never model-supplied statistics.

The policy is a conservative descriptive gate, not a significance/equivalence test.
It cannot establish a causal bottleneck or generalize beyond the recorded host.
"""

from math import isclose

from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.experiments.study import summarize_batches
from gpu_os_copilot.schemas import AgentDecision, ExperimentResult, TraceBundle

POLICY_VERSION = "descriptive-batches-v1"
MIN_BATCHES = 2
MIN_TRIALS = 5
MIN_EFFECT = 0.05
MIN_EFFECT_RETENTION = 0.5


def policy() -> dict:
    return {
        "version": POLICY_VERSION,
        "minimum_distinct_batches": MIN_BATCHES,
        "minimum_trials_per_condition_per_batch": MIN_TRIALS,
        "minimum_effect_fraction_in_every_batch": MIN_EFFECT,
        "minimum_smallest_to_largest_effect_ratio": MIN_EFFECT_RETENTION,
        "require_nonoverlapping_observed_wall_time_ranges": True,
        "claim_scope": "affinity_throughput",
        "evidence_level": "descriptive",
        "statistical_support": "not_evaluated",
    }


def assess_batches(results: list[ExperimentResult], *, unread_comparisons: int = 0) -> dict:
    """Recompute every consumed batch; do not expose private IDs/configurations.

    Repeated reads of one batch must be deduplicated by the executor. Distinct batches
    with reused trial IDs, different environments or placements are invalid evidence.
    """
    reasons = []
    comparisons = []
    if unread_comparisons:
        reasons.append("UNREAD_AVAILABLE_COMPARISONS")
    if len(results) < MIN_BATCHES:
        reasons.append("INSUFFICIENT_DISTINCT_BATCHES")
    try:
        for result in results:
            if (
                result.status != "ok"
                or result.provenance.confidence != "measured"
                or result.provenance.source != "benchmark"
            ):
                raise ValueError("Unverified batch")
            comparison = compare_trials(result.trials)
            if comparison.get("status") != "ok":
                raise ValueError("Invalid trials")
            for trial in result.trials:
                if trial.operation_count != trial.configuration.get("operations"):
                    raise ValueError("Operation count disagrees with workload")
                if not isclose(
                    trial.throughput_ops_s,
                    trial.operation_count * 1000 / trial.wall_time_ms,
                    rel_tol=1e-9,
                    abs_tol=0,
                ):
                    raise ValueError("Rate disagrees with duration")
            placement = {
                condition: {
                    tuple(sorted(t.cpu_set)) for t in result.trials if t.condition == condition
                }
                for condition in ("baseline", "treatment")
            }
            if any(len(sets) != 1 or not next(iter(sets)) for sets in placement.values()):
                raise ValueError("Inconsistent placement within batch")
            if placement["baseline"] == placement["treatment"]:
                raise ValueError("No placement intervention")
            comparisons.append(comparison)
        if len(results) >= MIN_BATCHES:
            summarize_batches(results)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        reasons.append("INVALID_OR_INCOMPARABLE_BATCHES")
        # Do not present a valid-looking subset of an invalid study as its result.
        comparisons = []

    effects = [c["median_throughput_change_fraction"] for c in comparisons]
    direction = None
    if effects:
        if all(effect >= MIN_EFFECT for effect in effects):
            direction = "supported"
        elif all(effect <= -MIN_EFFECT for effect in effects):
            direction = "rejected"
        else:
            reasons.append("SMALL_OR_CONFLICTING_EFFECTS")
        if min(effects) < 0 < max(effects):
            reasons.append("EFFECT_DIRECTION_CONFLICT")
        largest = max(abs(effect) for effect in effects)
        if largest and min(abs(effect) for effect in effects) / largest < MIN_EFFECT_RETENTION:
            reasons.append("EFFECT_SIZE_UNSTABLE")
        if any(min(c["baseline"]["n"], c["treatment"]["n"]) < MIN_TRIALS for c in comparisons):
            reasons.append("INSUFFICIENT_TRIALS_PER_BATCH")
        for c in comparisons:
            baseline, treatment = c["baseline"], c["treatment"]
            if not (
                treatment["max_wall_time_ms"] < baseline["min_wall_time_ms"]
                or baseline["max_wall_time_ms"] < treatment["min_wall_time_ms"]
            ):
                reasons.append("OBSERVED_RANGES_OVERLAP")
                break
    return {
        "policy": policy(),
        "batch_count": len(results),
        "unread_comparisons": unread_comparisons,
        "batches": [
            {"batch_ref": f"batch-{i + 1}", "comparison": comparison}
            for i, comparison in enumerate(comparisons)
        ],
        "throughput_change_fractions": effects,
        "eligible_decision": direction if not reasons and direction else "inconclusive",
        "reason_codes": list(dict.fromkeys(reasons)),
        "limitations": [
            "Descriptive observations only; no statistical significance or equivalence test.",
            "Distinct batches do not establish statistical independence or generalization.",
            "Affinity response cannot establish a sole CPU scheduling cause.",
        ],
    }


def verify_decision(
    proposed: AgentDecision,
    trace: TraceBundle,
    results: list[ExperimentResult],
    *,
    unread_comparisons: int = 0,
) -> tuple[AgentDecision, dict]:
    """Return an effective decision alongside the unchanged, separately logged proposal."""
    assessment = assess_batches(results, unread_comparisons=unread_comparisons)
    reasons = assessment["reason_codes"].copy()
    window = trace.window
    window_ids = {
        event.id
        for event in trace.events
        if event.end_ns > window.start_ns
        and event.start_ns < window.end_ns
        and event.device_id in {None, window.device_id}
    }
    if any(
        set(h.evidence_event_ids + h.contradicting_event_ids) - window_ids
        for h in proposed.hypotheses
    ):
        reasons.append("CITATION_OUTSIDE_TRACE_WINDOW")
    if trace.provenance.confidence != "measured" or trace.trace_completeness != "complete":
        reasons.append("TRACE_NOT_COMPLETE_MEASURED")
    required = {
        "scenario",
        "workload_version",
        "seed",
        "shape",
        "dtype",
        "warmup",
        "operations",
        "gpu",
        "cuda",
        "torch",
        "synchronized",
    }
    excluded = {"profiled", "capture", "scheduler_alignment", "scheduler_capture", "host"}
    if not required.issubset(trace.configuration) or any(
        result.workload_id != "launch_test"
        or not result.trials
        or any(
            type(result.trials[0].configuration.get(key)) is not type(value)
            or result.trials[0].configuration.get(key) != value
            for key, value in trace.configuration.items()
            if key not in excluded
        )
        or result.trials[0].configuration.get("profiled") is not False
        or result.trials[0].configuration.get("synchronized") is not True
        or not isinstance(result.trials[0].configuration.get("host"), dict)
        or not result.trials[0].configuration.get("host")
        for result in results
    ):
        reasons.append("TRACE_EXPERIMENT_BINDING_INCOMPLETE")
    conclusive = proposed.decision in {"supported", "rejected"}
    if conclusive and proposed.claim_scope != "affinity_throughput":
        reasons.append("UNSCOPED_PERFORMANCE_CLAIM")
    if conclusive and proposed.decision != assessment["eligible_decision"]:
        reasons.append("PROPOSAL_NOT_SUPPORTED_BY_BATCHES")

    effective = proposed.model_copy(deep=True)
    if conclusive:
        effective.decision = "inconclusive" if reasons else proposed.decision
        effective.claim_scope = "affinity_throughput"
        # Never pass through a conclusive model rationale containing invented numbers.
        effective.rationale = (
            "Deterministic verification: " + ", ".join(dict.fromkeys(reasons)) + ". "
            if reasons
            else "The recorded batches consistently show a descriptive affinity throughput "
            + ("improvement. " if proposed.decision == "supported" else "regression. ")
        ) + "No statistical significance, equivalence, or sole-cause conclusion is established."
    assessment.update(
        reason_codes=list(dict.fromkeys(reasons)),
        effective_decision=effective.decision,
        proposal_downgraded=effective.decision != proposed.decision,
        scope="Same-host recorded affinity throughput; hypotheses remain model proposals.",
    )
    return effective, assessment
