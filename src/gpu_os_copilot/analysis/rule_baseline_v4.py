"""Removal ranking with the mixed label's gpu_compute rule (rules v4).

Rules v3 ranked gpu_compute by kernel_only_busy_fraction on the removal scale. The mixed
label (ablation_labels.ground_truth_mixed) instead accepts gpu_compute whenever kernels are
busy for at least half the window and no single removal doubles throughput. v4 keeps the
capped removal-estimate argmax for the removable factors, lets measured removal shares
(gain / (1 + gain)) replace estimates, and applies that gpu_compute rule. Written from the
burned amendment-5 holdout (development data); any accuracy claim needs a fresh holdout.
"""

from collections.abc import Collection, Mapping

from gpu_os_copilot.analysis.ablation_labels import DOMINANT_GAIN
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.schemas import Hypothesis, MetricSummary, TraceBundle

REMOVABLE = ("synchronization", "data_transfer", "launch_overhead")
COMPUTE_BUSY = 0.5
DOMINANT_SHARE = DOMINANT_GAIN / (1 + DOMINANT_GAIN)
LABELS = {
    "synchronization": "host synchronization waits",
    "data_transfer": "data transfers",
    "launch_overhead": "kernel launch submission",
}
EVIDENCE_KINDS = {
    "synchronization": {"cuda_api"},
    "data_transfer": {"memcpy"},
    "launch_overhead": {"cuda_api", "gpu_kernel"},
    "gpu_compute": {"gpu_kernel"},
}
CAVEAT = " Estimated from the trace by rules v4; not causal proof."


def experiment_plan(
    estimates: Mapping[str, float | None], offered: Collection[str], limit: int = 2
) -> list[str]:
    """The offered removable categories with the largest estimates, at most `limit`."""
    ordered = sorted(REMOVABLE, key=lambda category: -(estimates.get(category) or 0.0))
    return [category for category in ordered if category in offered][:limit]


def rank_v4(
    estimates: Mapping[str, float | None],
    kernel_busy_fraction: float | None,
    measured: Mapping[str, float] | None = None,
) -> str:
    """Top category; measured shares replace estimates; unknown if evidence is missing or tied."""
    shares = {category: estimates.get(category) for category in REMOVABLE}
    shares.update(measured or {})
    if kernel_busy_fraction is None or None in shares.values():
        return "unknown"
    best = max(shares.values())
    if kernel_busy_fraction >= COMPUTE_BUSY and best < DOMINANT_SHARE:
        return "gpu_compute"
    leaders = [category for category in REMOVABLE if shares[category] == best]
    return leaders[0] if len(leaders) == 1 else "unknown"


def _evidence(trace: TraceBundle, kinds: set[str]) -> list[str]:
    window = trace.window
    return [
        event.id
        for event in trace.events
        if event.kind in kinds
        and event.end_ns > window.start_ns
        and event.start_ns < window.end_ns
        and (event.device_id is None or event.device_id == window.device_id)
    ][:100]


def ranking_rule(trace: TraceBundle, summary: MetricSummary) -> str:
    """The rule diagnose_v4 applies: rules_v4, or a rules_v1 fallback and its reason.

    CPU scheduling has no removal estimate, so rules v1's cpu_scheduling candidate (runnable
    CPU intervals overlapping GPU gaps) is kept unchanged. Without the estimates or kernel
    busy fraction, or on a tie, rules v1's single candidate is returned.
    """
    if diagnose(trace, summary)[0].category == "cpu_scheduling":
        return "rules_v1_fallback:cpu_scheduling_candidate"
    estimates = [getattr(summary, f"removal_estimate_{c}_fraction") for c in REMOVABLE]
    if summary.kernel_busy_fraction is None or None in estimates:
        return "rules_v1_fallback:removal_estimates_unavailable"
    if rank_v4(dict(zip(REMOVABLE, estimates, strict=True)), summary.kernel_busy_fraction) == (
        "unknown"
    ):
        return "rules_v1_fallback:tied_estimates"
    return "rules_v4"


def diagnose_v4(trace: TraceBundle, summary: MetricSummary) -> list[Hypothesis]:
    """Rules mode: rules v4 over the removal estimates, ranked candidates first to last."""
    if ranking_rule(trace, summary) != "rules_v4":
        return diagnose(trace, summary)
    estimates = {c: getattr(summary, f"removal_estimate_{c}_fraction") for c in REMOVABLE}
    busy = summary.kernel_busy_fraction
    top = rank_v4(estimates, busy)
    missing = [*summary.missing_data[:19], "Controlled repeated intervention"]
    others = sorted(
        (c for c in REMOVABLE if c != top and estimates[c] > 0), key=lambda c: -estimates[c]
    )
    ranked = [top, *others][:3]
    hypotheses = []
    for index, category in enumerate(ranked):
        if category == "gpu_compute":
            statement = (
                f"Kernels are busy for {busy:.0%} of the window and no single "
                "removable factor is estimated to recover half of it."
            )
            prediction = (
                "Faster kernels should shorten the window more than removing any "
                "single host-side factor."
            )
        else:
            share = estimates[category]
            statement = (
                f"Removing {LABELS[category]} would recover an estimated "
                f"{share:.0%} of the window"
                + (", the largest removable share." if index == 0 else ".")
            )
            prediction = (
                f"Removing {LABELS[category]} should raise throughput by roughly "
                f"{share / (1 - share):.0%}."
                if share < 1
                else ""
            )
        hypotheses.append(
            Hypothesis(
                category=category,
                statement=statement + CAVEAT,
                confidence="medium" if index == 0 else "low",
                evidence_event_ids=_evidence(trace, EVIDENCE_KINDS[category]),
                testable_prediction=prediction,
                missing_evidence=missing,
            )
        )
    return hypotheses
