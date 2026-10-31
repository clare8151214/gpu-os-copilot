"""Measured evidence catalog and checks for ranked, explicitly provisional hypotheses."""

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.observability import observability
from gpu_os_copilot.schemas import AgentDecision, TraceBundle

UNITS = {
    "gpu_idle_fraction": "fraction",
    "gpu_busy_ms": "ms",
    "transfer_busy_fraction": "fraction",
    "cuda_api_duration_ms": "ms",
    "cpu_runnable_delay_ms": "ms",
    "sched_idle_overlap_ms": "ms",
    "wall_time_ms": "ms",
    "throughput_ops_s": "operations/s",
    "kernel_count": "events",
    "kernel_median_ms": "ms",
    "kernel_total_ms": "ms",
}


def evidence_catalog(trace: TraceBundle) -> dict:
    """Anonymous bounded facts; no scenario labels, paths, PIDs or event attributes."""
    summary = summarize(trace)
    values = {key: getattr(summary, key) for key in UNITS if not key.startswith("kernel_")}
    values.update(
        kernel_count=summary.kernel_duration_ms.get("count"),
        kernel_median_ms=summary.kernel_duration_ms.get("median"),
        kernel_total_ms=summary.kernel_duration_ms.get("sum"),
    )
    # Distinguish an absent kernel in a complete GPU trace from unavailable telemetry.
    if summary.gpu_idle_fraction is not None and not summary.kernel_duration_ms:
        values.update(kernel_count=0, kernel_total_ms=0)
    window = trace.window
    events = sorted(
        (
            e
            for e in trace.events
            if e.end_ns > window.start_ns
            and e.start_ns < window.end_ns
            and e.device_id in {None, window.device_id}
        ),
        key=lambda e: (e.start_ns, e.end_ns, e.id),
    )
    samples = []
    counts = {}
    for kind in ("cuda_api", "gpu_kernel", "memcpy", "memset", "cpu_sched"):
        group = [e for e in events if e.kind == kind]
        counts[kind] = len(group)
        # First/middle/last samples avoid using only the first launch burst.
        for index in sorted({0, len(group) // 2, len(group) - 1}) if group else []:
            samples.append(
                group[index].model_dump(include={"id", "kind", "start_ns", "end_ns", "device_id"})
            )
    return {
        "version": "evidence-catalog-v1",
        "input_confidence": trace.provenance.confidence,
        "trace_completeness": trace.trace_completeness,
        "metrics": {
            name: {"value": value, "unit": UNITS[name], "available": value is not None}
            for name, value in values.items()
        },
        "missing_metric_refs": [name for name, value in values.items() if value is None],
        "event_counts_in_window": counts,
        "event_samples": samples,
        "scope": "Captured workload/device/window; counts are observed events, not coverage proof.",
        "causal_status": "Candidate explanations only; no sole cause is established.",
        "observability": observability(trace),
    }


def validate_investigation(decision: AgentDecision, catalog: dict) -> dict:
    """Check reference availability and diagnostic structure, not narrative truth."""
    errors = []
    hypotheses = decision.hypotheses
    explicit_unknown = (
        len(hypotheses) == 1
        and hypotheses[0].category == "unknown"
        and bool(hypotheses[0].missing_evidence or hypotheses[0].missing_metric_refs)
    )
    if not explicit_unknown and not 2 <= len(hypotheses) <= 3:
        errors.append("RANK_TWO_OR_THREE_COMPETING_HYPOTHESES_OR_EXPLICIT_UNKNOWN")
    if len({h.category for h in hypotheses}) != len(hypotheses):
        errors.append("DUPLICATE_HYPOTHESIS_CATEGORY")
    metrics = catalog["metrics"]
    ranked = []
    for rank, hypothesis in enumerate(hypotheses, 1):
        support = set(hypothesis.supporting_metric_refs)
        counter = set(hypothesis.contradicting_metric_refs)
        missing = set(hypothesis.missing_metric_refs)
        if any(ref not in metrics for ref in support | counter | missing):
            errors.append("UNKNOWN_METRIC_REFERENCE")
        if any(not metrics[ref]["available"] for ref in (support | counter) & metrics.keys()):
            errors.append("MISSING_METRIC_USED_AS_OBSERVATION")
        if any(metrics[ref]["available"] for ref in missing & metrics.keys()):
            errors.append("AVAILABLE_METRIC_MARKED_MISSING")
        if support & counter or set(hypothesis.evidence_event_ids) & set(
            hypothesis.contradicting_event_ids
        ):
            errors.append("SAME_FACT_USED_AS_SUPPORT_AND_CONTRADICTION")
        if not hypothesis.counterevidence_note.strip():
            errors.append("COUNTEREVIDENCE_ASSESSMENT_REQUIRED")
        if not hypothesis.testable_prediction.strip():
            errors.append("TESTABLE_PREDICTION_REQUIRED")
        if not (support or hypothesis.evidence_event_ids or hypothesis.missing_evidence or missing):
            errors.append("HYPOTHESIS_HAS_NO_EVIDENCE_OR_EXPLICIT_LIMITATION")
        if hypothesis.confidence == "high" and (
            hypothesis.category == "unknown"
            or catalog["input_confidence"] != "measured"
            or catalog["trace_completeness"] != "complete"
            or hypothesis.category == "cpu_scheduling"
            and (
                not metrics["sched_idle_overlap_ms"]["available"]
                or not catalog["observability"]["scheduling"]["all_observed_api_threads_covered"]
            )
        ):
            errors.append("HIGH_CONFIDENCE_WITH_INCOMPLETE_EVIDENCE")
        ranked.append(
            {
                "rank": rank,
                "category": hypothesis.category,
                "status": "candidate",
                "supporting_metrics": {
                    ref: metrics[ref] for ref in sorted(support & metrics.keys())
                },
                "contradicting_metrics": {
                    ref: metrics[ref] for ref in sorted(counter & metrics.keys())
                },
                "supporting_event_ids": hypothesis.evidence_event_ids,
                "contradicting_event_ids": hypothesis.contradicting_event_ids,
                "missing_metric_refs": sorted(missing),
            }
        )
    return {
        "version": "investigator-v1",
        "status": "valid" if not errors else "invalid",
        "error_codes": list(dict.fromkeys(errors)),
        "ranked_hypotheses": ranked,
        "missing_metric_refs": catalog["missing_metric_refs"],
        "causal_status": catalog["causal_status"],
        "limitations": [
            "Metric values are computed by the application; prose remains a model interpretation.",
            "Valid references do not prove that an interpretation or ranking is correct.",
        ],
    }
