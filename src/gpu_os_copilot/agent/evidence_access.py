"""Track offered but unread evidence separately from unavailable measurements."""

from gpu_os_copilot.analysis.scheduling_study import READ_TOOL


def unread_opportunities(catalog, tool_names):
    """Use application-owned availability, never infer results from an unread tool."""
    if READ_TOOL not in tool_names:
        return {}
    refs = {
        ref: metric
        for ref, metric in catalog["metrics"].items()
        if metric.get("availability_state") == "unread" and metric.get("read_tool") == READ_TOOL
    }
    if not refs:
        return {}
    return {
        READ_TOOL: {
            "status": "unread",
            "metric_refs": sorted(refs),
            "scope": next(iter(refs.values()))["scope"],
            "new_hardware_experiments": 0,
            "instruction": "This recorded evidence has not been read. Its results and quality "
            "are unknown. Input-trace GPU timing limits do not establish whether these separate "
            "CPU comparisons are valid. Read if informative, or explain a deferral.",
        }
    }


def assess_evidence_access(deferrals, catalog, tool_names, *, can_read_and_finish):
    opportunities = unread_opportunities(catalog, tool_names)
    errors, accepted, seen = [], {}, set()
    for item in deferrals:
        if item.tool_name in seen:
            errors.append("DUPLICATE_EVIDENCE_DEFERRAL")
        seen.add(item.tool_name)
        if item.tool_name not in opportunities:
            errors.append("DEFERRAL_NOT_AN_OFFERED_UNREAD_TOOL")
            continue
        if item.reason == "insufficient_budget" and can_read_and_finish:
            errors.append("EVIDENCE_READ_STILL_FITS_BUDGET")
            continue
        accepted[item.tool_name] = item.model_dump()
    unaccounted = sorted(set(opportunities) - accepted.keys())
    if unaccounted:
        errors.append("UNREAD_EVIDENCE_REQUIRES_READ_OR_DEFERRAL")
    return {
        "version": "evidence-access-v1",
        "status": "unresolved" if errors else "accounted",
        "error_codes": list(dict.fromkeys(errors)),
        "unread_opportunities": opportunities,
        "model_deferrals": list(accepted.values()),
        "unaccounted_tools": unaccounted,
        "can_read_and_finish": can_read_and_finish,
        "limitations": [
            "Explicit deferrals remain model judgments, not validated information-gain estimates.",
            "Accounting for a skipped tool does not establish that skipping it was useful.",
            "Unread results are not consumed; diagnosis and throughput scores are not changed.",
        ],
    }
