import json

from gpu_os_copilot.schemas import Hypothesis, MetricSummary


def render(summary: MetricSummary, hypotheses: list[Hypothesis]) -> str:
    label = summary.provenance.confidence.upper()
    idle = summary.gpu_idle_fraction
    lines = [
        f"GPU-OS Copilot | {label} input | {summary.run_id}",
        summary.provenance.description,
        f"Window: [{summary.window.start_ns}, {summary.window.end_ns}) ns; "
        f"device {summary.window.device_id}; captured workload, all captured streams",
        "Metric origin: deterministically derived from the labeled input above.",
        f"Observed workload GPU idle fraction: {idle:.1%}"
        if idle is not None
        else "Observed workload GPU idle fraction: unavailable",
        f"Kernel duration (ms): {json.dumps(summary.kernel_duration_ms)}",
        f"Transfer duration sums (ms): {json.dumps(summary.transfer_duration_ms)}",
        f"CPU runnable delay (ms): {summary.cpu_runnable_delay_ms}",
        "Candidate hypotheses (not established causes):",
    ]
    lines.extend(f"- {h.category} [{h.confidence}]: {h.statement}" for h in hypotheses)
    lines.extend(f"Missing/limited: {item}" for item in summary.missing_data)
    lines.append("Workload gaps do not establish whole-device idleness or a measured speedup.")
    return "\n".join(lines)
