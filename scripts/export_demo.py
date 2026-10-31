"""Export selected, privacy-reduced evidence from the recorded three-scenario demo."""

import argparse
import hashlib
import json
from pathlib import Path

from gpu_os_copilot.agent.tools import ToolExecutor
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import ExperimentResult, TraceBundle
from gpu_os_copilot.worker.local import LocalWorker


def export(root: Path, output: Path, trace_overrides: Path | None = None) -> None:
    scenarios = []
    traces = {}
    for scenario, prefix in (
        ("cpu_contention", "cpu"),
        ("transfer_bound", "transfer"),
        ("compute_bound", "compute"),
    ):
        report = json.loads((root / f"meta-measured-{prefix}/report.json").read_text())
        audits = list((root / f"meta-measured-{prefix}/audit").glob("*.jsonl"))
        if len(audits) != 1:
            raise ValueError("Expected exactly one successful audit per scenario")
        rows = [json.loads(line) for line in audits[0].read_text().splitlines()]
        if rows[-1]["kind"] != "decision" or any(row["mode"] != "meta" for row in rows):
            raise ValueError("Only completed real-model sessions may be exported")
        calls = [row for row in rows if row["kind"] == "tool"]
        if prefix == "cpu":
            result = next(
                row["result"]["result"]
                for row in calls
                if row["request"]["name"] == "run_affinity_ab"
            )
        else:
            result = json.loads((root / f"control-comparisons/{scenario}.json").read_text())
        experiment = ExperimentResult.model_validate(result)
        if experiment.status != "ok" or experiment.provenance.confidence != "measured":
            raise ValueError("Only completed measured experiments may be exported")
        comparison = compare_trials(experiment.trials)
        if comparison != experiment.comparison or comparison.get("status") != "ok":
            raise ValueError("Saved comparison does not match individual trials")
        trace_prefix = "cpu_baseline" if prefix == "cpu" else prefix
        trace_path = root / f"measured/{trace_prefix}.capture/trace.json"
        if trace_overrides is not None:
            trace_path = trace_overrides / f"{trace_prefix}-trace.json"
        trace = TraceBundle.model_validate_json(trace_path.read_text())
        executor = ToolExecutor(LocalWorker(), trace)
        if not executor.workload_bound() or trace.configuration["scenario"] != scenario:
            raise ValueError("Unbound measurement")
        # Exclude startup/shutdown, PID/TID and arbitrary attributes. Retain actual
        # timestamps, correlation IDs and GPU activity for offline analysis.
        trace.events = [
            event.model_copy(update={"pid": None, "tid": None, "attributes": {}})
            for event in executor.window_events()
        ]
        traces[scenario] = trace.model_dump(mode="json")
        decision = report["agent"]["decision"]
        summary = summarize(trace).model_dump(mode="json")
        scenarios.append(
            {
                "scenario": scenario,
                "created_at": result["created_at"],
                "trace_source_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                "trace": f"measured-traces/{scenario}.json",
                "configuration": trace.configuration,
                "profiled_metrics": {
                    key: summary[key]
                    for key in (
                        "window",
                        "gpu_idle_fraction",
                        "gpu_busy_ms",
                        "kernel_duration_ms",
                        "transfer_duration_ms",
                        "cuda_api_duration_ms",
                        "cpu_runnable_delay_ms",
                        "missing_data",
                    )
                },
                "rules": [hypothesis["category"] for hypothesis in report["hypotheses"]],
                "meta": {
                    "model": rows[0]["model"],
                    "recorded_input_gpu_idle_fraction": report["summary"]["gpu_idle_fraction"],
                    "input_confidence": report["agent"]["input_confidence"],
                    "tools": [row["request"]["name"] for row in calls],
                    "decision": decision["decision"],
                    "selected_experiment": (decision.get("selected_experiment") or {}).get("name"),
                    "hypothesis_categories": [h["category"] for h in decision["hypotheses"]],
                    "total_tokens": sum(
                        (row.get("response", {}).get("usage") or {}).get("total_tokens", 0)
                        for row in rows
                    ),
                },
                "experiment_origin": "meta_selected" if prefix == "cpu" else "manual_control",
                "experiment_plan": experiment.plan,
                "trials": [
                    trial.model_dump(mode="json", exclude={"configuration"})
                    for trial in experiment.trials
                ],
                "comparison": comparison,
                "trace_limitations": trace.provenance.limitations,
            }
        )
    data = {
        "schema_version": "1.0",
        "kind": "recorded_measured_demo",
        "environment": experiment.trials[0].configuration["host"],
        "limitations": [
            "Recorded measurements on one host; replay performs no new GPU/API work.",
            "Timing trials are unprofiled; diagnostic idle fractions come from separate captures.",
            "CPU scheduling alignment was unavailable; a sole scheduler cause is not established.",
            "Five trials per condition; show every trial and overlapping ranges.",
            "Negative-control A/B was run manually; Meta independently abstained on those traces.",
            "Scenarios are explicitly labeled: this is not a blinded diagnosis evaluation.",
            "No measured AI superiority or whole-device utilization claim.",
            "Raw captures and full local audits are excluded; exports omit process identities.",
            "Trace metrics recomputed; Meta choices refer to their recorded original inputs.",
        ],
        "scenarios": scenarios,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "measured-traces").mkdir()
    for scenario, trace in traces.items():
        (output / f"measured-traces/{scenario}.json").write_text(json.dumps(trace, indent=2) + "\n")
    (output / "measured-demo.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-overrides",
        type=Path,
        help="Directory of explicitly reimported <prefix>-trace.json files",
    )
    args = parser.parse_args()
    export(args.artifacts, args.output, args.trace_overrides)
