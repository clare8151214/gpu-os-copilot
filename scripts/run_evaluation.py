"""Run all matched replay cases, retaining failures; paid execution requires opt-in."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(result, key):
    successful = result["status"] == "ok"
    decision = result.get("agent", {}).get("decision", {})
    hypotheses = decision.get("hypotheses", [])
    first = hypotheses[0]["category"] if hypotheses else None
    reads = result.get("comparison_reads", [])
    rubric = key["rubric"]
    informative = rubric["comparison_read_informative"]
    # Score first actions from actual consumption, never from a terminal selected_experiment.
    return {
        "case_id": key["case_id"],
        "family": key["family"],
        "role": key["role"],
        "method": result["method"],
        "status": result["status"],
        "error_code": result.get("error_code"),
        "diagnosis_matches_symptom_rubric": successful and first in rubric["top_categories"]
        if key["role"] == "diagnosis"
        else None,
        "recorded_test_selection_matches": successful and bool(reads) == informative
        if isinstance(informative, bool)
        else None,
        "abstention_matches": successful and not reads if informative is False else None,
        "low_value_recorded_reads": (
            len(reads) if informative is False else len(reads) - len(set(reads))
        )
        if isinstance(informative, bool)
        else None,
        "repeated_recorded_reads": len(reads) - len(set(reads)),
        "terminal_uncertainty_matches": successful
        and decision.get("decision") in rubric["allowed_decisions"],
        "historical_revision_corrected": successful
        and decision.get("decision") == "inconclusive"
        and "read_repeated_comparisons" in reads
        if key["role"] == "revision"
        else None,
        "actual_unnecessary_hardware_interventions": None,
        "usage": result.get("usage", {}),
        "limitations": "Recorded-test consumption is replay, not a live hardware intervention.",
    }


def aggregate(rows):
    metrics = (
        "diagnosis_matches_symptom_rubric",
        "recorded_test_selection_matches",
        "abstention_matches",
        "terminal_uncertainty_matches",
        "historical_revision_corrected",
    )
    groups = []
    for method in ("rules", "legacy", "enhanced"):
        for role in ("diagnosis", "robustness", "revision"):
            subset = [row for row in rows if row["method"] == method and row["role"] == role]
            groups.append(
                {
                    "method": method,
                    "role": role,
                    "cases": len(subset),
                    "failures": sum(row["status"] != "ok" for row in subset),
                    "scores": {
                        metric: {
                            "matches": sum(row[metric] is True for row in subset),
                            "scored_cases": sum(row[metric] is not None for row in subset),
                        }
                        for metric in metrics
                    },
                    "low_value_recorded_reads": sum(
                        row["low_value_recorded_reads"]
                        for row in subset
                        if row["low_value_recorded_reads"] is not None
                    )
                    if any(row["low_value_recorded_reads"] is not None for row in subset)
                    else None,
                    "low_value_read_scored_cases": sum(
                        row["low_value_recorded_reads"] is not None for row in subset
                    ),
                    "repeated_recorded_reads": sum(
                        row["repeated_recorded_reads"] for row in subset
                    ),
                    "reported_total_tokens": sum(
                        row["usage"].get("reported_total_tokens", 0) for row in subset
                    ),
                    "usage_complete": bool(subset)
                    and all(row["usage"].get("usage_complete", False) for row in subset),
                }
            )
    return groups


def validate_suite(suite):
    suite = suite.resolve(strict=True)
    if (suite / "invalidated.json").exists():
        raise ValueError("Evaluation plan invalidated by a subsequent evidence audit")
    plan = json.loads((suite / "plan.json").read_text())
    if sha(suite / "private-answer-key.json") != plan["answer_key_sha256"]:
        raise ValueError("Answer key changed")
    for name, expected in plan["harness_sha256"].items():
        if sha(ROOT / "scripts" / name) != expected:
            raise ValueError("Shared evaluation harness changed")
    for method, files in plan["source_sha256"].items():
        for name, expected in files.items():
            if sha(suite / method / name) != expected:
                raise ValueError("Frozen agent source changed")
    for case in plan["cases"]:
        if sha(suite / case["path"]) != case["sha256"]:
            raise ValueError("Frozen case changed")
    for name, expected in plan.get("scheduler_data_sha256", {}).items():
        relative = Path(name)
        path = (suite / relative).resolve(strict=True)
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(suite):
            raise ValueError("Scheduling input outside frozen suite")
        if sha(path) != expected:
            raise ValueError("Frozen scheduling input changed")
    return plan


def validate_result_inputs(result, key, source_method):
    expected = (
        key["opening_sha256_by_method"][source_method]
        if "opening_sha256_by_method" in key
        else key["opening_sha256"]
    )
    if (
        result["opening_sha256"] != expected
        or result["common_tools_sha256"] != key["common_tools_sha256"]
    ):
        raise ValueError("Method did not receive its frozen evidence/tools")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--plan-sha256", help="Required for paid execution of the reviewed plan")
    args = parser.parse_args()
    if args.offline and args.allow_paid_api:
        parser.error("Offline and paid modes are mutually exclusive")
    suite = args.suite.resolve()
    plan = validate_suite(suite)
    if args.allow_paid_api and args.plan_sha256 != sha(suite / "plan.json"):
        parser.error("Paid execution requires the exact reviewed plan SHA-256")
    if not args.offline and not args.allow_paid_api:
        print(
            json.dumps(
                {
                    k: plan[k]
                    for k in [
                        "paid_sessions",
                        "max_provider_requests",
                        "max_reserved_completion_tokens",
                        "max_serialized_input_bytes",
                    ]
                },
                indent=2,
            )
        )
        return 0
    if args.out is None:
        parser.error("Execution requires a new output directory")
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.out.resolve()
    keys = {
        key["case_id"]: key for key in json.loads((suite / "private-answer-key.json").read_text())
    }
    rows = []
    for index, case in enumerate(plan["cases"]):
        for method in ["rules", *plan["method_order"][index]]:
            source_method = "legacy" if method == "rules" else method
            source = suite / source_method
            env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1")
            if args.offline or method == "rules":
                env.pop("MODEL_API_KEY", None)
            destination = out / case["case_id"] / method
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluation_case.py"),
                "--case",
                str(suite / case["path"]),
                "--out",
                str(destination),
                "--method",
                method,
                "--runner-sha256",
                plan["source_sha256"][source_method]["src/gpu_os_copilot/agent/runner.py"],
            ]
            if args.offline:
                command.append("--offline")
            elif method != "rules":
                command.append("--allow-paid-api")
            try:
                process = subprocess.run(
                    command, env=env, cwd=source, capture_output=True, text=True, timeout=240
                )
                if process.returncode != 0 or not (destination / "result.json").exists():
                    result = {
                        "method": method,
                        "status": "failed",
                        "error_code": "CASE_PROCESS_FAILED",
                    }
                else:
                    result = json.loads((destination / "result.json").read_text())
                    validate_result_inputs(result, keys[case["case_id"]], source_method)
            except subprocess.TimeoutExpired:
                result = {
                    "method": method,
                    "status": "failed",
                    "error_code": "CASE_PROCESS_TIMEOUT",
                }
            if result["status"] == "failed" and "usage" not in result:
                checkpoint = destination / "usage-checkpoint.json"
                result["usage"] = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
                result["usage"]["usage_complete"] = False
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "failure.json").write_text(json.dumps(result, indent=2) + "\n")
            row = score(result, keys[case["case_id"]])
            row["evaluation_mode"] = "offline_protocol" if args.offline else "live_meta_comparison"
            rows.append(row)
            (out / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                json.dumps({"case": case["case_id"], "method": method, "status": result["status"]}),
                flush=True,
            )
    summary = {
        "mode": "offline_protocol" if args.offline else "live_meta_comparison",
        "rows": len(rows),
        "successful_sessions": sum(row["status"] == "ok" for row in rows),
        "failures": sum(row["status"] != "ok" for row in rows),
        "provider_requests_reported": 0
        if args.offline
        else sum(row["usage"].get("requests", 0) for row in rows),
        "provider_usage_complete": False
        if args.offline
        else all(row["usage"].get("usage_complete", False) for row in rows),
        "new_hardware_experiments": 0,
        "limits": plan["limits_per_session"],
        "limitations": plan["limitations"],
        "source_plan_sha256": sha(suite / "plan.json"),
        "by_method_and_stratum": aggregate(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
