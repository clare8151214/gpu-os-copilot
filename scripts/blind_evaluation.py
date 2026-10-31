"""Evaluate real Meta tool choices on masked recorded traces, then reveal a later CPU batch."""

import argparse
import json
import random
from pathlib import Path
from uuid import uuid4

from gpu_os_copilot.agent.blind import BlindExecutor, audit_usage
from gpu_os_copilot.agent.meta_client import MetaClient
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.config import AgentConfig
from gpu_os_copilot.schemas import ExperimentResult, TraceBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-paid-api", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    if not 1 <= len(cases) <= 6:
        raise ValueError("Evaluation supports 1-6 bounded cases")
    prepared = []
    for item in cases:
        trace = TraceBundle.model_validate_json(Path(item["trace"]).read_text())
        initial = ExperimentResult.model_validate_json(Path(item["initial"]).read_text())
        repeat = ExperimentResult.model_validate_json(Path(item["repeat"]).read_text())
        case_id = "case-" + uuid4().hex[:12]
        executor = BlindExecutor(trace, initial, case_id=case_id)
        BlindExecutor(trace, initial, case_id=case_id, repeated=repeat)
        prepared.append((item, trace, initial, repeat, executor))
    plan = {
        "cases": len(cases),
        "feedback_sessions": sum(c.get("feedback", False) for c in cases),
        "max_responses_per_session": 4,
        "max_completion_tokens_per_response": 4096,
        "hardware_execution": False,
        "provider": "Meta",
        "automatic_retries": 0,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.allow_paid_api:
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    random.Random(20261030).shuffle(prepared)
    (args.out / "private-answer-key.json").write_text(
        json.dumps(
            [{"case_id": ex.trace.run_id, **item} for item, _, _, _, ex in prepared], indent=2
        )
    )
    config = AgentConfig(max_steps=4, max_experiments=0, reasoning_effort="low")
    client = MetaClient(config, allow_paid_api=True)
    records = []
    for item, trace, initial, repeat, executor in prepared:
        rule = diagnose(executor.trace, summarize(executor.trace))[0]
        # Fixed policy declared before seeing Meta results. Rules never read scenario labels.
        rule_selects = rule.category in {"launch_overhead", "cpu_scheduling"}
        expected = item["affinity_test_informative"]
        result = run_agent(client, executor, config, args.out / executor.trace.run_id)
        selects = result["decision"]["selected_experiment"] is not None
        row = {
            "case_id": executor.trace.run_id,
            "rules": rule.model_dump(),
            "rule_selects_affinity": rule_selects,
            "meta": result,
            "meta_selects_affinity": selects,
            "predeclared_informative": expected,
            "rule_selection_matches": rule_selects == expected,
            "meta_selection_matches": selects == expected,
            "recorded_comparison_reads": len(executor.reads),
            "usage": audit_usage(Path(result["audit_path"])),
        }
        records.append(row)
        (args.out / "results.json").write_text(json.dumps(records, indent=2))
        print(
            json.dumps(
                {
                    "case": executor.trace.run_id,
                    "decision": result["decision"]["decision"],
                    "tokens": row["usage"]["provider_tokens"],
                }
            ),
            flush=True,
        )
        if item.get("feedback"):
            feedback = BlindExecutor(
                trace,
                initial,
                case_id=executor.trace.run_id,
                repeated=repeat,
                prior_decision=result["decision"],
            )
            revision = run_agent(
                client, feedback, config, args.out / (executor.trace.run_id + "-revision")
            )
            row["revision"] = revision
            row["revision_read_later_batch"] = "read_repeated_comparisons" in feedback.reads
            row["revision_usage"] = audit_usage(Path(revision["audit_path"]))
            (args.out / "results.json").write_text(json.dumps(records, indent=2))
    summary = {
        "cases": len(records),
        "rule_selection_matches": sum(r["rule_selection_matches"] for r in records),
        "meta_selection_matches": sum(r["meta_selection_matches"] for r in records),
        "recorded_comparison_reads": sum(r["recorded_comparison_reads"] for r in records),
        "new_hardware_experiments": 0,
        "provider_tokens": sum(
            r["usage"]["provider_tokens"] + r.get("revision_usage", {}).get("provider_tokens", 0)
            for r in records
        ),
        "limitations": [
            "Small case set; the selection rubric does not establish scheduler causality.",
            "Real Meta inference over recorded measurements; no live experiment execution.",
            "Labels, source paths, configuration and answer key are withheld from model input.",
            "No superiority or statistical significance claim follows from this case count.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
