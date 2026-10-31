"""Run bounded CNN training/inference A/B batches, or print the plan without executing."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from gpu_os_copilot.experiments.study import summarize_batches
from gpu_os_copilot.schemas import ExperimentRequest, Provenance
from gpu_os_copilot.worker.local import LocalWorker
from gpu_os_copilot.workloads.specs import MODEL_SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--treatment-cpu", type=int, required=True)
    parser.add_argument("--batches", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    plan = {
        "scenarios": sorted(MODEL_SCENARIOS),
        "batches_per_scenario": args.batches,
        "trials_per_batch": 10,
        "treatment_cpu": args.treatment_cpu,
        "timeout_per_batch_s": 120,
        "maximum_total_batch_seconds": args.batches * len(MODEL_SCENARIOS) * 120,
        "cpu_budget": 2,
        "gpu_allocator_budget_mb": 256,
        "data": "Seeded synthetic images, no external dataset or pretrained model",
        "intervention": "Move launcher away from controlled CPU-only contender",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm:
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    worker = LocalWorker(args.out)
    (args.out / "environment.json").write_text(worker.capabilities().model_dump_json(indent=2))
    groups = {scenario: [] for scenario in sorted(MODEL_SCENARIOS)}
    for batch in range(args.batches):
        order = sorted(groups, reverse=bool(batch % 2))
        for scenario in order:
            result = worker.run_experiment(
                ExperimentRequest(
                    experiment_id=f"{scenario}-b{batch + 1}-{uuid4().hex[:8]}",
                    scenario=scenario,
                    candidate_cpu_set=[args.treatment_cpu],
                    repeats=5,
                    timeout_s=120,
                    owner_approved=True,
                    provenance=Provenance(
                        source="benchmark",
                        confidence="derived",
                        description="Owner-approved bounded model batch study",
                    ),
                )
            )
            print(
                json.dumps(
                    {
                        "scenario": scenario,
                        "batch": batch + 1,
                        "status": result.status,
                        "error_code": result.error_code,
                        "comparison": result.comparison,
                    }
                ),
                flush=True,
            )
            if result.status != "ok":
                return 2
            groups[scenario].append(result)
    summary = {scenario: summarize_batches(results) for scenario, results in groups.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
