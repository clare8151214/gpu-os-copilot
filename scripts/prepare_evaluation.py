"""Freeze a local, anonymous comparison suite and both agent sources; no API calls."""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tarfile
from pathlib import Path
from uuid import uuid4

from evaluation_common import LIMITS, ReplayExecutor, digest

from gpu_os_copilot.agent.investigator import evidence_catalog
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import GPU_ACTIVITY_KINDS, ExperimentResult, TraceBundle

ROOT = Path(__file__).resolve().parents[1]
HARNESS = [
    "evaluation_common.py",
    "evaluation_case.py",
    "prepare_evaluation.py",
    "run_evaluation.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_extract(archive, target):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) > 500 or sum(m.size for m in members) > 10 * 1024 * 1024:
            raise ValueError("Source archive exceeds budget")
        if any(
            not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("Unsafe source archive member")
        tar.extractall(target, filter="data")


def freeze_scheduler(item, source, directory, case_id):
    """Validate and copy recorded inputs; never change the original captures."""
    from gpu_os_copilot.analysis.scheduling_study import load_scheduling_study

    original = Path(item["scheduler_manifest"]).resolve(strict=True)
    load_scheduling_study(original, source)
    manifest = json.loads(original.read_text())
    derivation = item.get("scheduler_derivation", "none")
    if derivation not in {"none", "missing_capture", "reused_capture"}:
        raise ValueError("Unknown scheduling derivation")
    sources = {str(original): sha(original)}
    target = directory / "scheduler-data"
    target.mkdir(exist_ok=True)
    for pair in manifest["pairs"]:
        for arm in ("baseline", "treatment"):
            if pair[arm] is None:
                continue
            path = (original.parent / pair[arm]).resolve(strict=True)
            sources[str(path)] = sha(path)
            name = sha(path) + ".json"
            frozen = target / name
            if frozen.exists():
                if sha(frozen) != sha(path):
                    raise ValueError("Frozen scheduling input changed")
            else:
                frozen.write_bytes(path.read_bytes())
            pair[arm] = name
    if derivation == "missing_capture":
        manifest["pairs"][-1]["treatment"] = None
    elif derivation == "reused_capture":
        manifest["pairs"][-1]["treatment"] = manifest["pairs"][0]["treatment"]
    path = target / (case_id + ".json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    load_scheduling_study(path, source)
    return str(path.relative_to(directory)), sources


def inspect_input(case_path, source, method):
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1")
    environment.pop("MODEL_API_KEY", None)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation_case.py"),
            "--case",
            str(case_path),
            "--method",
            method,
            "--offline",
            "--inspect-input",
            "--runner-sha256",
            sha(source / "src/gpu_os_copilot/agent/runner.py"),
        ],
        env=environment,
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode:
        raise ValueError(
            "Offline version-specific input inspection failed: " + process.stderr[-2000:]
        )
    return json.loads(process.stdout)


def prepare_case(item, case_id, *, case_directory=None):
    source = TraceBundle.model_validate_json(Path(item["trace"]).read_text())
    if source.provenance.confidence != "measured":
        raise ValueError("Suite sources must be measured; transformations are declared separately")
    batches = [
        ExperimentResult.model_validate_json(Path(path).read_text()) for path in item["batches"]
    ]
    trace = source.model_copy(deep=True)
    derivation = item.get("derivation", "none")
    if derivation not in {"none", "partial_gpu_view"}:
        raise ValueError("Unknown derivation")
    if derivation == "partial_gpu_view":
        gpu_ids = [e.id for e in trace.events if e.kind in GPU_ACTIVITY_KINDS]
        removed = set(gpu_ids[::2])
        trace.events = [e for e in trace.events if e.id not in removed]
        trace.trace_completeness = "partial"
        trace.provenance.confidence = "derived"
        trace.provenance.description = (
            "Deliberately incomplete view of measured events; no new capture."
        )
    remap = {e.id: f"event-{i:05d}" for i, e in enumerate(trace.events)}
    for event in trace.events:
        event.id = remap[event.id]
    trace.run_id, trace.experiment_id, trace.workload_id = case_id, case_id, "anonymous"
    fields = {
        "window",
        "trace_completeness",
        "gpu_idle_fraction",
        "gpu_busy_ms",
        "kernel_duration_ms",
        "cuda_api_duration_ms",
        "transfer_duration_ms",
        "transfer_busy_fraction",
        "cpu_runnable_delay_ms",
        "sched_idle_overlap_ms",
        "wall_time_ms",
        "throughput_ops_s",
    }
    summary = summarize(trace).model_dump(mode="json", include=fields)
    summary.update(
        run_id=case_id,
        source=trace.provenance.confidence,
        clock_aligned=trace.clock_aligned,
        missing_metrics=[name for name in fields if summary[name] is None],
    )
    opening = {
        "run_id": case_id,
        "input_confidence": trace.provenance.confidence,
        "summary": summary,
        "evaluation": "Anonymous recorded-evidence evaluation; no new hardware execution.",
        "available_recorded_batches": len(batches),
        "evidence_catalog": evidence_catalog(trace),
    }
    sources = [item["trace"], *item["batches"]]
    if item.get("prior"):
        original = json.loads(Path(item["prior"]).read_text())
        prior = copy.deepcopy(original.get("agent", original)["decision"])
        for hypothesis in prior["hypotheses"]:
            hypothesis["evidence_event_ids"] = [
                remap[ref] for ref in hypothesis.get("evidence_event_ids", []) if ref in remap
            ]
        prior = json.loads(json.dumps(prior).replace("cpu_contention", "[masked workload]"))
        opening.update(
            prior_decision=prior,
            prior_measured_comparison=compare_trials(batches[0].trials),
            new_evidence="Later measured batches are available; read and reassess.",
        )
        sources.append(item["prior"])
    case = {
        "case_id": case_id,
        "trace": trace.model_dump(mode="json"),
        "batches": [r.model_dump(mode="json") for r in batches],
        "opening": opening,
    }
    scheduler_sources = {}
    if item.get("runtime_evidence", False) or "scheduler_manifest" in item:
        case["runtime_evidence"] = True
    if "scheduler_manifest" in item:
        if case_directory is None:
            raise ValueError("Scheduling inputs require a frozen case directory")
        case["scheduler_manifest"], scheduler_sources = freeze_scheduler(
            item, source, case_directory, case_id
        )
    ex = ReplayExecutor(case, case_directory=case_directory)
    key = {
        "case_id": case_id,
        "family": item["family"],
        "role": item["role"],
        "data_kind": "measured" if derivation == "none" else "derived_incomplete_view",
        "derivation": derivation,
        "rubric": item["rubric"],
        "sources_sha256": {**{path: sha(path) for path in sources}, **scheduler_sources},
        "opening_sha256": digest(ex.evidence()),
        "common_tools_sha256": digest(ex.tool_schemas()),
    }
    if "scheduler_manifest" in item:
        key["scheduler_derivation"] = item.get("scheduler_derivation", "none")
    return case, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not 4 <= len(manifest["cases"]) <= 12:
        raise ValueError("Expanded comparison requires 4-12 cases")
    archive = Path(manifest["baseline_archive"])
    if sha(archive) != manifest["baseline_archive_sha256"]:
        raise ValueError("Historical source archive changed")
    args.out.mkdir(parents=True, exist_ok=False)
    args.out = args.out.resolve()
    safe_extract(archive, args.out / "legacy")
    enhanced = args.out / "enhanced-source.tar.gz"
    with tarfile.open(enhanced, "x:gz") as tar:
        for path in sorted((ROOT / "src").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    safe_extract(enhanced, args.out / "enhanced")
    prepared = [
        prepare_case(item, "case-" + uuid4().hex[:12], case_directory=args.out)
        for item in manifest["cases"]
    ]
    random.Random(20261031).shuffle(prepared)
    keys, case_files = [], []
    for case, key in prepared:
        path = args.out / (case["case_id"] + ".json")
        path.write_text(json.dumps(case, indent=2) + "\n")
        if case.get("runtime_evidence"):
            inspected = {
                method: inspect_input(path, args.out / method, method)
                for method in ("legacy", "enhanced")
            }
            if any(
                entry["common_tools_sha256"] != key["common_tools_sha256"]
                for entry in inspected.values()
            ):
                raise ValueError("Version-specific tools are not identical")
            key["opening_sha256_by_method"] = {
                method: entry["opening_sha256"] for method, entry in inspected.items()
            }
        case_files.append({"path": path.name, "sha256": sha(path), "case_id": case["case_id"]})
        keys.append(key)
    (args.out / "private-answer-key.json").write_text(json.dumps(keys, indent=2) + "\n")
    plan = {
        "version": "matched-replay-v3",
        "model": "muse-spark-1.3",
        "cases": case_files,
        "methods": ["rules", "legacy", "enhanced"],
        "limits_per_session": LIMITS,
        "paid_sessions": 2 * len(prepared),
        "max_provider_requests": 2 * len(prepared) * LIMITS["max_steps"],
        "max_reserved_completion_tokens": 2 * len(prepared) * LIMITS["max_total_completion_tokens"],
        "max_serialized_input_bytes": 2 * len(prepared) * LIMITS["max_session_request_bytes"],
        "automatic_retries": 0,
        "new_hardware_experiments": 0,
        "method_order": [
            ["legacy", "enhanced"] if i % 2 == 0 else ["enhanced", "legacy"]
            for i in range(len(prepared))
        ],
        "source_archives_sha256": {"legacy": sha(archive), "enhanced": sha(enhanced)},
        "source_sha256": {
            method: {
                str(p.relative_to(args.out / method)): sha(p)
                for p in sorted((args.out / method / "src").rglob("*.py"))
            }
            for method in ["legacy", "enhanced"]
        },
        "harness_sha256": {name: sha(ROOT / "scripts" / name) for name in HARNESS},
        "answer_key_sha256": sha(args.out / "private-answer-key.json"),
        "scheduler_data_sha256": {
            str(path.relative_to(args.out)): sha(path)
            for path in sorted((args.out / "scheduler-data").glob("*.json"))
        },
        "limitations": [
            "Identical anonymous evidence, recorded outcomes, model and call/tool/input budgets.",
            "Historical rules consume the same screened summary; no legacy metric recomputation.",
            "CPU/GPU timestamp contradictions remain visible and their overlap is withheld.",
            "Enhanced planning/verification is part of the agent; shared read tools are identical.",
            "Development evidence, not unseen holdout; derived/revision cases are separate strata.",
            "Symptom-rubric accuracy is not causal ground truth or production generalization.",
            "Replay choices are scored; unnecessary live hardware interventions are not measured.",
            "Replay value is unscored where insufficient trace evidence does not settle its value.",
            "One session per method/case cannot establish statistical superiority.",
        ],
    }
    if any(case.get("runtime_evidence") for case, _ in prepared):
        plan["version"] = "matched-raw-replay-v4"
        plan["limitations"][:3] = [
            "Identical raw evidence, recorded outcomes, model, shared tools and budgets.",
            "Each runtime computes its own summaries/catalog; per-method input hashes are frozen. "
            "This compares pipelines, not models given identical derived measurements.",
            "Rules use legacy-derived metrics and read the same scheduling study when selecting "
            "affinity. This explicit extension reports the deterministic assessment only.",
            "Raw timestamp contradictions are retained, with qualification determined by each "
            "runtime. A clean ordering check is not independent GPU clock calibration.",
        ]
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
