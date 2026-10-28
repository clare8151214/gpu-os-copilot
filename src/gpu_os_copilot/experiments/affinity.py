"""Finite experiment runs, exclusive IDs, persisted partial results, child-only cleanup."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from uuid import uuid4

from gpu_os_copilot.experiments.compare import compare_trials
from gpu_os_copilot.schemas import BenchmarkTrial, ExperimentRequest, ExperimentResult, Provenance


def _invoke(argv: list[str], timeout: float, max_output: int) -> dict:
    """Fixed project argv; file-backed output capped before workload import."""
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        if name in os.environ
    }
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="42")
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            argv,
            stdout=output,
            stderr=errors,
            env=environment,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
            if process.returncode != 0:
                raise RuntimeError("Benchmark child failed")
            output.seek(0)
            raw = output.read(max_output + 1)
            errors.seek(0)
            stderr = errors.read(max_output + 1)
            if len(raw) + len(stderr) > max_output:
                raise RuntimeError("Benchmark output budget exceeded")
            return json.loads(raw)
        finally:
            # The process group belongs exclusively to this newly created subprocess.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_affinity_ab(worker, request: ExperimentRequest) -> ExperimentResult:
    request = ExperimentRequest.model_validate(request.model_dump())
    if request.worker_id != worker.worker_id:
        raise ValueError("Worker ID mismatch")
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if not available or not set(request.candidate_cpu_set).issubset(available):
        raise ValueError("Candidate CPU IDs are not available")
    baseline_cpu = available[0]
    if baseline_cpu in request.candidate_cpu_set:
        raise ValueError("Treatment CPU set must exclude the controlled contender's CPU")
    if len(set(request.candidate_cpu_set) | {baseline_cpu}) > request.max_cpu_cores:
        raise ValueError("Combined launcher/contender CPU set exceeds the budget")
    result = ExperimentResult(
        experiment_id=request.experiment_id,
        worker_id=request.worker_id,
        workload_id=request.workload_id,
        provenance=Provenance(
            source="benchmark",
            confidence="derived",
            description="Bounded project-child affinity A/B plan; no timing data yet",
        ),
        status="dry_run",
        plan={
            "scenario": request.scenario,
            "baseline_cpu_set": [baseline_cpu],
            "treatment_cpu_set": request.candidate_cpu_set,
            "contender_cpu": baseline_cpu,
            "repeats": request.repeats,
            "timeout_s": request.timeout_s,
            "max_cpu_cores": request.max_cpu_cores,
            "gpu_allocator_budget_mb": request.max_gpu_memory_mb,
            "limits": "GPU budget excludes CUDA context; fixed small tensors, no global tuning",
        },
    )
    if not request.owner_approved:
        return result
    caps = worker.capabilities()
    if caps.cuda_available is not True:
        result.status, result.error_code = "unsupported", "CUDA_UNAVAILABLE"
        return result
    store = worker.artifact_dir / "experiments"
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = store / request.experiment_id
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError:
        result.status, result.error_code = "duplicate", "DUPLICATE_EXPERIMENT_ID"
        return result

    def persist():
        temporary = run_dir / "result.tmp"
        temporary.write_text(result.model_dump_json(indent=2))
        temporary.replace(run_dir / "result.json")

    (run_dir / "request.json").write_text(request.model_dump_json(indent=2))
    result.status, result.error_code = "partial", "RUN_IN_PROGRESS"
    persist()
    deadline = time.monotonic() + request.timeout_s
    try:
        for repeat in range(request.repeats):
            order = ["baseline", "treatment"] if repeat % 2 == 0 else ["treatment", "baseline"]
            for condition in order:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Total experiment timeout")
                cpus = [baseline_cpu] if condition == "baseline" else request.candidate_cpu_set
                started_at = datetime.now(UTC)
                raw = _invoke(
                    [
                        sys.executable,
                        "-m",
                        "gpu_os_copilot.workloads.launch_test",
                        "--scenario",
                        request.scenario,
                        "--cpus",
                        ",".join(map(str, cpus)),
                        "--contender-cpu",
                        str(baseline_cpu),
                        "--gpu-memory-mb",
                        str(request.max_gpu_memory_mb),
                        "--output-limit",
                        str(request.max_output_bytes),
                    ],
                    timeout=min(remaining, 15),
                    max_output=request.max_output_bytes,
                )
                finished_at = datetime.now(UTC)
                trial = BenchmarkTrial(
                    run_id=f"trial-{uuid4().hex}",
                    condition=condition,
                    wall_time_ms=raw["wall_time_ms"],
                    operation_count=raw["operation_count"],
                    throughput_ops_s=raw["operation_count"] * 1000 / raw["wall_time_ms"],
                    cpu_set=cpus,
                    configuration={**raw["configuration"], "host": caps.model_dump()},
                    started_at=started_at,
                    finished_at=finished_at,
                )
                result.trials.append(trial)
                result.provenance = Provenance(
                    source="benchmark",
                    confidence="measured",
                    description="Synchronized, unprofiled CUDA timings from project subprocesses",
                )
                persist()
        result.status, result.error_code = "ok", None
        result.comparison = compare_trials(result.trials)
    except (TimeoutError, subprocess.TimeoutExpired):
        result.status = "partial" if result.trials else "failed"
        result.error_code = "EXPERIMENT_TIMEOUT"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ZeroDivisionError):
        result.status = "partial" if result.trials else "failed"
        result.error_code = "BENCHMARK_FAILED"
    except KeyboardInterrupt:
        result.status = "partial" if result.trials else "failed"
        result.error_code = "CANCELLED"
    finally:
        persist()
    return result
