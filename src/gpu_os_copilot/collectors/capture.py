"""Bounded profiling of the bundled workload and provenance-bound trace normalization."""

import hashlib
import json
import os
import resource
import shutil
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

from gpu_os_copilot.collectors.nsys import build_commands, discover_schema, import_sqlite
from gpu_os_copilot.schemas import CaptureRequest, Provenance, TraceBundle, TraceResult
from gpu_os_copilot.workloads.specs import MODEL_SCENARIOS, expected_configuration

MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_LOG_BYTES = 4 * 1024 * 1024


def _limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_TRACE_BYTES, MAX_TRACE_BYTES))


def _run(argv: list[str], log: Path, timeout: float) -> None:
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        if name in os.environ
    }
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="42")
    with log.open("x") as output:
        process = subprocess.Popen(
            argv,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            preexec_fn=_limits,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError("PROFILE_CHILD_FAILED")
            if log.stat().st_size > MAX_LOG_BYTES:
                raise RuntimeError("PROFILE_LOG_BUDGET_EXCEEDED")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def measurement_window(path: Path) -> tuple[int, int, list[str]]:
    """Require one closed benchmark NVTX range, never guess bounds from GPU events."""
    if not path.is_file() or path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("TRACE_SIZE_INVALID")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        work = 0

        def budget():
            nonlocal work
            work += 1
            return int(work > 10000)

        db.set_progress_handler(budget, 1000)
        schema = discover_schema(db)
        columns = schema.get("NVTX_EVENTS", [])
        if not {"start", "end"}.issubset(columns):
            raise ValueError("NVTX_WINDOW_UNAVAILABLE")
        label = "n.text" if "text" in columns else "NULL"
        join = ""
        if "textId" in columns and {"id", "value"}.issubset(schema.get("StringIds", [])):
            join = " LEFT JOIN StringIds s ON n.textId=s.id"
            label = f"COALESCE({label}, s.value)"
        rows = db.execute(
            f"SELECT n.start, n.end FROM NVTX_EVENTS n{join} WHERE {label}=? LIMIT 2",
            ("copilot_measure",),
        ).fetchall()
        if len(rows) != 1 or any(type(value) is not int for value in rows[0]):
            raise ValueError("NVTX_WINDOW_AMBIGUOUS")
        start, end = rows[0]
        if start < 0 or end <= start:
            raise ValueError("NVTX_WINDOW_INVALID")
        warnings = []
        if {"severity", "text"}.issubset(schema.get("DIAGNOSTIC_EVENT", [])):
            for severity, text in db.execute(
                "SELECT severity, text FROM DIAGNOSTIC_EVENT WHERE severity>=2 LIMIT 1000"
            ):
                message = str(text)
                lower = message.lower()
                if severity >= 3 or (
                    "cuda" in lower
                    and any(word in lower for word in ("dropped", "lost", "not all"))
                ):
                    raise ValueError("CAPTURE_DIAGNOSTIC_ERROR")
                warnings.append(message[:500])
    return start, end, warnings[:10]


def bind_measurement(
    path: Path,
    measurement: dict,
    request: CaptureRequest,
    capabilities,
) -> TraceBundle:
    configuration = measurement.get("configuration", {})
    expected = {
        **expected_configuration(request.scenario),
        "profiled": True,
        "torch": capabilities.torch_version,
        "cuda": capabilities.cuda_version,
        "gpu": capabilities.gpus[0]["model"],
    }
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError("WORKLOAD_CONFIGURATION_MISMATCH")
    if measurement.get("operation_count") != 200:
        raise ValueError("WORKLOAD_OPERATION_COUNT_MISMATCH")
    if (
        request.scenario in MODEL_SCENARIOS
        and measurement.get("validation", {}).get("validated") is not True
    ):
        raise ValueError("MODEL_OUTPUT_NOT_VALIDATED")
    start, end, warnings = measurement_window(path)
    trace = import_sqlite(path, start_ns=start, end_ns=end, capture_complete=True)
    relevant = [
        event
        for event in trace.events
        if event.start_ns >= start and event.end_ns <= end and event.device_id == 0
    ]
    if request.scenario == "transfer_bound":
        enough = sum(event.kind == "memcpy" for event in relevant) >= 400
    else:
        enough = sum(event.kind == "gpu_kernel" for event in relevant) >= 200
    if not enough or trace.trace_completeness != "complete":
        raise ValueError("WORKLOAD_CAPTURE_INCOMPLETE")
    data = trace.model_dump()
    data.update(
        experiment_id=request.experiment_id,
        run_id=request.experiment_id,
        worker_id=request.worker_id,
        workload_id="launch_test",
        wall_time_ms=measurement["wall_time_ms"],
        operation_count=200,
        configuration={
            **configuration,
            "capture": {
                "cpu_set": request.cpu_set,
                "contender_cpu": request.contender_cpu,
                "cuda_trace_mode": request.cuda_trace_mode,
                "cpu_context_switches_requested": request.cpu_context_switches,
                "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        },
    )
    data["provenance"]["description"] = "Measured bundled workload with a closed NVTX window"
    data["provenance"]["limitations"].extend(warnings)
    data["provenance"]["limitations"].append(
        "Trace wall time includes profiler overhead; compare unprofiled trials separately."
    )
    return TraceBundle.model_validate(data)


def capture_trace(worker, request: CaptureRequest, output: Path) -> TraceResult:
    request = CaptureRequest.model_validate(request.model_dump(warnings=False))
    if request.worker_id != worker.worker_id:
        raise ValueError("Worker ID mismatch")
    plan = build_commands(
        request.scenario,
        output,
        request.cpu_context_switches,
        request.cuda_trace_mode,
    )
    result = TraceResult(
        experiment_id=request.experiment_id,
        worker_id=worker.worker_id,
        workload_id=request.workload_id,
        status="unsupported",
        provenance=Provenance(
            source="nsys_sqlite",
            confidence="derived",
            description="Bounded trace request; capture not completed",
        ),
    )
    if not request.owner_approved:
        result.error_code = "PROFILE_APPROVAL_REQUIRED"
        return result
    if request.sudo_scheduler and (request.cpu_context_switches or os.getuid() == 0):
        raise ValueError(
            "Tracefs mode requires unprivileged Nsight without context-switch sampling"
        )
    allowed = set(os.sched_getaffinity(0))
    if not set(request.cpu_set + [request.contender_cpu]).issubset(allowed):
        raise ValueError("CPU set outside available affinity")
    capabilities = worker.capabilities()
    executable = shutil.which("nsys")
    if not executable or capabilities.cuda_available is not True:
        result.error_code = "PROFILE_CAPABILITIES_UNAVAILABLE"
        return result
    output = output.resolve()
    report, database = output.with_suffix(".nsys-rep"), output.with_suffix(".sqlite")
    state = output.with_name(output.name + ".capture")
    if report.exists() or database.exists():
        result.error_code = "CAPTURE_ALREADY_EXISTS"
        return result
    state.parent.mkdir(parents=True, exist_ok=True)
    try:
        state.mkdir(mode=0o700)
    except FileExistsError:
        result.error_code = "CAPTURE_ALREADY_EXISTS"
        return result
    (state / "request.json").write_text(request.model_dump_json(indent=2))
    for key in ("profile_argv", "export_argv"):
        plan[key][0] = executable
    plan["profile_argv"].extend(
        [
            "--cpus",
            ",".join(map(str, request.cpu_set)),
            "--contender-cpu",
            str(request.contender_cpu),
        ]
    )
    (state / "plan.json").write_text(json.dumps(plan, indent=2))
    deadline = time.monotonic() + request.timeout_s
    result.status, result.error_code = "failed", "CAPTURE_IN_PROGRESS"

    def persist():
        temporary = state / "result.tmp"
        temporary.write_text(result.model_dump_json(indent=2))
        temporary.replace(state / "result.json")

    persist()
    try:
        for stage in ("profile", "export"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if stage == "profile" and request.sudo_scheduler:
                from gpu_os_copilot.collectors.scheduler_capture import profile_with_scheduler

                profile_with_scheduler(
                    plan["profile_argv"], state / "profile.log", remaining, state
                )
            else:
                _run(plan[stage + "_argv"], state / (stage + ".log"), remaining)
            artifact = report if stage == "profile" else database
            if not artifact.is_file() or artifact.stat().st_size > MAX_TRACE_BYTES:
                raise RuntimeError("CAPTURE_ARTIFACT_INVALID")
        measurements = []
        for line in (state / "profile.log").read_text().splitlines():
            if line.startswith("{"):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and "configuration" in item:
                    measurements.append(item)
        if len(measurements) != 1:
            raise ValueError("WORKLOAD_MEASUREMENT_UNAVAILABLE")
        (state / "measurement.json").write_text(json.dumps(measurements[0], indent=2))
        trace = bind_measurement(database, measurements[0], request, capabilities)
        if request.sudo_scheduler:
            from gpu_os_copilot.collectors.scheduler import merge_scheduler

            trace = merge_scheduler(
                trace,
                database,
                (state / "scheduler.txt").read_text(),
                json.loads((state / "scheduler-meta.json").read_text()),
                measurements[0].get("clock_anchors", []),
            )
        (state / "trace.json").write_text(trace.model_dump_json(indent=2))
        result.trace, result.provenance = trace, trace.provenance
        result.status, result.error_code = "ok", None
    except (TimeoutError, subprocess.TimeoutExpired):
        result.error_code = "CAPTURE_TIMEOUT"
    except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError, IndexError) as error:
        result.error_code = "CAPTURE_FAILED"
        (state / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2)
        )
    except KeyboardInterrupt:
        result.error_code = "CAPTURE_CANCELLED"
    finally:
        persist()
    return result
