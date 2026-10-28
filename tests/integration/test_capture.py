import json
import sqlite3
import subprocess
import sys

import pytest

from gpu_os_copilot.collectors.capture import _run, bind_measurement, measurement_window
from gpu_os_copilot.schemas import CaptureRequest, Provenance, WorkerCapabilities
from gpu_os_copilot.worker.local import LocalWorker
from gpu_os_copilot.workloads.launch_test import WORKLOAD_VERSION


def synthetic_database(path, *, count=200, second_range=False):
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT);"
            "INSERT INTO NVTX_EVENTS VALUES (0, 10000, 'copilot_measure');"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, deviceId INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 10);"
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0)",
            [(100 + i * 10, 105 + i * 10) for i in range(count)],
        )
        if second_range:
            db.execute("INSERT INTO NVTX_EVENTS VALUES (20000, 30000, 'copilot_measure')")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr("gpu_os_copilot.collectors.capture.shutil.which", lambda name: "/test/nsys")
    worker = LocalWorker(tmp_path)
    worker._capabilities = WorkerCapabilities(
        os="test",
        python_version="test",
        cpu_affinity=[0, 1],
        cuda_available=True,
        torch_version="test",
        cuda_version="12.8",
        gpus=[{"model": "test GPU"}],
    )
    request = CaptureRequest(
        experiment_id="capture-test",
        cpu_set=[0],
        contender_cpu=0,
        provenance=Provenance(source="fixture", confidence="simulated", description="Test request"),
    )
    measurement = {
        "wall_time_ms": 0.01,
        "operation_count": 200,
        "configuration": {
            "scenario": "cpu_contention",
            "workload_version": WORKLOAD_VERSION,
            "seed": 42,
            "shape": [256, 256],
            "dtype": "float32",
            "warmup": 10,
            "operations": 200,
            "synchronized": True,
            "profiled": True,
            "torch": "test",
            "cuda": "12.8",
            "gpu": "test GPU",
        },
    }
    return worker, request, measurement


def test_capture_requires_approval_without_writes(setup, monkeypatch, tmp_path):
    worker, request, _ = setup
    monkeypatch.setattr(
        "gpu_os_copilot.collectors.capture._run", lambda *args: pytest.fail("Spawned")
    )
    result = worker.capture_trace(request, tmp_path / "report")
    assert result.error_code == "PROFILE_APPROVAL_REQUIRED"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change", [{"timeout_s": 900}, {"owner_approved": "true"}, {"scenario": "shell"}]
)
def test_capture_revalidates_request(setup, tmp_path, change):
    worker, request, _ = setup
    with pytest.raises(ValueError):
        worker.capture_trace(request.model_copy(update=change), tmp_path / "report")


def test_capture_binds_window_metadata_and_refuses_duplicate(setup, monkeypatch, tmp_path):
    worker, request, measurement = setup
    request.owner_approved = True
    calls = []

    def fake_run(argv, log, timeout):
        calls.append(argv)
        if argv[1] == "profile":
            prefix = tmp_path / "report"
            prefix.with_suffix(".nsys-rep").write_bytes(b"synthetic test report")
            log.write_text("Profiler output\n" + json.dumps(measurement) + "\n")
        else:
            synthetic_database(tmp_path / "report.sqlite")
            log.write_text("Synthetic export complete\n")

    monkeypatch.setattr("gpu_os_copilot.collectors.capture._run", fake_run)
    result = worker.capture_trace(request, tmp_path / "report")
    assert result.status == "ok"
    assert result.trace.workload_id == "launch_test"
    assert result.trace.window.end_ns == 10000
    assert result.trace.configuration["workload_version"] == WORKLOAD_VERSION
    assert result.trace.operation_count == 200
    assert (tmp_path / "report.capture/trace.json").is_file()
    assert worker.capture_trace(request, tmp_path / "report").error_code == "CAPTURE_ALREADY_EXISTS"
    assert len(calls) == 2


def test_capture_timeout_keeps_failure_record(setup, monkeypatch, tmp_path):
    worker, request, _ = setup
    request.owner_approved = True

    def timeout(*args):
        raise subprocess.TimeoutExpired("nsys", 1)

    monkeypatch.setattr("gpu_os_copilot.collectors.capture._run", timeout)
    result = worker.capture_trace(request, tmp_path / "report")
    assert result.error_code == "CAPTURE_TIMEOUT"
    saved = json.loads((tmp_path / "report.capture/result.json").read_text())
    assert saved["status"] == "failed" and saved["trace"] is None


def test_window_and_workload_evidence_must_match(setup, tmp_path):
    worker, request, measurement = setup
    path = tmp_path / "trace.sqlite"
    synthetic_database(path)
    measurement["configuration"]["seed"] = 999
    with pytest.raises(ValueError, match="CONFIGURATION_MISMATCH"):
        bind_measurement(path, measurement, request, worker.capabilities())
    path.unlink()
    synthetic_database(path, second_range=True)
    with pytest.raises(ValueError, match="WINDOW_AMBIGUOUS"):
        measurement_window(path)


def test_missing_gpu_operations_cannot_claim_complete_capture(setup, tmp_path):
    worker, request, measurement = setup
    path = tmp_path / "trace.sqlite"
    synthetic_database(path, count=199)
    with pytest.raises(ValueError, match="CAPTURE_INCOMPLETE"):
        bind_measurement(path, measurement, request, worker.capabilities())


def test_profiler_child_has_no_model_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", "test-only-credential")
    log = tmp_path / "child.log"
    _run(
        [sys.executable, "-c", "import os; print('MODEL_API_KEY' in os.environ)"],
        log,
        timeout=2,
    )
    assert log.read_text().strip() == "False"
