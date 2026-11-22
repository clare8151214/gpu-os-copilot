import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from gpu_os_copilot.experiments.affinity import _invoke
from gpu_os_copilot.schemas import ExperimentRequest, Provenance, WorkerCapabilities
from gpu_os_copilot.worker.local import LocalWorker


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1})
    worker = LocalWorker(tmp_path)
    worker._capabilities = WorkerCapabilities(
        os="test",
        python_version="test",
        cpu_affinity=[0, 1],
        cuda_available=True,
    )
    request = ExperimentRequest(
        experiment_id="test",
        candidate_cpu_set=[1],
        repeats=2,
        provenance=Provenance(source="benchmark", confidence="derived", description="Test"),
    )
    return worker, request


def test_default_dry_run_never_spawns(setup, monkeypatch):
    worker, request = setup
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    result = worker.run_experiment(request)
    assert result.status == "dry_run"
    assert result.trials == []
    assert not (worker.artifact_dir / "experiments").exists()


@pytest.mark.parametrize("scenario", ["cnn_inference", "cnn_training"])
@pytest.mark.parametrize("valid", [True, False])
def test_model_trials_require_validated_outputs(setup, monkeypatch, scenario, valid):
    from gpu_os_copilot.workloads.specs import expected_configuration

    worker, request = setup
    request.owner_approved = True
    request.scenario = scenario
    raw = {
        "wall_time_ms": 10,
        "operation_count": 200,
        "configuration": expected_configuration(scenario),
        "validation": {"validated": valid, "output_shape": [32, 10]},
    }
    monkeypatch.setattr("gpu_os_copilot.experiments.affinity._invoke", lambda *a, **kw: raw)
    result = worker.run_experiment(request)
    assert result.status == ("ok" if valid else "failed")
    if valid:
        assert all(t.validation["output_shape"] == [32, 10] for t in result.trials)
    else:
        assert result.trials == []


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "shell"},
        {"candidate_cpu_set": [-1]},
        {"repeats": 100},
        {"timeout_s": 121},
        {"candidate_cpu_set": [1, 1]},
        {"owner_approved": "true"},
        {"pid": 1},
        {"command": "anything"},
        {"experiment_id": "../escape"},
    ],
)
def test_request_rejects_unsafe_parameters(setup, change):
    _, request = setup
    data = request.model_dump()
    data.update(change)
    with pytest.raises(ValueError):
        ExperimentRequest.model_validate(data)


@pytest.mark.parametrize("cpus", [[0], [99]])
def test_cpu_validation(setup, cpus):
    worker, request = setup
    request.candidate_cpu_set = cpus
    with pytest.raises(ValueError):
        worker.run_experiment(request)


def test_worker_revalidates_model_construct_bypass(setup):
    worker, request = setup
    invalid = request.model_copy(update={"timeout_s": 999999})
    with pytest.raises(ValueError):
        worker.run_experiment(invalid)


def test_missing_hardware_is_unsupported(setup):
    worker, request = setup
    request.owner_approved = True
    worker._capabilities.cuda_available = False
    assert worker.run_experiment(request).error_code == "CUDA_UNAVAILABLE"


def test_alternating_trials_persistence_and_duplicate_refusal(setup, monkeypatch):
    worker, request = setup
    request.owner_approved = True
    calls = []
    child_intervals = []

    def simulated(argv, timeout, max_output):
        calls.append(argv)
        child_intervals.append((datetime.now(UTC), datetime.now(UTC)))
        return {"wall_time_ms": 10, "operation_count": 200, "configuration": {"test": True}}

    monkeypatch.setattr("gpu_os_copilot.experiments.affinity._invoke", simulated)
    result = worker.run_experiment(request)
    assert [t.condition for t in result.trials] == [
        "baseline",
        "treatment",
        "treatment",
        "baseline",
    ]
    assert result.status == "ok"
    assert len(calls) == 4
    # Finish = start + monotonic elapsed, so it can differ from separate wall-clock reads by
    # clock-read jitter (observed 1 us).
    jitter = timedelta(milliseconds=1)
    for trial, (child_start, child_finish) in zip(result.trials, child_intervals, strict=True):
        assert trial.started_at <= child_start <= child_finish <= trial.finished_at + jitter
    assert all(
        first.finished_at <= second.started_at + jitter
        for first, second in zip(result.trials, result.trials[1:], strict=False)
    )
    assert all(argv[1:3] == ["-m", "gpu_os_copilot.workloads.launch_test"] for argv in calls)
    stored = json.loads((worker.artifact_dir / "experiments/test/result.json").read_text())
    assert stored["comparison"]["baseline"]["n"] == 2
    assert all(trial["started_at"].endswith("Z") for trial in stored["trials"])
    assert all(trial["finished_at"].endswith("Z") for trial in stored["trials"])
    assert worker.run_experiment(request).status == "duplicate"


def test_partial_failure_persisted(setup, monkeypatch):
    worker, request = setup
    request.owner_approved = True
    responses = iter([{"wall_time_ms": 10, "operation_count": 200, "configuration": {}}])

    def fail_after_one(*args, **kwargs):
        try:
            return next(responses)
        except StopIteration:
            raise subprocess.TimeoutExpired("project-workload", 1) from None

    monkeypatch.setattr("gpu_os_copilot.experiments.affinity._invoke", fail_after_one)
    result = worker.run_experiment(request)
    assert result.status == "partial"
    assert result.error_code == "EXPERIMENT_TIMEOUT"
    assert len(result.trials) == 1
    assert result.comparison == {}


def test_child_timeout_and_secret_isolation(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "must-not-reach-child")
    data = _invoke(
        [
            sys.executable,
            "-c",
            ("import json, os; print(json.dumps({'has_key': 'MODEL_API_KEY' in os.environ}))"),
        ],
        timeout=2,
        max_output=4096,
    )
    assert data == {"has_key": False}
    with pytest.raises(subprocess.TimeoutExpired):
        _invoke([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1, max_output=4096)


def test_child_warnings_do_not_corrupt_json():
    result = _invoke(
        [
            sys.executable,
            "-c",
            "import sys; print('optional dependency warning', file=sys.stderr); "
            "print('{\"ok\": true}')",
        ],
        timeout=2,
        max_output=4096,
    )
    assert result == {"ok": True}
    with pytest.raises(RuntimeError, match="output budget"):
        _invoke(
            [sys.executable, "-c", "import sys; print('x' * 4096, file=sys.stderr); print('{}')"],
            timeout=2,
            max_output=4096,
        )


def test_child_failure_keeps_exit_status_and_local_stderr(setup, monkeypatch):
    with pytest.raises(RuntimeError, match="child failed") as raised:
        _invoke(
            [sys.executable, "-c", "import sys; print('x' * 9000, file=sys.stderr); sys.exit(3)"],
            timeout=2,
            max_output=4096,
        )
    assert raised.value.summary() == {"exit_code": 3, "signal": None, "stderr_bytes": 4097}
    assert len(raised.value.stderr_tail) == 4096
    with pytest.raises(RuntimeError) as killed:
        _invoke([sys.executable, "-c", "import os; os.kill(os.getpid(), 9)"], 2, 4096)
    assert killed.value.summary()["signal"] == "SIGKILL"
    with pytest.raises(RuntimeError, match="not JSON"):
        _invoke([sys.executable, "-c", "print('partial')"], timeout=2, max_output=4096)

    worker, request = setup
    request.owner_approved = True
    raw = {"wall_time_ms": 10, "operation_count": 200, "configuration": {}}
    calls = iter([raw, killed.value])

    def fail_second(*args, **kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("gpu_os_copilot.experiments.affinity._invoke", fail_second)
    result = worker.run_experiment(request)
    assert (result.status, result.error_code) == ("partial", "BENCHMARK_FAILED")
    assert result.failure == {
        "error_type": "ChildRunError",
        "trials_completed": 1,
        "exit_code": -9,
        "signal": "SIGKILL",
        "stderr_bytes": 0,
        "diagnostics_file": "failure.json",
    }
    assert "stderr_tail" not in result.model_dump_json()
    # Successful results serialize without the field (readable by earlier schema versions).
    assert "failure" not in result.model_copy(update={"failure": {}}).model_dump(mode="json")
    local = worker.artifact_dir / "experiments" / request.experiment_id / "failure.json"
    assert json.loads(local.read_text())["stderr_tail"] == ""
    assert local.stat().st_mode & 0o777 == 0o600


def test_trials_survive_backward_wall_clock_steps_and_keep_validator_messages(setup, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from gpu_os_copilot.experiments import affinity

    class SteppingClock:
        """Each reading is one second earlier, like a WSL2 time-sync step during a trial."""

        current = datetime(2026, 11, 2, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            cls.current -= timedelta(seconds=1)
            return cls.current

    worker, request = setup
    request.owner_approved = True
    monkeypatch.setattr(affinity, "datetime", SteppingClock)
    raw = {"wall_time_ms": 10, "operation_count": 200, "configuration": {}}
    monkeypatch.setattr(affinity, "_invoke", lambda *args, **kwargs: raw)
    result = worker.run_experiment(request)
    assert result.status == "ok"
    assert all(t.finished_at >= t.started_at for t in result.trials)

    monkeypatch.setattr(affinity, "_invoke", lambda *args, **kwargs: {**raw, "wall_time_ms": -7.25})
    failed = worker.run_experiment(request.model_copy(update={"experiment_id": "exp-invalid"}))
    assert (failed.status, failed.error_code) == ("failed", "BENCHMARK_FAILED")
    assert failed.failure["error_type"] == "ValidationError"
    assert failed.failure["validation_errors"][0]["type"] == "greater_than"
    assert "7.25" not in json.dumps(failed.failure)  # Messages carry no input values.
