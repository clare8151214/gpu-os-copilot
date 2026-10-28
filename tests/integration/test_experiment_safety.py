import json
import subprocess
import sys

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

    def simulated(argv, timeout, max_output):
        calls.append(argv)
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
    assert all(argv[1:3] == ["-m", "gpu_os_copilot.workloads.launch_test"] for argv in calls)
    stored = json.loads((worker.artifact_dir / "experiments/test/result.json").read_text())
    assert stored["comparison"]["baseline"]["n"] == 2
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
