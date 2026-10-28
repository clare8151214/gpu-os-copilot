import pytest

from gpu_os_copilot.schemas import Provenance, TraceRequest
from gpu_os_copilot.worker.local import LocalWorker
from gpu_os_copilot.worker.protocol import Worker


def test_worker_protocol_and_packaged_fixture(load_fixture):
    assert isinstance(LocalWorker(), Worker)
    assert load_fixture().worker_id == "local"
    assert load_fixture().provenance.confidence == "simulated"


def test_worker_revalidates_request():
    request = TraceRequest(
        experiment_id="test",
        worker_id="other",
        workload_id="test",
        scenario="compute_bound",
        provenance=Provenance(source="fixture", confidence="simulated", description="Test"),
    )
    with pytest.raises(ValueError, match="mismatch"):
        LocalWorker().collect_trace(request)


def test_path_injection_rejected():
    with pytest.raises(ValueError):
        TraceRequest(
            experiment_id="test",
            workload_id="test",
            scenario="../../secret",
            provenance=Provenance(source="fixture", confidence="simulated", description="Test"),
        )
