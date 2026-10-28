import socket

import pytest

from gpu_os_copilot.schemas import Provenance, TraceRequest
from gpu_os_copilot.worker.local import LocalWorker


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Offline tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


@pytest.fixture
def load_fixture():
    def load(scenario="cpu_starvation"):
        return (
            LocalWorker()
            .collect_trace(
                TraceRequest(
                    experiment_id="test",
                    workload_id=scenario,
                    scenario=scenario,
                    provenance=Provenance(
                        source="fixture", confidence="simulated", description="Test"
                    ),
                )
            )
            .trace
        )

    return load
