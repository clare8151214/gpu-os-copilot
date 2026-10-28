from pathlib import Path
from typing import Protocol, runtime_checkable

from gpu_os_copilot.schemas import (
    CaptureRequest,
    ExperimentRequest,
    ExperimentResult,
    TraceRequest,
    TraceResult,
    WorkerCapabilities,
    WorkerHealth,
)


@runtime_checkable
class Worker(Protocol):
    def capabilities(self) -> WorkerCapabilities: ...
    def collect_trace(self, request: TraceRequest) -> TraceResult: ...
    def capture_trace(self, request: CaptureRequest, output: Path) -> TraceResult: ...
    def run_experiment(self, request: ExperimentRequest) -> ExperimentResult: ...
    def health(self) -> WorkerHealth: ...
