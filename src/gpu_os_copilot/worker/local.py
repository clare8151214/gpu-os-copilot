from importlib.resources import files
from pathlib import Path

from gpu_os_copilot.collectors.environment import discover
from gpu_os_copilot.schemas import (
    CaptureRequest,
    ExperimentRequest,
    ExperimentResult,
    TraceBundle,
    TraceRequest,
    TraceResult,
    WorkerCapabilities,
    WorkerHealth,
)


class LocalWorker:
    worker_id = "local"

    def __init__(self, artifact_dir: Path = Path("artifacts")):
        self.artifact_dir = artifact_dir
        self._capabilities = None

    def capabilities(self) -> WorkerCapabilities:
        if self._capabilities is None:
            self._capabilities = discover()
        return self._capabilities.model_copy(deep=True)

    def health(self) -> WorkerHealth:
        caps = self.capabilities()
        return WorkerHealth(
            status="degraded" if caps.limitations else "ok", limitations=caps.limitations
        )

    def collect_trace(self, request: TraceRequest) -> TraceResult:
        request = TraceRequest.model_validate(request.model_dump())
        if request.worker_id != self.worker_id:
            raise ValueError("Worker ID mismatch")
        content = (
            files("gpu_os_copilot").joinpath("fixtures", f"{request.scenario}.json").read_text()
        )
        data = TraceBundle.model_validate_json(content).model_dump()
        data.update(experiment_id=request.experiment_id, workload_id=request.workload_id)
        trace = TraceBundle.model_validate(data)
        return TraceResult(
            experiment_id=request.experiment_id,
            workload_id=request.workload_id,
            provenance=trace.provenance,
            status="ok",
            trace=trace,
        )

    def run_experiment(self, request: ExperimentRequest) -> ExperimentResult:
        from gpu_os_copilot.experiments.affinity import run_affinity_ab

        return run_affinity_ab(self, request)

    def import_sqlite(self, path: Path, **kwargs) -> TraceBundle:
        from gpu_os_copilot.collectors.nsys import import_sqlite

        return import_sqlite(path, **kwargs)

    def capture_trace(self, request: CaptureRequest, output: Path) -> TraceResult:
        from gpu_os_copilot.collectors.capture import capture_trace

        return capture_trace(self, request, output)

    def plan_trace(
        self,
        scenario: str,
        output: Path,
        cpu_context_switches: bool = False,
        cuda_trace_mode: str = "cuda-sw",
    ) -> dict:
        from gpu_os_copilot.collectors.nsys import build_commands

        return build_commands(scenario, output, cpu_context_switches, cuda_trace_mode)
