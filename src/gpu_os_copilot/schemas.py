"""Version 1 contracts. Nanoseconds are the only accepted trace timestamp unit."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
EventKind = Literal["cuda_api", "gpu_kernel", "memcpy", "cpu_sched", "nvtx", "other"]
Source = Literal["fixture", "nsys_sqlite", "linux_trace", "benchmark", "environment"]
Confidence = Literal["measured", "derived", "simulated"]
Scenario = Literal["cpu_starvation", "transfer_bound", "compute_bound"]
WorkloadScenario = Literal["cpu_contention", "transfer_bound", "compute_bound"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Provenance(Model):
    source: Source
    confidence: Confidence
    description: str = Field(min_length=1, max_length=1000)
    limitations: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def fixture_is_simulated(self):
        if self.source == "fixture" and self.confidence != "simulated":
            raise ValueError("Fixtures must be labeled simulated")
        return self


class Envelope(Model):
    schema_version: Literal["1.0"] = "1.0"
    experiment_id: Identifier
    worker_id: Identifier = "local"
    workload_id: Identifier
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance: Provenance

    @model_validator(mode="after")
    def aware_timestamp(self):
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must have a timezone")
        return self


class TraceEvent(Model):
    id: Identifier
    source: Source
    kind: EventKind
    start_ns: NonNegativeInt
    end_ns: NonNegativeInt
    unit: Literal["ns"] = "ns"
    pid: NonNegativeInt | None = None
    tid: NonNegativeInt | None = None
    device_id: NonNegativeInt | None = None
    stream_id: NonNegativeInt | None = None
    correlation_id: NonNegativeInt | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must be at least start_ns")
        if self.kind in {"gpu_kernel", "memcpy"} and self.device_id is None:
            raise ValueError("GPU activity must identify its device")
        if self.source == "fixture" and self.confidence != "simulated":
            raise ValueError("Fixture events cannot be measured")
        return self


class ObservationWindow(Model):
    start_ns: NonNegativeInt
    end_ns: NonNegativeInt
    unit: Literal["ns"] = "ns"
    device_id: NonNegativeInt = 0
    scope: Literal["captured_workload_all_streams"] = "captured_workload_all_streams"
    excludes_warmup_shutdown: Literal[True] = True

    @model_validator(mode="after")
    def nonempty(self):
        if self.end_ns <= self.start_ns:
            raise ValueError("Observation window must have positive duration")
        return self


class TraceBundle(Envelope):
    run_id: Identifier
    timebase: str = Field(min_length=1, max_length=100)
    clock_aligned: bool = False
    window: ObservationWindow
    trace_completeness: Literal["complete", "partial", "unknown"] = "unknown"
    events: list[TraceEvent] = Field(max_length=100000)
    wall_time_ms: float | None = Field(default=None, gt=0)
    operation_count: NonNegativeInt | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if len({e.id for e in self.events}) != len(self.events):
            raise ValueError("Event IDs must be unique")
        if any(e.confidence == "simulated" for e in self.events):
            if self.provenance.confidence != "simulated":
                raise ValueError("Simulated events require simulated bundle provenance")
        if len({e.source for e in self.events}) > 1 and not self.clock_aligned:
            raise ValueError("Mixed sources require explicit clock alignment")
        if (self.wall_time_ms is None) != (self.operation_count is None):
            raise ValueError("Wall time and operation count must be supplied together")
        return self


class MetricSummary(Envelope):
    run_id: Identifier
    window: ObservationWindow
    trace_completeness: Literal["complete", "partial", "unknown"]
    gpu_idle_fraction: float | None = Field(default=None, ge=0, le=1)
    gpu_busy_ms: float | None = Field(default=None, ge=0)
    gpu_idle_intervals_ns: list[tuple[int, int]] = Field(default_factory=list)
    kernel_duration_ms: dict[str, float] = Field(default_factory=dict)
    cuda_api_duration_ms: float | None = Field(default=None, ge=0)
    transfer_duration_ms: dict[str, float] = Field(default_factory=dict)
    transfer_busy_fraction: float | None = Field(default=None, ge=0, le=1)
    cpu_runnable_delay_ms: float | None = Field(default=None, ge=0)
    sched_idle_overlap_ms: float | None = Field(default=None, ge=0)
    wall_time_ms: float | None = Field(default=None, gt=0)
    throughput_ops_s: float | None = Field(default=None, ge=0)
    missing_data: list[str] = Field(default_factory=list)


class Hypothesis(Model):
    category: Literal[
        "cpu_scheduling",
        "launch_overhead",
        "data_transfer",
        "synchronization",
        "gpu_compute",
        "unknown",
    ]
    statement: str = Field(min_length=1, max_length=2000)
    evidence_event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    missing_evidence: list[str] = Field(default_factory=list, max_length=20)
    confidence: Literal["low", "medium", "high"] = "low"


class ExperimentPlan(Model):
    name: Literal["affinity_ab"]
    justification: str = Field(min_length=1, max_length=2000)


class AgentDecision(Model):
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=6)
    selected_experiment: ExperimentPlan | None = None
    decision: Literal["test", "supported", "rejected", "inconclusive"]
    rationale: str = Field(min_length=1, max_length=3000)


class WorkerCapabilities(Model):
    schema_version: Literal["1.0"] = "1.0"
    worker_id: Identifier = "local"
    os: str
    python_version: str
    cpu_affinity: list[NonNegativeInt]
    gpu_count: NonNegativeInt | None = None
    gpus: list[dict[str, str]] = Field(default_factory=list)
    cuda_available: bool | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    nsys_version: str | None = None
    cpu_scheduling_trace: Literal["unverified", "unsupported"] = "unverified"
    perf_event_paranoid: int | None = None
    limitations: list[str] = Field(default_factory=list)


class WorkerHealth(Model):
    schema_version: Literal["1.0"] = "1.0"
    worker_id: Identifier = "local"
    status: Literal["ok", "degraded"]
    limitations: list[str]


class TraceRequest(Envelope):
    scenario: Scenario


class TraceResult(Envelope):
    status: Literal["ok", "unsupported", "failed"]
    trace: TraceBundle | None = None
    error_code: str | None = None


class CaptureRequest(Envelope):
    workload_id: Literal["launch_test"] = "launch_test"
    scenario: WorkloadScenario = "cpu_contention"
    cpu_set: list[NonNegativeInt] = Field(min_length=1, max_length=1)
    contender_cpu: NonNegativeInt
    timeout_s: Annotated[int, Field(strict=True, ge=5, le=90)] = 60
    cuda_trace_mode: Literal["cuda", "cuda-sw"] = "cuda-sw"
    cpu_context_switches: bool = False
    owner_approved: bool = Field(default=False, strict=True)


class ExperimentRequest(Envelope):
    kind: Literal["affinity_ab"] = "affinity_ab"
    workload_id: Literal["launch_test"] = "launch_test"
    scenario: WorkloadScenario = "cpu_contention"
    candidate_cpu_set: list[NonNegativeInt] = Field(min_length=1, max_length=4)
    repeats: Annotated[int, Field(strict=True, ge=2, le=5)] = 5
    timeout_s: Annotated[int, Field(strict=True, ge=1, le=120)] = 60
    max_cpu_cores: Annotated[int, Field(strict=True, ge=1, le=4)] = 2
    max_gpu_memory_mb: Annotated[int, Field(strict=True, ge=64, le=512)] = 256
    max_output_bytes: Annotated[int, Field(strict=True, ge=1024, le=1048576)] = 65536
    owner_approved: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def cpu_budget(self):
        if len(set(self.candidate_cpu_set)) != len(self.candidate_cpu_set):
            raise ValueError("Duplicate CPU IDs")
        if len(self.candidate_cpu_set) > self.max_cpu_cores:
            raise ValueError("Candidate CPU set exceeds CPU budget")
        return self


class BenchmarkTrial(Model):
    run_id: Identifier
    condition: Literal["baseline", "treatment"]
    wall_time_ms: float = Field(gt=0)
    operation_count: Annotated[int, Field(strict=True, gt=0)]
    throughput_ops_s: float = Field(gt=0)
    cpu_set: list[NonNegativeInt]
    configuration: dict[str, Any]
    gpu_idle_fraction: None = None
    missing_data: list[str] = Field(default_factory=lambda: ["No profiler trace for timed trial"])


class ExperimentResult(Envelope):
    status: Literal["dry_run", "ok", "partial", "failed", "unsupported", "duplicate"]
    error_code: str | None = None
    trials: list[BenchmarkTrial] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
