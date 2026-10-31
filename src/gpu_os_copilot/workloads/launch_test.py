"""Short deterministic CUDA benchmark; invoked only after explicit owner opt-in."""

import argparse
import json
import multiprocessing
import os
import resource
import time
from pathlib import Path
from uuid import uuid4

from gpu_os_copilot.workloads.specs import (
    MODEL_SCENARIOS,
    SCENARIO_SHAPES,
    expected_configuration,
)
from gpu_os_copilot.workloads.specs import WORKLOAD_VERSION as WORKLOAD_VERSION


def set_resource_limits(output_limit: int, *, profiled: bool):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    # Injected Nsight libraries write trace streams inside this process. The
    # timing-only stdout budget must not truncate those streams at 64 KiB.
    file_limit = 256 * 1024 * 1024 if profiled else output_limit
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))


def contend(cpu, stop, ready):
    os.sched_setaffinity(0, {cpu})
    ready.set()
    deadline = time.monotonic() + 12
    value = 0
    while not stop.is_set() and time.monotonic() < deadline:
        for _ in range(10000):
            value = (value + 1) % 104729


def enroll_scheduler_threads() -> list[dict]:
    """Name existing threads of this workload for namespace-safe, bounded enrollment."""
    tasks = sorted(Path("/proc/self/task").iterdir(), key=lambda p: int(p.name))
    if not 1 <= len(tasks) <= 32:
        raise RuntimeError("Scheduler enrollment exceeds the 32-thread budget")
    enrolled = []
    for task in tasks:
        comm = "copilot" + uuid4().hex[:8]
        birth = (task / "stat").read_text().rsplit(")", 1)[1].split()[19]
        (task / "comm").write_text(comm)
        if (task / "comm").read_text().strip() != comm:
            raise RuntimeError("Owned thread rename failed")
        enrolled.append({"tid": int(task.name), "comm": comm, "start_ticks": birth})
    return enrolled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_SHAPES),
        default="cpu_contention",
    )
    parser.add_argument("--cpus", default="")
    parser.add_argument("--contender-cpu", type=int)
    parser.add_argument("--gpu-memory-mb", type=int, default=256)
    parser.add_argument("--output-limit", type=int, default=65536)
    parser.add_argument("--profiled", action="store_true")
    parser.add_argument("--scheduler-gate", type=Path)
    args = parser.parse_args()
    if not 64 <= args.gpu_memory_mb <= 512 or not 1024 <= args.output_limit <= 1048576:
        parser.error("Resource limits outside permitted range")
    set_resource_limits(args.output_limit, profiled=args.profiled)
    available = sorted(os.sched_getaffinity(0))
    cpus = [int(cpu) for cpu in args.cpus.split(",")] if args.cpus else available[:1]
    contender_cpu = args.contender_cpu if args.contender_cpu is not None else cpus[0]
    if not cpus or len(cpus) > 4 or not set(cpus + [contender_cpu]).issubset(available):
        parser.error("CPU set outside available affinity")
    os.sched_setaffinity(0, set(cpus) | {contender_cpu})
    # Fork the CPU-only contender before importing/initializing CUDA. This avoids
    # injecting profiler libraries into multiprocessing's spawned resource tracker.
    context = multiprocessing.get_context("fork")
    stop, ready = context.Event(), context.Event()
    child = None
    try:
        os.sched_setaffinity(0, set(cpus))
        if args.scenario == "cpu_contention" or args.scenario in MODEL_SCENARIOS:
            child = context.Process(target=contend, args=(contender_cpu, stop, ready), daemon=True)
            child.start()
            if not ready.wait(3):
                raise RuntimeError("Contender did not become ready")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(42)
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.gpu_memory_mb * 1024 * 1024 / properties.total_memory),
            device=0,
        )
        shape = SCENARIO_SHAPES[args.scenario]
        value = torch.ones(shape, device="cuda")
        other = torch.ones_like(value)
        product = torch.empty_like(value)
        host = torch.ones(shape, pin_memory=True)

        def operation():
            if args.scenario == "compute_bound":
                torch.mm(value, other, out=product)
            elif args.scenario == "transfer_bound":
                value.copy_(host, non_blocking=True)
                host.copy_(value, non_blocking=True)
            else:
                value.add_(0.001)

        def validate():
            return {}

        if args.scenario in MODEL_SCENARIOS:
            from gpu_os_copilot.workloads.cnn import build_operation

            operation, validate = build_operation(torch, training=args.scenario == "cnn_training")

        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        anchors = []
        if args.scheduler_gate is not None:
            torch.cuda.nvtx.mark("copilot_clock_warmup")
            gate = args.scheduler_gate.resolve(strict=True)
            if not args.profiled or gate.stat().st_uid != os.getuid():
                raise RuntimeError("Scheduler gate requires an owned profiled capture directory")
            # Warmup creates the autograd CUDA launcher. Later threads remain explicitly
            # uncovered; the helper verifies each enrolled thread's owner and lifetime.
            enrolled = enroll_scheduler_threads()
            with (gate / "scheduler-ready.json").open("x") as ready_file:
                json.dump({"pid": os.getpid(), "uid": os.getuid(), "threads": enrolled}, ready_file)
            deadline = time.monotonic() + 8
            while not (gate / "scheduler.release").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Scheduler helper did not become ready")
                time.sleep(0.02)
            # Positive collection control outside the measured NVTX range.
            time.sleep(0.002)

        def clock_anchor(name):
            if args.scheduler_gate is not None:
                before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                torch.cuda.nvtx.mark(name)
                anchors.append(
                    {
                        "name": name,
                        "clock": "mono_raw",
                        "before_ns": before,
                        "after_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                    }
                )

        clock_anchor("copilot_clock_before")
        with torch.cuda.nvtx.range("copilot_measure"):
            started = time.perf_counter_ns()
            if args.scheduler_gate is None:
                for _ in range(200):
                    operation()
            else:
                for block in range(4):
                    for _ in range(50):
                        operation()
                    if block < 3:
                        clock_anchor(f"copilot_clock_{(block + 1) * 50}")
            torch.cuda.synchronize()
            wall_time_ms = (time.perf_counter_ns() - started) / 1e6
        clock_anchor("copilot_clock_after")
        if args.scheduler_gate is not None:
            # End collection before model validation and profiler/library shutdown.
            (gate / "scheduler.stop").touch(exist_ok=True)
            deadline = time.monotonic() + 3
            while not (gate / "scheduler.finished").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Scheduler helper did not verify enrolled thread lifetimes")
                time.sleep(0.02)
        if child is not None and not child.is_alive():
            raise RuntimeError("Contender exited before the measurement completed")
        print(
            json.dumps(
                {
                    "wall_time_ms": wall_time_ms,
                    "operation_count": 200,
                    "validation": validate(),
                    "clock_anchors": anchors,
                    "configuration": {
                        **expected_configuration(args.scenario),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": properties.name,
                        "profiled": args.profiled,
                    },
                }
            )
        )
    finally:
        stop.set()
        if child is not None:
            child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join()


if __name__ == "__main__":
    main()
