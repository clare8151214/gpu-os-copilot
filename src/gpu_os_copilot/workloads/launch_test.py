"""Short deterministic CUDA benchmark; invoked only after explicit owner opt-in."""

import argparse
import json
import multiprocessing
import os
import resource
import time

WORKLOAD_VERSION = 2
SCENARIO_SHAPES = {
    "cpu_contention": (256, 256),
    "transfer_bound": (2048, 2048),
    "compute_bound": (2048, 2048),
}


def contend(cpu, stop, ready):
    os.sched_setaffinity(0, {cpu})
    ready.set()
    deadline = time.monotonic() + 12
    value = 0
    while not stop.is_set() and time.monotonic() < deadline:
        for _ in range(10000):
            value = (value + 1) % 104729


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["cpu_contention", "transfer_bound", "compute_bound"],
        default="cpu_contention",
    )
    parser.add_argument("--cpus", default="")
    parser.add_argument("--contender-cpu", type=int)
    parser.add_argument("--gpu-memory-mb", type=int, default=256)
    parser.add_argument("--output-limit", type=int, default=65536)
    parser.add_argument("--profiled", action="store_true")
    args = parser.parse_args()
    if not 64 <= args.gpu_memory_mb <= 512 or not 1024 <= args.output_limit <= 1048576:
        parser.error("Resource limits outside permitted range")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_FSIZE, (args.output_limit, args.output_limit))
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
        if args.scenario == "cpu_contention":
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

        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("copilot_measure"):
            started = time.perf_counter_ns()
            for _ in range(200):
                operation()
            torch.cuda.synchronize()
            wall_time_ms = (time.perf_counter_ns() - started) / 1e6
        if child is not None and not child.is_alive():
            raise RuntimeError("Contender exited before the measurement completed")
        print(
            json.dumps(
                {
                    "wall_time_ms": wall_time_ms,
                    "operation_count": 200,
                    "configuration": {
                        "scenario": args.scenario,
                        "workload_version": WORKLOAD_VERSION,
                        "seed": 42,
                        "shape": list(shape),
                        "dtype": "float32",
                        "warmup": 10,
                        "operations": 200,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": properties.name,
                        "synchronized": True,
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
