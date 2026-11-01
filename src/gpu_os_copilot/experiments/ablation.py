"""Hidden ground truth: interleaved base/ablation timings for fixed pattern scenarios."""

import os
import sys
import time
from datetime import UTC, datetime
from statistics import median

from gpu_os_copilot.experiments.affinity import _invoke
from gpu_os_copilot.workloads.specs import PATTERN_SCENARIOS, expected_configuration


def run_ablation(
    scenario: str,
    cpu: int,
    *,
    factor: str | None = None,
    batches: int = 3,
    trials: int = 5,
    timeout_s: float = 600,
    gpu_memory_mb: int = 256,
    output_limit: int = 65536,
    invoke=_invoke,
) -> dict:
    """Never offered to agent tools; results label evaluation cases only."""
    if scenario not in PATTERN_SCENARIOS:
        raise ValueError("Ablation requires a fixed pattern scenario")
    expected_configuration(scenario, True, factor)  # Validates the factor before any run.
    if not (1 <= batches <= 5 and 2 <= trials <= 5 and 1 <= timeout_s <= 900):
        raise ValueError("Ablation budget outside permitted range")
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("CPU is not available")
    result = {
        "version": "ablation-v1",
        "scenario": scenario,
        "factor": factor,
        "cpu": cpu,
        "status": "partial",
        "trials": [],
        "batch_gains": [],
        "provenance": "Synchronized, unprofiled CUDA timings from project subprocesses",
    }
    deadline = time.monotonic() + timeout_s
    try:
        for batch in range(batches):
            for trial in range(trials):
                order = (False, True) if (batch + trial) % 2 == 0 else (True, False)
                for ablated in order:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("ABLATION_TIMEOUT")
                    started = datetime.now(UTC).isoformat()
                    raw = invoke(
                        [
                            sys.executable,
                            "-m",
                            "gpu_os_copilot.workloads.launch_test",
                            "--scenario",
                            scenario,
                            *(["--ablate"] if ablated else []),
                            *(["--factor", factor] if ablated and factor else []),
                            "--cpus",
                            str(cpu),
                            "--contender-cpu",
                            str(cpu),
                            "--gpu-memory-mb",
                            str(gpu_memory_mb),
                            "--output-limit",
                            str(output_limit),
                        ],
                        timeout=min(remaining, 30),
                        max_output=output_limit,
                    )
                    configuration = raw.get("configuration", {})
                    expected = expected_configuration(
                        scenario, ablated, factor if ablated else None
                    )
                    if any(configuration.get(k) != v for k, v in expected.items()):
                        raise ValueError("ABLATION_CONFIGURATION_MISMATCH")
                    if raw.get("operation_count") != 200 or not raw.get("wall_time_ms", 0) > 0:
                        raise ValueError("ABLATION_MEASUREMENT_INVALID")
                    result["trials"].append(
                        {
                            "batch": batch,
                            "condition": "ablated" if ablated else "base",
                            "wall_time_ms": raw["wall_time_ms"],
                            "throughput_ops_s": 200 * 1000 / raw["wall_time_ms"],
                            "started_at": started,
                            "finished_at": datetime.now(UTC).isoformat(),
                        }
                    )
            rows = [t for t in result["trials"] if t["batch"] == batch]
            base = median(t["throughput_ops_s"] for t in rows if t["condition"] == "base")
            ablated_median = median(
                t["throughput_ops_s"] for t in rows if t["condition"] == "ablated"
            )
            result["batch_gains"].append(ablated_median / base - 1)
        result["status"] = "ok"
    except (TimeoutError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        result["status"] = "failed"
        result["error"] = type(error).__name__ + ": " + str(error)[:200]
    return result
