"""Read-only, timeout-bounded discovery; never infer absent GPU capabilities."""

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from gpu_os_copilot.schemas import WorkerCapabilities


def _read_command(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def discover() -> WorkerCapabilities:
    limitations = []
    gpus = []
    gpu_count = None
    if shutil.which("nvidia-smi"):
        inventory = _read_command(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        if inventory is not None:
            gpus = [
                {"model": row[0].strip(), "driver": row[1].strip()}
                for row in csv.reader(inventory.splitlines())
                if len(row) == 2
            ]
            gpu_count = len(gpus)
        else:
            limitations.append("nvidia-smi failed; GPU inventory unknown")
    else:
        limitations.append("nvidia-smi not found; GPU inventory unknown")
    # An isolated interpreter contains torch import/driver initialization failures.
    torch_info = _read_command(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; print(json.dumps({'version': torch.__version__, "
                "'cuda': torch.cuda.is_available(), 'runtime': torch.version.cuda}))"
            ),
        ]
    )
    info = {}
    if torch_info:
        try:
            info = json.loads(torch_info)
        except json.JSONDecodeError:
            limitations.append("Unrecognized PyTorch probe response")
    else:
        limitations.append("PyTorch unavailable or CUDA probe failed")
    nsys_version = _read_command(["nsys", "--version"]) if shutil.which("nsys") else None
    if not nsys_version:
        limitations.append("Nsight Systems unavailable")
    limitations.append("CPU scheduling capture permissions and time alignment unverified")
    paranoid = None
    try:
        paranoid = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
    except (OSError, ValueError):
        pass
    return WorkerCapabilities(
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        gpu_count=gpu_count,
        gpus=gpus,
        cuda_available=info.get("cuda"),
        torch_version=info.get("version"),
        cuda_version=info.get("runtime"),
        nsys_version=nsys_version,
        perf_event_paranoid=paranoid,
        limitations=limitations,
    )
