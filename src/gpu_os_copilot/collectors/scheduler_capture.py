"""Coordinate an unprivileged profiler with an explicitly approved bounded root helper."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def profile_with_scheduler(argv: list[str], log: Path, timeout: float, state: Path):
    from gpu_os_copilot.collectors.capture import _limits

    env = {
        key: os.environ[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        if key in os.environ
    }
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="42")
    helper, helper_log = None, None
    with log.open("x") as output:
        process = subprocess.Popen(
            [*argv, "--scheduler-gate", str(state)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_limits,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Profile deadline exceeded")
                ready = state / "scheduler-ready.json"
                if helper is None and ready.exists():
                    try:
                        identity = json.loads(ready.read_text())
                    except json.JSONDecodeError:
                        time.sleep(0.02)
                        continue
                    if identity.get("uid") != os.getuid() or type(identity.get("pid")) is not int:
                        raise ValueError("Invalid workload scheduler identity")
                    helper_log = (state / "scheduler-helper.log").open("x")
                    helper = subprocess.Popen(
                        [
                            "sudo",
                            "-n",
                            "timeout",
                            "--signal=TERM",
                            "--kill-after=3",
                            "25",
                            sys.executable,
                            "-m",
                            "gpu_os_copilot.collectors.tracefs_helper",
                            "--state",
                            str(state),
                            "--uid",
                            str(os.getuid()),
                            "--pid",
                            str(identity["pid"]),
                        ],
                        stdout=helper_log,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                if helper is not None and helper.poll() not in (None, 0):
                    raise RuntimeError("Scheduler helper failed; inspect retained log")
                time.sleep(0.02)
            if process.returncode:
                raise RuntimeError(f"Profiled workload failed (exit code {process.returncode})")
            if helper is None:
                raise RuntimeError("Scheduler workload gate was never reached")
        finally:
            (state / "scheduler.stop").touch(exist_ok=True)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if helper is not None:
                helper.wait(timeout=30)
            if helper_log is not None:
                helper_log.close()
            (state / "subprocess-status.json").write_text(
                json.dumps(
                    {
                        "profiler_exit_code": process.returncode,
                        "helper_exit_code": helper.returncode if helper is not None else None,
                    },
                    indent=2,
                )
            )
        if helper.returncode:
            raise RuntimeError("Scheduler helper failed")
