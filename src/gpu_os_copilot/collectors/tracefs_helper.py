"""Bounded root helper: one private tracefs instance, at most 32 owned workload threads."""

import argparse
import json
import os
import re
import signal
import stat
import time
from pathlib import Path
from uuid import uuid4

TRACEFS = Path("/sys/kernel/tracing")
SCHED_DEBUG = Path("/sys/kernel/debug/sched/debug")


def kernel_tid_for_comm(text: str, comm: str) -> int:
    """sched/debug prints task_pid_nr; /proc/PID/sched uses the procfs namespace."""
    if re.fullmatch(r"copilot[0-9a-f]{8}", comm) is None:
        raise ValueError("Workload requires a unique scheduler task name")
    rows = re.findall(rf"^\s*>?[A-Za-z?]\s*{comm}\s+(\d+)\s", text, re.MULTILINE)
    if len(rows) != 1 or int(rows[0]) <= 1:
        raise ValueError("Kernel thread identity missing or ambiguous")
    return int(rows[0])


def resolve_kernel_identity(pid: int, uid: int, birth: str) -> dict:
    proc = Path(f"/proc/{pid}")
    comm = (proc / "comm").read_text().strip()
    with SCHED_DEBUG.open() as stream:
        snapshot = stream.read(8 * 1024 * 1024 + 1)
    if len(snapshot) > 8 * 1024 * 1024:
        raise ValueError("Scheduler identity snapshot exceeds budget")
    kernel_tid = kernel_tid_for_comm(snapshot, comm)
    if process_identity(pid) != (uid, birth) or (proc / "comm").read_text().strip() != comm:
        raise ValueError("Workload identity changed during kernel PID lookup")
    return {
        "method": "unique_task_comm_in_sched_debug",
        "comm": comm,
        "namespace_pid": pid,
        "kernel_tid": kernel_tid,
        "pid_namespace": os.readlink(proc / "ns/pid"),
        "process_start_ticks": birth,
    }


def process_identity(pid: int):
    proc = Path(f"/proc/{pid}")
    return proc.stat().st_uid, (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]


def verify_enrolled_thread(pid: int, uid: int, row: dict, *, proc_root=Path("/proc")):
    tid, comm, birth = row.get("tid"), row.get("comm"), row.get("start_ticks")
    if (
        type(tid) is not int
        or tid <= 1
        or not isinstance(comm, str)
        or re.fullmatch(r"copilot[0-9a-f]{8}", comm) is None
        or not isinstance(birth, str)
        or not birth.isdecimal()
    ):
        raise ValueError("Invalid owned thread enrollment")
    task = proc_root / str(pid) / "task" / str(tid)
    tgid = re.search(r"^Tgid:\s*(\d+)$", (task / "status").read_text(), re.MULTILINE)
    if (
        task.stat().st_uid != uid
        or tgid is None
        or int(tgid[1]) != pid
        or (task / "stat").read_text().rsplit(")", 1)[1].split()[19] != birth
        or (task / "comm").read_text().strip() != comm
    ):
        raise ValueError("Enrolled workload thread identity changed")


def resolve_thread_mappings(pid: int, uid: int, birth: str, enrolled: list) -> list[dict]:
    if not isinstance(enrolled, list) or not 1 <= len(enrolled) <= 32:
        raise ValueError("Expected one to 32 enrolled workload threads")
    if any(not isinstance(row, dict) for row in enrolled):
        raise ValueError("Invalid workload thread registration")
    for row in enrolled:
        verify_enrolled_thread(pid, uid, row)
    tids, names = [row["tid"] for row in enrolled], [row["comm"] for row in enrolled]
    if pid not in tids or len(set(tids)) != len(tids) or len(set(names)) != len(names):
        raise ValueError("Main thread required and enrolled identities must be unique")
    with SCHED_DEBUG.open() as stream:
        snapshot = stream.read(8 * 1024 * 1024 + 1)
    if len(snapshot) > 8 * 1024 * 1024:
        raise ValueError("Scheduler identity snapshot exceeds budget")
    namespace = os.readlink(f"/proc/{pid}/ns/pid")
    mappings = [
        {
            "method": "unique_task_comm_in_sched_debug",
            "namespace_pid": pid,
            "namespace_tid": row["tid"],
            "kernel_tid": kernel_tid_for_comm(snapshot, row["comm"]),
            "comm": row["comm"],
            "thread_start_ticks": row["start_ticks"],
            "pid_namespace": namespace,
        }
        for row in enrolled
    ]
    if len({row["kernel_tid"] for row in mappings}) != len(mappings):
        raise ValueError("Duplicate kernel thread mapping")
    for row in enrolled:
        verify_enrolled_thread(pid, uid, row)
    if process_identity(pid) != (uid, birth):
        raise ValueError("Workload process identity changed")
    return mappings


def thread_filters(kernel_tids: list[int]) -> dict:
    if (
        not 1 <= len(kernel_tids) <= 32
        or len(set(kernel_tids)) != len(kernel_tids)
        or any(type(tid) is not int or tid <= 1 for tid in kernel_tids)
    ):
        raise ValueError("Invalid bounded kernel thread set")
    return {
        "sched_switch": " || ".join(
            f"prev_pid == {tid} || next_pid == {tid}" for tid in sorted(kernel_tids)
        ),
        "sched_wakeup": " || ".join(f"pid == {tid}" for tid in sorted(kernel_tids)),
    }


def read_loss_stats(instance: Path) -> int:
    """Absence of counters is unknown, never zero observed event loss."""
    files = list((instance / "per_cpu").glob("cpu*/stats"))
    if not files:
        raise ValueError("Scheduler loss counters unavailable")
    required = {"overrun", "commit overrun", "dropped events"}
    lost = 0
    for path in files:
        values = {}
        for line in path.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in required:
                values[key.strip()] = int(value.strip())
        if not required.issubset(values) or any(value < 0 for value in values.values()):
            raise ValueError("Scheduler loss counters incomplete or invalid")
        lost += sum(values.values())
    return lost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or args.uid <= 0 or args.pid <= 1:
        raise ValueError("Root helper requires an unprivileged workload owner")
    state = args.state.resolve(strict=True)
    fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    if os.fstat(fd).st_uid != args.uid or os.fstat(fd).st_mode & 0o077:
        raise ValueError("State must be a private directory owned by the workload user")
    uid, birth = process_identity(args.pid)
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0")
    if uid != args.uid or b"gpu_os_copilot.workloads.launch_test" not in command:
        raise ValueError("Target is not an owned bundled workload")
    if b"--scheduler-gate" not in command or str(state).encode() not in command:
        raise ValueError("Workload has not opted into this capture directory")
    ready_fd = os.open("scheduler-ready.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(ready_fd) as ready:
        info = os.fstat(ready.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_size > 16384:
            raise ValueError("Invalid bounded enrollment file")
        registration = json.load(ready)
    if registration.get("pid") != args.pid or registration.get("uid") != uid:
        raise ValueError("Enrollment owner mismatch")
    enrolled = registration.get("threads")
    mappings = resolve_thread_mappings(args.pid, uid, birth, enrolled)
    mapping = next(row for row in mappings if row["namespace_tid"] == args.pid)
    kernel_tid = mapping["kernel_tid"]

    def save(name, text):
        out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            os.fchown(out, args.uid, -1)
            with os.fdopen(os.dup(out), "w") as stream:
                stream.write(text)
        finally:
            os.close(out)

    instance = TRACEFS / "instances" / f"copilot-{args.pid}-{uuid4().hex[:12]}"
    created = False

    def interrupted(*unused):
        raise InterruptedError("Capture interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        instance.mkdir()
        created = True
        (instance / "tracing_on").write_text("0")
        (instance / "trace_clock").write_text("mono_raw")
        if "[mono_raw]" not in (instance / "trace_clock").read_text():
            raise ValueError("Raw monotonic trace clock unavailable")
        (instance / "buffer_size_kb").write_text("512")
        filters = thread_filters([row["kernel_tid"] for row in mappings])
        settings = {}
        for event, expression in filters.items():
            directory = instance / "events/sched" / event
            (directory / "filter").write_text(expression)
            (directory / "enable").write_text("1")
            settings[event] = {
                "filter": (directory / "filter").read_text().strip(),
                "enable": (directory / "enable").read_text().strip(),
            }
            if settings[event] != {"filter": expression, "enable": "1"}:
                raise ValueError("Scheduler event configuration readback differs")
        (instance / "tracing_on").write_text("1")
        capture_start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        save("scheduler.release", "ready\n")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (state / "scheduler.stop").exists():
            try:
                if process_identity(args.pid) != (uid, birth):
                    break
            except FileNotFoundError:
                break
            time.sleep(0.02)
        capture_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        (instance / "tracing_on").write_text("0")
        if not (state / "scheduler.stop").exists():
            raise ValueError("Workload did not complete inside the bounded collection window")
        if resolve_thread_mappings(args.pid, uid, birth, enrolled) != mappings:
            raise ValueError("Enrolled thread mappings changed during collection")
        lost = read_loss_stats(instance)
        with (instance / "trace").open() as stream:
            text = stream.read(4 * 1024 * 1024 + 1)
        if len(text) > 4 * 1024 * 1024:
            raise ValueError("Scheduler output exceeds budget")
        save("scheduler.txt", text)
        save(
            "scheduler-meta.json",
            json.dumps(
                {
                    "pid": args.pid,
                    "tid": args.pid,
                    "kernel_tid": kernel_tid,
                    "pid_mapping": mapping,
                    "thread_mappings": mappings,
                    "thread_lifetimes_verified": True,
                    "enrollment_scope": "owned_threads_after_warmup",
                    "uid": args.uid,
                    "clock": "mono_raw",
                    "capture_start_monotonic_ns": capture_start_ns,
                    "capture_end_monotonic_ns": capture_end_ns,
                    "lost_events": lost,
                    "loss_stats_verified": True,
                    "process_start_ticks": birth,
                    "filters": filters,
                    "event_settings": settings,
                    "buffer_kb_per_cpu": 512,
                    "maximum_duration_s": 20,
                },
                indent=2,
            ),
        )
        save("scheduler.finished", "verified\n")
    finally:
        if created:
            (instance / "tracing_on").write_text("0")
            instance.rmdir()
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
