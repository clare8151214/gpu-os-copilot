"""Read-only SQLite adapter for recognized CUPTI tables; unsupported data stays explicit."""

import sqlite3
import sys
from pathlib import Path

from gpu_os_copilot.schemas import ObservationWindow, Provenance, TraceBundle, TraceEvent
from gpu_os_copilot.workloads.specs import SCENARIO_SHAPES

TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "gpu_kernel",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL": "gpu_kernel",
    "CUPTI_ACTIVITY_KIND_RUNTIME": "cuda_api",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
}

# These records describe host APIs, annotations or accounting, not intervals of
# device execution. Unknown CUPTI types must not silently pass the idle metric gate.
NON_EXECUTION_TABLES = {
    "CUPTI_ACTIVITY_KIND_DRIVER",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
    "CUPTI_ACTIVITY_KIND_OVERHEAD",
    "CUPTI_ACTIVITY_KIND_MEMORY",
    "CUPTI_ACTIVITY_KIND_MEMORY2",
    "CUPTI_ACTIVITY_KIND_MEMORY_POOL",
    "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION",
}


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def discover_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1001"
    ).fetchall()
    if len(tables) > 1000:
        raise ValueError("SQLite schema exceeds table inspection budget")
    return {
        name: [row[1] for row in connection.execute(f"PRAGMA table_info({_quote(name)})")]
        for (name,) in tables
    }


def import_sqlite(
    path: Path,
    *,
    start_ns: int,
    end_ns: int,
    device_id: int = 0,
    capture_complete: bool = False,
    simulated: bool = False,
    max_events: int = 100000,
) -> TraceBundle:
    """Window and capture completeness are explicit owner declarations, not guessed bounds."""
    if not 1 <= max_events <= 100000:
        raise ValueError("max_events outside parser budget")
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("SQLite input missing or exceeds 256 MiB")
    window = ObservationWindow(start_ns=start_ns, end_ns=end_ns, device_id=device_id)
    events, limitations = (
        [],
        [
            "Only captured workload activity is represented; other processes may use the GPU.",
            "CPU scheduling normalization/alignment is not implemented for Nsight exports.",
        ],
    )
    confidence = "simulated" if simulated else "measured"
    incomplete = False
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        # Cap SQL VM work in addition to file size and returned row count.
        work = 0

        def budget():
            nonlocal work
            work += 1
            return int(work > 10000)

        connection.set_progress_handler(budget, 1000)
        schema = discover_schema(connection)
        unsupported = []
        for table, columns in schema.items():
            if table in TABLES or table in NON_EXECUTION_TABLES:
                continue
            if table.startswith("CUPTI_ACTIVITY_KIND_") or {"start", "end", "deviceId"}.issubset(
                columns
            ):
                if connection.execute(f"SELECT 1 FROM {_quote(table)} LIMIT 1").fetchone():
                    unsupported.append(table)
                    incomplete = True
                    limitations.append(f"Unsupported activity table {table}; coverage incomplete")
        for table, kind in TABLES.items():
            columns = schema.get(table)
            if columns is None:
                continue
            required = {"start", "end"}
            if kind != "cuda_api":
                required.add("deviceId")
            if not required.issubset(columns):
                limitations.append(f"Unsupported columns in {table}; skipped")
                incomplete = True
                continue
            # Inspect both kernel tables: the alternate table may contain extra activity.
            # Only identical normalized rows in the alternate representation are deduplicated.
            primary_kernels = (
                {
                    (
                        e.start_ns,
                        e.end_ns,
                        e.device_id,
                        e.stream_id,
                        e.pid,
                        e.correlation_id,
                        e.attributes.get("vm_id"),
                        e.attributes.get("context_id"),
                    )
                    for e in events
                    if e.kind == "gpu_kernel"
                }
                if table == "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
                else set()
            )
            connection.row_factory = sqlite3.Row
            selected = [
                c
                for c in (
                    "start",
                    "end",
                    "deviceId",
                    "streamId",
                    "correlationId",
                    "processId",
                    "threadId",
                    "globalPid",
                    "globalTid",
                    "contextId",
                )
                if c in columns
            ]
            fields = ", ".join(_quote(c) for c in selected)
            row_budget = max_events - len(events)
            rows = connection.execute(
                f"SELECT {fields} FROM {_quote(table)} ORDER BY start, end LIMIT ?",
                (row_budget + 1,),
            ).fetchall()
            if len(rows) > row_budget:
                incomplete = True
                limitations.append(f"Row inspection budget reached in {table}; capture truncated")
            for number, row in enumerate(rows[:row_budget]):
                if len(events) >= max_events:
                    incomplete = True
                    limitations.append("Event budget reached; capture truncated")
                    break
                data = dict(row)
                try:
                    pid, tid = data.get("processId"), data.get("threadId")
                    global_id = data.get("globalTid")
                    if global_id is None:
                        global_id = data.get("globalPid")
                    attributes = {}
                    if data.get("contextId") is not None:
                        attributes["context_id"] = data["contextId"]
                    if type(global_id) is int and global_id >= 0:
                        # NVIDIA's shipped cuda_kern_exec_trace.py uses this encoding.
                        pid = pid if pid is not None else (global_id >> 24) & 0xFFFFFF
                        if data.get("globalTid") is not None and tid is None:
                            tid = global_id & 0xFFFFFF
                        attributes["vm_id"] = global_id >> 48
                    if (
                        data["start"],
                        data["end"],
                        data.get("deviceId"),
                        data.get("streamId"),
                        pid,
                        data.get("correlationId"),
                        attributes.get("vm_id"),
                        attributes.get("context_id"),
                    ) in primary_kernels:
                        continue
                    events.append(
                        TraceEvent(
                            id=f"{table}-{number}",
                            source="nsys_sqlite",
                            kind=kind,
                            start_ns=data["start"],
                            end_ns=data["end"],
                            device_id=data.get("deviceId"),
                            stream_id=data.get("streamId"),
                            correlation_id=data.get("correlationId"),
                            pid=pid,
                            tid=tid,
                            attributes=attributes,
                            confidence=confidence,
                        )
                    )
                except ValueError:
                    incomplete = True
                    limitations.append(f"Invalid row in {table}; skipped")
            if len(events) >= max_events:
                incomplete = True
                limitations.append("Event budget reached; remaining tables not inspected")
                break
    if not events:
        incomplete = True
        limitations.append("No supported CUPTI events found")
    if not capture_complete:
        limitations.append("Capture completeness has not been confirmed by the owner")
    if any(e.pid is None for e in events):
        limitations.append("Missing process ID mapping; ambiguous correlation IDs remain unlinked")
    return TraceBundle(
        experiment_id="sqlite-import",
        run_id="sqlite-import",
        workload_id="imported-workload",
        provenance=Provenance(
            source="nsys_sqlite",
            confidence=confidence,
            description="Synthetic SQLite test data"
            if simulated
            else "Imported Nsight SQLite data",
            limitations=list(dict.fromkeys(limitations))[:50],
        ),
        timebase="nsys_report_ns",
        window=window,
        clock_aligned=False,
        trace_completeness="complete" if capture_complete and not incomplete else "partial",
        events=sorted(events, key=lambda e: (e.start_ns, e.end_ns, e.id)),
        configuration={
            "adapter": "cupti_v2",
            "tables": sorted(schema),
            "unsupported_activity_tables": unsupported,
        },
    )


def build_commands(
    scenario: str,
    output: Path,
    cpu_context_switches: bool = False,
    cuda_trace_mode: str = "cuda-sw",
) -> dict:
    if scenario not in SCENARIO_SHAPES:
        raise ValueError("Workload scenario is not allowlisted")
    if output.name.startswith("-") or output.suffix or "%" in str(output):
        raise ValueError("Output must be a report prefix without a suffix")
    if cuda_trace_mode not in {"cuda", "cuda-sw"}:
        raise ValueError("Unsupported CUDA tracing mode")
    prefix = str(output.resolve())
    return {
        "status": "dry_run",
        "profile_argv": [
            "nsys",
            "profile",
            f"--trace={cuda_trace_mode},nvtx",
            "--sample=none",
            "--cpuctxsw=process-tree" if cpu_context_switches else "--cpuctxsw=none",
            "--force-overwrite=false",
            "--duration=15",
            "-o",
            prefix,
            sys.executable,
            "-m",
            "gpu_os_copilot.workloads.launch_test",
            "--scenario",
            scenario,
            "--profiled",
        ],
        "export_argv": [
            "nsys",
            "export",
            "--type=sqlite",
            "--output",
            prefix + ".sqlite",
            prefix + ".nsys-rep",
        ],
        "limitations": [
            "Commands are a plan only; inspect the installed nsys profile --help before running.",
            "No flags or scheduling permissions have been verified on this machine.",
            "Select the copilot_measure NVTX window when importing; exclude warmup/shutdown.",
        ],
    }
