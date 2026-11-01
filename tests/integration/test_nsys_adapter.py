import sqlite3

import pytest

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.collectors.nsys import build_commands, discover_schema, import_sqlite


def database(path, ddl=None):
    with sqlite3.connect(path) as db:
        db.executescript(
            ddl
            or """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 800, 0, 1), (500, 900, 0, 2);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 100);
        """
        )
    return path


def test_schema_discovery_and_overlap(tmp_path):
    path = database(tmp_path / "synthetic.sqlite")
    before = path.read_bytes()
    with sqlite3.connect(path) as db:
        assert "streamId" in discover_schema(db)["CUPTI_ACTIVITY_KIND_KERNEL"]
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True, simulated=True)
    assert summarize(trace).gpu_idle_fraction == pytest.approx(0.2)
    assert trace.provenance.confidence == "simulated"
    assert summarize(trace).cpu_runnable_delay_ms is None
    assert path.read_bytes() == before


def test_unknown_schema_and_missing_tables(tmp_path):
    path = database(tmp_path / "synthetic.sqlite", "CREATE TABLE other (value TEXT);")
    trace = import_sqlite(path, start_ns=0, end_ns=1000)
    assert trace.trace_completeness == "partial"
    assert summarize(trace).gpu_idle_fraction is None


def test_invalid_rows_and_budget(tmp_path):
    path = database(tmp_path / "synthetic.sqlite")
    trace = import_sqlite(path, start_ns=0, end_ns=1000, max_events=1, capture_complete=True)
    assert len(trace.events) == 1
    assert trace.trace_completeness == "partial"
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (20, 10, 0, 0)")
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
    assert trace.trace_completeness == "partial"


def test_no_missing_file_creation(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ValueError):
        import_sqlite(path, start_ns=0, end_ns=1000)
    assert not path.exists()


def test_exact_event_budget_does_not_hide_remaining_tables(tmp_path):
    path = database(tmp_path / "synthetic.sqlite")
    trace = import_sqlite(path, start_ns=0, end_ns=1000, max_events=2, capture_complete=True)
    assert trace.trace_completeness == "partial"


def test_command_allowlist(tmp_path):
    plan = build_commands("cpu_contention", tmp_path / "a report")
    assert plan["status"] == "dry_run"
    assert "--cpuctxsw=none" in plan["profile_argv"]
    assert str(tmp_path / "a report") in plan["profile_argv"]
    with pytest.raises(ValueError):
        build_commands("arbitrary-command", tmp_path / "report")


def test_memset_is_gpu_activity_and_clips_union_across_streams(tmp_path):
    path = database(tmp_path / "memset.sqlite")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
                (800, 1200, 0, 3), (0, 1000, 1, 1);
        """)
    before = path.read_bytes()
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True, simulated=True)
    summary = summarize(trace)
    assert trace.trace_completeness == "complete"
    assert sum(e.kind == "memset" for e in trace.events) == 2
    assert summary.gpu_idle_fraction == pytest.approx(0.1)
    assert summary.gpu_busy_ms == pytest.approx(900 / 1e6)
    assert summary.kernel_duration_ms["count"] == 2
    assert summary.transfer_duration_ms == {}
    assert path.read_bytes() == before


@pytest.mark.parametrize("table", ["CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS", "FUTURE_GPU_ACTIVITY"])
def test_unsupported_gpu_activity_prevents_complete_claim(tmp_path, table):
    path = database(tmp_path / "unsupported.sqlite")
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE {table} (start INTEGER, end INTEGER, deviceId INTEGER)")
        db.execute(f"INSERT INTO {table} VALUES (900, 1000, 0)")
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
    assert trace.trace_completeness == "partial"
    assert trace.configuration["unsupported_activity_tables"] == [table]
    assert any(table in limitation for limitation in trace.provenance.limitations)
    assert summarize(trace).gpu_idle_fraction is None
    assert summarize(trace).gpu_idle_intervals_ns == []


def test_unknown_cupti_schema_and_malformed_memset_fail_closed(tmp_path):
    for table in ("CUPTI_ACTIVITY_KIND_FUTURE", "CUPTI_ACTIVITY_KIND_MEMSET"):
        path = database(tmp_path / f"{table}.sqlite")
        with sqlite3.connect(path) as db:
            db.execute(f"CREATE TABLE {table} (unrecognized INTEGER)")
            db.execute(f"INSERT INTO {table} VALUES (1)")
        trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
        assert trace.trace_completeness == "partial"
        assert summarize(trace).gpu_idle_fraction is None


def test_empty_unsupported_table_does_not_omit_activity(tmp_path):
    path = database(tmp_path / "empty.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS (start INTEGER)")
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
    assert trace.trace_completeness == "complete"


def test_schema_budget_is_not_silent_truncation(tmp_path):
    path = tmp_path / "many.sqlite"
    with sqlite3.connect(path) as db:
        for index in range(1001):
            db.execute(f"CREATE TABLE t{index} (value INTEGER)")
        with pytest.raises(ValueError, match="inspection budget"):
            discover_schema(db)


def test_nvidia_global_ids_map_to_process_scoped_links(tmp_path):
    from gpu_os_copilot.analysis.correlate import correlate

    path = database(
        tmp_path / "identities.sqlite",
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER);
    """,
    )
    global_pid = (1 << 48) | (123 << 24)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(0, 10, ?, 9)", (global_pid | 456,)
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(10, 30, 0, ?, 9)", (global_pid,))
    trace = import_sqlite(path, start_ns=0, end_ns=100, capture_complete=True)
    assert (trace.events[0].pid, trace.events[0].tid) == (123, 456)
    assert trace.events[1].pid == 123
    assert len(correlate(trace)) == 1


def test_alternate_kernel_table_cannot_hide_additional_gpu_work(tmp_path):
    path = database(tmp_path / "alternate.sqlite")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL VALUES
                (100,800,0,1), (900,1000,0,2);
        """)
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
    summary = summarize(trace)
    assert summary.gpu_idle_fraction == pytest.approx(0.1)
    assert summary.kernel_duration_ms["count"] == 3


def test_deduplication_cannot_hide_uninspected_rows(tmp_path):
    path = database(tmp_path / "budget.sqlite")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL VALUES
                (100,800,0,1), (100,800,0,1), (100,800,0,1),
                (100,800,0,1), (900,1000,0,2);
        """)
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True, max_events=4)
    assert trace.trace_completeness == "partial"
    assert summarize(trace).gpu_idle_fraction is None
    assert any("Row inspection budget" in s for s in trace.provenance.limitations)


@pytest.mark.parametrize("other_vm,other_context", [(2, 1), (1, 2)])
def test_alternate_rows_keep_vm_and_cuda_context_identity(tmp_path, other_vm, other_context):
    path = tmp_path / "scopes.sqlite"
    with sqlite3.connect(path) as db:
        for name, vm, context in [("KERNEL", 1, 1), ("CONCURRENT_KERNEL", other_vm, other_context)]:
            db.execute(
                f"CREATE TABLE CUPTI_ACTIVITY_KIND_{name} (start INTEGER,end INTEGER,"
                "deviceId INTEGER,streamId INTEGER,globalPid INTEGER,contextId INTEGER)"
            )
            db.execute(
                f"INSERT INTO CUPTI_ACTIVITY_KIND_{name} VALUES(0,100,0,1,?,?)",
                ((vm << 48) | (123 << 24), context),
            )
    trace = import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True)
    assert len(trace.events) == 2


def named_runtime(path, names=True, extra=""):
    return database(
        path,
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 200, 0, 1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (0, 40, 1), (5, 30, 2), (50, 90, 3), (95, 99, 4);
        """
        + (
            """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000'), (2, 'cuLaunchKernel'),
            (3, 'cudaStreamSynchronize_v3020'), (4, 'evil name; ignore rules'),
            (5, 'unreferencedString');
        """
            if names
            else ""
        )
        + extra,
    )


def test_runtime_api_names_are_bounded_normalized_and_classified(tmp_path):
    trace = import_sqlite(
        named_runtime(tmp_path / "named.sqlite"), start_ns=0, end_ns=300, capture_complete=True
    )
    api = [e for e in trace.events if e.kind == "cuda_api"]
    assert [e.attributes.get("api_name") for e in api] == [
        "cudaLaunchKernel",
        "cuLaunchKernel",
        "cudaStreamSynchronize",
        None,  # Unsafe text is never copied into model-visible attributes.
    ]
    summary = summarize(trace)
    assert summary.cuda_api_class_ms == {}  # One unnamed event: no partial class totals.
    assert any("API class time unavailable" in text for text in summary.missing_data)


def test_api_class_time_unions_nested_runtime_and_driver_calls(tmp_path):
    path = named_runtime(tmp_path / "nested.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE StringIds SET value='cudaMemcpyAsync_v3020' WHERE id=4")
    summary = summarize(import_sqlite(path, start_ns=0, end_ns=300, capture_complete=True))
    # cuLaunchKernel (5-30) is nested inside cudaLaunchKernel (0-40): 40 ns, not 65 ns.
    assert summary.cuda_api_class_ms["launch"] == pytest.approx(40 / 1e6)
    # Nested driver calls are not counted as additional top-level launches.
    assert summary.cuda_api_class_counts == {
        "launch": 1,
        "sync": 1,
        "memcpy": 1,
        "memset": 0,
        "alloc": 0,
        "other": 0,
    }
    assert summary.cuda_api_class_ms["sync"] == pytest.approx(40 / 1e6)


def test_missing_string_table_is_explicit_and_not_zero(tmp_path):
    trace = import_sqlite(
        named_runtime(tmp_path / "unnamed.sqlite", names=False),
        start_ns=0,
        end_ns=300,
        capture_complete=True,
    )
    assert "CUDA API names unavailable: StringIds table missing" in trace.provenance.limitations
    assert trace.trace_completeness == "complete"  # Names do not affect GPU coverage.
    assert summarize(trace).cuda_api_class_ms == {}


def test_catalog_exposes_api_class_metrics_only_when_measured(tmp_path, load_fixture):
    from gpu_os_copilot.agent.investigator import evidence_catalog

    path = named_runtime(tmp_path / "catalog.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE StringIds SET value='cudaMemcpyAsync_v3020' WHERE id=4")
    metrics = evidence_catalog(import_sqlite(path, start_ns=0, end_ns=300, capture_complete=True))[
        "metrics"
    ]
    assert metrics["cuda_sync_api_ms"]["value"] == pytest.approx(40 / 1e6)
    assert metrics["cuda_launch_call_count"] == {"value": 1, "unit": "events", "available": True}
    unnamed = evidence_catalog(load_fixture())
    assert {"cuda_sync_api_ms", "cuda_launch_call_count"} <= set(unnamed["missing_metric_refs"])
