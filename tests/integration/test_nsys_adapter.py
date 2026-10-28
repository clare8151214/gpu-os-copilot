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
