import sqlite3

import pytest

from gpu_os_copilot.analysis.ablation_labels import ground_truth, ground_truth_mixed
from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.collectors.nsys import import_sqlite
from gpu_os_copilot.experiments import ablation
from gpu_os_copilot.schemas import CaptureRequest, Provenance
from gpu_os_copilot.workloads.patterns import FAMILIES, PATTERN_VARIANTS
from gpu_os_copilot.workloads.specs import SCENARIO_SHAPES, expected_configuration


def test_fixed_table_has_four_variants_per_family_and_is_allowlisted():
    families = [family for family, _ in PATTERN_VARIANTS.values()]
    expected = {**dict.fromkeys(FAMILIES, 4), "mixed": 650}
    assert {name: families.count(name) for name in FAMILIES} == expected
    for name in PATTERN_VARIANTS:
        assert name in SCENARIO_SHAPES
        CaptureRequest(
            experiment_id="pattern-test",
            provenance=Provenance(source="benchmark", confidence="measured", description="Test"),
            scenario=name,
            cpu_set=[0],
            contender_cpu=0,
        )


def test_only_patterns_have_ablation_twins_and_twins_differ_only_by_ablation():
    with pytest.raises(ValueError):
        expected_configuration("cpu_contention", True)
    base = expected_configuration("overlap_copy_b")
    twin = expected_configuration("overlap_copy_b", True)
    changed = {key for key in base if base[key] != twin[key]}
    assert changed == {"ablated", "ablation"}
    assert twin["pattern_parameters"] == {"shape": [2048, 2048], "copy_mb": 10}


def fake_invoke(gains):
    calls = []

    def invoke(argv, timeout, max_output):
        ablated = "--ablate" in argv
        scenario = argv[argv.index("--scenario") + 1]
        calls.append(ablated)
        batch = (len(calls) - 1) // 10
        wall = 10.0 / (1 + gains[batch]) if ablated else 10.0
        return {
            "wall_time_ms": wall,
            "operation_count": 200,
            "configuration": expected_configuration(scenario, ablated),
        }

    return invoke, calls


def test_ablation_batches_interleave_conditions_and_report_gains(monkeypatch):
    monkeypatch.setattr(ablation.os, "sched_getaffinity", lambda _: {0})
    invoke, calls = fake_invoke([0.5, 0.4, 0.6])
    result = ablation.run_ablation("sync_periodic_b", 0, invoke=invoke)
    assert result["status"] == "ok" and len(result["trials"]) == 30
    assert result["batch_gains"] == pytest.approx([0.5, 0.4, 0.6])
    assert calls[:4] == [False, True, True, False]  # Alternating order within each batch.


def test_ablation_rejects_configuration_drift_and_unknown_scenarios(monkeypatch):
    monkeypatch.setattr(ablation.os, "sched_getaffinity", lambda _: {0})

    def drifted(argv, timeout, max_output):
        return {"wall_time_ms": 1, "operation_count": 200, "configuration": {"ablated": "?"}}

    assert ablation.run_ablation("sync_device_a", 0, invoke=drifted)["status"] == "failed"

    def crashed(argv, timeout, max_output):
        raise ablation.ChildRunError("Benchmark child failed", 1, b"Traceback: CUDA error")

    failed = ablation.run_ablation("sync_device_a", 0, invoke=crashed)
    assert failed["child"] == {
        "exit_code": 1,
        "signal": None,
        "stderr_bytes": 21,
        "stderr_tail": "Traceback: CUDA error",
    }
    with pytest.raises(ValueError):
        ablation.run_ablation("cpu_contention", 0)
    with pytest.raises(ValueError):
        ablation.run_ablation("sync_device_a", 0, batches=9)


@pytest.mark.parametrize(
    "family,gains,busy,status,accepted",
    [
        ("sync_device", [1.2, 1.5, 1.4], 0.3, "verified", ["synchronization"]),
        ("sync_periodic", [0.2, 0.25, 0.3], 0.75, "verified", ["gpu_compute", "synchronization"]),
        ("launch_burst", [0.4, 0.5, 0.45], 0.3, "verified", ["launch_overhead"]),
        ("overlap_copy", [0.01, -0.02, 0.0], 0.97, "ruled_out", ["gpu_compute"]),
        ("overlap_copy", [0.01, 0.02, 0.0], 0.6, "excluded", None),
        ("serial_copy", [0.05, 0.2, 0.3], 0.6, "excluded", None),
        ("sync_periodic", [0.2, 0.3], 0.7, "excluded", None),
    ],
)
def test_preregistered_label_rule(family, gains, busy, status, accepted):
    label = ground_truth(family, gains, busy)
    assert label["status"] == status
    assert label.get("accepted_top_categories") == accepted


def test_kernel_busy_and_transfer_overlap_use_one_gpu_clock(tmp_path):
    path = tmp_path / "overlap.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 600, 0, 1), (700, 1000, 0, 1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (500, 800, 0, 2);
            """
        )
    complete = summarize(import_sqlite(path, start_ns=0, end_ns=1000, capture_complete=True))
    assert complete.kernel_busy_fraction == pytest.approx(0.9)
    assert complete.kernel_transfer_overlap_ms == pytest.approx(200 / 1e6)
    partial = summarize(import_sqlite(path, start_ns=0, end_ns=1000))
    assert partial.kernel_busy_fraction is None and partial.kernel_transfer_overlap_ms is None


def test_mixed_draws_have_two_factors_and_only_present_factors_ablate():
    from gpu_os_copilot.workloads.patterns import mixed_factors

    mixed = {k: p for k, (f, p) in PATTERN_VARIANTS.items() if f == "mixed"}
    assert all(len(mixed_factors(p)) >= 2 for p in mixed.values())
    assert expected_configuration("mixed_r6", True, "launch")["ablated_factor"] == "launch"
    for bad in [("mixed_r6", True, "copy"), ("mixed_r6", True, None), ("mixed_r6", False, "sync")]:
        with pytest.raises(ValueError):
            expected_configuration(*bad)
    with pytest.raises(ValueError):
        expected_configuration("sync_device_a", True, "sync")


@pytest.mark.parametrize(
    "gains,busy,status,accepted",
    [
        ({"sync": [1.5, 1.6, 1.4], "copy": [0.2, 0.2, 0.3]}, 0.4, "verified", ["synchronization"]),
        (
            {"sync": [0.5, 0.6, 0.5], "launch": [0.4, 0.45, 0.4]},
            0.3,
            "verified",
            ["launch_overhead", "synchronization"],
        ),
        (
            {"sync": [0.3, 0.3, 0.3], "copy": [0.0, 0.01, 0.0]},
            0.7,
            "verified",
            ["gpu_compute", "synchronization"],
        ),
        (
            {"copy": [0.0, 0.01, 0.0], "launch": [0.02, 0.0, 0.01]},
            0.9,
            "ruled_out",
            ["gpu_compute"],
        ),
        ({"copy": [0.05, 0.2, 0.0], "launch": [0.02, 0.0, 0.01]}, 0.9, "excluded", None),
    ],
)
def test_mixed_label_rule(gains, busy, status, accepted):
    label = ground_truth_mixed(gains, busy)
    assert label["status"] == status
    assert label.get("accepted_top_categories") == accepted


REMOVAL_SCHEMA = """
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER,
        globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER,
        deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER,
        deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER);
    CREATE TABLE StringIds (id INTEGER, value TEXT);
    INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000'),
        (2, 'cudaStreamSynchronize_v3020'), (3, 'cudaMemcpyAsync_v3020');
"""


def test_removal_estimates_partition_the_window_without_cross_clock(tmp_path):
    path = tmp_path / "removal.sqlite"
    process = (1 << 48) | (123 << 24)
    with sqlite3.connect(path) as db:
        db.executescript(REMOVAL_SCHEMA)
        # Host clock: kernel A, readback X then wait, kernel B, upload C, kernels D and E.
        # The GPU clock is offset by +1000 ns; no CPU/GPU timestamp is ever compared.
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (start, start + 10, process | 456, corr, name)
                for start, corr, name in [
                    (0, 1, 1),
                    (20, 2, 3),
                    (40, 3, 2),
                    (210, 4, 1),
                    (230, 5, 3),
                    (250, 6, 1),
                    (270, 7, 1),
                ]
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 7, ?, ?)",
            [(a, b, process, c) for a, b, c in [(1000, 1100, 1), (1400, 1500, 4)]]
            + [(a, b, process, c) for a, b, c in [(1600, 1700, 6), (1800, 1850, 7)]],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 0, 8, ?, ?)",
            [(1150, 1200, process, 2), (1550, 1650, process, 5)],
        )
    summary = summarize(import_sqlite(path, start_ns=0, end_ns=2000, capture_complete=True))
    # Awaited readback: idle before it (50) and its time (50), then 200 idle behind the wait.
    assert summary.removal_estimate_synchronization_fraction == pytest.approx(300 / 2000)
    # Upload C: idle before it (50) plus its time with no kernel running (50).
    assert summary.removal_estimate_data_transfer_fraction == pytest.approx(100 / 2000)
    assert summary.removal_estimate_launch_overhead_fraction == pytest.approx(100 / 2000)
    assert summary.kernel_only_busy_fraction == pytest.approx(300 / 2000)
    # 1000 ns before the first and 150 ns after the last work; overlap is the other 50 ns.
    assert summary.unattributed_idle_fraction == pytest.approx(1150 / 2000)
    # Host gaps (210 ns) plus six 10 ns calls; the synchronize wait costs nothing. No cap binds.
    assert summary.host_submission_fraction == pytest.approx(270 / 2000)

    with sqlite3.connect(path) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId = 99 WHERE start = 1800")
    unlinked = summarize(import_sqlite(path, start_ns=0, end_ns=2000, capture_complete=True))
    assert unlinked.removal_estimate_data_transfer_fraction is None
    assert unlinked.kernel_only_busy_fraction is None
    assert any("removal estimates unavailable" in text for text in unlinked.missing_data)


def test_removal_estimates_are_capped_by_host_submission_time(tmp_path):
    path = tmp_path / "host-paced.sqlite"
    process = (1 << 48) | (123 << 24)
    with sqlite3.connect(path) as db:
        db.executescript(REMOVAL_SCHEMA)
        # Host clock: one upload, then four launches 400 ns apart (390 ns of host work before
        # each). The host needs 1610 ns of the 2000 ns window.
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (start, start + 10, process | 456, corr, name)
                for start, corr, name in [
                    (0, 1, 3),
                    (400, 2, 1),
                    (800, 3, 1),
                    (1200, 4, 1),
                    (1600, 5, 1),
                ]
            ],
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (100, 900, 0, 8, ?, 1)", (process,)
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 7, ?, ?)",
            [(a, a + 50, process, c) for a, c in [(920, 2), (1100, 3), (1300, 4), (1500, 5)]],
        )
    summary = summarize(import_sqlite(path, start_ns=0, end_ns=2000, capture_complete=True))
    # The device timeline credits 800 ns to the upload, but without it the host still needs
    # 1600 ns, so at most 400 ns can be recovered; launch gaps (20 + 130 + 150 + 150) stay below
    # their cap (one launch per run: 410 ns of host work). The ranking flips to launch.
    assert summary.removal_estimate_data_transfer_fraction == pytest.approx(400 / 2000)
    assert summary.removal_estimate_launch_overhead_fraction == pytest.approx(450 / 2000)
    assert summary.removal_estimate_synchronization_fraction == 0
    assert summary.host_submission_fraction == pytest.approx(1610 / 2000)
    assert summary.kernel_only_busy_fraction == pytest.approx(200 / 2000)
    # Device-timeline edges only: 100 ns before the upload and 450 ns after the last kernel.
    assert summary.unattributed_idle_fraction == pytest.approx(550 / 2000)
