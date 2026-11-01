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
    expected = {**dict.fromkeys(FAMILIES, 4), "mixed": 8}
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
