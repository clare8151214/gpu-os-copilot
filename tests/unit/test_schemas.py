import pytest
from pydantic import ValidationError

from gpu_os_copilot.schemas import BenchmarkTrial, TraceBundle, TraceEvent


@pytest.mark.parametrize(
    "change",
    [
        {"start_ns": -1},
        {"end_ns": -1},
        {"unit": "ms"},
        {"start_ns": 1.5},
        {"start_ns": "1"},
        {"start_ns": True},
        {"confidence": "measured"},
        {"device_id": None},
    ],
)
def test_reject_invalid_events(load_fixture, change):
    data = load_fixture().events[1].model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "2.0"},
        {"experiment_id": "../escape"},
        {"created_at": "2026-10-28T00:00:00"},
        {"wall_time_ms": 0},
        {"wall_time_ms": float("nan")},
        {"operation_count": None},
    ],
)
def test_reject_invalid_bundle(load_fixture, change):
    data = load_fixture().model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        TraceBundle.model_validate(data)


def test_duplicate_ids_and_empty_window(load_fixture):
    data = load_fixture().model_dump()
    data["events"].append(data["events"][0])
    with pytest.raises(ValidationError, match="unique"):
        TraceBundle.model_validate(data)
    data = load_fixture().model_dump()
    data["window"]["end_ns"] = 0
    with pytest.raises(ValidationError, match="positive duration"):
        TraceBundle.model_validate(data)


def test_mixed_clocks_rejected(load_fixture):
    data = load_fixture().model_dump()
    data["events"][0]["source"] = "linux_trace"
    data["clock_aligned"] = False
    with pytest.raises(ValidationError, match="alignment"):
        TraceBundle.model_validate(data)


@pytest.mark.parametrize(
    "start,finish,valid",
    [
        (None, None, True),
        ("2026-10-29T00:00:00Z", "2026-10-29T00:00:01Z", True),
        ("2026-10-29T00:00:00Z", None, False),
        (None, "2026-10-29T00:00:01Z", False),
        ("2026-10-29T00:00:00", "2026-10-29T00:00:01Z", False),
        ("2026-10-29T00:00:02Z", "2026-10-29T00:00:01Z", False),
    ],
)
def test_trial_timestamps_and_legacy_records(start, finish, valid):
    data = {
        "run_id": "trial-test",
        "condition": "baseline",
        "wall_time_ms": 1,
        "operation_count": 1,
        "throughput_ops_s": 1000,
        "cpu_set": [0],
        "configuration": {},
        "started_at": start,
        "finished_at": finish,
    }
    if not valid:
        with pytest.raises(ValidationError, match="Trial"):
            BenchmarkTrial.model_validate(data)
        return
    trial = BenchmarkTrial.model_validate(data)
    restored = BenchmarkTrial.model_validate_json(trial.model_dump_json())
    assert restored == trial
    missing = [item for item in trial.missing_data if "timestamps unavailable" in item]
    assert len(missing) == (1 if start is None else 0)
    if start is None:
        assert restored.started_at is None and restored.finished_at is None
