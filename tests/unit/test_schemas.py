import pytest
from pydantic import ValidationError

from gpu_os_copilot.schemas import TraceBundle, TraceEvent


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
