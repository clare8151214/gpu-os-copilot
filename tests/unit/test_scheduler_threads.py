import os

import pytest
from test_scheduler import anchors_database, capture_metadata

from gpu_os_copilot.analysis.observability import observability
from gpu_os_copilot.collectors.scheduler import merge_scheduler, verify_enrolled_events
from gpu_os_copilot.collectors.tracefs_helper import thread_filters, verify_enrolled_thread
from gpu_os_copilot.schemas import ObservationWindow

TEXT = """
 t-9 [000] d..2. 0.000000: sched_wakeup: pid=107
 t-9 [000] d..2. 0.000001: sched_switch: prev_pid=9 prev_state=S ==> next_pid=107
 t-107 [000] d..2. 0.000002: sched_wakeup: pid=108
 t-107 [000] d..2. 0.000005: sched_switch: prev_pid=107 prev_state=R ==> next_pid=108
 t-108 [000] d..2. 0.000006: sched_switch: prev_pid=108 prev_state=R ==> next_pid=107
 t-107 [000] d..2. 0.000008: sched_switch: prev_pid=107 prev_state=S ==> next_pid=108
"""


def enrolled_case(load_fixture, tmp_path):
    trace = load_fixture()
    trace.window = ObservationWindow(start_ns=20000, end_ns=30000)
    api = next(e for e in trace.events if e.kind == "cuda_api")
    gpu = next(e for e in trace.events if e.kind == "gpu_kernel")
    trace.events = []
    for tid in (7, 8):
        trace.events += [
            api.model_copy(
                update={
                    "id": f"api-{tid}",
                    "pid": 7,
                    "tid": tid,
                    "start_ns": 20000,
                    "end_ns": 21000,
                    "correlation_id": tid,
                }
            ),
            gpu.model_copy(
                update={
                    "id": f"gpu-{tid}",
                    "pid": 7,
                    "tid": tid,
                    "start_ns": 26000,
                    "end_ns": 27000,
                    "correlation_id": tid,
                }
            ),
        ]
    metadata = capture_metadata()
    metadata.update(
        thread_lifetimes_verified=True,
        thread_mappings=[
            {
                "namespace_pid": 7,
                "namespace_tid": tid,
                "kernel_tid": tid + 100,
                "method": "unique_task_comm_in_sched_debug",
            }
            for tid in (7, 8)
        ],
        filters=thread_filters([107, 108]),
    )
    metadata["event_settings"] = {
        name: {"filter": expression, "enable": "1"}
        for name, expression in metadata["filters"].items()
    }
    db = tmp_path / "enrolled.sqlite"
    anchors = anchors_database(db)
    return trace, db, metadata, anchors


def test_both_api_threads_have_unique_intervals_and_validated_coverage(load_fixture, tmp_path):
    trace, db, metadata, anchors = enrolled_case(load_fixture, tmp_path)
    merged = merge_scheduler(trace, db, TEXT, metadata, anchors)
    events = [e for e in merged.events if e.kind == "cpu_sched"]
    assert {e.tid for e in events} == {7, 8}
    assert len({e.id for e in events}) == len(events)
    coverage = observability(merged)["scheduling"]
    assert coverage["all_observed_api_threads_covered"]
    assert coverage["covered_cuda_api_threads"] == 2


def test_enrolled_but_no_positive_events_remains_uncovered(load_fixture, tmp_path):
    trace, db, metadata, anchors = enrolled_case(load_fixture, tmp_path)
    text = TEXT.replace(" t-107 [000] d..2. 0.000002: sched_wakeup: pid=108\n", "")
    merged = merge_scheduler(trace, db, text, metadata, anchors)
    coverage = observability(merged)["scheduling"]
    assert coverage["covered_cuda_api_threads"] == 1
    assert not coverage["all_observed_api_threads_covered"]
    assert merged.configuration["scheduler_capture"]["uncovered_cuda_api_tids"] == [8]
    assert all(e.tid == 7 for e in merged.events if e.kind == "cpu_sched")


def test_new_api_thread_after_enrollment_is_reported_missing(load_fixture, tmp_path):
    trace, db, metadata, anchors = enrolled_case(load_fixture, tmp_path)
    trace.events.append(trace.events[0].model_copy(update={"id": "late-thread", "tid": 9}))
    merged = merge_scheduler(trace, db, TEXT, metadata, anchors)
    coverage = observability(merged)["scheduling"]
    assert coverage["observed_cuda_api_threads"] == 3
    assert coverage["covered_cuda_api_threads"] == 2
    assert not coverage["all_observed_api_threads_covered"]


@pytest.mark.parametrize("defect", ["lifetime", "duplicate", "foreign_process", "filter", "loss"])
def test_bad_enrollment_proof_rejected(load_fixture, tmp_path, defect):
    trace, db, metadata, anchors = enrolled_case(load_fixture, tmp_path)
    if defect == "lifetime":
        metadata["thread_lifetimes_verified"] = False
    elif defect == "duplicate":
        metadata["thread_mappings"][1] = metadata["thread_mappings"][0]
    elif defect == "foreign_process":
        metadata["thread_mappings"][1]["namespace_pid"] = 99
    elif defect == "filter":
        metadata["event_settings"]["sched_switch"]["filter"] = "0"
    else:
        metadata["lost_events"] = False
    with pytest.raises(ValueError):
        merge_scheduler(trace, db, TEXT, metadata, anchors)


def test_foreign_event_rejected_and_empty_thread_not_marked_complete():
    counts = verify_enrolled_events([(1, "sched_wakeup", {"pid": "107"})], {107, 108})
    assert counts[107]["wakeup"] == 1
    assert not any(counts[108].values())
    with pytest.raises(ValueError, match="outside"):
        verify_enrolled_events([(1, "sched_wakeup", {"pid": "109"})], {107, 108})


@pytest.mark.parametrize("tids", [[], [1], [True], [7, 7], list(range(2, 35))])
def test_filter_scope_cannot_exceed_owned_registration_budget(tids):
    with pytest.raises(ValueError):
        thread_filters(tids)


def test_thread_owner_tgid_birth_and_nonce_are_verified(tmp_path):
    task = tmp_path / "7/task/8"
    task.mkdir(parents=True)
    (task / "status").write_text("Tgid:\t7\n")
    (task / "stat").write_text("8 (test) " + " ".join(["S"] + ["0"] * 18 + ["123"]))
    (task / "comm").write_text("copilot1234abcd")
    row = {"tid": 8, "comm": "copilot1234abcd", "start_ticks": "123"}
    verify_enrolled_thread(7, os.getuid(), row, proc_root=tmp_path)
    for changed in [{**row, "start_ticks": "124"}, {**row, "comm": "copilot00000000"}]:
        with pytest.raises(ValueError, match="changed"):
            verify_enrolled_thread(7, os.getuid(), changed, proc_root=tmp_path)
    (task / "status").write_text("Tgid:\t9\n")
    with pytest.raises(ValueError, match="changed"):
        verify_enrolled_thread(7, os.getuid(), row, proc_root=tmp_path)
