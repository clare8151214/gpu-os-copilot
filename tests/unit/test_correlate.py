from gpu_os_copilot.analysis.correlate import correlate


def test_process_scoped_links(load_fixture):
    trace = load_fixture()
    assert len(correlate(trace)) == 2
    trace.events[0].pid = 999
    assert len(correlate(trace)) == 1
    trace.events[3].pid = None
    assert correlate(trace) == []


def test_duplicate_ids_cannot_create_ambiguous_correlation(load_fixture):
    trace = load_fixture()
    trace.events[3].correlation_id = 1
    assert correlate(trace) == []
