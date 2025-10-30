from insightspike.fallback.gedig_fallback_tracker import GeDIGFallbackTracker


def test_fallback_tracker_records_and_summarizes():
    tr = GeDIGFallbackTracker()
    class E1(RuntimeError):
        pass
    tr.record('pure_to_full', E1('x'))
    tr.record('pure_to_full', E1('y'))
    tr.record('full_failed', E1('z'))
    s = tr.summary()
    assert s['total'] == 3
    assert s['kinds']['pure_to_full'] == 2
    assert s['kinds']['full_failed'] == 1