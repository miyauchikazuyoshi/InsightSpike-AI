from insightspike.algorithms.gedig_core import GeDIGResult


def test_spike_alias_consistency():
    r = GeDIGResult(gedig_value=0.0, ged_value=0.0, ig_value=0.0, spike=True)
    assert r.spike is True
    assert r.has_spike is True
    r2 = GeDIGResult(gedig_value=0.0, ged_value=0.0, ig_value=0.0, spike=False)
    assert r2.spike is False
    assert r2.has_spike is False