from insightspike.layer1_error_monitor import uncertainty


def test_uncertainty():
    val = uncertainty([0.2, 0.8])
    assert val > 0
