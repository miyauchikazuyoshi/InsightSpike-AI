from insightspike.training.quantizer import quantize


def test_quantize_returns_input():
    obj = object()
    assert quantize(obj) is obj
