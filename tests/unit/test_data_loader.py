from insightspike.processing.loader import load_corpus as load_data


def test_load_data():
    assert isinstance(load_data(), list)
