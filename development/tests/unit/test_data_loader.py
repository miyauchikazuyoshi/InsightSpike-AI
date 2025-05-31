from insightspike.data_loader import load_data


def test_load_data():
    assert isinstance(load_data(), list)
