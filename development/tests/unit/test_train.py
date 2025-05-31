from insightspike.train import train


def test_train_noop():
    assert train() is None
