from insightspike.training.train import train


def test_train_noop():
    assert train() is None
