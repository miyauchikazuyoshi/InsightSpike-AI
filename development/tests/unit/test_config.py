from insightspike import config


def test_timestamp():
    ts = config.timestamp()
    assert isinstance(ts, str) and ts
