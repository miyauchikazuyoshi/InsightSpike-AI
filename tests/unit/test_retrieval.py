from insightspike.processing.retrieval import retrieve


def test_retrieve():
    assert retrieve("query") == []
