import sys, types, importlib

class DummyModel:
    def __init__(self, name, device=None):
        self.name = name

module = types.SimpleNamespace(SentenceTransformer=DummyModel)
sys.modules['sentence_transformers'] = module

from insightspike import embedder


def test_get_model_singleton():
    m1 = embedder.get_model()
    m2 = embedder.get_model()
    assert isinstance(m1, DummyModel) and m1 is m2
