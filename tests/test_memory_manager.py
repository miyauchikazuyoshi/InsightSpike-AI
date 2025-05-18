import types
import numpy as np
from insightspike.layer2_memory_manager import Memory, Episode
from unittest.mock import patch


class _DummyIndex:
    def __init__(self, k=5):
        self.d = 1
        self.k = k

    def search(self, q, top_k):
        D = np.zeros((1, top_k), dtype=np.float32)
        I = np.full((1, top_k), -1, dtype=np.int64)
        I[0, 0] = 0
        return D, I


def _make_memory():
    mem = Memory.__new__(Memory)
    mem.index = _DummyIndex()
    mem.dim = 1
    mem.episodes = [Episode(np.zeros(1), "doc", 0.5), Episode(np.zeros(1), "doc2", 0.5)]
    return mem


def test_search_skips_negative_indices():
    mem = _make_memory()
    results = mem.search(np.zeros((1, 1)), top_k=5)
    assert all(i >= 0 for _, i in results)
    assert len(results) == 1


def test_merge_trains_once():
    mem = _make_memory()
    calls = 0

    def train():
        nonlocal calls
        calls += 1

    mem.train_index = train
    mem.merge([0, 1])
    assert calls == 1
    # episodesが増えていることも確認
    assert len(mem.episodes) == 3


def test_split_trains_twice():
    mem = _make_memory()
    calls = 0

    def train():
        nonlocal calls
        calls += 1

    mem.train_index = train
    with patch("insightspike.layer2_memory_manager.KMeans"):
        mem.split(0)
    assert calls == 2
    # episodesが増えていることも確認
    assert len(mem.episodes) == 4
