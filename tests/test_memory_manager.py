import numpy as np
from insightspike.layer2_memory_manager import Memory, Episode
from unittest.mock import patch


class _DummyIndex:
    def __init__(self, k=5):
        self.d = 1
        self.k = k

    def search(self, q, top_k):
        D = np.zeros((1, top_k), dtype=np.float32)
        I = np.arange(top_k).reshape(1, -1)  # 0,1,2,3,4...
        return D, I


def _make_memory():
    mem = Memory.__new__(Memory)
    mem.index = _DummyIndex()
    mem.dim = 1
    # ファイルから10文を読み込む
    with open("data/raw/test_sentences.txt") as f:
        docs = [line.strip() for line in f if line.strip()]
    mem.episodes = [Episode(np.zeros(1), doc, 0.5) for doc in docs]
    return mem


def test_search_skips_negative_indices():
    mem = _make_memory()
    results = mem.search(np.zeros((1, 1)), top_k=5)
    assert all(i >= 0 for _, i in results)
    assert len(results) == 5


def test_merge_trains_once():
    mem = _make_memory()
    calls = 0

    def train():
        nonlocal calls
        calls += 1

    mem.train_index = train
    mem.merge([0, 1])
    assert calls == 1
    assert len(mem.episodes) == len([line for line in open("data/raw/test_sentences.txt") if line.strip()]) + 1


def test_split_trains_twice():
    mem = _make_memory()
    calls = 0

    def train():
        nonlocal calls
        calls += 1

    mem.train_index = train
    # KMeansのpatch先は実際に使われている場所に合わせて修正してください
    with patch("insightspike.graph_metrics.KMeans"):
        mem.split(0)
    assert calls == 2
    assert len(mem.episodes) == len([line for line in open("data/raw/test_sentences.txt") if line.strip()]) + 2
