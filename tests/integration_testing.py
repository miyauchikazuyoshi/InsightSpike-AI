import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.modules.setdefault("rich", types.SimpleNamespace(print=lambda *a, **k: None))
np_stub = types.SimpleNamespace(
    array=lambda x, dtype=float: x,
    vstack=lambda xs: xs,
    ndarray=list,
)
sys.modules.setdefault("numpy", np_stub)
class _GraphStub:
    def __init__(self):
        self.edges = []
    def add_nodes_from(self, it):
        pass
    def add_edges_from(self, it):
        self.edges.extend(it)

sys.modules.setdefault("networkx", types.SimpleNamespace(Graph=_GraphStub))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=lambda name, device=None: None),
)
sys.modules.setdefault("faiss", types.SimpleNamespace(index_factory=lambda *a, **k: None,
                                                       write_index=lambda *a, **k: None,
                                                       read_index=lambda *a, **k: None))
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("torch_geometric", types.SimpleNamespace())
sys.modules.setdefault("torch_geometric.data", types.SimpleNamespace(Data=lambda **k: None))
sys.modules.setdefault("sklearn", types.SimpleNamespace())
sys.modules.setdefault(
    "sklearn.metrics",
    types.SimpleNamespace(
        pairwise=types.SimpleNamespace(cosine_similarity=lambda a: [[1.0]]),
        silhouette_score=lambda a, b: 0.0,
    ),
)
sys.modules.setdefault("sklearn.metrics.pairwise", types.SimpleNamespace(cosine_similarity=lambda a: [[1.0]]))
sys.modules.setdefault("sklearn.cluster", types.SimpleNamespace(KMeans=lambda k: types.SimpleNamespace(fit=lambda x: None, labels_=[0])))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        pipeline=lambda *a, **k: lambda prompt, do_sample=False: [{"generated_text": ""}],
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: None),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: None),
    ),
)

# Stubs to avoid heavy dependencies ---------------------------
class _DummyModel:
    def encode(self, texts, normalize_embeddings=True):
        # Return 1D vector per text for simplicity
        return [[float(len(t))] for t in texts]

class _DummyGraph:
    def __init__(self):
        self.num_nodes = 1
        class _Arr(list):
            @property
            def T(self):
                return self
            def tolist(self):
                return self

        self.edge_index = types.SimpleNamespace(numpy=lambda: _Arr([[0, 0]]))

class _FakeEpisode:
    def __init__(self, vec, text, c=0.5):
        class _Vec(list):
            def any(self):
                return any(self)

        self.vec = _Vec(vec)
        self.text = text
        self.c = c

class _FakeMemory:
    def __init__(self):
        self.episodes = [_FakeEpisode([1.0], "doc1")]

    def search(self, q_vec, top_k):
        return [(1.0, 0)]

    def update_c(self, idxs, reward, eta=0.1):
        pass

    def train_index(self):
        pass

    def merge(self, idxs):
        pass

    def split(self, idx):
        pass

    def prune(self, c, n):
        pass

    def add_episode(self, vec, text, c_init=0.2):
        self.episodes.append(_FakeEpisode(vec, text, c_init))

    def save(self):
        from pathlib import Path
        path = Path("index.json")
        path.write_text("{}")
        return path

# -------------------------------------------------------------

class IntegrationTest(unittest.TestCase):
    def test_cycle_adds_episode(self):
        from insightspike import agent_loop

        mem = _FakeMemory()
        with patch("insightspike.embedder.get_model", return_value=_DummyModel()), \
             patch("insightspike.agent_loop.get_model", return_value=_DummyModel()), \
             patch("insightspike.layer3_graph_pyg.build_graph", return_value=(_DummyGraph(), None)), \
             patch("insightspike.agent_loop.build_graph", return_value=(_DummyGraph(), None)), \
             patch("insightspike.graph_metrics.delta_ged", return_value=0.0), \
             patch("insightspike.agent_loop.delta_ged", return_value=0.0), \
             patch("insightspike.graph_metrics.delta_ig", return_value=0.0), \
             patch("insightspike.agent_loop.delta_ig", return_value=0.0), \
             patch("insightspike.layer1_error_monitor.uncertainty", return_value=0.0), \
             patch("insightspike.agent_loop.uncertainty", return_value=0.0):
                        with patch("insightspike.layer4_llm.generate", return_value="answer"):
                            g_new = agent_loop.cycle(mem, "question")

        self.assertEqual(len(mem.episodes), 2)
        self.assertTrue(hasattr(g_new, "edges") or hasattr(g_new, "edge_index"))

if __name__ == "__main__":
    unittest.main()
