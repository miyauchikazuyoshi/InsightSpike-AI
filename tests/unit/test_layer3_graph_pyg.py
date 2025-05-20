import sys, types, importlib
import numpy as np

class DummyTensor:
    def __init__(self, data, dtype=None):
        self.data = np.array(data)
    def numpy(self):
        return self.data

class DummyTorch(types.SimpleNamespace):
    float32 = "float32"  # 追加
    def tensor(self, data, dtype=None):
        return DummyTensor(data, dtype)
    def save(self, obj, path):
        pass

class DummyNoGrad:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return False

class DummyModule:
    def eval(self):
        return self

torch_mod = DummyTorch()
sys.modules['torch'] = types.SimpleNamespace(
    tensor=torch_mod.tensor,      # ← ここを追加
    float32=torch_mod.float32,    # ← ここを追加
    nn=types.SimpleNamespace(Module=DummyModule),
    no_grad=DummyNoGrad
)
sys.modules['torch_geometric.data'] = types.SimpleNamespace(Data=lambda x, edge_index: types.SimpleNamespace(x=x, edge_index=edge_index, num_node_features=len(x.data)))
sys.modules['sklearn.metrics.pairwise'] = types.SimpleNamespace(
    cosine_similarity=lambda x, y=None: np.ones((x.shape[0], x.shape[0]))
)

class DummyKMeans:
    def __init__(self, k):
        self.k = k
        self.labels_ = None
    def fit(self, x):
        self.labels_ = [0] * len(x)
        return self

sys.modules['sklearn.cluster'] = types.SimpleNamespace(KMeans=DummyKMeans)
sys.modules['sklearn.metrics'] = types.SimpleNamespace(silhouette_score=lambda x, labels=None: 0.5)

lgp = importlib.import_module('insightspike.layer3_graph_pyg')


def test_build_graph(tmp_path):
    vecs = np.zeros((2,2))
    data, edge_index = lgp.build_graph(vecs, dest=tmp_path/'g.pt')
    assert hasattr(data, 'edge_index')
