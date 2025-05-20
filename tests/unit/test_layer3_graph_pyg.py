import sys, types, importlib
import numpy as np

class DummyTensor:
    def __init__(self, data, dtype=None):
        self.data = np.array(data)
    def numpy(self):
        return self.data

class DummyTorch(types.SimpleNamespace):
    def tensor(self, data, dtype=None):
        return DummyTensor(data, dtype)
    def save(self, obj, path):
        pass

torch_mod = DummyTorch()
sys.modules['torch'] = torch_mod
sys.modules['torch_geometric.data'] = types.SimpleNamespace(Data=lambda x, edge_index: types.SimpleNamespace(x=x, edge_index=edge_index, num_node_features=len(x.data)))
sys.modules['sklearn.metrics.pairwise'] = types.SimpleNamespace(
    cosine_similarity=lambda x, y=None: np.ones((x.shape[0], x.shape[0]))
)

lgp = importlib.import_module('insightspike.layer3_graph_pyg')


def test_build_graph(tmp_path):
    vecs = np.zeros((2,2))
    data, edge_index = lgp.build_graph(vecs, dest=tmp_path/'g.pt')
    assert hasattr(data, 'edge_index')
