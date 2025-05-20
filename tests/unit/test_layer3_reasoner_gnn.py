import sys, types, importlib
import numpy as np

class DummyModule:
    def eval(self):
        return self

# Stub dependencies
sys.modules['faiss'] = types.SimpleNamespace(IndexFlatIP=lambda d: types.SimpleNamespace(add=lambda x: None, search=lambda q,k:(np.zeros((1,k)), np.zeros((1,k), dtype=int))))
sys.modules['torch'] = types.SimpleNamespace(
    tensor=torch_mod.tensor,
    float32=torch_mod.float32,
    nn=types.SimpleNamespace(Module=DummyModule),
    no_grad=DummyNoGrad  
)
sys.modules['torch_geometric.nn'] = types.SimpleNamespace(SAGEConv=lambda d1,d2: types.SimpleNamespace())
sys.modules['torch_geometric.data'] = types.SimpleNamespace(Data=object)

lgp_mod = types.SimpleNamespace(load_graph=lambda: types.SimpleNamespace(num_node_features=2, x=np.zeros((2,2)), edge_index=None))
sys.modules['insightspike.layer3_graph_pyg'] = lgp_mod
sys.modules['insightspike.embedder'] = types.SimpleNamespace(get_model=lambda: types.SimpleNamespace(encode=lambda x, normalize_embeddings=True: np.zeros((1,2))))
sys.modules['insightspike.loader'] = types.SimpleNamespace(load_corpus=lambda: ['a','b'])

l3 = importlib.import_module('insightspike.layer3_reasoner_gnn')


def test_retrieve_gnn():
    ids, scores, corpus = l3.retrieve_gnn('q')
    assert isinstance(ids, list) and isinstance(corpus, list)
