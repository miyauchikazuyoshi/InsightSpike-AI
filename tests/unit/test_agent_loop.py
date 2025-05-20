import sys, types, importlib

# Patch dependencies
sys.modules['networkx'] = types.SimpleNamespace(Graph=lambda: types.SimpleNamespace(add_nodes_from=lambda x: None, add_edges_from=lambda x: None))
sys.modules['rich'] = types.SimpleNamespace(print=lambda *a, **k: None)

embed_mod = types.SimpleNamespace(get_model=lambda: types.SimpleNamespace(encode=lambda x, normalize_embeddings=True: [[0.0]]))
layer1 = types.SimpleNamespace(uncertainty=lambda x: 0.0)
layer2 = types.SimpleNamespace(Memory=types.SimpleNamespace(search=lambda self, q, k: [(1.0,0)], update_c=lambda self, idxs, r, eta=0.1: None, train_index=lambda self: None, merge=lambda self, idxs: None, split=lambda self, idx: None, prune=lambda self,c,i: None, add_episode=lambda self,v,t,c_init=0.2: None, save=lambda self: importlib.import_module('pathlib').Path('p'))))
layer3 = types.SimpleNamespace(build_graph=lambda vecs: (types.SimpleNamespace(num_nodes=0, edge_index=types.SimpleNamespace(numpy=lambda: [[0],[0]])), None))
metrics = types.SimpleNamespace(delta_ged=lambda g1, g2: 0.0, delta_ig=lambda v1,v2:0.0)
layer4 = types.SimpleNamespace(generate=lambda prompt: 'ans')

sys.modules['insightspike.embedder'] = embed_mod
sys.modules['insightspike.layer1_error_monitor'] = layer1
sys.modules['insightspike.layer2_memory_manager'] = types.SimpleNamespace(Memory=type('M', (), layer2.Memory.__dict__))
sys.modules['insightspike.layer3_graph_pyg'] = layer3
sys.modules['insightspike.graph_metrics'] = metrics
sys.modules['insightspike.layer4_llm'] = layer4

agent_loop = importlib.import_module('insightspike.agent_loop')


def test_cycle():
    mem = types.SimpleNamespace(search=lambda q,k: [(1.0,0)], update_c=lambda idxs,r,eta=0.1: None, train_index=lambda: None, merge=lambda idxs: None, split=lambda idx: None, prune=lambda c,i: None, add_episode=lambda v,t,c_init=0.2: None, save=lambda: importlib.import_module('pathlib').Path('p'), episodes=[types.SimpleNamespace(vec=[0], text='t')])
    g = agent_loop.cycle(mem, 'q')
    assert hasattr(g, 'add_nodes_from')
