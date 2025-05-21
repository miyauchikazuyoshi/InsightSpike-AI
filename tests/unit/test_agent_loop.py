import sys, types, importlib
import numpy as np

# Patch dependencies
sys.modules['networkx'] = types.SimpleNamespace(Graph=lambda: types.SimpleNamespace(add_nodes_from=lambda x: None, add_edges_from=lambda x: None))
sys.modules['rich'] = types.SimpleNamespace(print=lambda *a, **k: None)

# カスタムPathクラスを追加
class MockPath:
    def __init__(self, path):
        self.path = path
    
    def replace(self, target):
        # 何もせずに成功を装う
        return MockPath(target)
    
    def __str__(self):
        return self.path

embed_mod = types.SimpleNamespace(get_model=lambda: types.SimpleNamespace(encode=lambda x, normalize_embeddings=True: [[0.0]]))
layer1 = types.SimpleNamespace(uncertainty=lambda x: 0.0)
layer2 = types.SimpleNamespace(Memory=types.SimpleNamespace(search=lambda self, q, k: [(1.0,0)], update_c=lambda self, idxs, r, eta=0.1: None, train_index=lambda self: None, merge=lambda self, idxs: None, split=lambda self, idx: None, prune=lambda self,c,i: None, add_episode=lambda self,v,t,c_init=0.2: None, save=lambda self: MockPath('p')))
layer3 = types.SimpleNamespace(
    build_graph=lambda vecs: (
        types.SimpleNamespace(
            num_nodes=0,
            edge_index=types.SimpleNamespace(
                numpy=lambda: np.array([[0, 1], [1, 0]])  # ← np.arrayで返す
            )
        ),
        None
    )
)
metrics = types.SimpleNamespace(delta_ged=lambda g1, g2: 0.0, delta_ig=lambda v1,v2:0.0)
layer4 = types.SimpleNamespace(generate=lambda prompt: 'ans')

sys.modules['insightspike.embedder'] = embed_mod
sys.modules['insightspike.layer1_error_monitor'] = layer1
sys.modules['insightspike.layer2_memory_manager'] = types.SimpleNamespace(Memory=type('M', (), layer2.Memory.__dict__))
sys.modules['insightspike.layer3_graph_pyg'] = layer3
sys.modules['insightspike.graph_metrics'] = metrics
sys.modules['insightspike.layer4_llm'] = layer4

agent_loop = importlib.import_module('insightspike.agent_loop')


# test_cycle 関数の修正
def test_cycle():
    mem = types.SimpleNamespace(
        # 変更点: [(1.0, 0)] ではなく ([1.0], [0]) を返すように修正
        search=lambda q,k: ([1.0], [0]),
        update_c=lambda idxs,r,eta=0.1: None,
        # 他の属性は同じまま
        train_index=lambda: None,
        merge=lambda idxs: None,
        split=lambda idx: None,
        prune=lambda c,i: None,
        add_episode=lambda v,t,c_init=0.2: None,
        save=lambda: MockPath('p'),
        episodes=[types.SimpleNamespace(vec=np.array([0]), text='t')]
    )
    g = agent_loop.cycle(mem, 'q')
    assert hasattr(g, 'add_nodes_from')


# tests/unit/test_agent_loop.py
import pytest
from insightspike.agent_loop import cycle
from insightspike.layer2_memory_manager import Memory
import numpy as np

# test_cycle_with_empty_memory 関数の修正
def test_cycle_with_empty_memory():
    """空のメモリでも機能することを確認"""
    class MockMemory:
        def __init__(self):
            self.episodes = []
        def search(self, vec, k):
            return [], []
        # 変更点: 必要なメソッドを追加
        def update_c(self, idxs, r, eta=0.1):
            pass
        def train_index(self):
            pass

    mem = MockMemory()
    result = cycle(mem, "What is quantum physics?")
    # クラッシュせずに何らかの結果を返すことを確認
    assert result is not None

# test_cycle_with_single_document 関数の修正
def test_cycle_with_single_document():
    """1つのドキュメントでも機能することを確認"""
    class MockMemory:
        def __init__(self):
            self.episodes = [type('obj', (object,), {'vec': np.random.random(384)})]
        def search(self, vec, k):
            return [0.8], [0]
        # 変更点: 必要なメソッドを追加
        def update_c(self, idxs, r, eta=0.1):
            pass
        def train_index(self):
            pass

    mem = MockMemory()
    result = cycle(mem, "What is a single document?")
    assert result is not None
