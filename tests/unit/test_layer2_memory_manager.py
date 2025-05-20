import sys, types, importlib
import numpy as np

class DummyIndex:
    d = 1
    ntotal = 0
    def search(self, q, k):
        return np.zeros((1,k), dtype=np.float32), -np.ones((1,k), dtype=int)
    def add(self, vecs):
        self.ntotal += len(vecs)
    def train(self, vecs):
        pass

def index_factory(dim, spec):
    return DummyIndex()

def IndexFlatL2(dim):
    return DummyIndex()

class IndexIVFPQ(DummyIndex):
    def __init__(self, *a, **k):
        pass

faiss_stub = types.SimpleNamespace(index_factory=index_factory, IndexFlatL2=IndexFlatL2, IndexIVFPQ=IndexIVFPQ, write_index=lambda i,p: None, read_index=lambda p: DummyIndex())
sys.modules['faiss'] = faiss_stub

mm = importlib.import_module('insightspike.layer2_memory_manager')


def test_memory_init():
    mem = mm.Memory(1)
    assert mem.dim == 1
