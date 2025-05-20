import sys, types, importlib

torch_mod = types.SimpleNamespace(save=lambda *a, **k: None, load=lambda *a, **k: 'graph')
sys.modules['torch'] = torch_mod
cm = importlib.import_module('insightspike.cache_manager')

class DummyMem:
    def save(self, path):
        DummyMem.saved = path


def test_save_cache(tmp_path):
    cm.GRAPH_PATH = tmp_path / 'g.pt'
    cm.MEMORY_JSON = tmp_path / 'm.json'
    cm.save_cache('graph', DummyMem())
    assert DummyMem.saved == cm.MEMORY_JSON
