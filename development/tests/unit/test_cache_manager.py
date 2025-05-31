import insightspike.cache_manager as cm
from unittest.mock import patch

class DummyMem:
    def save(self, path):
        DummyMem.saved = path


def test_save_cache(tmp_path):
    """Test cache saving - should work regardless of PyTorch availability"""
    cm.GRAPH_PATH = tmp_path / 'g.pt'
    cm.MEMORY_JSON = tmp_path / 'm.json'
    
    # Force fallback to pickle for consistent testing
    original_torch_available = cm.TORCH_AVAILABLE
    cm.TORCH_AVAILABLE = False
    
    try:
        # Test that save_cache function works and calls memory.save()
        cm.save_cache('test_graph', DummyMem())
        
        # Verify that memory.save() was called with the correct path
        assert DummyMem.saved == cm.MEMORY_JSON
        
        # Verify that some graph file was created (pickle)
        assert cm.GRAPH_PATH.exists()
    finally:
        cm.TORCH_AVAILABLE = original_torch_available
