import numpy as np
from unittest.mock import patch, MagicMock

def test_memory_init():
    """Test Memory class initialization with proper mocking."""
    
    class DummyIndex:
        def __init__(self, dim):
            self.d = dim
            self.ntotal = 0
        def add(self, vecs):
            pass
        def train(self, vecs):
            pass

    # Mock faiss.index_factory to return our dummy index
    with patch('faiss.index_factory') as mock_factory:
        mock_factory.return_value = DummyIndex(8)  # Use dimension compatible with PQ
        
        # Import the module after setting up mocks
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager as Memory
        
        mem = Memory(8)  # Use compatible dimension
        assert mem.dim == 8
        assert hasattr(mem, 'episodes')
        assert hasattr(mem, 'index')
