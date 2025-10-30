"""Tests for config summary utilities."""
import json
import pytest

from insightspike.config.summary import summarize_memory_config
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager

class TestConfigSummaryMemory:
    def test_memory_defaults_applied_summary(self):
        mgr = L2MemoryManager({'memory': {'max_episodes': 321}})
        summ = summarize_memory_config(mgr.config)
        assert summ['max_episodes'] == 321
        # Default injected values
        assert summ['faiss_index_type'] == 'FlatL2'
        assert summ['metric'] == 'l2'
        assert 'defaults_applied' in summ
        assert set(['faiss_index_type','metric']).issubset(set(summ['defaults_applied']))

    def test_memory_no_defaults_when_explicit(self):
        mgr = L2MemoryManager({'memory': {
            'max_episodes': 50,
            'faiss_index_type': 'IVF',
            'metric': 'ip'
        }})
        summ = summarize_memory_config(mgr.config)
        assert summ['faiss_index_type'] == 'IVF'
        assert summ['metric'] == 'ip'
        # Both provided so defaults_applied should not list them
        assert 'defaults_applied' in summ
        assert 'faiss_index_type' not in summ['defaults_applied']
        assert 'metric' not in summ['defaults_applied']

if __name__ == '__main__':
    pytest.main([__file__, '-q'])
