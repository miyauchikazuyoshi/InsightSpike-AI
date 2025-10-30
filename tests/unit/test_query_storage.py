"""
Unit tests for query storage functionality
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch

from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager


class TestDataStoreQueryMethods:
    """Test query methods in DataStore implementations"""
    
    def test_save_single_query_memory(self):
        """Test saving a single query in MemoryStore"""
        datastore = InMemoryDataStore()
        
        query = {
            "id": "query_123",
            "text": "What is insight?",
            "vec": np.random.rand(384),
            "has_spike": False,
            "response": "An insight is...",
            "metadata": {"source": "test"}
        }
        
        # Save query
        assert datastore.save_queries([query])
        
        # Load and verify
        loaded = datastore.load_queries()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "query_123"
        assert loaded[0]["text"] == "What is insight?"
        assert loaded[0]["has_spike"] == False
        
    def test_save_multiple_queries_memory(self):
        """Test saving multiple queries in MemoryStore"""
        datastore = InMemoryDataStore()
        
        queries = [
            {
                "id": "q1",
                "text": "Query 1",
                "vec": np.random.rand(384),
                "has_spike": True,
                "timestamp": 1000
            },
            {
                "id": "q2", 
                "text": "Query 2",
                "vec": np.random.rand(384),
                "has_spike": False,
                "timestamp": 2000
            }
        ]
        
        assert datastore.save_queries(queries)
        loaded = datastore.load_queries()
        assert len(loaded) == 2
        
    def test_load_queries_with_filter(self):
        """Test loading queries with has_spike filter"""
        datastore = InMemoryDataStore()
        
        # Save queries with different spike status
        queries = [
            {"id": "q1", "text": "Q1", "has_spike": True, "timestamp": 1000},
            {"id": "q2", "text": "Q2", "has_spike": False, "timestamp": 2000},
            {"id": "q3", "text": "Q3", "has_spike": True, "timestamp": 3000}
        ]
        datastore.save_queries(queries)
        
        # Filter by has_spike=True
        spike_queries = datastore.load_queries(has_spike=True)
        assert len(spike_queries) == 2
        assert all(q["has_spike"] for q in spike_queries)
        
        # Filter by has_spike=False
        no_spike_queries = datastore.load_queries(has_spike=False)
        assert len(no_spike_queries) == 1
        assert not no_spike_queries[0]["has_spike"]
        
    def test_load_queries_with_limit(self):
        """Test loading queries with limit"""
        datastore = InMemoryDataStore()
        
        # Save 5 queries
        queries = [
            {"id": f"q{i}", "text": f"Query {i}", "timestamp": i * 1000}
            for i in range(5)
        ]
        datastore.save_queries(queries)
        
        # Load with limit
        limited = datastore.load_queries(limit=3)
        assert len(limited) == 3
        # Should be sorted by timestamp descending
        assert limited[0]["id"] == "q4"
        assert limited[1]["id"] == "q3"
        assert limited[2]["id"] == "q2"
        
    def test_filesystem_query_persistence(self):
        """Test query persistence in FileSystemDataStore"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First datastore instance
            ds1 = FileSystemDataStore(tmpdir)
            
            query = {
                "id": "query_test",
                "text": "Test query",
                "vec": np.random.rand(384),
                "has_spike": True,
                "metadata": {"test": "data"}
            }
            
            assert ds1.save_queries([query])
            
            # Second datastore instance
            ds2 = FileSystemDataStore(tmpdir)
            loaded = ds2.load_queries()
            
            assert len(loaded) == 1
            assert loaded[0]["id"] == "query_test"
            assert loaded[0]["text"] == "Test query"
            assert loaded[0]["has_spike"] == True
            

class TestCachedMemoryManagerQueryMethods:
    """Test query methods in CachedMemoryManager"""
    
    def test_save_query_record(self):
        """Test saving a query record"""
        datastore = InMemoryDataStore()
        manager = CachedMemoryManager(datastore, cache_size=10)
        
        query_id = manager.save_query(
            query_text="What is apple?",
            query_vec=np.random.rand(384),
            has_spike=False,
            response="Apple is a fruit...",
            metadata={"source": "test"}
        )
        
        assert query_id is not None
        assert query_id.startswith("query_")
        
        # Verify saved to datastore
        queries = datastore.load_queries()
        assert len(queries) == 1
        assert queries[0]["text"] == "What is apple?"
        
    def test_save_query_with_spike(self):
        """Test saving a query that generated a spike"""
        datastore = InMemoryDataStore()
        manager = CachedMemoryManager(datastore, cache_size=10)
        
        query_id = manager.save_query(
            query_text="What is insight spike?",
            has_spike=True,
            spike_episode_id="episode_123",
            response="An insight spike is...",
            metadata={"processing_time": 1.5}
        )
        
        queries = datastore.load_queries()
        assert len(queries) == 1
        assert queries[0]["has_spike"] == True
        assert queries[0]["spike_episode_id"] == "episode_123"
        
    def test_get_recent_queries(self):
        """Test getting recent queries"""
        datastore = InMemoryDataStore()
        manager = CachedMemoryManager(datastore, cache_size=10)
        
        # Save multiple queries
        for i in range(5):
            manager.save_query(
                query_text=f"Query {i}",
                has_spike=(i % 2 == 0),
                response=f"Response {i}"
            )
        
        # Get all recent queries
        recent = manager.get_recent_queries(limit=10)
        assert len(recent) == 5
        
        # Get only spike queries
        spike_queries = manager.get_recent_queries(has_spike=True)
        assert len(spike_queries) == 3  # 0, 2, 4
        
    def test_get_query_statistics(self):
        """Test getting query statistics"""
        datastore = InMemoryDataStore()
        manager = CachedMemoryManager(datastore, cache_size=10)
        
        # Save queries with different properties
        manager.save_query(
            query_text="Q1",
            has_spike=True,
            metadata={"processing_time": 1.0, "llm_provider": "MockProvider"}
        )
        manager.save_query(
            query_text="Q2",
            has_spike=False,
            metadata={"processing_time": 2.0, "llm_provider": "MockProvider"}
        )
        manager.save_query(
            query_text="Q3",
            has_spike=True,
            metadata={"processing_time": 1.5, "llm_provider": "OpenAIProvider"}
        )
        
        stats = manager.get_query_statistics()
        
        assert stats["total_queries"] == 3
        assert stats["spike_queries"] == 2
        assert stats["non_spike_queries"] == 1
        assert stats["spike_rate"] == pytest.approx(2/3)
        assert stats["avg_processing_time"] == pytest.approx(1.5)
        assert stats["llm_providers"]["MockProvider"] == 2
        assert stats["llm_providers"]["OpenAIProvider"] == 1
        
    def test_add_query_to_graph(self):
        """Test adding query node to graph"""
        import networkx as nx
        
        datastore = InMemoryDataStore()
        manager = CachedMemoryManager(datastore, cache_size=10)
        
        # Create a simple graph
        graph = nx.Graph()
        graph.add_node("episode_1", text="Episode 1", vec=np.random.rand(384))
        graph.add_node("episode_2", text="Episode 2", vec=np.random.rand(384))
        
        # Add query to graph
        success = manager.add_query_to_graph(
            graph=graph,
            query_id="query_123",
            query_text="Test query",
            query_vec=np.random.rand(384),
            has_spike=True,
            spike_episode_id="episode_1",
            retrieved_episode_ids=["episode_1", "episode_2"],
            metadata={"test": "data"}
        )
        
        assert success
        assert "query_123" in graph
        assert graph.nodes["query_123"]["type"] == "query"
        assert graph.nodes["query_123"]["has_spike"] == True
        
        # Check edges
        assert graph.has_edge("query_123", "episode_1")
        assert graph.edges["query_123", "episode_1"]["relation"] == "query_spike"
        
        assert graph.has_edge("query_123", "episode_2")
        assert graph.edges["query_123", "episode_2"]["relation"] == "query_retrieval"


class TestQuerySavingIntegration:
    """Integration tests for query saving in MainAgent and AdaptiveProcessor"""
    
    @patch('insightspike.implementations.agents.main_agent.get_llm_provider')
    def test_main_agent_saves_query(self, mock_get_llm):
        """Test that MainAgent saves queries after processing"""
        from insightspike.implementations.agents.main_agent import MainAgent
        from insightspike.config.models import InsightSpikeConfig
        
        # Mock LLM provider
        mock_llm = Mock()
        mock_llm.initialize.return_value = True
        mock_llm.generate_response_detailed.return_value = {
            "response": "Test response",
            "success": True,
            "confidence": 0.8
        }
        mock_get_llm.return_value = mock_llm
        
        # Create MainAgent with datastore
        datastore = InMemoryDataStore()
        config = InsightSpikeConfig()
        agent = MainAgent(config=config, datastore=datastore)
        
        # Process a question
        result = agent.process_question("What is test?", max_cycles=1)
        
        # Check that query was saved
        queries = datastore.load_queries()
        assert len(queries) > 0
        assert queries[0]["text"] == "What is test?"
        assert result.query_id is not None
        
    def test_adaptive_processor_saves_query(self):
        """Test that AdaptiveProcessor saves queries"""
        from insightspike.adaptive.core.adaptive_processor import AdaptiveProcessor
        from insightspike.adaptive.core.exploration_loop import ExplorationLoop
        from insightspike.adaptive.strategies.exponential_strategy import ExponentialStrategy
        from insightspike.adaptive.calculators.simple_topk import SimpleTopKCalculator
        
        # Mock components
        mock_loop = Mock(spec=ExplorationLoop)
        mock_result = Mock()
        mock_result.spike_detected = True
        mock_result.confidence = 0.9
        mock_result.retrieved_docs = []
        mock_result.graph_analysis = {}
        mock_result.metrics = {}
        mock_result.params = Mock()
        mock_result.params.to_dict.return_value = {}
        mock_loop.explore_once.return_value = mock_result
        
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "Test response"
        
        # Create processor with datastore
        datastore = InMemoryDataStore()
        processor = AdaptiveProcessor(
            exploration_loop=mock_loop,
            strategy=ExponentialStrategy(),
            topk_calculator=SimpleTopKCalculator(),
            l4_llm=mock_llm,
            datastore=datastore
        )
        
        # Process a question
        result = processor.process("What is adaptive?")
        
        # Check that query was saved
        queries = datastore.load_queries()
        assert len(queries) == 1
        assert queries[0]["text"] == "What is adaptive?"
        assert queries[0]["has_spike"] == True
        assert "query_id" in result


if __name__ == "__main__":
    pytest.main([__file__])