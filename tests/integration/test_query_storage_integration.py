"""
Integration tests for query storage functionality across the system
"""

import pytest
import tempfile
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch

from insightspike.config.models import InsightSpikeConfig
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager
from insightspike.implementations.agents.main_agent import MainAgent, CycleResult
from insightspike.adaptive.core.adaptive_processor import AdaptiveProcessor


class TestQueryStorageEndToEnd:
    """Test complete query storage workflow from user query to persistence"""
    
    @pytest.fixture
    def temp_datastore(self):
        """Create temporary filesystem datastore"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileSystemDataStore(tmpdir)
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider"""
        mock_llm = Mock()
        mock_llm.initialize.return_value = True
        mock_llm.__class__.__name__ = "MockLLMProvider"
        mock_llm.generate_response_detailed.return_value = {
            "response": "This is a test response about insights.",
            "success": True,
            "confidence": 0.85,
            "reasoning": "Based on the retrieved documents..."
        }
        return mock_llm
    
    @patch('insightspike.implementations.agents.main_agent.get_llm_provider')
    @patch('insightspike.implementations.agents.main_agent.GRAPH_REASONER_AVAILABLE', True)
    def test_main_agent_query_storage_workflow(self, mock_get_llm, temp_datastore, mock_llm_provider):
        """Test query storage through MainAgent processing"""
        mock_get_llm.return_value = mock_llm_provider
        
        # Create config
        config = InsightSpikeConfig()
        
        # Create MainAgent with datastore
        agent = MainAgent(config=config, datastore=temp_datastore)
        
        # Process a question that should generate a spike
        question = "What is the relationship between insights and knowledge discovery?"
        result = agent.process_question(question, max_cycles=2, verbose=True)
        
        # Verify result has query_id
        assert result.query_id is not None
        assert result.query_id.startswith("query_")
        
        # Verify query was saved to datastore
        saved_queries = temp_datastore.load_queries()
        assert len(saved_queries) >= 1
        
        # Find our query
        our_query = None
        for q in saved_queries:
            if q["id"] == result.query_id:
                our_query = q
                break
        
        assert our_query is not None
        assert our_query["text"] == question
        assert our_query["response"] == result.response
        assert our_query["has_spike"] == result.spike_detected
        
        # Check metadata
        assert "processing_time" in our_query["metadata"]
        assert our_query["metadata"]["llm_provider"] == "MockLLMProvider"
        assert our_query["metadata"]["total_cycles"] == len(result.graph_analysis.get("cycle_history", [1]))
        assert our_query["metadata"]["reasoning_quality"] == result.reasoning_quality
    
    def test_cached_memory_manager_with_graph_integration(self, temp_datastore):
        """Test CachedMemoryManager saving queries and adding to graph"""
        # Create manager
        manager = CachedMemoryManager(datastore=temp_datastore, cache_size=10)
        
        # Create a knowledge graph
        graph = nx.Graph()
        
        # Add some episodes to graph
        episode_ids = []
        for i in range(3):
            ep_id = f"episode_{i}"
            episode_ids.append(ep_id)
            graph.add_node(
                ep_id,
                text=f"Episode {i} about insights",
                vec=np.random.rand(384),
                metadata={"index": i}
            )
        
        # Save a query that retrieved these episodes
        query_id = manager.save_query(
            query_text="What are insights?",
            query_vec=np.random.rand(384),
            has_spike=True,
            spike_episode_id=episode_ids[0],
            response="Insights are sudden realizations...",
            metadata={"test": True}
        )
        
        # Add query to graph
        success = manager.add_query_to_graph(
            graph=graph,
            query_id=query_id,
            query_text="What are insights?",
            query_vec=np.random.rand(384),
            has_spike=True,
            spike_episode_id=episode_ids[0],
            retrieved_episode_ids=episode_ids
        )
        
        assert success
        
        # Verify graph structure
        assert query_id in graph
        assert graph.nodes[query_id]["type"] == "query"
        assert graph.nodes[query_id]["has_spike"] == True
        
        # Check edges
        # Should have query_spike edge to spike episode
        assert graph.has_edge(query_id, episode_ids[0])
        assert graph.edges[query_id, episode_ids[0]]["relation"] == "query_spike"
        
        # Should have query_retrieval edges to all retrieved episodes
        for ep_id in episode_ids:
            assert graph.has_edge(query_id, ep_id)
            if ep_id != episode_ids[0]:  # Not the spike episode
                assert graph.edges[query_id, ep_id]["relation"] == "query_retrieval"
    
    def test_query_persistence_across_sessions(self, temp_datastore):
        """Test that queries persist across different manager instances"""
        # First session - save queries
        manager1 = CachedMemoryManager(datastore=temp_datastore, cache_size=5)
        
        query_ids = []
        for i in range(3):
            qid = manager1.save_query(
                query_text=f"Query {i}",
                has_spike=(i == 1),  # Only second query has spike
                response=f"Response {i}",
                metadata={"session": 1, "index": i}
            )
            query_ids.append(qid)
        
        # Second session - new manager instance
        manager2 = CachedMemoryManager(datastore=temp_datastore, cache_size=5)
        
        # Should be able to retrieve queries from first session
        all_queries = manager2.get_recent_queries()
        assert len(all_queries) == 3
        
        # Verify query IDs match
        saved_ids = [q["id"] for q in all_queries]
        for qid in query_ids:
            assert qid in saved_ids
        
        # Add more queries in second session
        manager2.save_query(
            query_text="New session query",
            has_spike=False,
            response="New response",
            metadata={"session": 2}
        )
        
        # Third session - verify all queries
        manager3 = CachedMemoryManager(datastore=temp_datastore, cache_size=5)
        final_queries = manager3.get_recent_queries()
        assert len(final_queries) == 4
        
        # Check statistics across all sessions
        stats = manager3.get_query_statistics()
        assert stats["total_queries"] == 4
        assert stats["spike_queries"] == 1  # Only one spike query
        assert stats["spike_rate"] == 0.25
    
    def test_adaptive_processor_integration(self, temp_datastore):
        """Test AdaptiveProcessor saving queries with exploration metadata"""
        from insightspike.adaptive.core.exploration_loop import ExplorationLoop
        from insightspike.adaptive.strategies.exponential_strategy import ExponentialStrategy
        from insightspike.adaptive.calculators.simple_topk import SimpleTopKCalculator
        
        # Mock exploration loop
        mock_loop = Mock(spec=ExplorationLoop)
        
        # Create mock results for multiple exploration attempts
        mock_results = []
        for i in range(3):
            result = Mock()
            result.spike_detected = (i == 2)  # Spike on third attempt
            result.confidence = 0.3 + i * 0.3
            result.retrieved_docs = [{"text": f"Doc {j}", "index": j} for j in range(i + 1)]
            result.graph_analysis = {"attempt": i + 1}
            result.metrics = {"score": result.confidence}
            result.params = Mock()
            result.params.to_dict.return_value = {"radius": 0.8 - i * 0.1, "topk": 10 + i * 5}
            mock_results.append(result)
        
        # Configure mock to return results in sequence
        mock_loop.explore_once.side_effect = mock_results
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "AdaptiveLLM"
        mock_llm.generate_response.return_value = "Adaptive response with insight"
        
        # Create processor
        processor = AdaptiveProcessor(
            exploration_loop=mock_loop,
            strategy=ExponentialStrategy(),
            topk_calculator=SimpleTopKCalculator(),
            l4_llm=mock_llm,
            datastore=temp_datastore,
            max_attempts=5
        )
        
        # Process question
        result = processor.process("How does adaptive exploration find insights?", verbose=True)
        
        # Verify result
        assert result["spike_detected"] == True
        assert result["adaptive_metadata"]["total_attempts"] == 3
        assert "query_id" in result
        
        # Check saved query
        queries = temp_datastore.load_queries()
        assert len(queries) == 1
        
        query = queries[0]
        assert query["text"] == "How does adaptive exploration find insights?"
        assert query["has_spike"] == True
        
        # Verify exploration path is saved
        assert "exploration_path" in query["metadata"]
        assert len(query["metadata"]["exploration_path"]) == 3
        
        # Check exploration parameters progression
        path = query["metadata"]["exploration_path"]
        assert path[0]["radius"] == 0.8
        assert path[1]["radius"] == 0.7
        assert path[2]["radius"] == 0.6
    
    def test_query_analysis_over_time(self, temp_datastore):
        """Test analyzing query patterns over multiple sessions"""
        manager = CachedMemoryManager(datastore=temp_datastore, cache_size=10)
        
        # Simulate queries over time with different patterns
        
        # Session 1: Learning phase - mostly no spikes
        for i in range(5):
            manager.save_query(
                query_text=f"Basic question {i}",
                has_spike=False,
                response=f"Basic response {i}",
                metadata={
                    "session": "learning",
                    "processing_time": 0.5 + i * 0.1,
                    "reasoning_quality": 0.3 + i * 0.05
                }
            )
        
        # Session 2: Improving - some spikes
        for i in range(5):
            manager.save_query(
                query_text=f"Intermediate question {i}",
                has_spike=(i % 2 == 0),
                response=f"Better response {i}",
                metadata={
                    "session": "improving", 
                    "processing_time": 1.0 + i * 0.15,
                    "reasoning_quality": 0.6 + i * 0.05
                }
            )
        
        # Session 3: Expert phase - mostly spikes
        for i in range(5):
            manager.save_query(
                query_text=f"Advanced question {i}",
                has_spike=(i != 2),  # All but one generate spikes
                response=f"Insightful response {i}",
                metadata={
                    "session": "expert",
                    "processing_time": 1.5 + i * 0.2,
                    "reasoning_quality": 0.8 + i * 0.03
                }
            )
        
        # Analyze progression
        all_queries = temp_datastore.load_queries()
        assert len(all_queries) == 15
        
        # Group by session
        sessions = {}
        for q in all_queries:
            session = q["metadata"]["session"]
            if session not in sessions:
                sessions[session] = []
            sessions[session].append(q)
        
        # Calculate spike rates per session
        spike_rates = {}
        for session, queries in sessions.items():
            spike_count = sum(1 for q in queries if q["has_spike"])
            spike_rates[session] = spike_count / len(queries)
        
        # Verify improvement over sessions
        assert spike_rates["learning"] == 0.0
        assert spike_rates["improving"] == 0.6
        assert spike_rates["expert"] == 0.8
        
        # Verify reasoning quality improvement
        avg_qualities = {}
        for session, queries in sessions.items():
            qualities = [q["metadata"]["reasoning_quality"] for q in queries]
            avg_qualities[session] = sum(qualities) / len(qualities)
        
        assert avg_qualities["learning"] < avg_qualities["improving"]
        assert avg_qualities["improving"] < avg_qualities["expert"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])