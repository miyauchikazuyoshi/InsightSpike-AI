#!/usr/bin/env python3
"""
Configuration Pattern Tests
===========================

Test various configuration patterns and combinations.
"""

import pytest
import tempfile
from pathlib import Path

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory


class TestConfigurationPatterns:
    """Test different configuration patterns"""
    
    @pytest.fixture
    def temp_datastore(self):
        """Create temporary datastore"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataStoreFactory.create("filesystem", base_path=tmpdir)
            
    def create_config(self, **options):
        """Create config with specified options"""
        config = {
            "processing": {
                "enable_learning": options.get("enable_learning", False),
                "enable_layer1_bypass": options.get("enable_layer1_bypass", False),
                "enable_insight_registration": options.get("enable_insight_registration", True),
                "enable_insight_search": options.get("enable_insight_search", False),
                "max_insights_per_query": options.get("max_insights_per_query", 5),
                "dynamic_doc_adjustment": options.get("dynamic_doc_adjustment", False),
                "enable_adaptive_loop": options.get("enable_adaptive_loop", False),
            },
            "memory": {
                "max_retrieved_docs": options.get("max_retrieved_docs", 5),
                "enable_graph_search": options.get("enable_graph_search", False),
            },
            "l4_config": {
                "provider": options.get("llm_provider", "mock"),
                "model": options.get("llm_model", "mock-model"),
                "temperature": options.get("temperature", 0.7),
                "max_tokens": options.get("max_tokens", 500),
            },
            "graph": {
                "similarity_threshold": options.get("similarity_threshold", 0.7),
                "hop_limit": options.get("hop_limit", 2),
                "path_decay": options.get("path_decay", 0.8),
            },
            "vector_search": {
                "backend": "numpy",  # Force numpy backend to avoid FAISS segfaults
                "optimize": True,
                "batch_size": 1000,
            }
        }
        
        # Add optional configs
        if options.get("enable_scalable_graph", False):
            config["scalable_graph"] = {
                "top_k_neighbors": 50,
                "batch_size": 1000
            }
            
        if options.get("enable_embeddings", True):
            config["embedding"] = {
                "dimension": 384,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
        return config
        
    def test_minimal_config(self, temp_datastore):
        """Test minimal configuration"""
        config = self.create_config()
        agent = MainAgent(config=config, datastore=temp_datastore)
        
        assert agent.initialize()
        
        # Test basic operations
        agent.add_knowledge("Test knowledge")
        result = agent.process_question("What is test?")
        
        assert hasattr(result, 'response') or isinstance(result, dict)
        
    def test_learning_enabled(self, temp_datastore):
        """Test with learning enabled"""
        config = self.create_config(enable_learning=True)
        agent = MainAgent(config=config, datastore=temp_datastore)
        
        assert agent.initialize()
        
        # Add knowledge and process questions
        agent.add_knowledge("Machine learning is about patterns")
        result1 = agent.process_question("What is machine learning?")
        result2 = agent.process_question("Tell me about patterns")
        
        # Save state to datastore
        assert agent.save_state()
        
        # Should create episodes for questions
        episodes = temp_datastore.load_episodes(namespace="agent_state")
        assert len(episodes) >= 2
        
    def test_all_features_enabled(self, temp_datastore):
        """Test with all features enabled"""
        config = self.create_config(
            enable_learning=True,
            enable_insight_search=True,
            enable_graph_search=True,
            dynamic_doc_adjustment=True,
            enable_adaptive_loop=True,
            enable_scalable_graph=True
        )
        
        agent = MainAgent(config=config, datastore=temp_datastore)
        assert agent.initialize()
        
        # Complex operations
        for i in range(5):
            agent.add_knowledge(f"Knowledge item {i}")
            
        result = agent.process_question("What did you learn?")
        assert result is not None
        
    def test_graph_configurations(self, temp_datastore):
        """Test different graph configurations"""
        # High similarity threshold
        config1 = self.create_config(similarity_threshold=0.9)
        agent1 = MainAgent(config=config1, datastore=temp_datastore)
        assert agent1.initialize()
        
        # Deep graph search (hop_limit max is 3)
        config2 = self.create_config(
            enable_graph_search=True,
            hop_limit=3,
            path_decay=0.5
        )
        agent2 = MainAgent(config=config2, datastore=temp_datastore)
        assert agent2.initialize()
        
    def test_memory_patterns(self, temp_datastore):
        """Test different memory configurations"""
        # Large retrieval
        config1 = self.create_config(max_retrieved_docs=20)
        agent1 = MainAgent(config=config1, datastore=temp_datastore)
        assert agent1.initialize()
        
        # Minimal retrieval
        config2 = self.create_config(max_retrieved_docs=1)
        agent2 = MainAgent(config=config2, datastore=temp_datastore)
        assert agent2.initialize()
        
    @pytest.mark.parametrize("datastore_type", ["memory", "filesystem"])
    def test_datastore_types(self, datastore_type):
        """Test different datastore types"""
        with tempfile.TemporaryDirectory() as tmpdir:
            if datastore_type == "filesystem":
                datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
            else:
                datastore = DataStoreFactory.create("memory")
                
            config = self.create_config(enable_learning=True)
            agent = MainAgent(config=config, datastore=datastore)
            
            assert agent.initialize()
            agent.add_knowledge("Test")
            
            # Save state explicitly for filesystem datastore
            if datastore_type == "filesystem":
                agent.save_state()
            
            # Verify persistence with correct namespace
            episodes = datastore.load_episodes(namespace="agent_state")
            assert len(episodes) >= 1
            
    def test_edge_cases(self, temp_datastore):
        """Test edge case configurations"""
        # Everything disabled
        config1 = self.create_config(
            enable_learning=False,
            enable_layer1_bypass=False,
            enable_insight_registration=False,
            enable_insight_search=False,
            enable_graph_search=False,
            dynamic_doc_adjustment=False,
            enable_adaptive_loop=False,
        )
        agent1 = MainAgent(config=config1, datastore=temp_datastore)
        assert agent1.initialize()
        
        # No embeddings
        config2 = self.create_config(enable_embeddings=False)
        agent2 = MainAgent(config=config2, datastore=temp_datastore)
        assert agent2.initialize()
        
    def test_stress_patterns(self, temp_datastore):
        """Test stress configurations"""
        config = self.create_config(
            enable_insight_registration=True,
            max_insights_per_query=50,
            max_retrieved_docs=100
        )
        
        agent = MainAgent(config=config, datastore=temp_datastore)
        assert agent.initialize()
        
        # Add many knowledge items
        for i in range(20):
            agent.add_knowledge(f"Knowledge item {i} with various details")
            
        # Process with high limits
        result = agent.process_question("Summarize all knowledge")
        assert result is not None