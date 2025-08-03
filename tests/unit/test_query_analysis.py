"""
Unit tests for query history retrieval and analysis functionality
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta

from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager


class TestQueryHistoryAnalysis:
    """Test query history retrieval and analysis features"""
    
    def setup_method(self):
        """Set up test data"""
        self.datastore = InMemoryDataStore()
        self.manager = CachedMemoryManager(self.datastore, cache_size=10)
        
        # Create sample queries with different properties
        self.sample_queries = [
            {
                "text": "What is machine learning?",
                "has_spike": True,
                "metadata": {
                    "processing_time": 1.2,
                    "llm_provider": "OpenAIProvider",
                    "reasoning_quality": 0.85,
                    "retrieved_doc_count": 5
                }
            },
            {
                "text": "How does deep learning work?",
                "has_spike": True,
                "metadata": {
                    "processing_time": 1.5,
                    "llm_provider": "OpenAIProvider",
                    "reasoning_quality": 0.92,
                    "retrieved_doc_count": 8
                }
            },
            {
                "text": "What is the weather today?",
                "has_spike": False,
                "metadata": {
                    "processing_time": 0.8,
                    "llm_provider": "MockProvider",
                    "reasoning_quality": 0.3,
                    "retrieved_doc_count": 0
                }
            },
            {
                "text": "Explain reinforcement learning",
                "has_spike": True,
                "metadata": {
                    "processing_time": 2.1,
                    "llm_provider": "ClaudeProvider",
                    "reasoning_quality": 0.88,
                    "retrieved_doc_count": 6
                }
            },
            {
                "text": "Hello",
                "has_spike": False,
                "metadata": {
                    "processing_time": 0.5,
                    "llm_provider": "MockProvider",
                    "reasoning_quality": 0.1,
                    "retrieved_doc_count": 0
                }
            }
        ]
        
        # Save sample queries
        for i, query_data in enumerate(self.sample_queries):
            self.manager.save_query(
                query_text=query_data["text"],
                has_spike=query_data["has_spike"],
                response=f"Response {i}",
                metadata=query_data["metadata"]
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps
    
    def test_query_time_range_filtering(self):
        """Test filtering queries by time range"""
        # Get all queries
        all_queries = self.manager.get_recent_queries()
        assert len(all_queries) == 5
        
        # Test filtering by recent time (should get all)
        recent_queries = self.manager.get_recent_queries(limit=3)
        assert len(recent_queries) == 3
        
        # Verify ordering (most recent first)
        timestamps = [q["timestamp"] for q in recent_queries]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_query_pattern_analysis(self):
        """Test analyzing query patterns"""
        stats = self.manager.get_query_statistics()
        
        # Basic statistics
        assert stats["total_queries"] == 5
        assert stats["spike_queries"] == 3
        assert stats["spike_rate"] == 0.6
        
        # Provider distribution
        assert stats["llm_providers"]["OpenAIProvider"] == 2
        assert stats["llm_providers"]["MockProvider"] == 2
        assert stats["llm_providers"]["ClaudeProvider"] == 1
        
        # Average processing time
        expected_avg = (1.2 + 1.5 + 0.8 + 2.1 + 0.5) / 5
        assert stats["avg_processing_time"] == pytest.approx(expected_avg)
    
    def test_spike_success_analysis(self):
        """Test analyzing spike success patterns"""
        # Get spike queries
        spike_queries = self.datastore.load_queries(has_spike=True)
        
        # Analyze reasoning quality for spike queries
        spike_qualities = [
            q["metadata"]["reasoning_quality"] 
            for q in spike_queries
        ]
        avg_spike_quality = sum(spike_qualities) / len(spike_qualities)
        
        # Get non-spike queries
        no_spike_queries = self.datastore.load_queries(has_spike=False)
        no_spike_qualities = [
            q["metadata"]["reasoning_quality"]
            for q in no_spike_queries
        ]
        avg_no_spike_quality = sum(no_spike_qualities) / len(no_spike_qualities)
        
        # Spike queries should have higher quality
        assert avg_spike_quality > avg_no_spike_quality
        assert avg_spike_quality > 0.8
        assert avg_no_spike_quality < 0.5
    
    def test_document_retrieval_patterns(self):
        """Test analyzing document retrieval patterns"""
        all_queries = self.datastore.load_queries()
        
        # Calculate average documents retrieved
        doc_counts = [q["metadata"]["retrieved_doc_count"] for q in all_queries]
        avg_docs = sum(doc_counts) / len(doc_counts)
        
        # Spike queries should retrieve more documents
        spike_doc_counts = [
            q["metadata"]["retrieved_doc_count"]
            for q in all_queries if q["has_spike"]
        ]
        no_spike_doc_counts = [
            q["metadata"]["retrieved_doc_count"]
            for q in all_queries if not q["has_spike"]
        ]
        
        assert sum(spike_doc_counts) / len(spike_doc_counts) > 5
        assert sum(no_spike_doc_counts) / len(no_spike_doc_counts) < 1
    
    def test_query_complexity_analysis(self):
        """Test analyzing query complexity patterns"""
        all_queries = self.datastore.load_queries()
        
        # Simple complexity metric: word count
        query_complexities = []
        for q in all_queries:
            word_count = len(q["text"].split())
            complexity = {
                "text": q["text"],
                "word_count": word_count,
                "has_spike": q["has_spike"],
                "processing_time": q["metadata"]["processing_time"]
            }
            query_complexities.append(complexity)
        
        # Sort by word count
        query_complexities.sort(key=lambda x: x["word_count"])
        
        # Verify simple queries (1-2 words) tend to not generate spikes
        simple_queries = [q for q in query_complexities if q["word_count"] <= 2]
        assert all(not q["has_spike"] for q in simple_queries)
        
        # Complex queries (3+ words) tend to generate spikes
        complex_queries = [q for q in query_complexities if q["word_count"] >= 3]
        spike_ratio = sum(1 for q in complex_queries if q["has_spike"]) / len(complex_queries)
        assert spike_ratio >= 0.5
    
    def test_provider_performance_comparison(self):
        """Test comparing performance across LLM providers"""
        all_queries = self.datastore.load_queries()
        
        # Group by provider
        provider_stats = {}
        for q in all_queries:
            provider = q["metadata"]["llm_provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "count": 0,
                    "total_time": 0,
                    "spike_count": 0,
                    "total_quality": 0
                }
            
            stats = provider_stats[provider]
            stats["count"] += 1
            stats["total_time"] += q["metadata"]["processing_time"]
            stats["total_quality"] += q["metadata"]["reasoning_quality"]
            if q["has_spike"]:
                stats["spike_count"] += 1
        
        # Calculate averages
        for provider, stats in provider_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_quality"] = stats["total_quality"] / stats["count"]
            stats["spike_rate"] = stats["spike_count"] / stats["count"]
        
        # Verify mock provider has lower quality
        assert provider_stats["MockProvider"]["avg_quality"] < 0.5
        
        # Real providers should have higher quality
        if "OpenAIProvider" in provider_stats:
            assert provider_stats["OpenAIProvider"]["avg_quality"] > 0.8
    
    def test_query_similarity_clustering(self):
        """Test finding similar queries"""
        # Add some similar queries
        similar_queries = [
            "What is neural network?",
            "What are neural networks?",
            "Explain neural networks"
        ]
        
        for q in similar_queries:
            self.manager.save_query(
                query_text=q,
                has_spike=True,
                response="Neural networks are..."
            )
        
        # Get all queries
        all_queries = self.datastore.load_queries()
        
        # Simple similarity check - queries about neural networks
        nn_queries = [
            q for q in all_queries 
            if "neural" in q["text"].lower()
        ]
        
        assert len(nn_queries) == 3
        # All neural network queries should have generated spikes
        assert all(q["has_spike"] for q in nn_queries)


class TestQueryExportFormats:
    """Test exporting query history in different formats"""
    
    def setup_method(self):
        """Set up test data"""
        self.datastore = InMemoryDataStore()
        self.manager = CachedMemoryManager(self.datastore, cache_size=10)
        
        # Add some test queries
        for i in range(3):
            self.manager.save_query(
                query_text=f"Test query {i}",
                has_spike=(i % 2 == 0),
                response=f"Response {i}",
                metadata={"index": i}
            )
    
    def test_export_to_dict(self):
        """Test exporting queries as dictionary"""
        queries = self.datastore.load_queries()
        
        # Convert to export format
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_queries": len(queries),
            "queries": queries
        }
        
        assert export_data["total_queries"] == 3
        assert len(export_data["queries"]) == 3
    
    def test_export_summary_stats(self):
        """Test exporting summary statistics"""
        stats = self.manager.get_query_statistics()
        
        summary = {
            "period": "all_time",
            "total_queries": stats["total_queries"],
            "spike_rate": f"{stats['spike_rate'] * 100:.1f}%",
            "avg_processing_time": f"{stats['avg_processing_time']:.2f}s",
            "providers": stats["llm_providers"]
        }
        
        assert summary["total_queries"] == 3
        assert "spike_rate" in summary
        assert "providers" in summary


if __name__ == "__main__":
    pytest.main([__file__])