"""Frequency-based RAG system."""

import time
import hashlib
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np

from .base_rag import BaseRAGSystem, UpdateDecision, RetrievalResult
from ..core.gedig_evaluator import UpdateType, GraphUpdate
from ..core.config import ExperimentConfig


class FrequencyBasedRAG(BaseRAGSystem):
    """RAG system that updates knowledge based on query frequency and temporal patterns.
    
    This baseline represents simple heuristic-based knowledge management:
    - Adds knowledge for infrequent queries (assuming new information)
    - Considers temporal patterns (time since last update)
    - Uses access frequency as a proxy for importance
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Frequency-based RAG system.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, method_name="frequency")
        
        # Frequency tracking
        self.query_counts: Dict[str, int] = defaultdict(int)
        self.last_update_times: Dict[str, float] = defaultdict(float)
        self.query_history: List[str] = []
        
        # Configuration parameters
        self.frequency_threshold = config.frequency_threshold
        self.time_threshold = config.time_threshold_hours * 3600  # Convert to seconds
        
        # Simple heuristics for knowledge selection
        self.low_retrieval_threshold = 0.3  # If avg similarity < this, consider adding knowledge
    
    def should_update_knowledge(self, 
                               query: str, 
                               response: str,
                               retrieval_result: RetrievalResult) -> UpdateDecision:
        """Decide based on query frequency and temporal patterns.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Result from knowledge retrieval
            
        Returns:
            Update decision based on frequency heuristics
        """
        # Create normalized query hash for tracking
        query_hash = self._get_query_hash(query)
        current_time = time.time()
        
        # Update tracking
        self.query_counts[query_hash] += 1
        self.query_history.append(query)
        
        # Decision factors
        query_count = self.query_counts[query_hash]
        time_since_update = current_time - self.last_update_times[query_hash]
        avg_retrieval_similarity = retrieval_result.stats.get('avg_similarity', 0.0)
        n_retrieved = retrieval_result.stats.get('n_retrieved', 0)
        
        # Decision logic
        should_update = False
        reason_parts = []
        confidence = 0.0
        
        # Factor 1: Low frequency queries (likely new information)
        if query_count <= self.frequency_threshold:
            should_update = True
            reason_parts.append(f"low_frequency({query_count}<={self.frequency_threshold})")
            confidence += 0.4
        
        # Factor 2: Poor retrieval performance (knowledge gap)
        if avg_retrieval_similarity < self.low_retrieval_threshold:
            should_update = True
            reason_parts.append(f"poor_retrieval({avg_retrieval_similarity:.3f}<{self.low_retrieval_threshold})")
            confidence += 0.3
        
        # Factor 3: No relevant knowledge found
        if n_retrieved == 0:
            should_update = True
            reason_parts.append("no_knowledge_found")
            confidence += 0.5
        
        # Factor 4: Sufficient time has passed since last update
        if time_since_update > self.time_threshold:
            should_update = True
            reason_parts.append(f"time_elapsed({time_since_update/3600:.1f}h>{self.time_threshold/3600:.1f}h)")
            confidence += 0.2
        
        # Create update if decision is positive
        update = None
        if should_update:
            # Record update time
            self.last_update_times[query_hash] = current_time
            
            # Create simple knowledge addition update
            update = self._create_knowledge_addition_update(query, response, retrieval_result)
            
            reason = f"Frequency-based: {', '.join(reason_parts)}"
        else:
            reason = f"Frequency-based: thresholds not met (freq={query_count}, similarity={avg_retrieval_similarity:.3f})"
        
        return UpdateDecision(
            should_update=should_update,
            update_type=UpdateType.ADD,
            reason=reason,
            confidence=min(confidence, 1.0),
            update=update,
            metadata={
                'query_hash': query_hash,
                'query_count': query_count,
                'time_since_update': time_since_update,
                'avg_retrieval_similarity': avg_retrieval_similarity,
                'n_retrieved': n_retrieved,
                'decision_factors': reason_parts
            }
        )
    
    def _get_query_hash(self, query: str) -> str:
        """Create normalized hash for query tracking.
        
        Args:
            query: User query
            
        Returns:
            Normalized hash string
        """
        # Simple normalization: lowercase, remove extra whitespace
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def _create_knowledge_addition_update(self, 
                                        query: str, 
                                        response: str,
                                        retrieval_result: RetrievalResult) -> GraphUpdate:
        """Create a knowledge addition update based on query/response.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Retrieval result for context
            
        Returns:
            Graph update for knowledge addition
        """
        # Create embedding for the new knowledge
        knowledge_text = f"Q: {query} A: {response}"
        embedding = self.embedder.encode([knowledge_text])[0]
        
        # Create new node data
        new_node_id = f"freq_node_{int(time.time())}"
        new_node_data = {
            'id': new_node_id,
            'text': knowledge_text,
            'embedding': embedding,
            'node_type': 'qa_pair',
            'confidence': 0.7,  # Medium confidence for frequency-based additions
            'timestamp': time.time(),
            'metadata': {
                'source': 'frequency_based_addition',
                'query': query,
                'response': response,
                'retrieval_quality': retrieval_result.stats.get('avg_similarity', 0.0)
            }
        }
        
        # Create edges to retrieved nodes (if any)
        new_edges = []
        for node_id, similarity in retrieval_result.retrieved_nodes[:3]:  # Top 3
            if similarity > 0.1:  # Minimum similarity threshold
                new_edges.append((new_node_id, node_id, {
                    'relation': 'semantic',
                    'weight': similarity,
                    'semantic_similarity': similarity
                }))
        
        return GraphUpdate(
            update_type=UpdateType.ADD,
            target_nodes=[],  # New addition
            new_node_data=new_node_data,
            new_edges=new_edges,
            metadata={
                'method': 'frequency_based',
                'query_hash': self._get_query_hash(query)
            }
        )
    
    def get_frequency_statistics(self) -> Dict[str, Any]:
        """Get frequency-specific statistics.
        
        Returns:
            Dictionary of frequency tracking statistics
        """
        total_queries = len(self.query_history)
        unique_queries = len(self.query_counts)
        
        if unique_queries > 0:
            avg_frequency = total_queries / unique_queries
            max_frequency = max(self.query_counts.values())
            min_frequency = min(self.query_counts.values())
        else:
            avg_frequency = max_frequency = min_frequency = 0
        
        # Most common query patterns
        sorted_queries = sorted(self.query_counts.items(), key=lambda x: x[1], reverse=True)
        top_queries = sorted_queries[:5] if sorted_queries else []
        
        return {
            'total_queries': total_queries,
            'unique_queries': unique_queries,
            'avg_query_frequency': avg_frequency,
            'max_query_frequency': max_frequency,
            'min_query_frequency': min_frequency,
            'top_query_patterns': top_queries,
            'frequency_threshold': self.frequency_threshold,
            'time_threshold_hours': self.time_threshold / 3600
        }
    
    def reset(self):
        """Reset frequency tracking and parent state."""
        super().reset()
        self.query_counts.clear()
        self.last_update_times.clear()
        self.query_history.clear()
