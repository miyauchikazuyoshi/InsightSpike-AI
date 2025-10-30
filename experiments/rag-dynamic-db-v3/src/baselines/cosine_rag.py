"""Cosine similarity-based RAG system."""

import time
from typing import Dict, Any, List

import numpy as np

from .base_rag import BaseRAGSystem, UpdateDecision, RetrievalResult
from ..core.gedig_evaluator import UpdateType, GraphUpdate
from ..core.config import ExperimentConfig


class CosineOnlyRAG(BaseRAGSystem):
    """RAG system that updates knowledge based only on cosine similarity thresholds.
    
    This baseline represents embedding-based knowledge management:
    - Adds knowledge when query similarity to existing knowledge is low
    - Uses cosine similarity as the primary decision criterion
    - No temporal or frequency considerations
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Cosine-only RAG system.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, method_name="cosine")
        
        # Configuration parameters
        self.similarity_threshold = config.cosine_similarity_threshold
        self.high_similarity_threshold = 0.9  # Very similar - likely redundant
        self.medium_similarity_threshold = 0.5  # Moderate similarity
        
        # Statistics tracking
        self.similarity_decisions = {
            'below_threshold': 0,
            'above_threshold': 0,
            'redundant_avoided': 0
        }
    
    def should_update_knowledge(self, 
                               query: str, 
                               response: str,
                               retrieval_result: RetrievalResult) -> UpdateDecision:
        """Decide based purely on cosine similarity to existing knowledge.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Result from knowledge retrieval
            
        Returns:
            Update decision based on similarity thresholds
        """
        # Get similarity statistics from retrieval
        max_similarity = retrieval_result.stats.get('max_similarity', 0.0)
        avg_similarity = retrieval_result.stats.get('avg_similarity', 0.0)
        n_retrieved = retrieval_result.stats.get('n_retrieved', 0)
        
        # Decision logic based on similarity
        should_update = False
        confidence = 0.0
        reason_parts = []
        
        if n_retrieved == 0:
            # No existing knowledge - definitely add
            should_update = True
            confidence = 1.0
            reason_parts.append("no_existing_knowledge")
            self.similarity_decisions['below_threshold'] += 1
            
        elif max_similarity >= self.high_similarity_threshold:
            # Very high similarity - likely redundant
            should_update = False
            confidence = 0.9
            reason_parts.append(f"high_similarity({max_similarity:.3f}>={self.high_similarity_threshold})")
            self.similarity_decisions['redundant_avoided'] += 1
            
        elif max_similarity < self.similarity_threshold:
            # Low similarity - new information likely
            should_update = True
            confidence = 1.0 - max_similarity  # Higher confidence for lower similarity
            reason_parts.append(f"low_similarity({max_similarity:.3f}<{self.similarity_threshold})")
            self.similarity_decisions['below_threshold'] += 1
            
        else:
            # Moderate similarity - don't add
            should_update = False
            confidence = max_similarity
            reason_parts.append(f"moderate_similarity({max_similarity:.3f}>={self.similarity_threshold})")
            self.similarity_decisions['above_threshold'] += 1
        
        # Additional factors for fine-tuning
        if should_update:
            # Check average similarity to avoid adding to very dense regions
            if avg_similarity > self.medium_similarity_threshold and n_retrieved > 3:
                confidence *= 0.7  # Reduce confidence in dense areas
                reason_parts.append("dense_region_penalty")
        
        # Create update if decision is positive
        update = None
        if should_update:
            update = self._create_similarity_based_update(query, response, retrieval_result, max_similarity)
            reason = f"Cosine-only: {', '.join(reason_parts)}"
        else:
            reason = f"Cosine-only: {', '.join(reason_parts)}"
        
        return UpdateDecision(
            should_update=should_update,
            update_type=UpdateType.ADD,
            reason=reason,
            confidence=confidence,
            update=update,
            metadata={
                'max_similarity': max_similarity,
                'avg_similarity': avg_similarity,
                'n_retrieved': n_retrieved,
                'similarity_threshold': self.similarity_threshold,
                'decision_factors': reason_parts
            }
        )
    
    def _create_similarity_based_update(self, 
                                      query: str, 
                                      response: str,
                                      retrieval_result: RetrievalResult,
                                      max_similarity: float) -> GraphUpdate:
        """Create knowledge addition update based on similarity analysis.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Retrieval result for context
            max_similarity: Maximum similarity to existing knowledge
            
        Returns:
            Graph update for knowledge addition
        """
        # Create knowledge text - combine query and response
        knowledge_text = f"Query: {query}\nAnswer: {response}"
        
        # Generate embedding
        embedding = self.embedder.encode([knowledge_text])[0]
        
        # Create new node data
        new_node_id = f"cosine_node_{int(time.time())}"
        new_node_data = {
            'id': new_node_id,
            'text': knowledge_text,
            'embedding': embedding,
            'node_type': 'qa_pair',
            'confidence': 1.0 - max_similarity,  # Higher confidence for more novel information
            'timestamp': time.time(),
            'metadata': {
                'source': 'cosine_similarity_addition',
                'query': query,
                'response': response,
                'max_similarity_to_existing': max_similarity,
                'n_similar_nodes': retrieval_result.stats.get('n_retrieved', 0)
            }
        }
        
        # Create edges to moderately similar nodes (not too similar, not too dissimilar)
        new_edges = []
        for node_id, similarity in retrieval_result.retrieved_nodes:
            # Create edges to nodes with moderate similarity
            if 0.2 <= similarity <= 0.7:  # Sweet spot for connections
                edge_weight = similarity * 0.8  # Slightly discounted
                new_edges.append((new_node_id, node_id, {
                    'relation': 'semantic',
                    'weight': edge_weight,
                    'semantic_similarity': similarity
                }))
        
        return GraphUpdate(
            update_type=UpdateType.ADD,
            target_nodes=[],
            new_node_data=new_node_data,
            new_edges=new_edges,
            metadata={
                'method': 'cosine_similarity',
                'decision_similarity': max_similarity,
                'n_edges_created': len(new_edges)
            }
        )
    
    def _calculate_embeddings_for_evaluation(self, texts: List[str]) -> np.ndarray:
        """Calculate embeddings for similarity evaluation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        return self.embedder.encode(texts)
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get cosine similarity-specific statistics.
        
        Returns:
            Dictionary of similarity-based decision statistics
        """
        total_decisions = sum(self.similarity_decisions.values())
        
        if total_decisions > 0:
            decision_rates = {
                f"{key}_rate": count / total_decisions 
                for key, count in self.similarity_decisions.items()
            }
        else:
            decision_rates = {}
        
        return {
            'decision_counts': self.similarity_decisions,
            'total_decisions': total_decisions,
            **decision_rates,
            'similarity_threshold': self.similarity_threshold,
            'high_similarity_threshold': self.high_similarity_threshold,
            'medium_similarity_threshold': self.medium_similarity_threshold
        }
    
    def analyze_knowledge_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of knowledge based on embeddings.
        
        Returns:
            Analysis of knowledge embedding space
        """
        if len(self.knowledge_graph.nodes) < 2:
            return {'status': 'insufficient_data'}
        
        # Get all embeddings
        embeddings = []
        node_ids = []
        
        for node_id, node in self.knowledge_graph.nodes.items():
            embeddings.append(node.embedding)
            node_ids.append(node_id)
        
        embeddings = np.array(embeddings)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        if similarities:
            return {
                'n_nodes': len(embeddings),
                'avg_pairwise_similarity': np.mean(similarities),
                'max_pairwise_similarity': np.max(similarities),
                'min_pairwise_similarity': np.min(similarities),
                'similarity_std': np.std(similarities),
                'dense_pairs': sum(1 for s in similarities if s > self.high_similarity_threshold),
                'sparse_pairs': sum(1 for s in similarities if s < 0.1)
            }
        else:
            return {'status': 'no_similarities_calculated'}
    
    def reset(self):
        """Reset similarity tracking and parent state."""
        super().reset()
        self.similarity_decisions = {
            'below_threshold': 0,
            'above_threshold': 0,
            'redundant_avoided': 0
        }
