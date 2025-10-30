"""
Response evaluation utilities with both distance and cosine metrics
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .similarity_converter import SimilarityConverter
import logging

logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """Evaluate response quality using multiple similarity metrics."""
    
    def __init__(self, embedding_model=None):
        """
        Initialize evaluator.
        
        Args:
            embedding_model: Model to encode text (will use default if None)
        """
        self.embedding_model = embedding_model
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy load the encoder."""
        if self._encoder is None:
            if self.embedding_model:
                self._encoder = self.embedding_model
            else:
                # Use default model
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._encoder
    
    def evaluate_response(
        self, 
        query: str, 
        response: str,
        context_docs: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate response quality with multiple metrics.
        
        Args:
            query: Original query
            response: Generated response
            context_docs: Optional list of context documents used
            
        Returns:
            Dictionary with evaluation metrics
        """
        encoder = self._get_encoder()
        
        # Encode texts
        query_vec = encoder.encode([query])[0]
        response_vec = encoder.encode([response])[0]
        
        # Normalize vectors
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        response_vec_norm = response_vec / (np.linalg.norm(response_vec) + 1e-8)
        
        # Calculate both metrics
        distance, cosine_sim = SimilarityConverter.get_both_metrics(
            query_vec, response_vec
        )
        
        evaluation = {
            "query_response_distance": distance,
            "query_response_cosine": cosine_sim,
            "query_response_relevance": SimilarityConverter.distance_to_relevance(distance),
            "metrics_formatted": SimilarityConverter.format_metrics(distance, cosine_sim)
        }
        
        # Evaluate against context if provided
        if context_docs:
            evaluation["context_evaluation"] = self._evaluate_context_usage(
                response_vec_norm, context_docs, encoder
            )
        
        return evaluation
    
    def _evaluate_context_usage(
        self, 
        response_vec: np.ndarray, 
        context_docs: list,
        encoder
    ) -> Dict[str, Any]:
        """
        Evaluate how well the response uses the provided context.
        
        Args:
            response_vec: Normalized response vector
            context_docs: List of context documents
            encoder: Text encoder
            
        Returns:
            Context usage metrics
        """
        if not context_docs:
            return {}
        
        # Get context vectors
        context_texts = [doc.get("text", "") for doc in context_docs]
        context_vecs = encoder.encode(context_texts)
        
        # Calculate metrics for each context doc
        context_metrics = []
        for i, (doc, vec) in enumerate(zip(context_docs, context_vecs)):
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            
            # Calculate both metrics
            distance = np.linalg.norm(response_vec - vec_norm)
            cosine_sim = np.dot(response_vec, vec_norm)
            
            context_metrics.append({
                "doc_index": i,
                "text_preview": doc.get("text", "")[:100] + "...",
                "distance_to_response": distance,
                "cosine_to_response": cosine_sim,
                "relevance_score": SimilarityConverter.distance_to_relevance(distance)
            })
        
        # Sort by relevance (closest distance)
        context_metrics.sort(key=lambda x: x["distance_to_response"])
        
        # Calculate aggregate metrics
        avg_distance = np.mean([m["distance_to_response"] for m in context_metrics])
        avg_cosine = np.mean([m["cosine_to_response"] for m in context_metrics])
        min_distance = min(m["distance_to_response"] for m in context_metrics)
        max_cosine = max(m["cosine_to_response"] for m in context_metrics)
        
        return {
            "avg_context_distance": avg_distance,
            "avg_context_cosine": avg_cosine,
            "closest_context_distance": min_distance,
            "highest_context_cosine": max_cosine,
            "context_usage_score": SimilarityConverter.distance_to_relevance(min_distance),
            "top_relevant_contexts": context_metrics[:3]  # Top 3
        }
    
    def format_evaluation_summary(self, evaluation: Dict[str, Any]) -> str:
        """
        Format evaluation results for display.
        
        Args:
            evaluation: Evaluation results from evaluate_response
            
        Returns:
            Formatted string
        """
        lines = [
            "Response Evaluation:",
            f"  Query-Response: {evaluation['metrics_formatted']}",
            f"  Relevance Score: {evaluation['query_response_relevance']:.3f}"
        ]
        
        if "context_evaluation" in evaluation:
            ctx = evaluation["context_evaluation"]
            if ctx:
                lines.extend([
                    "",
                    "Context Usage:",
                    f"  Average Distance: {ctx['avg_context_distance']:.3f}",
                    f"  Average Cosine: {ctx['avg_context_cosine']:.3f}",
                    f"  Context Usage Score: {ctx['context_usage_score']:.3f}"
                ])
        
        return "\n".join(lines)