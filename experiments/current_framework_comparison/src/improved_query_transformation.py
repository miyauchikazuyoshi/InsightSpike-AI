"""
Improved Query Transformation with LLM-based reformulation
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ImprovedQueryTransformer:
    """
    Enhanced query transformer with better reformulation strategies
    """
    
    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        
    def generate_transformed_query_with_llm(
        self,
        original_query: str,
        discovered_concepts: List[str],
        connections: List[str],
        insights: List[str]
    ) -> str:
        """Use LLM to generate natural query transformation"""
        
        if not self.llm:
            # Fallback to rule-based
            return self._rule_based_transformation(original_query, discovered_concepts)
        
        prompt = f"""
You are helping refine a query based on discovered knowledge connections.

Original Query: {original_query}

Discovered Concepts:
{chr(10).join(f'- {c}' for c in discovered_concepts[:5])}

Key Connections:
{chr(10).join(f'- {c}' for c in connections[:3])}

Insights:
{chr(10).join(f'- {i}' for i in insights[:2])}

Generate a refined query that:
1. Maintains the original intent
2. Incorporates the discovered connections
3. Is more specific and targeted
4. Leads to deeper understanding

Refined Query:"""
        
        try:
            refined = self.llm.generate_response(prompt, "")
            return refined.strip()
        except:
            return self._rule_based_transformation(original_query, discovered_concepts)
    
    def _rule_based_transformation(self, query: str, concepts: List[str]) -> str:
        """Rule-based query transformation"""
        
        # Extract key question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        query_lower = query.lower()
        
        question_type = None
        for qw in question_words:
            if qw in query_lower:
                question_type = qw
                break
        
        # Transform based on question type and concepts
        if question_type == 'what' and len(concepts) >= 2:
            # What -> How (deeper understanding)
            return f"How does {concepts[0]} relate to {concepts[1]}?"
        
        elif question_type == 'how' and len(concepts) >= 1:
            # How -> Why (causal understanding)
            return f"Why does {concepts[0]} work this way, and what are the implications?"
        
        elif question_type == 'why' and len(concepts) >= 2:
            # Why -> What if (hypothetical reasoning)
            return f"What would happen if {concepts[0]} didn't involve {concepts[1]}?"
        
        else:
            # General expansion
            if concepts:
                return f"{query.rstrip('?')} in relation to {', '.join(concepts[:2])}?"
            return query


class GraphAwareQueryTransformer:
    """
    Query transformer that uses graph structure for better transformations
    """
    
    def __init__(self, base_transformer):
        self.base_transformer = base_transformer
        
    def transform_with_graph_context(
        self,
        query: str,
        graph_metrics: Dict[str, float],
        node_activations: Dict[int, float],
        edge_weights: Optional[np.ndarray] = None
    ) -> str:
        """Transform query using graph structural information"""
        
        # Identify highly activated nodes
        if node_activations:
            top_nodes = sorted(node_activations.items(), key=lambda x: x[1], reverse=True)[:5]
            activated_concepts = [f"node_{idx}" for idx, score in top_nodes if score > 0.7]
        else:
            activated_concepts = []
        
        # Check for insight patterns
        insights = []
        if graph_metrics.get('delta_ged', 0) < -0.2:
            insights.append("Graph structure simplified - fundamental connection discovered")
        
        if graph_metrics.get('delta_ig', 0) > 0.3:
            insights.append("High information gain - new knowledge integrated")
        
        # Detect connection patterns
        connections = []
        if edge_weights is not None and len(edge_weights) > 0:
            strong_connections = np.where(edge_weights > 0.8)[0]
            if len(strong_connections) > 0:
                connections.append("Strong conceptual links detected")
        
        # Generate transformed query
        return self.base_transformer.generate_transformed_query_with_llm(
            query,
            activated_concepts,
            connections,
            insights
        )


@dataclass
class QueryTransformationStrategy:
    """Defines how queries should be transformed based on context"""
    
    name: str
    condition: callable  # Function that returns True if strategy applies
    transform: callable  # Function that transforms the query
    
    def applies(self, context: Dict[str, Any]) -> bool:
        """Check if this strategy applies to current context"""
        return self.condition(context)
    
    def apply(self, query: str, context: Dict[str, Any]) -> str:
        """Apply transformation strategy"""
        return self.transform(query, context)


# Example strategies
TRANSFORMATION_STRATEGIES = [
    QueryTransformationStrategy(
        name="exploration_to_specific",
        condition=lambda ctx: ctx.get('stage') == 'exploring' and ctx.get('confidence', 0) < 0.5,
        transform=lambda q, ctx: f"{q.rstrip('?')} specifically regarding {ctx.get('top_concept', 'the main topic')}?"
    ),
    
    QueryTransformationStrategy(
        name="insight_integration",
        condition=lambda ctx: ctx.get('spike_detected', False),
        transform=lambda q, ctx: f"Given that {ctx.get('insight', 'a new connection was found')}, {q.lower()}"
    ),
    
    QueryTransformationStrategy(
        name="convergence_refinement",
        condition=lambda ctx: ctx.get('confidence', 0) > 0.8,
        transform=lambda q, ctx: f"What are the deeper implications of {q.rstrip('?').lower()}?"
    )
]


def select_transformation_strategy(context: Dict[str, Any]) -> Optional[QueryTransformationStrategy]:
    """Select the best transformation strategy based on context"""
    for strategy in TRANSFORMATION_STRATEGIES:
        if strategy.applies(context):
            return strategy
    return None