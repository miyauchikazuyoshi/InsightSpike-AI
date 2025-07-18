"""
Query Transformation Module
===========================

Implements query transformation through message passing on knowledge graphs.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class QueryState:
    """Represents the state of a query during transformation"""
    
    text: str  # Current query text
    embedding: Optional[np.ndarray] = None  # Query embedding
    confidence: float = 0.5  # Confidence in current formulation
    stage: str = "initial"  # initial, exploring, refined, insight, complete
    insights: List[str] = field(default_factory=list)
    node_positions: Optional[Dict[int, float]] = None  # Node activation strengths
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "stage": self.stage,
            "insights": self.insights,
            "node_positions": self.node_positions
        }
    
    def add_insight(self, insight: str):
        """Add a new insight discovered during transformation"""
        self.insights.append(insight)
        logger.info(f"New insight: {insight}")


@dataclass
class QueryTransformationHistory:
    """Tracks the evolution of a query through transformation cycles"""
    
    initial_query: str
    states: List[QueryState] = field(default_factory=list)
    
    def add_state(self, state: QueryState):
        """Add a new state to the history"""
        self.states.append(state)
    
    def get_current_state(self) -> Optional[QueryState]:
        """Get the most recent state"""
        return self.states[-1] if self.states else None
    
    def get_total_insights(self) -> List[str]:
        """Get all insights discovered"""
        insights = []
        for state in self.states:
            insights.extend(state.insights)
        return insights
    
    def get_confidence_trajectory(self) -> List[float]:
        """Get confidence values over time"""
        return [state.confidence for state in self.states]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_query": self.initial_query,
            "states": [state.to_dict() for state in self.states],
            "total_transformations": len(self.states) - 1
        }


class QueryTransformer:
    """
    Transforms queries through message passing on knowledge graphs.
    
    Key mechanisms:
    1. Query embedding and graph placement
    2. Message passing between query and knowledge nodes
    3. Query refinement based on activated knowledge
    4. Insight detection through convergence patterns
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", use_gnn: bool = True):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_gnn = use_gnn
        
        # Message passing parameters
        self.num_propagation_steps = 3
        self.attention_temperature = 0.5
        self.transformation_threshold = 0.3
        
    def place_query_on_graph(self, query: str, knowledge_graph: Optional[Any] = None) -> QueryState:
        """Place the initial query on the knowledge graph"""
        
        # Embed the query
        embedding = self.embedding_model.encode(query)
        
        # Initialize query state
        state = QueryState(
            text=query,
            embedding=embedding,
            confidence=0.5,
            stage="initial"
        )
        
        # If we have a knowledge graph, compute initial activations
        if knowledge_graph is not None and hasattr(knowledge_graph, 'x'):
            node_positions = self._compute_initial_activations(embedding, knowledge_graph)
            state.node_positions = node_positions
        
        return state
    
    def transform_query(
        self, 
        current_state: QueryState, 
        knowledge_graph: Any,
        retrieved_documents: List[Dict[str, Any]]
    ) -> QueryState:
        """Transform query through message passing"""
        
        logger.info(f"Transforming query: {current_state.text}")
        
        # Extract embeddings from documents
        doc_embeddings = self._extract_document_embeddings(retrieved_documents)
        
        if self.use_gnn and hasattr(knowledge_graph, 'edge_index'):
            # Use GNN-based message passing
            new_embedding = self._gnn_message_passing(
                current_state.embedding,
                doc_embeddings,
                knowledge_graph
            )
        else:
            # Use attention-based message passing
            new_embedding = self._attention_message_passing(
                current_state.embedding,
                doc_embeddings
            )
        
        # Compute transformation strength
        similarity = np.dot(current_state.embedding, new_embedding) / (
            np.linalg.norm(current_state.embedding) * np.linalg.norm(new_embedding)
        )
        
        # Decide if transformation is significant
        if 1 - similarity > self.transformation_threshold:
            # Generate new query text
            new_query_text = self._generate_transformed_query(
                current_state.text,
                new_embedding,
                retrieved_documents
            )
            
            # Create new state
            new_state = QueryState(
                text=new_query_text,
                embedding=new_embedding,
                confidence=min(1.0, current_state.confidence + 0.1),
                stage="exploring" if current_state.stage == "initial" else "refined"
            )
            
            # Detect insights
            if similarity < 0.7:  # Major transformation
                new_state.add_insight(f"Query evolved from '{current_state.text}' to '{new_query_text}'")
                new_state.stage = "insight"
                
        else:
            # Minor update
            new_state = QueryState(
                text=current_state.text,
                embedding=new_embedding,
                confidence=min(1.0, current_state.confidence + 0.05),
                stage=current_state.stage
            )
        
        # Check for convergence
        if new_state.confidence > 0.85:
            new_state.stage = "complete"
        
        return new_state
    
    def _compute_initial_activations(self, query_embedding: np.ndarray, knowledge_graph: Any) -> Dict[int, float]:
        """Compute initial node activations based on query similarity"""
        activations = {}
        
        if hasattr(knowledge_graph, 'x'):
            node_features = knowledge_graph.x.numpy() if torch.is_tensor(knowledge_graph.x) else knowledge_graph.x
            
            for i, node_feat in enumerate(node_features):
                similarity = np.dot(query_embedding, node_feat) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node_feat) + 1e-8
                )
                activations[i] = float(similarity)
        
        return activations
    
    def _extract_document_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Extract embeddings from retrieved documents"""
        embeddings = []
        
        for doc in documents:
            if 'embedding' in doc and doc['embedding'] is not None:
                embeddings.append(doc['embedding'])
            else:
                # Encode on the fly if needed
                text = doc.get('text', str(doc))
                embedding = self.embedding_model.encode(text)
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _attention_message_passing(
        self, 
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """Simple attention-based message passing"""
        
        # Compute attention weights
        scores = np.dot(doc_embeddings, query_embedding)
        weights = np.exp(scores / self.attention_temperature)
        weights = weights / np.sum(weights)
        
        # Weighted combination
        new_embedding = query_embedding.copy()
        for step in range(self.num_propagation_steps):
            # Message: weighted sum of document embeddings
            message = np.sum(doc_embeddings * weights[:, np.newaxis], axis=0)
            
            # Update: blend with current embedding
            new_embedding = 0.7 * new_embedding + 0.3 * message
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
        
        return new_embedding
    
    def _gnn_message_passing(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        knowledge_graph: Any
    ) -> np.ndarray:
        """GNN-based message passing (simplified without full GNN library)"""
        
        # For now, fall back to attention-based
        # TODO: Implement proper GNN message passing with PyTorch Geometric
        logger.warning("GNN message passing not fully implemented, using attention-based")
        return self._attention_message_passing(query_embedding, doc_embeddings)
    
    def _generate_transformed_query(
        self,
        original_query: str,
        new_embedding: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> str:
        """Generate new query text based on transformed embedding"""
        
        # Find most relevant phrases from documents
        relevant_phrases = []
        for doc in documents[:3]:  # Top 3 documents
            text = doc.get('text', '')
            # Simple keyword extraction (could be improved)
            words = text.split()
            key_phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            relevant_phrases.extend(key_phrases[:2])
        
        # Combine with original query
        # This is a simplified version - could use LLM for better reformulation
        if relevant_phrases:
            expanded_query = f"{original_query} (related to: {', '.join(relevant_phrases[:3])})"
        else:
            expanded_query = original_query
        
        return expanded_query