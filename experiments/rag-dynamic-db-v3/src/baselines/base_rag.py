"""Abstract base class for all RAG systems in geDIG-RAG v3."""

from __future__ import annotations

import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..core.knowledge_graph import KnowledgeGraph, KnowledgeNode
from ..core.gedig_evaluator import GraphUpdate, UpdateType, GeDIGResult
from ..core.config import ExperimentConfig


@dataclass
class RetrievalResult:
    """Result of knowledge retrieval."""
    retrieved_nodes: List[Tuple[str, float]]  # (node_id, score)
    context_texts: List[str]
    retrieval_time: float
    stats: Dict[str, Any]


@dataclass  
class UpdateDecision:
    """Decision result for knowledge update."""
    should_update: bool
    update_type: UpdateType
    reason: str
    confidence: float = 0.0
    update: Optional[GraphUpdate] = None
    gedig_result: Optional[GeDIGResult] = None
    metadata: Optional[Dict] = None


@dataclass
class RAGResponse:
    """Response from RAG system with detailed logging."""
    query: str
    response: str = ""
    query_id: Optional[str] = None
    session_id: Optional[int] = None
    
    # Retrieval information
    retrieval_result: Optional[RetrievalResult] = None
    
    # Update information  
    update_decision: Optional[UpdateDecision] = None
    knowledge_updated: bool = False
    
    # Performance tracking
    total_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    update_time: float = 0.0
    
    # Graph state
    graph_size_before: int = 0
    graph_size_after: int = 0
    
    # Metadata for analysis
    method_name: str = ""
    ground_truth: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query_id': self.query_id,
            'session_id': self.session_id,
            'query': self.query,
            'response': self.response,
            'method_name': self.method_name,
            'knowledge_updated': self.knowledge_updated,
            'total_time': self.total_time,
            'retrieval_time': self.retrieval_time,
            'generation_time': self.generation_time,
            'update_time': self.update_time,
            'graph_size_before': self.graph_size_before,
            'graph_size_after': self.graph_size_after,
            'update_reason': self.update_decision.reason if self.update_decision else None,
            'update_confidence': self.update_decision.confidence if self.update_decision else 0.0,
            'metadata': self.metadata
        }


class DummyEmbedder:
    """Placeholder embedder for development."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate dummy embeddings."""
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        return np.array(embeddings)


class DummyGenerator:
    """Placeholder text generator for development."""
    
    def generate(self, query: str, context: str) -> str:
        """Generate simple response based on query and context."""
        if not context.strip():
            return f"I don't have enough information to answer '{query}'."
        
        # Simple template-based response
        context_snippets = context.split('\n')[:3]  # Take first 3 context pieces
        
        response_parts = [f"Based on the available information:"]
        for i, snippet in enumerate(context_snippets):
            if snippet.strip():
                response_parts.append(f"{i+1}. {snippet.strip()}")
        
        if len(response_parts) == 1:
            return f"I found some relevant information but cannot provide a specific answer to '{query}'."
        
        response_parts.append(f"This addresses your question about {query.lower()}.")
        return " ".join(response_parts)


class BaseRAGSystem(ABC):
    """Abstract base class for all RAG systems."""
    
    def __init__(self, config: ExperimentConfig, method_name: str):
        """Initialize RAG system.
        
        Args:
            config: Experiment configuration
            method_name: Name identifying this method
        """
        self.config = config
        self.method_name = method_name
        
        # Core components
        self.knowledge_graph = KnowledgeGraph(
            embedding_dim=384  # Default sentence-transformer dimension
        )
        
        # Placeholder components (to be replaced with real implementations)
        self.embedder = DummyEmbedder(embedding_dim=384)
        self.generator = DummyGenerator()
        
        # Performance tracking
        self.query_count = 0
        self.update_count = 0
        self.total_processing_time = 0.0
        
        # Response history for analysis
        self.response_history: List[RAGResponse] = []
    
    def add_initial_knowledge(self, documents: List[str]) -> int:
        """Add initial documents to knowledge base.
        
        Args:
            documents: List of document texts
            
        Returns:
            Number of nodes added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents)
        
        nodes_added = 0
        for i, (text, embedding) in enumerate(zip(documents, embeddings)):
            node_id = self.knowledge_graph.add_node(
                text=text,
                embedding=embedding,
                node_type="fact",
                confidence=1.0,
                metadata={'source': 'initial_documents', 'doc_index': i}
            )
            
            if node_id:
                nodes_added += 1
        
        return nodes_added
    
    def process_query(self, 
                     query: str, 
                     query_id: Optional[str] = None,
                     session_id: Optional[int] = None) -> RAGResponse:
        """Process a query through the complete RAG pipeline.
        
        Args:
            query: User query
            query_id: Optional query identifier
            session_id: Optional session identifier
            
        Returns:
            RAG response with detailed logging
        """
        start_time = time.time()
        
        # Initialize response
        response_obj = RAGResponse(
            query=query,
            query_id=query_id,
            session_id=session_id,
            method_name=self.method_name,
            graph_size_before=len(self.knowledge_graph.nodes)
        )
        
        try:
            # Step 1: Retrieve relevant context
            retrieval_start = time.time()
            retrieval_result = self.retrieve_context(query)
            response_obj.retrieval_result = retrieval_result
            response_obj.retrieval_time = time.time() - retrieval_start
            
            # Step 2: Generate response
            generation_start = time.time()
            context_text = "\n".join(retrieval_result.context_texts)
            generated_response = self.generator.generate(query, context_text)
            response_obj.response = generated_response
            response_obj.generation_time = time.time() - generation_start
            
            # Step 3: Decide on knowledge update
            update_start = time.time()
            update_decision = self.should_update_knowledge(
                query, generated_response, retrieval_result
            )
            response_obj.update_decision = update_decision
            
            # Step 4: Apply update if decided
            if update_decision.should_update and update_decision.update:
                success = self.apply_update(update_decision.update)
                response_obj.knowledge_updated = success
                if success:
                    self.update_count += 1
            
            response_obj.update_time = time.time() - update_start
            response_obj.graph_size_after = len(self.knowledge_graph.nodes)
            
        except Exception as e:
            # Handle errors gracefully
            response_obj.response = f"Error processing query: {str(e)}"
            response_obj.metadata = {'error': str(e)}
        
        # Finalize timing and statistics
        response_obj.total_time = time.time() - start_time
        self.total_processing_time += response_obj.total_time
        self.query_count += 1
        
        # Store for analysis
        self.response_history.append(response_obj)
        
        return response_obj
    
    def retrieve_context(self, query: str) -> RetrievalResult:
        """Retrieve relevant context from knowledge graph.
        
        Args:
            query: User query
            
        Returns:
            Retrieval result with context and statistics
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Find similar nodes
        similar_nodes = self.knowledge_graph.find_similar_nodes(
            query_embedding=query_embedding,
            k=self.config.top_k_values[0] if self.config.top_k_values else 5,
            min_similarity=0.1  # Low threshold to get some results
        )
        
        # Extract context texts
        context_texts = []
        for node_id, similarity in similar_nodes:
            if node_id in self.knowledge_graph.nodes:
                node_text = self.knowledge_graph.nodes[node_id].text
                context_texts.append(node_text)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            retrieved_nodes=similar_nodes,
            context_texts=context_texts,
            retrieval_time=retrieval_time,
            stats={
                'n_retrieved': len(similar_nodes),
                'avg_similarity': np.mean([sim for _, sim in similar_nodes]) if similar_nodes else 0.0,
                'max_similarity': max([sim for _, sim in similar_nodes]) if similar_nodes else 0.0
            }
        )
    
    @abstractmethod
    def should_update_knowledge(self, 
                               query: str, 
                               response: str,
                               retrieval_result: RetrievalResult) -> UpdateDecision:
        """Decide whether to update knowledge based on query/response.
        
        This is the key method that differentiates the various RAG approaches.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Result from knowledge retrieval
            
        Returns:
            Update decision with reasoning
        """
        pass
    
    def apply_update(self, update: GraphUpdate) -> bool:
        """Apply a knowledge graph update.
        
        Args:
            update: Graph update to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if update.update_type == UpdateType.ADD:
                return self._apply_addition_update(update)
            elif update.update_type == UpdateType.PRUNE:
                return self._apply_pruning_update(update)
            elif update.update_type == UpdateType.MERGE:
                return self._apply_merge_update(update)
            else:
                return False
        except Exception as e:
            print(f"Failed to apply update: {e}")
            return False
    
    def _apply_addition_update(self, update: GraphUpdate) -> bool:
        """Apply node addition update."""
        if not update.new_node_data:
            return False
        
        node_data = update.new_node_data
        
        # Add new node
        node_id = self.knowledge_graph.add_node(
            text=node_data.get('text', ''),
            embedding=node_data.get('embedding'),
            node_type=node_data.get('node_type', 'fact'),
            confidence=node_data.get('confidence', 1.0),
            metadata=node_data.get('metadata', {})
        )
        
        # Add edges if specified
        if update.new_edges:
            for source, target, edge_data in update.new_edges:
                # Replace placeholder IDs with actual node ID
                actual_source = node_id if source == node_data.get('id') else source
                actual_target = node_id if target == node_data.get('id') else target
                
                self.knowledge_graph.add_edge(
                    source=actual_source,
                    target=actual_target,
                    **edge_data
                )
        
        return True
    
    def _apply_pruning_update(self, update: GraphUpdate) -> bool:
        """Apply node pruning update."""
        if not update.remove_nodes:
            return False
        
        for node_id in update.remove_nodes:
            self.knowledge_graph.remove_node(node_id)
        
        return True
    
    def _apply_merge_update(self, update: GraphUpdate) -> bool:
        """Apply node merging update."""
        # Remove old nodes
        if update.remove_nodes:
            for node_id in update.remove_nodes:
                self.knowledge_graph.remove_node(node_id)
        
        # Add merged node
        if update.new_node_data:
            self.knowledge_graph.add_node(
                text=update.new_node_data['text'],
                embedding=update.new_node_data['embedding'],
                node_type=update.new_node_data.get('node_type', 'summary'),
                confidence=update.new_node_data.get('confidence', 1.0)
            )
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics.
        
        Returns:
            Dictionary of performance and usage statistics
        """
        graph_stats = self.knowledge_graph.get_statistics()
        
        # Processing statistics
        avg_response_time = (self.total_processing_time / self.query_count 
                           if self.query_count > 0 else 0.0)
        
        # Update statistics
        update_rate = self.update_count / self.query_count if self.query_count > 0 else 0.0
        
        return {
            'method_name': self.method_name,
            'queries_processed': self.query_count,
            'updates_applied': self.update_count,
            'update_rate': update_rate,
            'avg_response_time': avg_response_time,
            'total_processing_time': self.total_processing_time,
            'graph_statistics': graph_stats,
            'response_history_length': len(self.response_history)
        }
    
    def reset(self):
        """Reset the system to initial state."""
        self.knowledge_graph = KnowledgeGraph(embedding_dim=384)
        self.query_count = 0
        self.update_count = 0
        self.total_processing_time = 0.0
        self.response_history.clear()
    
    def save_state(self, file_path: str):
        """Save system state to file."""
        import json
        from pathlib import Path
        
        state_data = {
            'method_name': self.method_name,
            'statistics': self.get_statistics(),
            'response_history': [resp.to_dict() for resp in self.response_history]
        }
        
        # Save graph separately
        graph_path = Path(file_path).with_suffix('.graph.json')
        self.knowledge_graph.save_to_file(graph_path)
        
        # Save state
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, file_path: str):
        """Load system state from file."""
        import json
        from pathlib import Path
        
        # Load graph
        graph_path = Path(file_path).with_suffix('.graph.json')
        if graph_path.exists():
            self.knowledge_graph.load_from_file(graph_path)
        
        # Load state data
        with open(file_path, 'r') as f:
            state_data = json.load(f)
        
        # Restore basic statistics
        stats = state_data.get('statistics', {})
        self.query_count = stats.get('queries_processed', 0)
        self.update_count = stats.get('updates_applied', 0)
        self.total_processing_time = stats.get('total_processing_time', 0.0)
