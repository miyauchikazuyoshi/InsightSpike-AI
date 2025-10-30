"""Static RAG baseline - no knowledge updates."""

from .base_rag import BaseRAGSystem, UpdateDecision, RetrievalResult
from ..core.gedig_evaluator import UpdateType
from ..core.config import ExperimentConfig


class StaticRAG(BaseRAGSystem):
    """Static RAG system that never updates its knowledge base.
    
    This serves as the primary baseline, representing traditional RAG systems
    that maintain a fixed knowledge base without any learning or adaptation.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Static RAG system.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, method_name="static")
    
    def should_update_knowledge(self, 
                               query: str, 
                               response: str,
                               retrieval_result: RetrievalResult) -> UpdateDecision:
        """Static RAG never updates knowledge.
        
        Args:
            query: User query
            response: Generated response  
            retrieval_result: Result from knowledge retrieval
            
        Returns:
            Update decision (always False for static system)
        """
        return UpdateDecision(
            should_update=False,
            update_type=UpdateType.ADD,  # Placeholder, not used
            reason="Static RAG: Knowledge updates disabled",
            confidence=1.0,
            metadata={
                'system_type': 'static',
                'retrieval_quality': retrieval_result.stats.get('avg_similarity', 0.0),
                'n_retrieved': retrieval_result.stats.get('n_retrieved', 0)
            }
        )