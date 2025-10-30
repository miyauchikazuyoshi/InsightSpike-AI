"""
Integrated Memory Manager - Orchestrates Evaluation and Confidence Updates
=========================================================================

Combines evaluation, confidence management, and memory operations.
Demonstrates proper separation of concerns.
"""

import logging
from typing import List, Optional, Tuple

from ..confidence.confidence_manager import ConfidenceManager
from ..core.episode import Episode
from ..evaluation.gedig_evaluator import GeDIGEvaluator
from ..generation.insight_generator import InsightGenerator
from ..memory.confidence_pruning import ConfidenceBasedPruning
from ..search.similarity_search import SimilaritySearch

logger = logging.getLogger(__name__)


class IntegratedMemoryManager:
    """
    High-level memory manager that orchestrates different components.
    
    Responsibilities:
    - Coordinate search, evaluation, and generation
    - Manage confidence updates through ConfidenceManager
    - Handle memory pruning
    - Provide unified interface for memory operations
    """
    
    def __init__(
        self,
        weight_manager=None,
        gedig_k: float = 0.5,
        max_episodes: int = 10000
    ):
        """
        Initialize integrated memory manager.
        
        Args:
            weight_manager: Optional weight vector manager
            gedig_k: Balance coefficient for geDIG evaluation
            max_episodes: Maximum number of episodes to keep
        """
        # Core components
        self.searcher = SimilaritySearch(weight_manager=weight_manager)
        self.evaluator = GeDIGEvaluator(k=gedig_k)
        self.generator = InsightGenerator()
        
        # Confidence and memory management
        self.confidence_manager = ConfidenceManager()
        self.pruner = ConfidenceBasedPruning(max_episodes=max_episodes)
        
        # Episode storage
        self.episodes: List[Episode] = []
    
    def process_query(
        self,
        query_vec,
        k_search: int = 30,
        generate_insight: bool = True
    ) -> Tuple[List[Episode], Optional[Episode]]:
        """
        Process a query through the full pipeline.
        
        1. Search for similar episodes
        2. Evaluate and rerank using geDIG
        3. Update confidence values
        4. Optionally generate new insight
        5. Prune if needed
        
        Args:
            query_vec: Query vector
            k_search: Number of candidates to retrieve
            generate_insight: Whether to generate new insight
            
        Returns:
            Tuple of (selected_episodes, new_insight)
        """
        # Step 1: Search (no C-values used)
        candidates = self.searcher.search(query_vec, self.episodes, k=k_search)
        
        if not candidates:
            logger.warning("No candidates found for query")
            return [], None
        
        # Step 2: Evaluate and rerank (pure evaluation)
        selected, all_candidates = self.evaluator.rerank(candidates)
        
        # Step 3: Update confidence (separated concern)
        self.confidence_manager.update_after_selection(selected, all_candidates)
        
        # Step 4: Generate insight if requested
        new_insight = None
        if generate_insight and len(selected) >= 2:
            new_insight = self.generator.generate(selected)
            self.add_episode(new_insight)
        
        # Step 5: Prune if needed
        self._prune_if_needed()
        
        return selected, new_insight
    
    def add_episode(self, episode: Episode) -> None:
        """
        Add new episode to memory.
        
        Args:
            episode: Episode to add
        """
        self.episodes.append(episode)
        logger.debug(
            f"Added {episode.episode_type} with C={episode.c:.2f}: "
            f"{episode.text[:50]}"
        )
    
    def add_experience(
        self,
        text: str,
        vec,
        initial_confidence: float = 0.5
    ) -> Episode:
        """
        Add new experience episode.
        
        Args:
            text: Experience description
            vec: Vector representation
            initial_confidence: Initial C-value
            
        Returns:
            Created episode
        """
        episode = Episode(
            text=text,
            vec=vec,
            c=initial_confidence,
            episode_type="experience"
        )
        self.add_episode(episode)
        return episode
    
    def evaluate_and_select(
        self,
        candidates: List[Episode]
    ) -> List[Episode]:
        """
        Evaluate candidates and update confidence.
        
        Demonstrates clean separation: evaluation then update.
        
        Args:
            candidates: Candidate episodes
            
        Returns:
            Selected episodes
        """
        # Pure evaluation
        selected, all_candidates = self.evaluator.rerank(candidates)
        
        # Separate confidence update
        self.confidence_manager.update_after_selection(selected, all_candidates)
        
        return selected
    
    def _prune_if_needed(self) -> None:
        """
        Prune memory if approaching limits.
        """
        if len(self.episodes) > self.pruner.max_episodes * 0.9:
            self.episodes = self.pruner.adaptive_threshold(self.episodes)
            logger.info(f"Pruned to {len(self.episodes)} episodes")
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_episodes": len(self.episodes),
            "confidence_stats": self.confidence_manager.get_confidence_stats(self.episodes),
            "pruning_stats": self.pruner.get_statistics(self.episodes),
            "type_distribution": {
                "experiences": sum(1 for ep in self.episodes if ep.episode_type == "experience"),
                "insights": sum(1 for ep in self.episodes if ep.episode_type == "insight")
            }
        }
    
    def reset(self) -> None:
        """
        Reset memory manager to initial state.
        """
        self.episodes.clear()
        logger.info("Memory manager reset")