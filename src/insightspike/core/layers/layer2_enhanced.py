"""
Enhanced L2 Memory Manager with Graph-Based Integration and Splitting
====================================================================

Implements dynamic episode management using graph structure from Layer3.
Features:
- Graph-aware episode integration
- Conflict-based automatic splitting
- Self-organizing knowledge structure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import faiss
import numpy as np
import torch

from .layer2_memory_manager import L2MemoryManager, Episode
from ...config import get_config
from ..config import get_config as get_core_config
from ...utils.embedder import get_model

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for graph-aware integration"""

    # Traditional thresholds
    similarity_threshold: float = 0.85
    content_overlap_threshold: float = 0.70
    c_value_diff_threshold: float = 0.30

    # Graph-aware parameters
    graph_weight: float = 0.3
    graph_connection_bonus: float = 0.1
    enable_graph_integration: bool = True


@dataclass
class SplittingConfig:
    """Configuration for conflict-based splitting"""

    conflict_threshold: float = 0.7
    min_connections_for_split: int = 3
    max_episode_length: int = 500
    split_decay_factor: float = 0.8
    enable_auto_split: bool = True
    max_splits_per_episode: int = 3


class EnhancedL2MemoryManager(L2MemoryManager):
    """
    Enhanced memory manager with graph-aware episode management.

    This extends the base L2MemoryManager with:
    - Graph structure awareness from Layer3
    - Intelligent integration based on graph connections
    - Automatic conflict-based splitting
    - Self-organizing knowledge structure
    """

    def __init__(self, dim: int = 384, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)

        # Enhanced configurations
        self.integration_config = IntegrationConfig()
        self.splitting_config = SplittingConfig()

        # Reference to Layer3 graph (will be set externally)
        self.l3_graph = None

        # Statistics tracking
        self.integration_stats = {
            "total_attempts": 0,
            "graph_assisted": 0,
            "successful_integrations": 0,
            "graph_bonus_applied": 0,
        }

        self.splitting_stats = {
            "conflicts_detected": 0,
            "episodes_split": 0,
            "total_new_episodes": 0,
        }

        logger.info("Enhanced L2 Memory Manager initialized with graph awareness")

    def set_layer3_graph(self, l3_graph):
        """Set reference to Layer3 graph reasoner"""
        self.l3_graph = l3_graph
        logger.info("Layer3 graph reference set")

    def add_episode(self, vector: np.ndarray, text: str, c_value: float = 0.2) -> int:
        """
        Enhanced add_episode with graph-aware integration and auto-splitting.

        Args:
            vector: Episode embedding vector
            text: Episode text content
            c_value: Initial C-value

        Returns:
            Index of added/integrated episode
        """
        try:
            # Check for integration with graph awareness
            integration_result = self._check_episode_integration_enhanced(
                vector, text, c_value
            )

            if integration_result["should_integrate"]:
                # Integrate with existing episode
                target_idx = integration_result["target_index"]
                logger.info(f"Graph-aware integration with episode {target_idx}")

                idx = self._integrate_with_existing(target_idx, vector, text, c_value)

                # After integration, check if the episode needs splitting
                self._check_and_split_if_needed(idx)

                return idx
            else:
                # Add as new episode
                episode = Episode(vector, text, c_value)
                self.episodes.append(episode)

                # Re-train index if needed
                if len(self.episodes) >= 2 and not self.is_trained:
                    self._train_index()
                elif self.is_trained:
                    self._add_to_index(episode)

                new_idx = len(self.episodes) - 1

                # Check if any existing episodes need splitting due to new connection
                if self.splitting_config.enable_auto_split:
                    self._check_global_conflicts()

                logger.debug(f"Added new episode node {new_idx}")
                return new_idx

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return -1

    def _check_episode_integration_enhanced(
        self, vector: np.ndarray, text: str, c_value: float
    ) -> Dict[str, Any]:
        """
        Enhanced integration check using graph structure.

        Combines traditional similarity metrics with graph connection information
        for more intelligent integration decisions.
        """
        self.integration_stats["total_attempts"] += 1

        if len(self.episodes) == 0:
            return {
                "should_integrate": False,
                "target_index": -1,
                "reason": "no_existing_episodes",
            }

        # Get traditional integration check result
        traditional_result = self._check_episode_integration(vector, text, c_value)

        # If graph integration is disabled or Layer3 not available, return traditional result
        if (
            not self.integration_config.enable_graph_integration
            or self.l3_graph is None
        ):
            return traditional_result

        # Enhance with graph information
        best_candidate = traditional_result.get("best_candidate", {})
        if not best_candidate or best_candidate.get("index", -1) < 0:
            return traditional_result

        # Get graph connection strength
        graph_connection = self._get_graph_connection_strength(
            len(self.episodes),  # New episode would be at this index
            best_candidate["index"],
        )

        if graph_connection > 0:
            self.integration_stats["graph_assisted"] += 1

        # Combine scores with graph information
        original_score = best_candidate.get("integration_score", 0.0)

        # Dynamic weighting based on graph connection
        graph_weight = self.integration_config.graph_weight
        vector_weight = 1.0 - graph_weight

        combined_score = (
            vector_weight * original_score + graph_weight * graph_connection
        )

        # Adjust threshold if strong graph connection exists
        threshold = self.integration_config.similarity_threshold
        if graph_connection > 0.5:
            threshold -= self.integration_config.graph_connection_bonus
            self.integration_stats["graph_bonus_applied"] += 1

        # Update decision
        should_integrate = (
            combined_score >= threshold
            and best_candidate["content_overlap"]
            >= self.integration_config.content_overlap_threshold
        )

        if should_integrate:
            self.integration_stats["successful_integrations"] += 1

        return {
            "should_integrate": should_integrate,
            "target_index": best_candidate["index"] if should_integrate else -1,
            "best_candidate": best_candidate,
            "graph_connection": graph_connection,
            "combined_score": combined_score,
            "threshold_used": threshold,
            "reason": "graph_enhanced" if graph_connection > 0 else "traditional",
        }

    def _get_graph_connection_strength(self, idx1: int, idx2: int) -> float:
        """
        Get connection strength between two episodes from Layer3 graph.

        Returns:
            Connection strength (0-1), where:
            - 0: No connection
            - 0.4: 2-hop connection
            - 0.8: Direct connection
            - Variable: If edge weights are available
        """
        if self.l3_graph is None or not hasattr(self.l3_graph, "previous_graph"):
            return 0.0

        graph = self.l3_graph.previous_graph
        if graph is None or not hasattr(graph, "edge_index"):
            return 0.0

        edge_index = graph.edge_index

        # Check direct connection
        mask = ((edge_index[0] == idx1) & (edge_index[1] == idx2)) | (
            (edge_index[0] == idx2) & (edge_index[1] == idx1)
        )

        if mask.any():
            # Direct connection found
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                # Use actual edge weight if available
                edge_idx = torch.where(mask)[0][0]
                return float(graph.edge_attr[edge_idx])
            else:
                return 0.8  # Default direct connection strength

        # Check 2-hop connection
        # Get neighbors of both nodes
        neighbors1 = edge_index[1][edge_index[0] == idx1]
        neighbors2 = edge_index[1][edge_index[0] == idx2]

        # Check for common neighbors
        if len(neighbors1) > 0 and len(neighbors2) > 0:
            common = set(neighbors1.tolist()) & set(neighbors2.tolist())
            if common:
                return 0.4  # 2-hop connection strength

        return 0.0  # No connection

    def _check_and_split_if_needed(self, episode_idx: int):
        """
        Check if an episode should be split due to conflicts or size.
        """
        if not self.splitting_config.enable_auto_split:
            return

        if not (0 <= episode_idx < len(self.episodes)):
            return

        episode = self.episodes[episode_idx]

        # Check split conditions
        should_split = False
        split_reason = ""

        # 1. Length-based split
        if len(episode.text) > self.splitting_config.max_episode_length:
            should_split = True
            split_reason = "text_too_long"

        # 2. Conflict-based split
        elif self.l3_graph is not None:
            conflict_score = self._calculate_episode_conflict(episode_idx)
            if conflict_score > self.splitting_config.conflict_threshold:
                should_split = True
                split_reason = f"high_conflict:{conflict_score:.2f}"
                self.splitting_stats["conflicts_detected"] += 1

        if should_split:
            self._perform_episode_split(episode_idx, split_reason)

    def _calculate_episode_conflict(self, episode_idx: int) -> float:
        """
        Calculate conflict score for an episode based on its graph connections.

        High conflict indicates the episode connects disparate concepts.
        """
        if self.l3_graph is None or not hasattr(self.l3_graph, "previous_graph"):
            return 0.0

        graph = self.l3_graph.previous_graph
        if graph is None or not hasattr(graph, "edge_index"):
            return 0.0

        # Get neighbors
        edge_index = graph.edge_index
        neighbors = edge_index[1][edge_index[0] == episode_idx].tolist()

        if len(neighbors) < self.splitting_config.min_connections_for_split:
            return 0.0

        # Calculate pairwise conflicts among neighbors
        conflicts = []

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[i] < len(self.episodes) and neighbors[j] < len(
                    self.episodes
                ):
                    vec1 = self.episodes[neighbors[i]].vec
                    vec2 = self.episodes[neighbors[j]].vec

                    # Conflict = 1 - similarity
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    conflict = 1.0 - similarity
                    conflicts.append(conflict)

        return np.mean(conflicts) if conflicts else 0.0

    def _perform_episode_split(self, episode_idx: int, reason: str):
        """
        Split an episode into multiple coherent parts.
        """
        if not (0 <= episode_idx < len(self.episodes)):
            return

        episode = self.episodes[episode_idx]

        # Simple sentence-based splitting
        sentences = [s.strip() for s in episode.text.split(".") if s.strip()]

        if len(sentences) < 2:
            logger.info(f"Episode {episode_idx} cannot be split (too few sentences)")
            return

        # Determine number of splits
        n_splits = min(
            self.splitting_config.max_splits_per_episode,
            len(sentences),
            3,  # Reasonable maximum
        )

        # Create new episodes
        new_episodes = []
        base_c = episode.c * self.splitting_config.split_decay_factor

        for i in range(n_splits):
            # Distribute sentences
            start_idx = i * len(sentences) // n_splits
            end_idx = (i + 1) * len(sentences) // n_splits
            split_text = ". ".join(sentences[start_idx:end_idx]) + "."

            # Create vector variation
            noise = np.random.normal(0, 0.05, episode.vec.shape)
            split_vec = episode.vec + noise
            split_vec = split_vec / np.linalg.norm(split_vec)

            # Create metadata
            metadata = episode.metadata.copy()
            metadata.update(
                {
                    "split_from": episode_idx,
                    "split_reason": reason,
                    "split_part": i + 1,
                    "split_total": n_splits,
                }
            )

            new_episode = Episode(
                split_vec.astype(np.float32), split_text, base_c, metadata
            )
            new_episodes.append(new_episode)

        # Remove original episode
        del self.episodes[episode_idx]

        # Add new episodes
        for new_ep in new_episodes:
            self.episodes.append(new_ep)

        # Retrain index
        if len(self.episodes) >= 2:
            self._train_index()

        self.splitting_stats["episodes_split"] += 1
        self.splitting_stats["total_new_episodes"] += len(new_episodes)

        logger.info(
            f"Split episode {episode_idx} into {len(new_episodes)} parts (reason: {reason})"
        )

    def _check_global_conflicts(self):
        """
        Check all episodes for potential conflicts after graph update.
        """
        if not self.splitting_config.enable_auto_split:
            return

        # Check each episode for conflicts
        episodes_to_split = []

        for i in range(len(self.episodes)):
            conflict_score = self._calculate_episode_conflict(i)
            if conflict_score > self.splitting_config.conflict_threshold:
                episodes_to_split.append((i, conflict_score))

        # Split episodes with highest conflicts first
        episodes_to_split.sort(key=lambda x: x[1], reverse=True)

        # Process splits (in reverse order to maintain indices)
        for idx, score in reversed(episodes_to_split):
            if idx < len(self.episodes):  # Check if still valid after previous splits
                self._perform_episode_split(idx, f"global_conflict:{score:.2f}")

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get statistics about enhanced episode management"""
        base_stats = self.get_memory_stats()

        base_stats.update(
            {
                "integration_stats": self.integration_stats,
                "splitting_stats": self.splitting_stats,
                "graph_enabled": self.l3_graph is not None,
                "config": {
                    "integration": {
                        "similarity_threshold": self.integration_config.similarity_threshold,
                        "graph_weight": self.integration_config.graph_weight,
                        "graph_bonus": self.integration_config.graph_connection_bonus,
                    },
                    "splitting": {
                        "conflict_threshold": self.splitting_config.conflict_threshold,
                        "auto_split_enabled": self.splitting_config.enable_auto_split,
                        "max_length": self.splitting_config.max_episode_length,
                    },
                },
            }
        )

        # Calculate rates
        if self.integration_stats["total_attempts"] > 0:
            base_stats["integration_rate"] = (
                self.integration_stats["successful_integrations"]
                / self.integration_stats["total_attempts"]
            )
            base_stats["graph_assist_rate"] = (
                self.integration_stats["graph_assisted"]
                / self.integration_stats["total_attempts"]
            )

        return base_stats


# Convenience function to upgrade existing L2MemoryManager
def upgrade_to_enhanced(existing_manager: L2MemoryManager) -> EnhancedL2MemoryManager:
    """
    Upgrade an existing L2MemoryManager to enhanced version.

    Args:
        existing_manager: Current L2MemoryManager instance

    Returns:
        EnhancedL2MemoryManager with copied episodes
    """
    enhanced = EnhancedL2MemoryManager(dim=existing_manager.dim)

    # Copy episodes
    enhanced.episodes = existing_manager.episodes.copy()

    # Copy index state
    enhanced.is_trained = existing_manager.is_trained
    if hasattr(existing_manager, "index"):
        enhanced.index = existing_manager.index

    # Copy configurations
    enhanced.config = existing_manager.config
    enhanced.c_min = existing_manager.c_min
    enhanced.c_max = existing_manager.c_max
    if hasattr(existing_manager, "nlist"):
        enhanced.nlist = existing_manager.nlist
    if hasattr(existing_manager, "nprobe"):
        enhanced.nprobe = existing_manager.nprobe

    logger.info(
        f"Upgraded to enhanced memory manager with {len(enhanced.episodes)} episodes"
    )

    return enhanced
