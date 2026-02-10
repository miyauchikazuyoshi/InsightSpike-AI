"""Three-layer search engine: unified L0 → L1 → L2 controller.

L0: VectorHashIndex   — O(1) revisit detection
L1: AttentionGraphWalker — O(degree) attention-filtered graph walk
L2: Full memory search — O(N log N) weighted distance sort (existing logic)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from .hash_index import VectorHashIndex
from .graph_walker import AttentionGraphWalker

Node = Tuple[int, int, int]


@dataclass
class SearchResult:
    """Result from the three-layer search engine."""

    candidates: List[Dict]
    layer_used: int  # 0, 1, or 2
    is_revisit: bool
    revisit_similarity: float = 0.0
    search_time_ms: float = 0.0


class ThreeLayerSearchEngine:
    """Unified three-layer search controller.

    Routes queries through L0 → L1 → L2, stopping at the first
    layer that provides sufficient candidates.
    """

    def __init__(
        self,
        *,
        hash_resolution: float = 0.05,
        theta_revisit: float = 0.95,
        theta_attention: float = 0.3,
        attention_alpha: float = 0.5,
        dg_gate_tau: float = 1.0,
        tau_dg_3att: float = 0.3,
        tau_reward: float = 0.3,
        score_mode: str = "legacy",
        weight_vector: np.ndarray,
        top_k: int = 32,
        min_layer1_candidates: int = 2,
    ):
        self.hash_index = VectorHashIndex(resolution=hash_resolution)
        self.graph_walker = AttentionGraphWalker(
            theta=theta_attention, alpha=attention_alpha,
            dg_gate_tau=dg_gate_tau,
            tau_dg_3att=tau_dg_3att, tau_reward=tau_reward,
            score_mode=score_mode,
        )
        self.theta_revisit = theta_revisit
        self.min_layer1 = min_layer1_candidates
        self.weight_vector = weight_vector
        self.top_k = top_k
        self._stats: Dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "total": 0}

    def search(
        self,
        query_vector: np.ndarray,
        graph: nx.Graph,
        memory_pool: List[Dict] | None = None,
    ) -> SearchResult:
        """Execute three-layer search.

        Args:
            query_vector: Current observation vector (raw, for L0 hash).
            graph: The knowledge graph (for L1 walk).
            memory_pool: Full memory pool (for L2 fallback). If None,
                         L2 returns empty candidates.

        Returns:
            SearchResult with candidates and metadata.
        """
        t0 = time.monotonic()
        self._stats["total"] += 1

        # --- Layer 0: Hash lookup ---
        revisit = self.hash_index.lookup(query_vector, self.theta_revisit)

        if revisit:
            self._stats["L0"] += 1

            # --- Layer 1: Graph walk ---
            cands = self.graph_walker.get_candidates(
                graph, revisit, query_vector, self.weight_vector
            )
            if len(cands) >= self.min_layer1:
                self._stats["L1"] += 1
                return SearchResult(
                    candidates=cands,
                    layer_used=1,
                    is_revisit=True,
                    revisit_similarity=revisit[0][1],
                    search_time_ms=(time.monotonic() - t0) * 1000,
                )

        # --- Layer 2: Full search (handled by caller) ---
        self._stats["L2"] += 1
        return SearchResult(
            candidates=[],
            layer_used=2,
            is_revisit=bool(revisit),
            revisit_similarity=revisit[0][1] if revisit else 0.0,
            search_time_ms=(time.monotonic() - t0) * 1000,
        )

    def register(self, node_id: Node, raw_vector: np.ndarray) -> None:
        """Register a node in the hash index after commit."""
        self.hash_index.add(node_id, raw_vector)

    def get_stats(self) -> Dict[str, Any]:
        """Return search statistics."""
        t = max(1, self._stats["total"])
        return {
            **self._stats,
            "L1_skip_rate": self._stats["L1"] / t,
        }
