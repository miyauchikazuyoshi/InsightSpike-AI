"""
Enhanced Query Transformer with Advanced Features
===============================================

Phase 2 implementation with multi-hop reasoning, adaptive exploration,
and query branching capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .query_state import QueryState
from .query_transformer import QueryGraphGNN, QueryTransformer

logger = logging.getLogger(__name__)


@dataclass
class QueryBranch:
    """Represents a branch in query exploration"""

    branch_id: str
    parent_state: QueryState
    current_state: QueryState
    exploration_direction: str  # e.g., "theoretical", "practical", "historical"
    priority: float = 0.5
    discovered_paths: List[List[str]] = field(default_factory=list)

    def merge_insights(self, other_branch: "QueryBranch") -> List[str]:
        """Merge insights from another branch"""
        combined_insights = []

        # Combine unique insights
        all_insights = set(
            self.current_state.insights + other_branch.current_state.insights
        )

        # Synthesize new insights from combination
        if self.exploration_direction != other_branch.exploration_direction:
            combined_insights.append(
                f"Connecting {self.exploration_direction} and {other_branch.exploration_direction} perspectives"
            )

        combined_insights.extend(list(all_insights))
        return combined_insights


class MultiHopGNN(nn.Module):
    """GNN with multi-hop message passing and attention"""

    def __init__(
        self, feature_dim: int = 384, hidden_dim: int = 256, num_hops: int = 3
    ):
        super().__init__()
        self.num_hops = num_hops
        self.feature_dim = feature_dim

        # Multi-hop layers
        self.hop_layers = nn.ModuleList(
            [
                nn.Linear(feature_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_hops)
            ]
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        # Hop-specific gates
        self.hop_gates = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_hops)]
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        query_idx: int,
        return_paths: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Multi-hop message passing with attention
        Returns: (final_features, query_transformation, [optional] reasoning_paths)
        """
        batch_size = 1
        num_nodes = node_features.shape[0]

        # Initialize hidden states
        h = node_features
        hop_outputs = []
        reasoning_paths = []

        for hop in range(self.num_hops):
            # Message passing
            messages = torch.matmul(adjacency, h)

            # Apply hop-specific transformation
            h_new = self.hop_layers[hop](h if hop == 0 else h)

            # Gated combination with messages
            gate_input = torch.cat([h_new, messages], dim=-1)
            gate = torch.sigmoid(self.hop_gates[hop](gate_input))
            h = gate * h_new + (1 - gate) * messages

            h = self.activation(h)
            h = self.dropout(h)

            hop_outputs.append(h)

            # Track reasoning paths
            if return_paths and hop > 0:
                # Find strongest connections at this hop
                query_connections = adjacency[query_idx] * (
                    1 - torch.eye(num_nodes)[query_idx]
                )
                top_k = min(3, (query_connections > 0).sum().item())
                if top_k > 0:
                    _, top_indices = torch.topk(query_connections, top_k)
                    reasoning_paths.append(top_indices.tolist())

        # Apply attention across all hop outputs
        if len(hop_outputs) > 1:
            # Stack hop outputs for attention
            hop_stack = torch.stack(hop_outputs, dim=0).unsqueeze(
                0
            )  # [1, hops, nodes, hidden]
            hop_stack = hop_stack.permute(0, 2, 1, 3).reshape(
                num_nodes, self.num_hops, -1
            )

            # Self-attention across hops
            attended, _ = self.attention(hop_stack, hop_stack, hop_stack)
            h = attended.mean(dim=1)  # Average across hops

        # Final projection
        output = self.output_proj(h)

        # Calculate query transformation
        query_transformation = output[query_idx] - node_features[query_idx]

        if return_paths:
            return output, query_transformation, reasoning_paths
        return output, query_transformation, None


class AdaptiveExplorer:
    """Adaptive exploration strategy for query transformation"""

    def __init__(self, exploration_temperature: float = 1.0):
        self.temperature = exploration_temperature
        self.exploration_history = []
        self.successful_patterns = []

    def select_exploration_direction(
        self,
        current_state: QueryState,
        knowledge_graph: nx.Graph,
        available_directions: List[str],
    ) -> str:
        """Select exploration direction based on current state and history"""

        if not available_directions:
            return "general"

        # Calculate scores for each direction
        direction_scores = {}

        for direction in available_directions:
            score = self._calculate_direction_score(
                direction, current_state, knowledge_graph
            )
            direction_scores[direction] = score

        # Apply temperature-based selection
        if self.temperature > 0:
            # Softmax with temperature
            scores = np.array(list(direction_scores.values()))
            probabilities = np.exp(scores / self.temperature)
            probabilities /= probabilities.sum()

            selected_idx = np.random.choice(len(available_directions), p=probabilities)
            selected_direction = available_directions[selected_idx]
        else:
            # Greedy selection
            selected_direction = max(direction_scores, key=direction_scores.get)

        self.exploration_history.append((current_state.stage, selected_direction))
        return selected_direction

    def _calculate_direction_score(
        self, direction: str, current_state: QueryState, knowledge_graph: nx.Graph
    ) -> float:
        """Calculate score for an exploration direction"""

        score = 0.0

        # Favor unexplored directions
        exploration_count = sum(
            1 for _, d in self.exploration_history if d == direction
        )
        score -= exploration_count * 0.1

        # Favor directions with more connections
        direction_nodes = [
            n
            for n in knowledge_graph.nodes()
            if knowledge_graph.nodes[n].get("category") == direction
        ]
        score += len(direction_nodes) * 0.2

        # Favor directions that previously led to insights
        successful_count = sum(
            1 for pattern in self.successful_patterns if direction in pattern
        )
        score += successful_count * 0.3

        # Consider current state confidence
        if current_state.confidence < 0.5:
            # Explore more when confidence is low
            score += 0.2

        return score

    def update_success_pattern(self, path: List[str], insight_quality: float):
        """Update successful exploration patterns"""
        if insight_quality > 0.7:
            self.successful_patterns.append(path)


class EnhancedQueryTransformer(QueryTransformer):
    """Enhanced query transformer with advanced features"""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_multi_hop: bool = True,
        enable_branching: bool = True,
        max_branches: int = 3,
    ):
        super().__init__(embedding_model_name, use_gnn=False)

        self.use_multi_hop = use_multi_hop
        self.enable_branching = enable_branching
        self.max_branches = max_branches

        if use_multi_hop:
            self.multi_hop_gnn = MultiHopGNN(feature_dim=self.embedding_dim)

        self.adaptive_explorer = AdaptiveExplorer()
        self.active_branches: Dict[str, QueryBranch] = {}

    def transform_query_advanced(
        self,
        query_state: QueryState,
        knowledge_graph: nx.Graph,
        documents: List[Dict[str, Any]],
        allow_branching: bool = True,
    ) -> Tuple[QueryState, List[QueryBranch]]:
        """Advanced query transformation with branching and multi-hop reasoning"""

        # Check if we should branch
        branches = []
        if (
            allow_branching
            and self.enable_branching
            and self._should_branch(query_state)
        ):
            branches = self._create_branches(query_state, knowledge_graph)

        # Perform multi-hop transformation
        if self.use_multi_hop and len(knowledge_graph.nodes()) > 0:
            new_state = self._multi_hop_transform(
                query_state, knowledge_graph, documents
            )
        else:
            # Fallback to standard transformation
            new_state = self.transform_query(query_state, knowledge_graph, documents)

        # Explore adaptively
        exploration_direction = self.adaptive_explorer.select_exploration_direction(
            new_state, knowledge_graph, self._get_available_directions(knowledge_graph)
        )
        new_state.metadata["exploration_direction"] = exploration_direction

        # Merge branch insights if available
        if branches:
            branch_insights = self._merge_branch_insights(branches)
            for insight in branch_insights:
                new_state.add_insight(insight)

        return new_state, branches

    def _should_branch(self, query_state: QueryState) -> bool:
        """Determine if query should branch"""

        # Branch when confidence is moderate and we're exploring
        if 0.3 < query_state.confidence < 0.7 and query_state.stage == "exploring":
            return True

        # Branch when we have multiple strong connections
        if hasattr(query_state, "edge_weights") and len(query_state.edge_weights) > 3:
            strong_connections = sum(
                1 for w in query_state.edge_weights.values() if w > 0.6
            )
            return strong_connections >= 2

        return False

    def _create_branches(
        self, parent_state: QueryState, knowledge_graph: nx.Graph
    ) -> List[QueryBranch]:
        """Create exploration branches"""

        branches = []
        directions = self._get_available_directions(knowledge_graph)[
            : self.max_branches
        ]

        for i, direction in enumerate(directions):
            branch_id = f"branch_{i}_{direction}"

            # Create branch with different exploration focus
            branch_state = QueryState(
                text=parent_state.text,
                embedding=parent_state.embedding.clone()
                if parent_state.embedding is not None
                else None,
                stage="exploring",
                confidence=parent_state.confidence * 0.8,  # Slightly lower confidence
                absorbed_concepts=parent_state.absorbed_concepts.copy(),
                insights=parent_state.insights.copy(),
            )

            branch = QueryBranch(
                branch_id=branch_id,
                parent_state=parent_state,
                current_state=branch_state,
                exploration_direction=direction,
                priority=1.0 / (i + 1),  # Prioritize earlier branches
            )

            branches.append(branch)
            self.active_branches[branch_id] = branch

        return branches

    def _multi_hop_transform(
        self,
        query_state: QueryState,
        knowledge_graph: nx.Graph,
        documents: List[Dict[str, Any]],
    ) -> QueryState:
        """Perform multi-hop transformation"""

        # Prepare graph with query
        node_features, adjacency, query_idx = self._prepare_graph_with_query(
            query_state, knowledge_graph, documents
        )

        # Multi-hop reasoning
        new_features, query_change, reasoning_paths = self.multi_hop_gnn(
            node_features, adjacency, query_idx, return_paths=True
        )

        # Create new state
        new_state = QueryState(
            text=query_state.text,
            embedding=new_features[query_idx],
            stage=query_state.stage,
            confidence=query_state.confidence,
            absorbed_concepts=query_state.absorbed_concepts.copy(),
            insights=query_state.insights.copy(),
        )

        # Update based on multi-hop insights
        change_magnitude = torch.norm(query_change).item()
        new_state.transformation_magnitude = change_magnitude

        # Extract insights from reasoning paths
        if reasoning_paths:
            path_insights = self._extract_path_insights(
                reasoning_paths, knowledge_graph, list(knowledge_graph.nodes())
            )
            for insight in path_insights:
                new_state.add_insight(insight)

        # Update confidence based on path coherence
        if reasoning_paths:
            coherence = self._calculate_path_coherence(reasoning_paths)
            new_state.confidence = min(1.0, new_state.confidence + coherence * 0.1)

        return new_state

    def _extract_path_insights(
        self,
        reasoning_paths: List[List[int]],
        knowledge_graph: nx.Graph,
        node_list: List[str],
    ) -> List[str]:
        """Extract insights from multi-hop reasoning paths"""

        insights = []

        for hop_idx, path in enumerate(reasoning_paths):
            if not path:
                continue

            # Get node names from indices
            path_nodes = [node_list[idx] for idx in path if idx < len(node_list)]

            if len(path_nodes) >= 2:
                # Create insight about connections
                insight = f"Hop {hop_idx + 1}: Connecting through {' → '.join(path_nodes[:2])}"
                insights.append(insight)

        # Check for cycles or patterns
        if len(reasoning_paths) > 2:
            all_nodes = set()
            for path in reasoning_paths:
                all_nodes.update(path)

            if len(all_nodes) < sum(len(p) for p in reasoning_paths) * 0.5:
                insights.append("Discovered convergent reasoning pattern")

        return insights

    def _calculate_path_coherence(self, reasoning_paths: List[List[int]]) -> float:
        """Calculate coherence of reasoning paths"""

        if not reasoning_paths:
            return 0.0

        # Check for consistent progression
        coherence = 0.0

        # Measure path overlap
        if len(reasoning_paths) > 1:
            for i in range(len(reasoning_paths) - 1):
                current = set(reasoning_paths[i])
                next_hop = set(reasoning_paths[i + 1])

                # Some overlap is good (shows progression)
                overlap = len(current & next_hop) / max(len(current), len(next_hop), 1)
                coherence += overlap * 0.3

        # Normalize
        coherence = min(1.0, coherence / max(len(reasoning_paths) - 1, 1))

        return coherence

    def _merge_branch_insights(self, branches: List[QueryBranch]) -> List[str]:
        """Merge insights from multiple branches"""

        if len(branches) < 2:
            return []

        merged_insights = []

        # Pairwise merging
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                branch_insights = branches[i].merge_insights(branches[j])
                merged_insights.extend(branch_insights)

        # Add synthesis insight
        directions = [b.exploration_direction for b in branches]
        merged_insights.append(
            f"Synthesized understanding from {', '.join(directions)} perspectives"
        )

        return list(set(merged_insights))  # Remove duplicates

    def _get_available_directions(self, knowledge_graph: nx.Graph) -> List[str]:
        """Get available exploration directions from graph"""

        categories = set()
        for node in knowledge_graph.nodes():
            category = knowledge_graph.nodes[node].get("category", "general")
            categories.add(category)

        return list(categories)

    def prune_branches(self, confidence_threshold: float = 0.2):
        """Prune low-confidence branches"""

        branches_to_remove = []

        for branch_id, branch in self.active_branches.items():
            if branch.current_state.confidence < confidence_threshold:
                branches_to_remove.append(branch_id)

        for branch_id in branches_to_remove:
            del self.active_branches[branch_id]

        return len(branches_to_remove)
