"""
Query Transformer
Core logic for transforming queries through graph exploration
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from ...algorithms.graph_edit_distance import GraphEditDistance
from .query_state import QueryState, QueryTransformationHistory

logger = logging.getLogger(__name__)


class QueryGraphGNN(nn.Module):
    """GNN that includes query as a special node"""

    def __init__(self, feature_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor, query_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process nodes including the query node
        Returns: (all_features, query_transformation)
        """
        # Ensure input dimension matches
        if node_features.shape[1] != self.feature_dim:
            # Pad or truncate to match expected dimension
            if node_features.shape[1] < self.feature_dim:
                # Pad with zeros
                padding = torch.zeros(
                    node_features.shape[0], self.feature_dim - node_features.shape[1]
                )
                node_features = torch.cat([node_features, padding], dim=1)
            else:
                # Truncate
                node_features = node_features[:, : self.feature_dim]

        original_query = node_features[query_idx].clone()

        # Message passing
        h = node_features
        for _ in range(3):  # 3 layers of message passing
            # Aggregate messages from neighbors
            messages = torch.matmul(adjacency, h)
            h = self.activation(self.fc1(h + messages))
            h = self.dropout(h)

        # Final transformation
        h = self.fc3(self.fc2(h))

        # Calculate query transformation
        query_transformation = h[query_idx] - original_query

        return h, query_transformation


class QueryTransformer:
    """Transforms queries through graph exploration"""

    def __init__(
        self, embedding_model_name: str = "all-MiniLM-L6-v2", use_gnn: bool = True
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = QueryGraphGNN(feature_dim=self.embedding_dim)
        self.ged_calculator = GraphEditDistance()

    def place_query_on_graph(self, query: str, knowledge_graph: nx.Graph) -> QueryState:
        """Place query as a node on the knowledge graph"""

        # Create initial query state
        query_embedding = torch.tensor(
            self.embedding_model.encode(query), dtype=torch.float32
        )

        query_state = QueryState(text=query, embedding=query_embedding, stage="initial")

        # Find optimal placement by calculating similarity to existing nodes
        best_connections = self._find_best_connections(query_embedding, knowledge_graph)

        query_state.connected_nodes = [node for node, _ in best_connections]
        query_state.edge_weights = {node: weight for node, weight in best_connections}

        # Evaluate geDIG potential
        placement_score = self._evaluate_placement(query_state, knowledge_graph)

        query_state.graph_position = {
            "placement_score": placement_score,
            "initial_connections": len(best_connections),
        }

        logger.info(
            f"Query placed with {len(best_connections)} connections, "
            f"placement score: {placement_score:.3f}"
        )

        return query_state

    def transform_query(
        self,
        query_state: QueryState,
        knowledge_graph: nx.Graph,
        documents: List[Dict[str, Any]],
    ) -> QueryState:
        """Transform query through one cycle of graph exploration"""

        new_state = QueryState(
            text=query_state.text,
            embedding=query_state.embedding.clone(),
            stage="exploring",
            confidence=query_state.confidence,
            absorbed_concepts=query_state.absorbed_concepts.copy(),
            insights=query_state.insights.copy(),
        )

        if self.use_gnn:
            # Prepare graph data including query node
            node_features, adjacency, query_idx = self._prepare_graph_with_query(
                new_state, knowledge_graph, documents
            )

            # Apply GNN transformation
            transformed_features, query_change = self.gnn(
                node_features, adjacency, query_idx
            )

            # Update query state based on transformation
            new_state.embedding = transformed_features[query_idx]
            new_state.transformation_magnitude = torch.norm(query_change).item()

            # Extract insights from transformation
            insights = self._extract_insights_from_transformation(
                query_change, knowledge_graph, new_state.connected_nodes
            )

            for insight in insights:
                new_state.add_insight(insight)

            # Absorb concepts from strongly connected nodes
            self._absorb_concepts(new_state, knowledge_graph, adjacency[query_idx])

            # Update stage based on progress
            if new_state.transformation_magnitude > 0.8:
                new_state.stage = "transforming"
            if new_state.insights and new_state.confidence > 0.7:
                new_state.stage = "insight"

        else:
            # Fallback: Simple concept absorption without GNN
            self._simple_concept_absorption(new_state, knowledge_graph)

        return new_state

    def _find_best_connections(
        self, query_embedding: torch.Tensor, knowledge_graph: nx.Graph, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find best nodes to connect the query to"""
        similarities = []

        for node in knowledge_graph.nodes():
            node_data = knowledge_graph.nodes[node]
            if "embedding" in node_data:
                node_embedding = node_data["embedding"]
                similarity = torch.cosine_similarity(
                    query_embedding, torch.tensor(node_embedding), dim=0
                ).item()
                similarities.append((node, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _evaluate_placement(
        self, query_state: QueryState, knowledge_graph: nx.Graph
    ) -> float:
        """Evaluate how good the query placement is for potential insights"""

        # Factors to consider:
        # 1. Connection strength (average edge weight)
        avg_weight = (
            sum(query_state.edge_weights.values()) / len(query_state.edge_weights)
            if query_state.edge_weights
            else 0
        )

        # 2. Diversity of connected concepts
        concept_diversity = len(
            set(
                knowledge_graph.nodes[node].get("category", "unknown")
                for node in query_state.connected_nodes
            )
        )

        # 3. Potential for new connections (nodes that could be bridged)
        bridge_potential = self._calculate_bridge_potential(
            query_state.connected_nodes, knowledge_graph
        )

        # Combined score
        placement_score = (
            0.4 * avg_weight
            + 0.3 * (concept_diversity / 5.0)
            + 0.3 * bridge_potential  # Normalize to [0, 1]
        )

        return placement_score

    def _calculate_bridge_potential(
        self, connected_nodes: List[str], knowledge_graph: nx.Graph
    ) -> float:
        """Calculate potential for creating new connections"""

        # Check how many currently disconnected node pairs could be connected
        potential_bridges = 0

        for i, node1 in enumerate(connected_nodes):
            for node2 in connected_nodes[i + 1 :]:
                if not knowledge_graph.has_edge(node1, node2):
                    # These nodes could potentially be bridged
                    potential_bridges += 1

        max_possible = len(connected_nodes) * (len(connected_nodes) - 1) / 2
        return potential_bridges / max_possible if max_possible > 0 else 0

    def _prepare_graph_with_query(
        self,
        query_state: QueryState,
        knowledge_graph: nx.Graph,
        documents: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare graph data with query as a node"""

        # Get embeddings for all nodes
        node_list = list(knowledge_graph.nodes())
        embeddings = []

        for node in node_list:
            node_data = knowledge_graph.nodes[node]
            if "embedding" in node_data:
                emb = torch.tensor(node_data["embedding"])
                # Ensure correct dimension
                if emb.shape[0] != self.embedding_dim:
                    # Regenerate embedding if dimension mismatch
                    text = node_data.get("text", str(node))
                    embedding = self.embedding_model.encode(text)
                    emb = torch.tensor(embedding)
                embeddings.append(emb)
            else:
                # Generate embedding for nodes without one
                text = node_data.get("text", str(node))
                embedding = self.embedding_model.encode(text)
                embeddings.append(torch.tensor(embedding))

        # Add query embedding
        embeddings.append(query_state.embedding)
        query_idx = len(embeddings) - 1

        # Stack into tensor
        node_features = torch.stack(embeddings)

        # Build adjacency matrix
        n = len(embeddings)
        adjacency = torch.zeros(n, n)

        # Existing edges
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                if knowledge_graph.has_edge(node1, node2):
                    adjacency[i, j] = 1.0

        # Query connections
        for node, weight in query_state.edge_weights.items():
            if node in node_list:
                idx = node_list.index(node)
                adjacency[query_idx, idx] = weight
                adjacency[idx, query_idx] = weight

        # Normalize adjacency
        degree = adjacency.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adjacency = adjacency / degree

        return node_features, adjacency, query_idx

    def _extract_insights_from_transformation(
        self,
        query_change: torch.Tensor,
        knowledge_graph: nx.Graph,
        connected_nodes: List[str],
    ) -> List[str]:
        """Extract insights based on how the query transformed"""

        insights = []
        change_magnitude = torch.norm(query_change).item()

        # Threshold-based insight detection
        if change_magnitude > 0.5:
            # Analyze which dimensions changed most
            top_changes = torch.topk(torch.abs(query_change), k=5)

            # Generate insight based on connected concepts
            if len(connected_nodes) >= 2:
                insight = f"Discovered connection between {connected_nodes[0]} and {connected_nodes[1]}"
                insights.append(insight)

        if change_magnitude > 0.8:
            insights.append("Significant conceptual transformation detected")

        return insights

    def _absorb_concepts(
        self,
        query_state: QueryState,
        knowledge_graph: nx.Graph,
        connection_strengths: torch.Tensor,
    ):
        """Absorb concepts from strongly connected nodes"""

        node_list = list(knowledge_graph.nodes())

        for i, strength in enumerate(connection_strengths):
            if i < len(node_list) and strength > 0.3:
                node = node_list[i]
                node_data = knowledge_graph.nodes[node]
                concept = node_data.get("concept", str(node))
                query_state.absorb_concept(concept, strength.item())

    def _simple_concept_absorption(
        self, query_state: QueryState, knowledge_graph: nx.Graph
    ):
        """Simple concept absorption without GNN"""

        for node in query_state.connected_nodes[:3]:  # Top 3 connections
            if node in knowledge_graph:
                node_data = knowledge_graph.nodes[node]
                concept = node_data.get("concept", str(node))
                weight = query_state.edge_weights.get(node, 0.5)
                query_state.absorb_concept(concept, weight)

        # Simple heuristic for stage progression
        if len(query_state.absorbed_concepts) >= 3:
            query_state.stage = "transforming"
        if query_state.confidence > 0.5:
            query_state.add_insight(
                f"Integrated concepts: {', '.join(query_state.absorbed_concepts[-3:])}"
            )
