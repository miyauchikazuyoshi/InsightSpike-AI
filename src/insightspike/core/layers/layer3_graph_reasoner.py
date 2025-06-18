"""
L3 Graph Reasoning - Enhanced GNN with ΔGED/ΔIG Analysis
======================================================

Implements graph-based reasoning with spike detection and conflict analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from ...utils.graph_metrics import delta_ged, delta_ig
from ..config import get_config
from ..interfaces import L3GraphReasonerInterface, LayerInput, LayerOutput

logger = logging.getLogger(__name__)

__all__ = ["L3GraphReasoner", "ConflictScore", "GraphBuilder"]


class ConflictScore:
    """Conflict detection and scoring for graph reasoning."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.conflict_threshold = self.config.reasoning.conflict_threshold

    def calculate_conflict(
        self, graph_old: Data, graph_new: Data, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate conflict scores between graphs."""
        try:
            # Basic structural conflict
            structural_conflict = self._structural_conflict(graph_old, graph_new)

            # Semantic conflict based on node features
            semantic_conflict = self._semantic_conflict(graph_old, graph_new)

            # Temporal conflict (if context provides timing info)
            temporal_conflict = self._temporal_conflict(context)

            return {
                "structural": float(structural_conflict),
                "semantic": float(semantic_conflict),
                "temporal": float(temporal_conflict),
                "total": float(
                    structural_conflict + semantic_conflict + temporal_conflict
                )
                / 3,
            }

        except Exception as e:
            logger.error(f"Conflict calculation failed: {e}")
            return {"structural": 0.0, "semantic": 0.0, "temporal": 0.0, "total": 0.0}

    def _structural_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate structural differences between graphs."""
        if graph_old is None or graph_new is None:
            return 0.0

        # Edge count difference
        edge_diff = abs(graph_old.edge_index.size(1) - graph_new.edge_index.size(1))
        node_diff = abs(graph_old.x.size(0) - graph_new.x.size(0))

        # Normalize by graph size
        max_edges = max(graph_old.edge_index.size(1), graph_new.edge_index.size(1), 1)
        max_nodes = max(graph_old.x.size(0), graph_new.x.size(0), 1)

        return (edge_diff / max_edges + node_diff / max_nodes) / 2

    def _semantic_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate semantic differences in node features."""
        if graph_old is None or graph_new is None:
            return 0.0

        try:
            # Compare node feature distributions
            old_features = graph_old.x.cpu().numpy() if hasattr(graph_old.x, 'cpu') else graph_old.x.numpy()
            new_features = graph_new.x.cpu().numpy() if hasattr(graph_new.x, 'cpu') else graph_new.x.numpy()

            # Handle empty features
            if old_features.size == 0 or new_features.size == 0:
                return 0.0

            # Use cosine similarity for feature comparison
            if old_features.shape[1] == new_features.shape[1]:
                old_mean = np.mean(old_features, axis=0, keepdims=True)
                new_mean = np.mean(new_features, axis=0, keepdims=True)

                # Check for non-zero vectors
                old_norm = np.linalg.norm(old_mean)
                new_norm = np.linalg.norm(new_mean)
                
                if old_norm == 0 or new_norm == 0:
                    return 0.0
                
                similarity = cosine_similarity(old_mean, new_mean)[0, 0]
                
                # Handle NaN results
                if not np.isfinite(similarity):
                    return 0.0
                    
                return float(1.0 - similarity)  # Convert similarity to conflict

        except Exception as e:
            logger.warning(f"Semantic conflict calculation failed: {e}")

        return 0.0  # Default to no conflict on error

    def _temporal_conflict(self, context: Dict[str, Any]) -> float:
        """Calculate temporal inconsistencies."""
        # Simple heuristic based on context
        if "previous_confidence" in context and "current_confidence" in context:
            conf_diff = abs(
                context["previous_confidence"] - context["current_confidence"]
            )
            return min(conf_diff, 1.0)
        return 0.0


class GraphBuilder:
    """Build and manage PyTorch Geometric graphs from documents."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.similarity_threshold = self.config.reasoning.similarity_threshold

    def build_graph(
        self, documents: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None
    ) -> Data:
        """Build a graph from documents and their embeddings."""
        if not documents:
            return self._empty_graph()

        try:
            # Extract embeddings
            if embeddings is None:
                embeddings = self._get_embeddings(documents)

            # For very small graphs, create simple structures
            if len(documents) < 3:
                # Create simple chain for small graphs
                edge_list = []
                if len(documents) == 2:
                    edge_list = [[0, 1], [1, 0]]
                elif len(documents) == 1:
                    edge_list = [[0, 0]]  # Self-loop for single node
            else:
                # Build similarity matrix for larger graphs
                sim_matrix = cosine_similarity(embeddings)

                # Create edges based on similarity threshold
                edge_list = []
                for i in range(len(documents)):
                    for j in range(i + 1, len(documents)):
                        if sim_matrix[i, j] > self.similarity_threshold:
                            edge_list.extend([[i, j], [j, i]])  # Undirected edges

                if not edge_list:
                    # Create a simple chain if no similarities found
                    edge_list = [[i, i + 1] for i in range(len(documents) - 1)]
                    edge_list.extend([[i + 1, i] for i in range(len(documents) - 1)])

            # Ensure we have at least some edges
            if not edge_list and len(documents) > 0:
                edge_list = [[0, 0]]  # Self-loop fallback

            # Convert to PyG format
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            x = torch.tensor(embeddings, dtype=torch.float)

            graph = Data(x=x, edge_index=edge_index)
            graph.num_nodes = len(documents)

            # Add document metadata
            graph.documents = documents

            logger.debug(
                f"Built graph with {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges"
            )
            return graph

        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            return self._empty_graph()

    def _get_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Extract or compute embeddings for documents."""
        embeddings = []

        for doc in documents:
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
            else:
                # Fallback: use text hash as simple embedding
                text = doc.get("text", "")
                simple_emb = np.random.random(384)  # Default dimension
                embeddings.append(simple_emb)

        return np.array(embeddings)

    def _empty_graph(self) -> Data:
        """Create an empty graph for error cases."""
        return Data(
            x=torch.empty(0, 384), edge_index=torch.empty(2, 0, dtype=torch.long)
        )


class L3GraphReasoner(L3GraphReasonerInterface):
    """
    Enhanced graph reasoning layer with GNN processing and spike detection.

    Features:
    - PyTorch Geometric graph construction
    - ΔGED and ΔIG calculation for insight detection
    - Conflict scoring between reasoning states
    - Reward calculation for memory updates
    """

    def __init__(self, config=None):
        # Set layer_id for LayerInterface
        super().__init__("layer3_graph_reasoner", config)
        self.config = config or get_config()
        self.graph_builder = GraphBuilder(config)
        self.conflict_scorer = ConflictScore(config)
        self.previous_graph = None

        # Initialize simple GNN if needed
        self.gnn = None
        if self.config.reasoning.use_gnn:
            self._init_gnn()

    def initialize(self) -> bool:
        """Initialize the layer"""
        try:
            # Any initialization needed
            self._is_initialized = True
            logger.info("L3GraphReasoner initialized successfully")
            return True
        except Exception as e:
            logger.error(f"L3GraphReasoner initialization failed: {e}")
            return False

    def process(self, input_data) -> Any:
        """Process input through this layer"""
        try:
            # Handle LayerInput format if provided
            if hasattr(input_data, "data"):
                documents = input_data.data
                context = input_data.context or {}
            else:
                documents = input_data
                context = {}

            return self.analyze_documents(documents, context)
        except Exception as e:
            logger.error(f"L3GraphReasoner processing failed: {e}")
            return self._fallback_result()

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.previous_graph = None
            self.gnn = None
            self._is_initialized = False
            logger.info("L3GraphReasoner cleaned up successfully")
        except Exception as e:
            logger.error(f"L3GraphReasoner cleanup failed: {e}")

    def analyze_documents(
        self, documents: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze documents and detect insights through graph reasoning."""
        context = context or {}

        try:
            # Handle empty documents case - create a minimal synthetic graph
            if not documents:
                # Create a minimal single-node graph for empty document case
                synthetic_embedding = np.random.normal(0, 0.1, (1, 384))  # Small variance
                current_graph = Data(
                    x=torch.tensor(synthetic_embedding, dtype=torch.float),
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Self-loop
                    num_nodes=1
                )
                logger.debug("Created synthetic graph for empty documents")
            else:
                # Build current graph from documents
                current_graph = self.graph_builder.build_graph(documents)

            # Get previous graph from context or instance variable
            previous_graph = context.get("previous_graph", self.previous_graph)

            # Calculate metrics if we have a previous graph
            metrics = self._calculate_metrics(current_graph, previous_graph)

            # Detect conflicts
            conflicts = self.conflict_scorer.calculate_conflict(
                previous_graph, current_graph, context
            )

            # Calculate reward signal
            reward = self._calculate_reward(metrics, conflicts)

            # Apply GNN if enabled
            graph_features = self._process_with_gnn(current_graph) if self.gnn else None

            # Store current graph for next iteration
            self.previous_graph = current_graph

            result = {
                "graph": current_graph,
                "metrics": metrics,
                "conflicts": conflicts,
                "reward": reward,
                "spike_detected": self._detect_spike(metrics, conflicts),
                "graph_features": graph_features,
                "reasoning_quality": self._assess_reasoning_quality(metrics, conflicts),
            }

            logger.debug(f"Graph analysis complete: {metrics}")
            return result

        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return self._fallback_result()

    def _calculate_metrics(
        self, current_graph: Data, previous_graph: Optional[Data]
    ) -> Dict[str, float]:
        """Calculate ΔGED and ΔIG metrics."""
        if previous_graph is None:
            return {"delta_ged": 0.0, "delta_ig": 0.0}

        try:
            # Calculate graph edit distance change
            ged = delta_ged(previous_graph, current_graph)

            # Calculate information gain change
            ig = delta_ig(previous_graph, current_graph)

            return {
                "delta_ged": float(ged),
                "delta_ig": float(ig),
                "graph_size_current": current_graph.num_nodes,
                "graph_size_previous": previous_graph.num_nodes,
            }

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {"delta_ged": 0.0, "delta_ig": 0.0}

    def _calculate_reward(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate reward signal for memory updates."""
        # Get weights from config
        w1 = self.config.reasoning.weight_ged
        w2 = self.config.reasoning.weight_ig
        w3 = self.config.reasoning.weight_conflict

        # Base reward calculation: R = w1*ΔGED + w2*ΔIG - w3*conflict
        base_reward = (
            w1 * metrics.get("delta_ged", 0)
            + w2 * metrics.get("delta_ig", 0)
            - w3 * conflicts.get("total", 0)
        )

        # Additional reward components
        structure_reward = self._structure_reward(metrics)
        novelty_reward = self._novelty_reward(metrics, conflicts)

        return {
            "base": float(base_reward),
            "structure": float(structure_reward),
            "novelty": float(novelty_reward),
            "total": float(base_reward + structure_reward + novelty_reward),
        }

    def _structure_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for good graph structure."""
        current_size = metrics.get("graph_size_current", 0)
        if current_size == 0:
            return 0.0

        # Reward moderate-sized graphs (not too sparse, not too dense)
        optimal_size = 10  # Configurable
        size_penalty = abs(current_size - optimal_size) / optimal_size
        return max(0.0, 1.0 - size_penalty)

    def _novelty_reward(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Reward for novel insights while penalizing excessive conflict."""
        novelty = metrics.get("delta_ig", 0)
        conflict = conflicts.get("total", 0)

        # Balance novelty with stability
        return max(0.0, novelty - 0.5 * conflict)

    def _detect_spike(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> bool:
        """Detect if current state represents an insight spike."""
        ged_threshold = self.config.reasoning.spike_ged_threshold
        ig_threshold = self.config.reasoning.spike_ig_threshold
        conflict_threshold = self.config.reasoning.conflict_threshold

        high_ged = metrics.get("delta_ged", 0) > ged_threshold
        high_ig = metrics.get("delta_ig", 0) > ig_threshold
        low_conflict = conflicts.get("total", 1.0) < conflict_threshold

        return high_ged and high_ig and low_conflict

    def _assess_reasoning_quality(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Assess overall quality of reasoning process."""
        # Combine multiple factors
        metric_score = (metrics.get("delta_ged", 0) + metrics.get("delta_ig", 0)) / 2
        conflict_penalty = conflicts.get("total", 0)

        quality = max(0.0, min(1.0, metric_score - conflict_penalty))
        return float(quality)

    def _init_gnn(self):
        """Initialize a simple GNN for graph processing."""
        try:
            hidden_dim = self.config.reasoning.gnn_hidden_dim
            input_dim = self.config.embedding.dimension

            self.gnn = torch.nn.Sequential(
                GCNConv(input_dim, hidden_dim),
                torch.nn.ReLU(),
                GCNConv(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                GCNConv(hidden_dim, input_dim),
            )
            logger.info("Initialized GNN for graph processing")

        except Exception as e:
            logger.warning(f"GNN initialization failed: {e}")
            self.gnn = None

    def _process_with_gnn(self, graph: Data) -> Optional[torch.Tensor]:
        """Process graph through GNN if available."""
        if self.gnn is None or graph.num_nodes == 0:
            return None

        try:
            with torch.no_grad():
                x = self.gnn(graph.x, graph.edge_index)
                # Global pooling to get graph-level representation
                graph_repr = global_mean_pool(
                    x, torch.zeros(graph.num_nodes, dtype=torch.long)
                )
                return graph_repr

        except Exception as e:
            logger.error(f"GNN processing failed: {e}")
            return None

    def _fallback_result(self) -> Dict[str, Any]:
        """Fallback result for error cases."""
        return {
            "graph": self.graph_builder._empty_graph(),
            "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
            "conflicts": {
                "structural": 0.0,
                "semantic": 0.0,
                "temporal": 0.0,
                "total": 0.0,
            },
            "reward": {"base": 0.0, "structure": 0.0, "novelty": 0.0, "total": 0.0},
            "spike_detected": False,
            "graph_features": None,
            "reasoning_quality": 0.0,
        }

    def save_graph(self, graph: Data, path: Optional[Path] = None) -> Path:
        """Save graph to disk."""
        if path is None:
            path = Path(self.config.reasoning.graph_file)

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(graph, path)
            logger.info(f"Saved graph to {path}")

        except Exception as e:
            # Fallback: save as dict
            save_data = {
                "x": graph.x.detach().cpu().numpy(),
                "edge_index": graph.edge_index.detach().cpu().numpy(),
                "num_nodes": graph.num_nodes,
            }
            torch.save(save_data, path)
            logger.warning(f"Saved graph as dict due to error: {e}")

        return path

    def load_graph(self, path: Optional[Path] = None) -> Optional[Data]:
        """Load graph from disk."""
        if path is None:
            path = Path(self.config.reasoning.graph_file)

        if not path.exists():
            logger.warning(f"Graph file not found: {path}")
            return None

        try:
            loaded = torch.load(path)

            if isinstance(loaded, Data):
                return loaded
            elif isinstance(loaded, dict):
                # Reconstruct from dict
                x = torch.tensor(loaded["x"], dtype=torch.float)
                edge_index = torch.tensor(loaded["edge_index"], dtype=torch.long)
                graph = Data(x=x, edge_index=edge_index)
                graph.num_nodes = loaded.get("num_nodes", x.size(0))
                return graph

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")

        return None

    # Interface methods implementation
    def build_graph(self, vectors: np.ndarray) -> Any:
        """Build similarity graph from vectors"""
        # Convert vectors to documents format
        documents = [
            {"vector": vec, "text": f"doc_{i}"} for i, vec in enumerate(vectors)
        ]
        return self.graph_builder.build_graph(documents, vectors)

    def calculate_ged(self, graph1: Any, graph2: Any) -> float:
        """Calculate graph edit distance"""
        try:
            return delta_ged(graph1, graph2)
        except Exception as e:
            logger.error(f"GED calculation failed: {e}")
            return 0.0

    def calculate_ig(self, old_state: Any, new_state: Any) -> float:
        """Calculate information gain"""
        try:
            return delta_ig(old_state, new_state)
        except Exception as e:
            logger.error(f"IG calculation failed: {e}")
            return 0.0

    def detect_eureka_spike(self, delta_ged: float, delta_ig: float) -> bool:
        """Detect if current state constitutes a eureka spike"""
        metrics = {"delta_ged": delta_ged, "delta_ig": delta_ig}
        conflicts = {"total": 0.0}  # No conflicts for direct call
        return self._detect_spike(metrics, conflicts)

        return None
