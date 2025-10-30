"""
Standalone L3 Graph Reasoner
============================

A self-contained version of the L3 Graph Reasoner that can be used
independently of the full InsightSpike system for other projects.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import PyTorch components, fallback gracefully
try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics.pairwise import cosine_similarity
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Data = None

logger = logging.getLogger(__name__)


class StandaloneConfig:
    """Minimal configuration for standalone use"""

    def __init__(self, **kwargs):
        # Default values
        self.conflict_threshold = kwargs.get("conflict_threshold", 0.5)
        self.use_gnn = kwargs.get("use_gnn", TORCH_AVAILABLE)
        self.embedding_dim = kwargs.get("embedding_dim", 128)
        self.gnn_hidden_dim = kwargs.get("gnn_hidden_dim", 64)
        self.device = kwargs.get("device", "cpu")

        # Insight detection thresholds
        self.spike_threshold = kwargs.get("spike_threshold", 0.3)
        self.ged_threshold = kwargs.get("ged_threshold", 0.4)
        self.ig_threshold = kwargs.get("ig_threshold", 0.2)


class DocumentProcessor:
    """Process documents into graph-compatible format"""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Convert documents to vector representations"""
        vectors = []

        for doc in documents:
            # Extract text content
            if isinstance(doc, dict):
                text = doc.get("text", doc.get("content", str(doc)))
            else:
                text = str(doc)

            # Simple text vectorization (replace with proper embeddings in production)
            vector = self._text_to_vector(text)
            vectors.append(vector)

        return vectors

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector (placeholder implementation)"""
        # This is a simple hash-based vectorization
        # In production, use proper embeddings (BERT, etc.)
        hash_val = hash(text.lower())
        np.random.seed(abs(hash_val) % (2**31))
        vector = np.random.normal(0, 1, self.embedding_dim)
        return vector / np.linalg.norm(vector)


class GraphBuilder:
    """Build similarity graphs from document vectors"""

    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold

    def build_graph(self, vectors: List[np.ndarray]) -> Optional[Data]:
        """Build graph from document vectors"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning None")
            return None

        if not vectors:
            return None

        # Convert to tensor
        node_features = torch.tensor(np.array(vectors), dtype=torch.float32)
        num_nodes = node_features.size(0)

        # Build edges based on similarity
        edge_indices = []
        edge_weights = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = cosine_similarity(
                    vectors[i].reshape(1, -1), vectors[j].reshape(1, -1)
                )[0, 0]

                if similarity > self.similarity_threshold:
                    edge_indices.extend([[i, j], [j, i]])  # Undirected edge
                    edge_weights.extend([similarity, similarity])

        # Create edge tensor
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            # No edges - create self-loops
            edge_index = (
                torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long)
                .t()
                .contiguous()
            )
            edge_attr = torch.ones(num_nodes, dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class ConflictAnalyzer:
    """Analyze conflicts between graph states"""

    def __init__(self, config: StandaloneConfig):
        self.config = config

    def calculate_conflict(
        self,
        graph_old: Optional[Data],
        graph_new: Optional[Data],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Calculate conflict scores between graphs"""

        if not TORCH_AVAILABLE or graph_old is None or graph_new is None:
            return {"structural": 0.0, "semantic": 0.0, "temporal": 0.0, "total": 0.0}

        try:
            # Structural conflict
            structural = self._structural_conflict(graph_old, graph_new)

            # Semantic conflict
            semantic = self._semantic_conflict(graph_old, graph_new)

            # Temporal conflict (placeholder)
            temporal = self._temporal_conflict(context or {})

            total = (structural + semantic + temporal) / 3

            return {
                "structural": float(structural),
                "semantic": float(semantic),
                "temporal": float(temporal),
                "total": float(total),
            }

        except Exception as e:
            logger.error(f"Conflict calculation failed: {e}")
            return {"structural": 0.0, "semantic": 0.0, "temporal": 0.0, "total": 0.0}

    def _structural_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate structural differences"""
        edge_diff = abs(graph_old.edge_index.size(1) - graph_new.edge_index.size(1))
        node_diff = abs(graph_old.x.size(0) - graph_new.x.size(0))

        max_edges = max(graph_old.edge_index.size(1), graph_new.edge_index.size(1), 1)
        max_nodes = max(graph_old.x.size(0), graph_new.x.size(0), 1)

        return (edge_diff / max_edges + node_diff / max_nodes) / 2

    def _semantic_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate semantic differences"""
        try:
            old_features = graph_old.x.cpu().numpy()
            new_features = graph_new.x.cpu().numpy()

            if old_features.shape[1] == new_features.shape[1]:
                old_mean = np.mean(old_features, axis=0, keepdims=True)
                new_mean = np.mean(new_features, axis=0, keepdims=True)

                similarity = cosine_similarity(old_mean, new_mean)[0, 0]
                return 1.0 - similarity
        except Exception:
            pass

        return 0.5

    def _temporal_conflict(self, context: Dict[str, Any]) -> float:
        """Calculate temporal conflict (placeholder)"""
        return context.get("temporal_conflict", 0.0)


class MetricsCalculator:
    """Calculate ΔGED and ΔIG metrics"""

    def calculate_metrics(
        self, graph_old: Optional[Data], graph_new: Optional[Data]
    ) -> Dict[str, float]:
        """Calculate graph comparison metrics"""

        if not TORCH_AVAILABLE or graph_old is None or graph_new is None:
            return {"delta_ged": 0.0, "delta_ig": 0.0}

        try:
            # Calculate graph edit distance approximation
            delta_ged = self._approximate_ged(graph_old, graph_new)

            # Calculate information gain approximation
            delta_ig = self._approximate_ig(graph_old, graph_new)

            return {"delta_ged": float(delta_ged), "delta_ig": float(delta_ig)}

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {"delta_ged": 0.0, "delta_ig": 0.0}

    def _approximate_ged(self, graph_old: Data, graph_new: Data) -> float:
        """Approximate graph edit distance"""
        # Simple approximation based on structural differences
        edge_diff = abs(graph_old.edge_index.size(1) - graph_new.edge_index.size(1))
        node_diff = abs(graph_old.x.size(0) - graph_new.x.size(0))

        # Normalize by graph size
        total_changes = edge_diff + node_diff
        max_size = max(graph_old.x.size(0), graph_new.x.size(0), 1)

        return total_changes / max_size

    def _approximate_ig(self, graph_old: Data, graph_new: Data) -> float:
        """Approximate information gain"""
        # Calculate based on feature diversity
        old_diversity = self._calculate_diversity(graph_old.x.cpu().numpy())
        new_diversity = self._calculate_diversity(graph_new.x.cpu().numpy())

        return abs(new_diversity - old_diversity)

    def _calculate_diversity(self, features: np.ndarray) -> float:
        """Calculate feature diversity in a graph"""
        if features.shape[0] <= 1:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(features.shape[0]):
            for j in range(i + 1, features.shape[0]):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0


class StandaloneL3GraphReasoner:
    """Standalone version of L3 Graph Reasoner"""

    def __init__(
        self, config: Optional[Union[Dict[str, Any], StandaloneConfig]] = None
    ):
        if isinstance(config, dict):
            self.config = StandaloneConfig(**config)
        elif config is None:
            self.config = StandaloneConfig()
        else:
            self.config = config

        # Initialize components
        self.document_processor = DocumentProcessor(self.config.embedding_dim)
        self.graph_builder = GraphBuilder()
        self.conflict_analyzer = ConflictAnalyzer(self.config)
        self.metrics_calculator = MetricsCalculator()

        # State tracking
        self.previous_graph = None
        self.analysis_history = []

    def analyze_documents(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main analysis method - processes documents and detects insights"""

        try:
            # Convert documents to dict format if needed
            doc_list = []
            for doc in documents:
                if isinstance(doc, str):
                    doc_list.append({"text": doc})
                else:
                    doc_list.append(doc)

            # Process documents to vectors
            vectors = self.document_processor.process_documents(doc_list)

            # Build graph
            current_graph = self.graph_builder.build_graph(vectors)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                self.previous_graph, current_graph
            )

            # Detect conflicts
            conflicts = self.conflict_analyzer.calculate_conflict(
                self.previous_graph, current_graph, context
            )

            # Detect spike
            spike_detected = self._detect_spike(metrics, conflicts)

            # Calculate reward signal
            reward = self._calculate_reward(metrics, conflicts)

            # Assessment
            reasoning_quality = self._assess_reasoning_quality(metrics, conflicts)

            # Prepare result
            result = {
                "graph": current_graph,
                "metrics": metrics,
                "conflicts": conflicts,
                "spike_detected": spike_detected,
                "reward": reward,
                "reasoning_quality": reasoning_quality,
                "document_count": len(doc_list),
                "has_previous_state": self.previous_graph is not None,
                "analysis_id": len(self.analysis_history),
            }

            # Store for next analysis
            self.previous_graph = current_graph
            self.analysis_history.append(result)

            logger.info(f"Graph analysis complete. Spike detected: {spike_detected}")
            return result

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return self._fallback_result()

    def _detect_spike(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> bool:
        """Detect if current state constitutes an insight spike"""
        delta_ged = metrics.get("delta_ged", 0.0)
        delta_ig = metrics.get("delta_ig", 0.0)
        total_conflict = conflicts.get("total", 0.0)

        # Spike conditions
        high_change = delta_ged > self.config.ged_threshold
        high_information = delta_ig > self.config.ig_threshold
        significant_conflict = total_conflict > self.config.conflict_threshold

        # Combined spike detection
        spike_score = (delta_ged + delta_ig + total_conflict) / 3
        return spike_score > self.config.spike_threshold

    def _calculate_reward(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Calculate reward signal based on graph analysis"""
        delta_ged = metrics.get("delta_ged", 0.0)
        delta_ig = metrics.get("delta_ig", 0.0)
        total_conflict = conflicts.get("total", 0.0)

        # Reward based on information gain and manageable conflict
        reward = delta_ig * 2.0  # Information gain is positive
        reward += max(0, 0.5 - total_conflict)  # Penalty for high conflict
        reward += min(delta_ged, 0.3)  # Small bonus for structural change

        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]

    def _assess_reasoning_quality(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Assess the quality of reasoning in this cycle"""
        delta_ig = metrics.get("delta_ig", 0.0)
        total_conflict = conflicts.get("total", 0.0)

        # Quality based on information gain vs conflict
        quality = delta_ig / (1.0 + total_conflict)
        return max(0.0, min(1.0, quality))

    def _fallback_result(self) -> Dict[str, Any]:
        """Fallback result when analysis fails"""
        return {
            "graph": None,
            "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
            "conflicts": {
                "structural": 0.0,
                "semantic": 0.0,
                "temporal": 0.0,
                "total": 0.0,
            },
            "spike_detected": False,
            "reward": 0.0,
            "reasoning_quality": 0.0,
            "document_count": 0,
            "has_previous_state": False,
            "error": True,
        }

    def reset_state(self):
        """Reset internal state"""
        self.previous_graph = None
        self.analysis_history.clear()

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed"""
        if not self.analysis_history:
            return {"total_analyses": 0}

        spikes = [a for a in self.analysis_history if a.get("spike_detected", False)]
        rewards = [a.get("reward", 0) for a in self.analysis_history]
        qualities = [a.get("reasoning_quality", 0) for a in self.analysis_history]

        return {
            "total_analyses": len(self.analysis_history),
            "total_spikes": len(spikes),
            "spike_rate": len(spikes) / len(self.analysis_history),
            "avg_reward": np.mean(rewards),
            "avg_quality": np.mean(qualities),
            "torch_available": TORCH_AVAILABLE,
        }


# Convenience functions for easy usage
def create_standalone_reasoner(
    config: Optional[Dict[str, Any]] = None
) -> StandaloneL3GraphReasoner:
    """Create a standalone graph reasoner with optional configuration"""
    return StandaloneL3GraphReasoner(config)


def analyze_documents_simple(documents: List[str]) -> Dict[str, Any]:
    """Simple function to analyze a list of text documents"""
    reasoner = create_standalone_reasoner()
    return reasoner.analyze_documents(documents)


# Export main classes and functions
__all__ = [
    "StandaloneL3GraphReasoner",
    "StandaloneConfig",
    "create_standalone_reasoner",
    "analyze_documents_simple",
]
