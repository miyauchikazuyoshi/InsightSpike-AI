"""
L3 Graph Reasoning - Enhanced GNN with ΔGED/ΔIG Analysis
======================================================

Implements graph-based reasoning with spike detection and conflict analysis.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ...core.episode import Episode
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Import refactored components
from ...features.graph_reasoning import GraphAnalyzer, RewardCalculator

# Import both simple and advanced metrics
from ...metrics.graph_metrics import delta_ged as simple_delta_ged
from ...metrics.graph_metrics import delta_ig as simple_delta_ig

# Import message passing components
from ...graph.message_passing import MessagePassing
from ...graph.edge_reevaluator import EdgeReevaluator

try:
    from ...metrics.advanced_graph_metrics import delta_ged as advanced_delta_ged
    from ...metrics.advanced_graph_metrics import delta_ig as advanced_delta_ig

    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False
    advanced_delta_ged = simple_delta_ged
    advanced_delta_ig = simple_delta_ig
from ...config import get_config
from ...config.legacy_adapter import LegacyConfigAdapter
from ...core.base import L3GraphReasonerInterface, LayerInput, LayerOutput
from .scalable_graph_builder import ScalableGraphBuilder

logger = logging.getLogger(__name__)

__all__ = ["L3GraphReasoner", "ConflictScore", "GraphBuilder", "ScalableGraphBuilder"]


class ConflictScore:
    """Conflict detection and scoring for graph reasoning."""

    def __init__(self, config=None):
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self.conflict_threshold = self.config.graph.conflict_threshold

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
            old_features = (
                graph_old.x.cpu().numpy()
                if hasattr(graph_old.x, "cpu")
                else graph_old.x.numpy()
            )
            new_features = (
                graph_new.x.cpu().numpy()
                if hasattr(graph_new.x, "cpu")
                else graph_new.x.numpy()
            )

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
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self.similarity_threshold = self.config.graph.similarity_threshold

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
        # Store original config for message passing settings
        self._original_config = config
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        # Use ScalableGraphBuilder for better performance
        self.graph_builder = ScalableGraphBuilder(self.config)
        self.conflict_scorer = ConflictScore(self.config)
        self.previous_graph = None

        # Initialize refactored components
        self.graph_analyzer = GraphAnalyzer(self.config)
        self.reward_calculator = RewardCalculator(self.config)

        # Initialize metrics selector with configuration
        from ...algorithms.metrics_selector import MetricsSelector

        self.metrics_selector = MetricsSelector(config)

        # Set methods from selector
        self.delta_ged = self.metrics_selector.delta_ged
        self.delta_ig = self.metrics_selector.delta_ig

        # Log algorithm selection
        algo_info = self.metrics_selector.get_algorithm_info()
        logger.info(
            f"Metrics algorithms - GED: {algo_info['ged_algorithm']}, IG: {algo_info['ig_algorithm']}"
        )

        # Initialize simple GNN if needed
        self.gnn = None
        if self.config.graph.use_gnn:
            self._init_gnn()
        
        # Initialize message passing components
        self._init_message_passing()

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
            # Check if a pre-built graph is provided in context
            if context.get("graph") is not None:
                current_graph = context["graph"]
                logger.debug(
                    f"Using pre-built graph with {current_graph.num_nodes} nodes"
                )
            elif not documents:
                # Handle empty documents case - create a minimal synthetic graph
                synthetic_embedding = np.random.normal(
                    0, 0.1, (1, 384)
                )  # Small variance
                current_graph = Data(
                    x=torch.tensor(synthetic_embedding, dtype=torch.float),
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Self-loop
                    num_nodes=1,
                )
                logger.debug("Created synthetic graph for empty documents")
            else:
                # Build current graph from documents
                if self.previous_graph is not None and documents:
                    # Preserve existing graph structure when processing new documents
                    logger.debug(f"Building incremental graph with {len(documents)} new documents")
                    # First, try incremental build
                    current_graph = self.graph_builder.build_graph(
                        documents, 
                        incremental=True
                    )
                    # Verify that incremental build preserved existing nodes
                    if current_graph.num_nodes < self.previous_graph.num_nodes:
                        logger.warning("Incremental build resulted in smaller graph, preserving existing structure")
                        # Incremental mode might not be working properly, manually preserve
                        current_graph = self.previous_graph
                else:
                    # No previous graph or no documents - build new
                    current_graph = self.graph_builder.build_graph(documents)

            # Get query vector from context
            query_vector = context.get("query_vector")
            
            # Apply message passing if enabled and query vector is available
            if self.message_passing_enabled and query_vector is not None:
                logger.info("Applying question-aware message passing")
                
                # Perform message passing
                updated_representations = self.message_passing.forward(
                    current_graph, query_vector
                )
                
                # Re-evaluate edges after message passing
                current_graph = self.edge_reevaluator.reevaluate(
                    current_graph, updated_representations, query_vector,
                    return_edge_scores=True
                )
                
                # Log edge statistics
                if hasattr(current_graph, 'edge_info') and current_graph.edge_info:
                    new_edges = sum(1 for e in current_graph.edge_info if e['type'] == 'new')
                    logger.info(f"Edge re-evaluation: {new_edges} new edges discovered")
            
            # Get previous graph from context or instance variable
            previous_graph = context.get("previous_graph", self.previous_graph)

            # Calculate metrics if we have a previous graph
            metrics = self.graph_analyzer.calculate_metrics(
                current_graph, previous_graph, self.delta_ged, self.delta_ig
            )
            
            # Log metrics values
            logger.info(f"Metrics calculated - GED: {metrics.get('delta_ged', 'N/A')}, IG: {metrics.get('delta_ig', 'N/A')}")
            if previous_graph is not None:
                logger.info(f"Graph comparison - Previous: {previous_graph.num_nodes} nodes, Current: {current_graph.num_nodes} nodes")

            # Detect conflicts
            conflicts = self.conflict_scorer.calculate_conflict(
                previous_graph, current_graph, context
            )

            # Calculate reward signal
            reward = self.reward_calculator.calculate_reward(metrics, conflicts)

            # Apply GNN if enabled
            graph_features = self._process_with_gnn(current_graph) if self.gnn else None

            # Store current graph for next iteration
            self.previous_graph = current_graph

            # Note: Graph saving is now handled by MainAgent via DataStore
            # self.save_graph(current_graph)  # Deprecated - removed

            # Calculate reasoning quality
            base_quality = self.graph_analyzer.assess_quality(metrics, conflicts)
            
            # Enhance quality assessment with graph features if available
            if graph_features is not None and hasattr(graph_features, 'mean'):
                # Use mean graph feature activation as a signal
                feature_signal = float(graph_features.mean().item())
                # Combine base quality with feature signal (weighted average)
                enhanced_quality = 0.8 * base_quality + 0.2 * min(1.0, abs(feature_signal))
                logger.debug(f"Enhanced reasoning quality from {base_quality:.3f} to {enhanced_quality:.3f} using graph features")
            else:
                enhanced_quality = base_quality

            # Log spike detection details
            spike_thresholds = self._get_spike_thresholds()
            logger.info(f"Spike detection thresholds - GED: {spike_thresholds['ged']}, IG: {spike_thresholds['ig']}, Conflict: {spike_thresholds['conflict']}")
            
            spike_detected = self.graph_analyzer.detect_spike(
                metrics, conflicts, spike_thresholds
            )
            
            # Log spike detection result
            if spike_detected:
                logger.warning(f"🎯 SPIKE DETECTED! GED: {metrics.get('delta_ged', 'N/A')}, IG: {metrics.get('delta_ig', 'N/A')}")
            else:
                logger.debug(f"No spike. GED: {metrics.get('delta_ged', 'N/A')} >= {spike_thresholds['ged']} (need <), IG: {metrics.get('delta_ig', 'N/A')} <= {spike_thresholds['ig']} (need >)")
            
            result = {
                "graph": current_graph,
                "metrics": metrics,
                "conflicts": conflicts,
                "reward": reward,
                "spike_detected": spike_detected,
                "graph_features": graph_features,
                "reasoning_quality": enhanced_quality,
            }

            logger.debug(f"Graph analysis complete: {metrics}")
            return result

        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return self._fallback_result()

    def _get_spike_thresholds(self) -> Dict[str, float]:
        """Get spike detection thresholds from config."""
        return {
            "ged": self.config.graph.spike_ged_threshold,
            "ig": self.config.graph.spike_ig_threshold,
            "conflict": self.config.graph.conflict_threshold,
        }

    def _init_message_passing(self):
        """Initialize message passing components."""
        try:
            # Check if message passing is enabled in config
            # First check original config (dict format)
            if isinstance(self._original_config, dict):
                self.message_passing_enabled = self._original_config.get('graph', {}).get(
                    'enable_message_passing', False
                )
                logger.info(f"Message passing from original config: {self.message_passing_enabled}")
            # Then check Pydantic config
            elif hasattr(self.config, 'graph'):
                self.message_passing_enabled = getattr(
                    self.config.graph, 'enable_message_passing', False
                )
            else:
                self.message_passing_enabled = False
            
            if self.message_passing_enabled:
                # Get message passing config from original config
                if isinstance(self._original_config, dict):
                    mp_config = self._original_config.get('graph', {}).get('message_passing', {})
                else:
                    mp_config = getattr(self.config.graph, 'message_passing', {})
                
                # Check if we should use optimized version
                use_optimized = mp_config.get('enable_batch_computation', True)
                max_hops = mp_config.get('max_hops', 1)
                
                if use_optimized:
                    from ...graph.message_passing_optimized import OptimizedMessagePassing
                    self.message_passing = OptimizedMessagePassing(
                        alpha=mp_config.get('alpha', 0.3),
                        iterations=mp_config.get('iterations', 2),
                        max_hops=max_hops,
                        aggregation=mp_config.get('aggregation', 'weighted_mean'),
                        self_loop_weight=mp_config.get('self_loop_weight', 0.5),
                        decay_factor=mp_config.get('decay_factor', 0.8),
                        convergence_threshold=mp_config.get('convergence_threshold', 1e-4),
                        similarity_threshold=mp_config.get('similarity_threshold', 0.3)
                    )
                    logger.info(f"Using OptimizedMessagePassing with max_hops={max_hops}")
                else:
                    self.message_passing = MessagePassing(
                        alpha=mp_config.get('alpha', 0.3),
                        iterations=mp_config.get('iterations', 2),
                        aggregation=mp_config.get('aggregation', 'weighted_mean'),
                        self_loop_weight=mp_config.get('self_loop_weight', 0.5),
                        decay_factor=mp_config.get('decay_factor', 0.8)
                    )
                
                # Get edge re-evaluation config from original config
                if isinstance(self._original_config, dict):
                    er_config = self._original_config.get('graph', {}).get('edge_reevaluation', {})
                else:
                    er_config = getattr(self.config.graph, 'edge_reevaluation', {})
                
                self.edge_reevaluator = EdgeReevaluator(
                    similarity_threshold=er_config.get('similarity_threshold', 0.7),
                    new_edge_threshold=er_config.get('new_edge_threshold', 0.8),
                    max_new_edges_per_node=er_config.get('max_new_edges_per_node', 5),
                    edge_decay_factor=er_config.get('edge_decay_factor', 0.9)
                )
                
                logger.info("Message passing components initialized")
            else:
                self.message_passing = None
                self.edge_reevaluator = None
                logger.info("Message passing disabled in config")
                
        except Exception as e:
            logger.warning(f"Message passing initialization failed: {e}")
            self.message_passing_enabled = False
            self.message_passing = None
            self.edge_reevaluator = None

    def _init_gnn(self):
        """Initialize a simple GNN for graph processing."""
        try:
            hidden_dim = self.config.graph.gnn_hidden_dim
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
            return advanced_delta_ged(graph1, graph2)
        except Exception as e:
            logger.error(f"GED calculation failed: {e}")
            return 0.0

    def calculate_ig(self, old_state: Any, new_state: Any) -> float:
        """Calculate information gain"""
        try:
            return advanced_delta_ig(old_state, new_state)
        except Exception as e:
            logger.error(f"IG calculation failed: {e}")
            return 0.0

    def detect_eureka_spike(self, delta_ged: float, delta_ig: float) -> bool:
        """Detect if current state constitutes a eureka spike"""
        metrics = {"delta_ged": delta_ged, "delta_ig": delta_ig}
        conflicts = {"total": 0.0}  # No conflicts for direct call
        return self.graph_analyzer.detect_spike(
            metrics, conflicts, self._get_spike_thresholds()
        )
    
    def update_graph(self, episodes: List[Episode]):
        """
        Update graph with new episodes.
        
        This method is called by MainAgent but was missing from the implementation.
        For now, we store the reference to episodes for future graph building.
        
        Args:
            episodes: List of new episodes to incorporate into the graph
        """
        # Log the update request
        logger.debug(f"Graph update requested with {len(episodes)} episodes")
        
        # In a full implementation, this would:
        # 1. Extract vectors from episodes
        # 2. Update the existing graph structure
        # 3. Recalculate graph metrics
        
        # For now, we just acknowledge the request
        # The actual graph update happens in build_graph when needed
        pass
