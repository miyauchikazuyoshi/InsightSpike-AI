"""geDIG-based navigator for maze exploration.

Day2 integration: refactored GeDIGCore (feature flag + optional dual evaluation) and
structural improvement guided action energy adjustment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ...environments.maze import MazeObservation
try:  # sphere_search prototype may be absent; provide fallback
    from insightspike.maze_experimental.query.sphere_search import SphereSearch  # type: ignore
except Exception:  # pragma: no cover - fallback path
    class SphereSearch:  # minimal placeholder
        def __init__(self, *args, **kwargs):
            pass
from insightspike.algorithms.gedig_core import GeDIGCore  # absolute to avoid relative resolution issues
from insightspike.algorithms.gedig_factory import GeDIGFactory, dual_evaluate
from insightspike.algorithms.gedig_core import GeDIGMonitor
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class NavigationMemoryNode:
    """Memory node for navigation."""
    position: Tuple[int, int]
    features: Dict[str, Any]
    vector: np.ndarray
    creation_energy: float
    visits: int = 1
    last_visited: int = 0
    
    def should_merge(self, other: 'NavigationMemoryNode', threshold: float = 0.9) -> bool:
        """Check if this node should be merged with another."""
        if np.linalg.norm(self.vector - other.vector) < (1 - threshold):
            return True
        return False


class GeDIGNavigator:
    """Navigator using geDIG principles for maze exploration."""
    
    def __init__(self, config: MazeNavigatorConfig):
        """Initialize the navigator.
        
        Args:
            config: Navigation configuration
        """
        self.config = config
        self.memory_nodes: Dict[Tuple[int, int], NavigationMemoryNode] = {}
        self.sphere_search = SphereSearch()
        # Day2: integrate feature-flagged GeDIGCore
        self._legacy_core: Optional[GeDIGCore] = None
        self._ref_core: GeDIGCore = GeDIGFactory.create({
            'use_refactored_gedig': self.config.use_refactored_gedig,
            'tau_s': self.config.tau_s,
            'tau_i': self.config.tau_i,
            'spike_detection_mode': self.config.spike_detection_mode,
        })
        if self.config.enable_dual_evaluate:
            # Prepare legacy core for divergence tracking
            self._legacy_core = GeDIGFactory.create({
                'use_refactored_gedig': False,
                'tau_s': self.config.tau_s,
                'tau_i': self.config.tau_i,
                'spike_detection_mode': self.config.spike_detection_mode,
            })
        self._last_dual_delta: Optional[float] = None

        # Attach monitor to reference core (legacy optional)
        # Attach monitor with proper parameter name (window_size)
        self._ref_core.monitor = GeDIGMonitor(window_size=50)
        if self._legacy_core:
            self._legacy_core.monitor = GeDIGMonitor(window_size=50)
        
        # Episode tracking
        self.current_episode = 0
        self.total_energy_spent = 0.0
        
        # Simple feature embedder (can be replaced with sentence transformer)
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        if self.config.use_pretrained_embedder:
            # TODO: Load sentence transformer
            logger.info("Using pretrained embedder not implemented, falling back to simple embedder")
        
        # Simple embedder: map feature types to random vectors
        self.feature_vectors = {
            'wall': np.random.randn(self.feature_dim),
            'junction': np.random.randn(self.feature_dim),
            'dead_end': np.random.randn(self.feature_dim),
            'corridor': np.random.randn(self.feature_dim),
            'goal': np.random.randn(self.feature_dim),
            'unknown': np.random.randn(self.feature_dim)
        }
        
        # Normalize
        for key in self.feature_vectors:
            self.feature_vectors[key] /= np.linalg.norm(self.feature_vectors[key])
    
    def embed_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to vector embedding."""
        feature_type = features.get('type', 'unknown')
        base_vector = self.feature_vectors.get(feature_type, self.feature_vectors['unknown'])
        
        # Add some variation based on position
        position = features.get('position', (0, 0))
        position_encoding = np.array([
            np.sin(position[0] / 10),
            np.cos(position[0] / 10),
            np.sin(position[1] / 10),
            np.cos(position[1] / 10)
        ])
        
        # Combine base vector with position encoding
        if len(position_encoding) < self.feature_dim:
            position_encoding = np.pad(position_encoding, (0, self.feature_dim - len(position_encoding)))
        else:
            position_encoding = position_encoding[:self.feature_dim]
        
        vector = 0.8 * base_vector + 0.2 * position_encoding
        return vector / np.linalg.norm(vector)
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide next action based on geDIG principles.
        
        Args:
            observation: Current observation from maze
            maze: Maze environment (for looking ahead)
            
        Returns:
            Action to take (0-3)
        """
        current_pos = observation.position
        
        # Check if we should create a memory node
        if self._should_create_node(observation):
            self._create_memory_node(observation)
        
        # Sphere search for nearby memories
        query_vector = self.embed_features(observation.to_features())
        nearby_memories = self._search_nearby_memories(current_pos, query_vector)
        
        # Evaluate each possible action (capture geDIG result for each)
        action_energies: Dict[int, float] = {}
        action_results: Dict[int, Any] = {}
        for action in observation.possible_moves:
            energy, ref_res = self._evaluate_action_energy(
                current_pos, action, nearby_memories, maze, observation
            )
            action_energies[action] = energy
            action_results[action] = ref_res
        
        # Epsilon-greedy exploration
        if np.random.random() < self.config.exploration_epsilon:
            chosen = int(np.random.choice(observation.possible_moves))
        else:
            chosen = min(action_energies.keys(), key=lambda a: action_energies[a])

        # Store last geDIG result (may be None if evaluation failed)
        self.last_result = action_results.get(chosen)
        if self.last_result is not None:
            self.last_reward = getattr(self.last_result, 'reward', getattr(self.last_result, 'hop0_reward', None))
            self.last_spike = getattr(self.last_result, 'spike', False)
            self.last_structural_improvement = getattr(self.last_result, 'structural_improvement', 0.0)
        else:
            self.last_reward = None
            self.last_spike = False
            self.last_structural_improvement = 0.0
        return chosen
    
    def _should_create_node(self, observation: MazeObservation) -> bool:
        """Determine if current location should be memorized."""
        # Always create node for important locations
        if observation.is_goal or observation.is_junction or observation.is_dead_end:
            return True
        
        # Create node if hit wall (to remember obstacles)
        if observation.hit_wall:
            return True
        
        # Check if we already have a node here
        if observation.position in self.memory_nodes:
            # Update visit count instead
            self.memory_nodes[observation.position].visits += 1
            self.memory_nodes[observation.position].last_visited = self.current_episode
            return False
        
        # Don't create nodes for simple corridors (unless first time)
        if observation.num_paths == 2 and len(self.memory_nodes) > 10:
            return False
        
        return True
    
    def _create_memory_node(self, observation: MazeObservation):
        """Create a new memory node."""
        features = observation.to_features()
        vector = self.embed_features(features)
        
        node = NavigationMemoryNode(
            position=observation.position,
            features=features,
            vector=vector,
            creation_energy=self.config.node_creation_cost,
            last_visited=self.current_episode
        )
        
        self.memory_nodes[observation.position] = node
        self.total_energy_spent += self.config.node_creation_cost
        
        logger.debug(f"Created memory node at {observation.position}, type: {features['type']}")
    
    def _search_nearby_memories(self, position: Tuple[int, int], 
                               query_vector: np.ndarray) -> List[NavigationMemoryNode]:
        """Search for nearby memory nodes."""
        nearby = []
        
        for node in self.memory_nodes.values():
            # Spatial distance
            spatial_dist = np.sqrt(
                (node.position[0] - position[0])**2 + 
                (node.position[1] - position[1])**2
            )
            
            # Vector distance
            vector_dist = np.linalg.norm(query_vector - node.vector)
            
            # Combined distance (weighted)
            combined_dist = 0.7 * spatial_dist + 0.3 * vector_dist * 10
            
            if combined_dist <= self.config.search_radius:
                nearby.append(node)
        
        return nearby
    
    def _evaluate_action_energy(self, current_pos: Tuple[int, int], 
                               action: int,
                               nearby_memories: List[NavigationMemoryNode],
                               maze,
                               observation: MazeObservation) -> Tuple[float, Optional[Any]]:
        """Evaluate energy cost of taking an action.

        Integrates refactored GeDIG structural_improvement as an energy reduction term.
        If dual evaluation enabled, divergence is tracked but does not alter decision yet.
        """
        # Get next position
        delta = maze.ACTIONS[action]
        next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
        
        energy = 0.0
        
        # Check for walls ahead (donut search)
        walls_ahead = self._count_walls_ahead(next_pos, nearby_memories)
        energy += walls_ahead * self.config.wall_penalty
        
        # Node creation cost (if new area)
        if next_pos not in self.memory_nodes:
            energy += self.config.node_creation_cost
        else:
            # Visiting known area is cheaper
            energy += 0.1
        
        # Bonus for unexplored areas (if not too many walls)
        if next_pos not in self.memory_nodes and walls_ahead < 2:
            energy -= self.config.unknown_bonus
        
        # Consider dead ends (high cost)
        dead_end_memories = [m for m in nearby_memories if m.features['type'] == 'dead_end']
        for dead_end in dead_end_memories:
            dist = np.sqrt((dead_end.position[0] - next_pos[0])**2 + 
                          (dead_end.position[1] - next_pos[1])**2)
            if dist < 3:
                energy += 2.0 / (dist + 1)
        
        # GeDIG structural improvement (graph-level): build minimal graphs from memory nodes
        try:
            import networkx as nx  # Lazy import
            g_prev = nx.Graph()
            g_now = nx.Graph()
            # Represent memory nodes as nodes; edges for spatial adjacency (4-neighborhood)
            for pos in self.memory_nodes.keys():
                g_prev.add_node(pos)
                g_now.add_node(pos)
            # Add candidate next position to g_now (simulating exploration)
            g_now.add_node(next_pos)
            # Simple adjacency edges (prev)
            for pos in self.memory_nodes.keys():
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    np2 = (pos[0]+dx, pos[1]+dy)
                    if np2 in self.memory_nodes:
                        g_prev.add_edge(pos, np2)
                        g_now.add_edge(pos, np2)
            # Edges from new node to existing neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                np2 = (next_pos[0]+dx, next_pos[1]+dy)
                if np2 in self.memory_nodes:
                    g_now.add_edge(next_pos, np2)

            if self.config.enable_dual_evaluate and self._legacy_core is not None:
                ref_res, delta_div = dual_evaluate(
                    self._legacy_core,
                    self._ref_core,
                    g_prev=g_prev,
                    g_now=g_now,
                    delta_threshold=self.config.dual_delta_threshold
                )
                self._last_dual_delta = delta_div
            else:
                try:
                    from insightspike.algorithms.linkset_adapter import build_linkset_info as _build_ls  # type: ignore
                except Exception:
                    _build_ls = None  # type: ignore
                _ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
                if _ls is not None:
                    ref_res = self._ref_core.calculate(g_prev=g_prev, g_now=g_now, linkset_info=_ls)
                else:
                    ref_res = self._ref_core.calculate(g_prev=g_prev, g_now=g_now)
            structural_improvement = ref_res.structural_improvement if ref_res else 0.0

            # Monitor recording is already handled inside GeDIGCore.calculate; avoid duplicate writes.
            # (Previously: self._ref_core.monitor.record_prediction / record_outcome)

            # Positive structural improvement reduces energy
            energy -= self.config.structural_improvement_weight * max(0.0, structural_improvement)
            return energy, ref_res
        except Exception as e:  # Fallback if graph calc fails
            logger.debug(f"GeDIG structural energy adjustment skipped: {e}")
            return energy, None

    # --- Exposed last computation results ---
    last_result: Optional[Any] = None
    last_reward: Optional[float] = None
    last_spike: bool = False
    last_structural_improvement: float = 0.0

    def step(self, observation: MazeObservation, maze) -> Tuple[int, MazeObservation]:
        """High-level step helper: decide action and advance environment.

        Returns (action, new_observation). Side effects: populates last_result / last_reward / last_spike.
        """
        action = self.decide_action(observation, maze)
        new_obs, _, done, _ = maze.step(action)
        # If episode ends, we keep last_result as-is.
        return action, new_obs

    @property
    def last_dual_delta(self) -> Optional[float]:
        return self._last_dual_delta

    # --- Day3: spike ground-truth derivation ---
    def _derive_spike_outcome(self, result) -> bool:
        mode = getattr(self.config, 'spike_outcome_mode', 'mirror')
        if mode == 'mirror':  # identical to prediction
            return result.spike
        if mode == 'structural_positive':
            return result.structural_improvement > 0
        if mode == 'spike_threshold':
            return result.gedig_value < self._ref_core.spike_threshold
        # fallback
        return result.spike

    # --- Day2 D2: Wiring strategy (optional) ---
    def evaluate_wiring_candidates(self, candidates: List[Tuple[Tuple[int,int], Tuple[int,int]]]) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
        """Evaluate structural improvement for hypothetical edge additions.

        Each candidate is an edge (pos_a, pos_b). We build a lightweight graph of current
        memory nodes and score adding that edge. The edge with maximal structural_improvement
        (post-add) is returned. If dual evaluate is enabled we still rely on ref result but
        update last_dual_delta with the candidate that wins.
        """
        if not candidates:
            return None
        try:
            import networkx as nx  # Lazy import
            # Base graph from memory nodes + adjacency edges
            base = nx.Graph()
            for pos in self.memory_nodes.keys():
                base.add_node(pos)
            for pos in self.memory_nodes.keys():
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    np2 = (pos[0]+dx, pos[1]+dy)
                    if np2 in self.memory_nodes:
                        base.add_edge(pos, np2)
            best_edge = None
            best_score = -1e9
            best_delta = None
            for edge in candidates:
                g_now = base.copy()
                a, b = edge
                g_now.add_node(a)
                g_now.add_node(b)
                g_now.add_edge(a, b)
                if self.config.enable_dual_evaluate and self._legacy_core is not None:
                    ref_res, delta_div = dual_evaluate(
                        self._legacy_core,
                        self._ref_core,
                        g_prev=base,
                        g_now=g_now,
                        delta_threshold=self.config.dual_delta_threshold
                    )
                    score = ref_res.structural_improvement
                    candidate_delta = delta_div
                else:
                    try:
                        from insightspike.algorithms.linkset_adapter import build_linkset_info as _build_ls  # type: ignore
                    except Exception:
                        _build_ls = None  # type: ignore
                    _ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
                    if _ls is not None:
                        ref_res = self._ref_core.calculate(g_prev=base, g_now=g_now, linkset_info=_ls)
                    else:
                        ref_res = self._ref_core.calculate(g_prev=base, g_now=g_now)
                    score = ref_res.structural_improvement
                    candidate_delta = None
                if score > best_score:
                    best_score = score
                    best_edge = edge
                    best_delta = candidate_delta
            # Store divergence only for chosen edge
            if best_delta is not None:
                self._last_dual_delta = best_delta
            return best_edge
        except Exception as e:  # pragma: no cover
            logger.debug(f"evaluate_wiring_candidates failed: {e}")
            return candidates[0]
    
    def _count_walls_ahead(self, position: Tuple[int, int], 
                          nearby_memories: List[NavigationMemoryNode]) -> int:
        """Count walls in the direction of movement using donut search."""
        wall_count = 0
        
        for memory in nearby_memories:
            if memory.features['type'] == 'wall':
                # Distance from position
                dist = np.sqrt(
                    (memory.position[0] - position[0])**2 + 
                    (memory.position[1] - position[1])**2
                )
                
                # Donut search: only count if within outer radius but not too close
                if self.config.donut_inner_radius < dist <= self.config.donut_outer_radius:
                    wall_count += 1
        
        return wall_count
    
    def sleep_phase(self):
        """Optimize memory during sleep phase."""
        logger.info(f"Entering sleep phase after episode {self.current_episode}")
        
        # 1. Merge similar nodes
        self._merge_similar_nodes()
        
        # 2. Forget rarely visited nodes
        self._forget_unused_nodes()
        
        # 3. Discover shortcuts
        self._discover_shortcuts()
        
        # 4. Extract patterns
        patterns = self._extract_patterns()
        logger.info(f"Extracted patterns: {patterns}")
        
    def _merge_similar_nodes(self):
        """Merge nodes that are very similar."""
        merged = set()
        nodes = list(self.memory_nodes.values())
        
        for i, node1 in enumerate(nodes):
            if id(node1) in merged:
                continue
                
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if id(node2) in merged:
                    continue
                    
                if node1.should_merge(node2):
                    # Keep the more visited node
                    if node1.visits >= node2.visits:
                        del self.memory_nodes[node2.position]
                        merged.add(id(node2))
                    else:
                        del self.memory_nodes[node1.position]
                        merged.add(id(node1))
                        break
    
    def _forget_unused_nodes(self):
        """Remove nodes that haven't been visited recently."""
        threshold_episode = self.current_episode - 50
        to_remove = []
        
        for pos, node in self.memory_nodes.items():
            if node.last_visited < threshold_episode and node.visits < 3:
                to_remove.append(pos)
        
        for pos in to_remove:
            del self.memory_nodes[pos]
            
        if to_remove:
            logger.info(f"Forgot {len(to_remove)} unused nodes")
    
    def _discover_shortcuts(self):
        """Find potential shortcuts between nodes."""
        # TODO: Implement shortcut discovery
        pass
    
    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract navigation patterns from memory."""
        patterns = {
            'junction_count': sum(1 for n in self.memory_nodes.values() 
                                 if n.features['type'] == 'junction'),
            'dead_end_count': sum(1 for n in self.memory_nodes.values() 
                                 if n.features['type'] == 'dead_end'),
            'wall_count': sum(1 for n in self.memory_nodes.values() 
                             if n.features['type'] == 'wall'),
            'total_nodes': len(self.memory_nodes),
            'avg_visits': np.mean([n.visits for n in self.memory_nodes.values()])
                         if self.memory_nodes else 0
        }
        return patterns
    
    def new_episode(self):
        """Called at the start of a new episode."""
        self.current_episode += 1
        
        # Sleep phase at intervals
        if self.current_episode % self.config.sleep_interval == 0:
            self.sleep_phase()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {
            'total_nodes': len(self.memory_nodes),
            'total_energy': self.total_energy_spent,
            'episodes': self.current_episode,
            'node_positions': list(self.memory_nodes.keys()),
            'node_types': [n.features['type'] for n in self.memory_nodes.values()]
        }
