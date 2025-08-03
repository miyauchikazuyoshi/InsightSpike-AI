# Maze Experiment Design for InsightSpike

## Overview
Apply InsightSpike's wake-sleep cycle and geDIG optimization to maze solving, demonstrating the generality of the approach beyond language tasks.

## Core Design Principle: Unified Memory Model

### Consistency with Language Implementation
- **Nodes = Memories** (both in language and maze domains)
- **Edges = Relationships** between memories
- Same geDIG algorithm works across domains

### Language Space
```python
# Nodes: Conceptual memories
node = {
    "id": "energy_conservation",
    "vector": sentence_transformer.encode("Energy is conserved"),
    "content": "Energy cannot be created or destroyed"
}

# Edges: Semantic relationships
edge = {
    "from": "energy_conservation",
    "to": "thermodynamics",
    "relation": "fundamental_principle"
}
```

### Maze Space
```python
# Nodes: Spatial memories (feature points)
node = {
    "id": "junction_A",
    "position": (5, 5),
    "features": {
        "type": "4_way_junction",
        "choices": 4,
        "distinctive": ["wide_space", "pillar_in_center"]
    }
}

# Edges: Reachability relationships
edge = {
    "from": "entrance",
    "to": "junction_A", 
    "relation": "reachable",
    "distance": 7
}
```

## Feature-Based Memory Model

### What to Remember (Nodes)
1. **Significant Locations Only**
   - Junctions (decision points)
   - Dead ends (termination points)
   - Corners (direction changes)
   - Goal (target location)
   - Distinctive features (landmarks)

2. **NOT Every Position**
   - Skip boring corridors
   - Skip redundant waypoints
   - Similar to how humans remember space

### Human-Like Spatial Memory
```python
class MazeMemoryNode:
    def __init__(self, position, observation):
        self.position = position
        self.memory_type = self.classify_location(observation)
        self.features = self.extract_features(observation)
        self.timestamp = current_time()
        
    def classify_location(self, obs):
        """Determine if this location is worth remembering"""
        choice_count = len(obs["possible_moves"])
        
        if choice_count == 0:
            return "dead_end"
        elif choice_count >= 3:
            return "junction"
        elif self.has_direction_change(obs):
            return "corner"
        elif self.has_landmark(obs):
            return "landmark"
        else:
            return None  # Don't create node
```

## Wake-Sleep Cycle in Maze

### Wake Phase (Exploration)
```python
def explore_wake_mode(current_pos):
    # 1. Query-centric search (current position as origin)
    nearby_memories = find_memories_in_sphere(current_pos, radius=5)
    
    # 2. Cost minimization for next move
    candidates = get_unexplored_directions(current_pos)
    best_direction = minimize_exploration_cost(candidates, nearby_memories)
    
    # 3. Move and create memory if significant
    new_pos = move(current_pos, best_direction)
    if is_significant_location(new_pos):
        create_memory_node(new_pos)
```

### Sleep Phase (Consolidation)
```python
def consolidate_maze_knowledge():
    # 1. Merge redundant corridors
    for node in graph.nodes:
        if node.memory_type == "corridor":
            merge_through_corridor(node)
    
    # 2. Discover shortcuts
    for node_pair in potential_connections:
        if direct_path_exists(node_pair):
            add_shortcut_edge(node_pair)
    
    # 3. Extract patterns
    patterns = {
        "maze_structure": detect_regularity(),
        "effective_paths": find_efficient_routes(),
        "dead_end_clusters": identify_avoid_zones()
    }
    
    # 4. Optimize using geDIG
    maximize_reward(lambda_ig * delta_IG - mu * (churn + accretion))
```

## Key Advantages

### 1. Cognitive Plausibility
- Matches human spatial memory
- "Turn left at the big junction, then right at the T-intersection"
- Details of corridors are forgotten

### 2. Computational Efficiency
- Fewer nodes = faster graph operations
- Meaningful units for reasoning
- Natural hierarchy emerges

### 3. geDIG Effectiveness
- Node merger = memory compression (GED reduction)
- Edge addition = shortcut discovery (IG increase)
- Same algorithm as language tasks

## Implementation Notes

### Node Creation Criteria
```python
def should_create_node(observation):
    return any([
        observation['choice_count'] >= 3,      # Junction
        observation['choice_count'] == 0,      # Dead end
        observation['direction_change'] > 45,  # Sharp turn
        observation['has_landmark'],           # Distinctive feature
        observation['is_goal']                 # Goal found
    ])
```

### Edge Relationships
```python
# Direct reachability only
edge_types = {
    "reachable": True,      # Can navigate between nodes
    "distance": integer,    # Steps between nodes
    "discovered_order": int # When this connection was found
}
# NO path details stored in edges (not needed)
```

### Comparison with Traditional Approaches
- **DFS/BFS**: Remember every cell (inefficient)
- **A***: No memory between runs
- **InsightSpike**: Remember only what matters, optimize during sleep

## Expected Results

1. **First Run**: Similar to other algorithms
2. **After Sleep**: Significant improvement in path efficiency
3. **Maze Variations**: Transfer learning from structural patterns
4. **Memory Efficiency**: 10-20 nodes vs 100+ cells

## Metrics

- Path length (before/after sleep)
- Memory usage (node count)
- Adaptation speed (when maze changes)
- Exploration efficiency (% of maze covered vs steps)

## Implementation Strategy for MacBook

### Development Environment Constraints
- Limited GPU/CPU resources
- Memory constraints (8-16GB RAM typical)
- Need for real-time visualization during development

### Recommended: 2.5D Multi-level Maze

#### Why 2.5D?
- **Algorithmic complexity of 3D**: Multiple floors, vertical connections
- **Computational efficiency of 2D**: No 3D rendering overhead
- **Best of both worlds**: Complex navigation, simple visualization

#### Technical Implementation
```python
class MultiLevelMaze:
    """
    2.5D Maze: Multiple 2D floors connected vertically
    """
    def __init__(self, config):
        self.floors = config['num_floors']  # 3-5 floors
        self.floor_size = config['floor_size']  # (30, 30)
        self.connections = {
            'stairs': [(x, y, z) for stairs],
            'elevators': [(x, y, z_range) for elevators],
            'holes': [(x, y, z) for drops]
        }
        
    def render_mode(self, mode='development'):
        if mode == 'development':
            # Fast 2D top-down view of current floor
            return self.render_2d_current_floor()
        elif mode == 'demo':
            # Pseudo-3D for presentations
            return self.render_pseudo_3d()
        elif mode == 'paper':
            # High-quality static visualization
            return self.render_for_publication()
```

### Pseudo-3D Rendering (Optional)
```python
class LightweightRenderer:
    """
    Wolfenstein 3D-style raycasting renderer
    No GPU required, runs on CPU
    """
    def __init__(self):
        self.resolution = (640, 480)
        self.fov = 60
        self.render_distance = 15
        
    def render_frame(self, player_pos, player_angle):
        # Raycasting: O(width) complexity
        # 30-60 FPS achievable on MacBook
        for x in range(0, self.resolution[0], 2):  # Skip pixels for speed
            ray = self.cast_ray(player_pos, player_angle, x)
            self.draw_column(x, ray.distance, ray.texture)
```

### Development Phases

#### Phase 1: Algorithm Development (2D)
- Use MiniGrid or custom 2D maze
- Fast iteration on InsightSpike algorithms
- No rendering overhead
- Complete wake-sleep cycle implementation

#### Phase 2: Multi-level Extension (2.5D)
- Add vertical dimension
- Test hierarchical memory formation
- Validate 3D navigation strategies
- Still computationally efficient

#### Phase 3: Visualization Enhancement
- Add pseudo-3D rendering for demos
- Create publication-quality figures
- Optional: Record demo videos

### Benchmark Configuration for MacBook

```python
# Realistic experiment parameters
experiment_config = {
    # Maze complexity
    "maze_type": "multi_level",
    "num_floors": 3,
    "floor_size": (30, 30),
    "complexity": "medium",
    
    # Computational limits
    "max_episodes": 1000,
    "render_every_n": 10,  # Only render every 10th episode
    "parallel_envs": 1,    # No parallelization
    "max_memory_mb": 4096, # Memory limit
    
    # Optimization
    "use_vectorized_ops": True,
    "cache_distances": True,
    "sparse_graph": True,  # Don't store every connection
}
```

### Performance Optimization Tips

1. **Graph Sparsity**
   - Only store significant nodes (junctions, landmarks)
   - Use lazy edge computation

2. **Batch Processing**
   - Process multiple steps before rendering
   - Update visualization in chunks

3. **Adaptive Quality**
   - High quality for paper figures
   - Low quality for training
   - Medium quality for demos

### Expected Performance
- Training: 100-200 episodes/minute
- Memory usage: < 2GB for typical experiments
- Rendering: 30+ FPS for pseudo-3D view
- Full experiment: 2-4 hours on MacBook Pro

## Revolutionary Approach: Obstacles as Queries

### Core Innovation
Treating obstacle encounters as implicit query submissions, unifying spatial navigation with language understanding through the same geDIG mechanism.

### Traditional vs Query-Driven Planning

#### Traditional Path Planning
```python
# Pre-planned approach
def traditional_planner(start, goal, complete_map):
    obstacles = identify_all_obstacles(complete_map)
    path = find_optimal_path(start, goal, avoiding=obstacles)
    return execute_path(path)
```

#### InsightSpike Query-Driven Planning
```python
# Dynamic query-based approach
def query_driven_navigation(start, goal):
    while current_pos != goal:
        # Every obstacle becomes a query
        if obstacle_detected():
            query = generate_query(obstacle_context)
            # e.g., "Wall at (5,5), goal at (10,10), how to proceed?"
            
            # Wake mode: Find minimal cost solution
            solution = wake_mode_query_resolution(query)
            
            # Execute and learn
            execute_solution(solution)
            add_to_knowledge_graph(query, solution)
```

### Query Types in Navigation

```python
class NavigationQuery:
    # Physical obstacles
    wall_query = Query(
        type="physical_barrier",
        context={"position": (5,5), "target": (10,10)}
    )
    
    # Conditional obstacles
    locked_door_query = Query(
        type="conditional_access",
        context={"barrier": "door", "condition": "needs_key"}
    )
    
    # Dynamic obstacles
    moving_obstacle_query = Query(
        type="temporal_barrier",
        context={"obstacle": "patrol", "pattern": "periodic"}
    )
    
    # Cognitive obstacles (uncertainty)
    junction_query = Query(
        type="decision_point",
        context={"choices": 4, "goal_direction": "unknown"}
    )
```

### Autonomous Path Discovery via Contradiction Resolution

```python
class ContradictionDrivenPlanner:
    def plan_path(self, start, goal):
        # 1. Initial hypothesis: direct path exists
        hypothesis = DirectPath(start, goal)
        
        # 2. Test hypothesis (simulation or execution)
        result = test_hypothesis(hypothesis)
        
        # 3. If contradiction found, resolve via branching
        if result.contradiction:
            # Generate intermediate waypoints
            resolution = resolve_contradiction(
                result.contradiction,
                method="minimal_branching"
            )
            
            # Recursive resolution
            self.plan_path(start, resolution.intermediate)
            self.plan_path(resolution.intermediate, goal)
```

### Wake-Sleep Cycle in Path Planning

#### Wake Phase (Exploration & Query Resolution)
- Encounter obstacle → Generate query
- Search relevant memories in sphere
- Find minimum cost solution
- Create intermediate nodes as needed

#### Sleep Phase (Path Optimization & Pattern Extraction)
- Consolidate redundant waypoints
- Extract reusable strategies ("always go left around walls")
- Optimize graph structure for future queries
- Transfer patterns across similar obstacles

### Why This Approach Can Achieve SOTA

1. **Unified Framework**
   - Same algorithm for language and spatial tasks
   - Obstacles and questions are both "queries"
   - Knowledge transfer between domains

2. **Adaptive Learning**
   - No need for complete maps
   - Learns from each obstacle encounter
   - Improves with experience via sleep cycles

3. **Robustness**
   - Handles dynamic environments naturally
   - Graceful degradation with partial information
   - Self-improving through wake-sleep cycles

4. **Efficiency**
   - Sparse graph representation (only significant points)
   - Reuses previous solutions
   - Optimizes during idle time (sleep)

### Expected Performance Gains

```python
expected_improvements = {
    "sample_efficiency": "10x fewer episodes to solve new mazes",
    "transfer_learning": "80% knowledge reuse across maze types",
    "dynamic_adaptation": "Real-time response to environment changes",
    "memory_efficiency": "90% reduction in storage vs grid-based methods",
    "interpretability": "Complete audit trail of decisions"
}
```

### Research Impact

This "Obstacles as Queries" paradigm could revolutionize:
- Robotic navigation in unknown environments
- Game AI that learns from player behavior
- General problem-solving frameworks
- Understanding of biological navigation

### Implementation Priority

1. **Core Query-Driven Planner** (Week 1-2)
2. **Contradiction Resolution Engine** (Week 3-4)
3. **Wake-Sleep Integration** (Week 5-6)
4. **Benchmark Evaluation** (Week 7-8)
5. **Paper Writing** (Week 9-12)

## Autonomous Map Formation Through Feature Memory

### Wall as Query and Gradient Formation

When an agent hits a wall, it creates a node with:
```python
wall_node = {
    "position": (x, y),
    "feature": "wall",
    "vector": embed("obstacle_wall"),
    "timestamp": t
}
```

Through donut search logic:
```python
def check_direction(current_pos, direction):
    # Search for walls in that direction
    nearby_walls = donut_search(current_pos, "wall", inner=0, outer=5)
    
    if count_walls_in_direction(nearby_walls, direction) > threshold:
        # High wall density = low information gain
        # Naturally avoid without explicit programming
        return low_priority
```

### Natural Gradient Circuit Formation

The system learns spatial gradients without explicit programming:

1. **Node Creation Cost**
   - New area = High energy (node creation)
   - Known area = Low energy (edge traversal)
   - Creates natural preference for explored paths

2. **Information Gain Gradient**
   - Wall-dense areas = Low ΔIG (repetitive "wall" features)
   - Open areas = High ΔIG (new discoveries)
   - Agent naturally flows toward high IG regions

3. **Emergent Behaviors**
   ```
   Early exploration: Many nodes created (everything is new)
   Mid exploration: Reuse existing nodes, find connections
   Late exploration: Minimal nodes, optimal paths emerge
   ```

### Cognitive Map Emergence

Without being told to "create a map", the system naturally forms:
- Boundary representations (wall clusters)
- Pathway networks (connected open spaces)
- Landmark memories (distinctive features)

This is remarkably similar to:
- Hippocampal place cells
- Entorhinal grid cells
- Border cells in biological navigation

### Key Insight: Laziness as Intelligence

The principle of **avoiding node creation** (high energy cost) leads to:
- Efficient memory usage
- Natural abstraction (similar places → same node)
- Optimal granularity (not too detailed, not too coarse)

"Being lazy about creating new memories" paradoxically creates smarter navigation!

---
*This revolutionary approach positions InsightSpike as a potential paradigm shift in both navigation and AI problem-solving, with strong potential for achieving SOTA results across multiple benchmarks.*