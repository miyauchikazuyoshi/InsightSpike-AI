# Advanced Metrics System (January 2025)

This document details the advanced metric implementations added to InsightSpike-AI in January 2025, focusing on theoretically grounded and scale-invariant approaches to insight detection.

## Table of Contents
1. [Overview](#overview)
2. [Normalized GED](#normalized-ged)
3. [Local Information Gain (IG) V2](#local-information-gain-ig-v2)
4. [Entropy Variance IG](#entropy-variance-ig)
5. [Multi-hop geDIG](#multi-hop-gedig)
6. [Configuration and Usage](#configuration-and-usage)
7. [Integration Example](#integration-example)

---

## Overview

The January 2025 update introduces four major improvements to the geDIG (Graph Edit Distance × Information Gain) calculation system:

1. **Scale Invariance**: Metrics now produce comparable values across different graph sizes
2. **Theoretical Grounding**: Information-theoretic foundations for all calculations
3. **Multi-level Analysis**: Ability to detect insights at different abstraction levels
4. **Clear Responsibility Separation**: GED handles structure, IG handles information flow

### Key Principles

1. **GED (Structure)**: Measures how graph connectivity changes
2. **IG (Information)**: Measures how information distribution changes
3. **geDIG (Insight)**: Product of structural and informational changes

---

## Normalized GED

### Normalized GED Overview

Normalizes Graph Edit Distance to a [-1, 1] range, making values comparable across different graph sizes.

### Normalized GED Algorithm

```python
def calculate_normalized_ged(graph_before, graph_after):
    # Raw GED calculation
    raw_ged = compute_edit_distance(graph_before, graph_after)
    
    # Maximum possible GED
    max_nodes = max(graph_before.num_nodes, graph_after.num_nodes)
    max_edges = max(graph_before.num_edges, graph_after.num_edges)
    max_possible_ged = max_nodes + max_edges
    
    # Normalize to [0, 1]
    normalized_ged = raw_ged / max_possible_ged
    
    # Calculate structural improvement metric
    complexity_before = compute_complexity(graph_before)
    complexity_after = compute_complexity(graph_after)
    
    if complexity_after < complexity_before:
        # Simplification (good for understanding)
        return -normalized_ged
    else:
        # Complexification
        return normalized_ged
```

### Normalized GED Interpretation
 
1. **[-1, 0]**: Structural simplification (insight through reduction)
2. **[0, 1]**: Structural complexification (added connections)
3. **Magnitude**: Degree of change relative to graph size
 

### Normalized GED Benefits
 
1. Compare small (10 nodes) and large (10K nodes) graphs fairly
2. Consistent thresholds across different domains
3. Clear interpretation of improvement vs. degradation
 

---

## Local Information Gain (IG) V2


### Overview (Local IG)
Measures information flow changes using diffusion-based propagation and local entropy calculations.

### Key Concepts

1. **Surprise Calculation**: How different a node is from its neighbors
2. **Information Diffusion**: PageRank-like propagation through edges
3. **Local Entropy**: Information diversity in node neighborhoods

### Algorithm Components

#### 1. Surprise Metrics

```python
def calculate_surprise(node, neighbors, features):
    # Distance-based surprise
    distances = [distance(features[node], features[n]) for n in neighbors]
    distance_surprise = np.mean(distances)
    
    # Entropy-based surprise
    combined_features = np.vstack([features[n] for n in neighbors])
    local_entropy = calculate_entropy(combined_features)
    entropy_surprise = 1.0 / (1.0 + local_entropy)
    
    return (distance_surprise + entropy_surprise) / 2
```

#### 2. Information Diffusion

```python
def diffuse_information(graph, surprises, iterations=5):
    info = surprises.copy()
    
    for _ in range(iterations):
        new_info = np.zeros_like(info)
        for node in graph.nodes():
            # Keep some original information
            new_info[node] = 0.5 * info[node]
            
            # Receive from neighbors
            for neighbor in graph.neighbors(node):
                weight = 1.0 / graph.degree(neighbor)
                new_info[node] += 0.5 * weight * info[neighbor]
        
        info = new_info
    
    return info
```

### Metrics Calculated
1. **Global IG**: Overall information change
2. **Homogenization**: Reduction in information variance
3. **Edge Tension Reduction**: Decreased differences across edges

---

## Entropy Variance IG


### Overview (Entropy Variance IG)
A theoretically cleaner approach that measures information integration through entropy variance reduction.

### Theory

- **Local Entropy**: Shannon entropy of features in each node's neighborhood
- **Variance**: Spread of entropy values across the network
- **Integration**: Variance reduction indicates knowledge consolidation

### Algorithm (Entropy Variance IG)

```python
def calculate_entropy_variance_ig(graph, features_before, features_after):
    # Calculate local entropies
    entropies_before = []
    entropies_after = []
    
    for node in graph.nodes():
        # Get neighborhood features
        neighbors = list(graph.neighbors(node))
        if include_self:
            neighbors.append(node)
        
        # Calculate Shannon entropy
        features_neighborhood_before = features_before[neighbors]
        features_neighborhood_after = features_after[neighbors]
        
        entropy_before = shannon_entropy(features_neighborhood_before)
        entropy_after = shannon_entropy(features_neighborhood_after)
        
        entropies_before.append(entropy_before)
        entropies_after.append(entropy_after)
    
    # Calculate variances
    variance_before = np.var(entropies_before)
    variance_after = np.var(entropies_after)
    
    # IG is variance reduction
    ig = variance_before - variance_after
    
    return ig
```

### Entropy Variance IG Interpretation

- **Positive IG**: Information became more integrated
- **Negative IG**: Information became more dispersed
- **Zero IG**: No change in information distribution

### Entropy Variance IG Advantages

- Purely information-theoretic (no structural overlap with GED)
- Scale-invariant through normalization
- Captures global information integration patterns

---

## Multi-hop geDIG


### Multi-hop Overview

Extends geDIG calculation to different neighborhood sizes, enabling insight detection at multiple abstraction levels.

### Concept
 
1. **0-hop**: Direct node changes only
2. **1-hop**: Include immediate neighbors
3. **2-hop**: Include neighbors of neighbors
4. **k-hop**: Include k-distance neighborhood
 

### Algorithm (Multi-hop)

```python
def calculate_multihop_gedig(graph_before, graph_after, focal_nodes, max_hops=3):
    results = {}
    
    for hop in range(max_hops + 1):
        # Extract k-hop subgraph
        subgraph_before = extract_k_hop_subgraph(graph_before, focal_nodes, hop)
        subgraph_after = extract_k_hop_subgraph(graph_after, focal_nodes, hop)
        
        # Calculate GED and IG for subgraph
        ged = calculate_ged(subgraph_before, subgraph_after)
        ig = calculate_ig(subgraph_before, subgraph_after)
        
        # Apply decay factor
        weight = decay_factor ** hop
        
        # Store results
        results[hop] = {
            'ged': ged,
            'ig': ig,
            'gedig': ged * ig,
            'weight': weight,
            'weighted_gedig': weight * ged * ig
        }
    
    # Total weighted geDIG
    total = sum(r['weighted_gedig'] for r in results.values())
    
    return total, results
```

### Use Cases

1. **Immediate Impact** (0-hop): Direct conceptual changes
2. **Local Context** (1-hop): How change affects immediate connections
3. **Broader Impact** (2-hop): Ripple effects through the network
4. **Deep Context** (3+ hop): Long-range conceptual relationships

### Adaptive Stopping
```python
if adaptive_hops:
    improvement = abs(results[hop]['gedig'] - results[hop-1]['gedig'])
    if improvement < min_improvement:
        break  # Stop if no significant change
```

---

## Configuration and Usage

### Enabling Advanced Metrics

In `config.yaml`:
```yaml
graph:
  # Algorithm selection
  ged_algorithm: simple
  ig_algorithm: simple
  
  # Advanced metrics flags
  use_normalized_ged: true        # Enable scale-invariant GED
  use_entropy_variance_ig: true   # Enable entropy variance IG
  use_local_ig: false            # Use Local IG V2 (alternative to entropy)
  
  # Multi-hop configuration
  use_multihop_gedig: true
  multihop_config:
    max_hops: 3                  # Maximum hop distance
    decay_factor: 0.7            # Weight decay per hop
    adaptive_hops: true          # Stop early if no improvement
    min_improvement: 0.1         # Minimum change to continue
    ged_weight: 0.5             # Weight for GED in geDIG
    ig_weight: 0.5              # Weight for IG in geDIG
```

### Programmatic Usage

```python
from insightspike.algorithms.gedig_calculator import GeDIGCalculator

# Load configuration
config = {
    'graph': {
        'use_normalized_ged': True,
        'use_entropy_variance_ig': True,
        'use_multihop_gedig': True,
        'multihop_config': {
            'max_hops': 2,
            'decay_factor': 0.7
        }
    }
}

# Create calculator
calculator = GeDIGCalculator(config)

# Calculate geDIG
result = calculator.calculate(
    graph_before, graph_after,
    features_before, features_after,
    focal_nodes=[new_insight_node]
)

# Access results
print(f"Total geDIG: {result['gedig']}")
print(f"Components: GED={result['ged']}, IG={result['ig']}")

if 'multihop_results' in result:
    for hop, details in result['multihop_results']['hop_details'].items():
        print(f"Hop {hop}: {details['gedig']}")
```

---

## Integration Example

### Complete Insight Detection Pipeline

```python
from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent
from insightspike.algorithms.gedig_calculator import GeDIGCalculator

# Configure with advanced metrics
config = load_config()
config.graph.use_normalized_ged = True
config.graph.use_entropy_variance_ig = True
config.graph.use_multihop_gedig = True
config.graph.multihop_config.max_hops = 3

# Initialize agent
agent = MainAgent(config)
agent.initialize()

# Add knowledge
agent.add_knowledge("Energy equals mass times speed of light squared")
agent.add_knowledge("This is Einstein's famous equation E=mc²")
agent.add_knowledge("It shows mass-energy equivalence")

# Process question
result = agent.process_question("What does E=mc² mean for nuclear physics?")

# Multi-hop analysis shows:
# - Hop 0: Direct connection to equation
# - Hop 1: Links to mass-energy concept
# - Hop 2: Broader physics implications
# - Hop 3: Deep theoretical connections

if result.has_spike:
    print("Multi-level insight detected!")
    # The system recognized connections at different abstraction levels
```

### Interpreting Multi-hop Results

```python
# Example output interpretation
hop_results = result.multihop_results['hop_details']

for hop, details in sorted(hop_results.items()):
    if hop == 0:
        print(f"Immediate insight: {details['gedig']:.3f}")
        # Direct conceptual breakthrough
    elif hop == 1:
        print(f"Local connections: {details['gedig']:.3f}")
        # How insight affects nearby concepts
    elif hop == 2:
        print(f"Broader impact: {details['gedig']:.3f}")
        # Ripple effects through knowledge
    else:
        print(f"Deep structure: {details['gedig']:.3f}")
        # Fundamental reorganization
```

---

## Performance Considerations

### Computational Complexity
- **Normalized GED**: O(V + E) - same as regular GED
- **Entropy Variance IG**: O(V × D × F) - V nodes, D degree, F features
- **Multi-hop**: O(H × V^H) - H hops, exponential in hop count

### Optimization Strategies
1. **Limit max_hops** to 2-3 for real-time applications
2. **Use adaptive stopping** to avoid unnecessary computation
3. **Cache subgraph extractions** when analyzing multiple focal nodes
4. **Increase decay_factor** to emphasize local changes

### Memory Usage
- Multi-hop requires storing H subgraphs
- Entropy calculations need neighborhood features
- Consider batch processing for large graphs

---

## Best Practices

1. **Choose Metrics Wisely**
   - Use entropy variance for information-centric domains
   - Use local IG for surprise-based discovery
   - Enable multi-hop for research/analysis tasks

---

## August 2025 Refactor Update (geDIG Phase A Completion)

### Summary
The August 2025 refactor unified previously fragmented implementations into a single `GeDIGCore` with:

- Minimal conservation base: `|GED_raw| + |IG_raw|` (removed secondary normalizations)
- Explicit separation of structural improvement vs. information integration
- Reward layer with warmup + dual forms: `hop0_reward` and `aggregate_reward`
- Online IG statistics (Welford O(1)) producing `ig_z_score`
- Feature flag & dual evaluation (`GeDIGFactory`, `dual_evaluate`) for safe rollout
- Rotating CSV logger (`GeDIGLogger`) capturing normalized & raw metrics

### Revised Core Formulas
Structural Improvement (SI): sign-aware normalized GED blended with efficiency (and optional spectral) change.

Information Gain (IG): entropy variance reduction (variance_before - variance_after).

Raw geDIG core value (current refactor):

```text
gedig_value = SI - IG   # product removed to decouple magnitude blending
```
(Historical docs referencing `GED * IG` are now legacy.)

Reward (hop0):

```text
R_hop0 = λ * Z(IG_raw) + μ * SI
    where λ=0 during warmup_steps (stability period)
```

Aggregate Reward (multihop enabled):

```text
R_agg = λ * Z(IG_raw) + μ * Σ_w (w_h * SI_h) / Σ_w w_h,  w_h = decay_factor^hop
```

### New Runtime Fields (GeDIGResult)

| Field | Description |
|-------|-------------|
| `raw_ged` | Unnormalized edit distance (node + edge ops) |
| `ged_value` | Normalized GED value |
| `structural_improvement` | Signed, efficiency/spectral adjusted structural change |
| `ig_raw` | Raw IG (variance reduction) |
| `ig_z_score` | Z-score of IG (running stats) |
| `hop0_reward` | Reward using only hop0 structural improvement |
| `aggregate_reward` | Reward using weighted multihop structural improvements |
| `reward` | Alias (currently hop0) |
| `hop_results` | Per-hop breakdown (if multihop enabled) |

### Logging Enhancements

CSV columns (initial set):

```text
step,raw_ged,ged_value,structural_improvement,ig_raw,ig_z_score,hop0_reward,aggregate_reward,reward,spike,version
```
Rotation triggers: ≥50,000 lines OR ≥50MB file size.

### Feature Flag & Dual Evaluation

```python
from insightspike.algorithms.gedig_factory import GeDIGFactory, dual_evaluate
legacy = GeDIGFactory.create({'use_refactored_gedig': False})
ref    = GeDIGFactory.create({'use_refactored_gedig': True})
result, delta = dual_evaluate(legacy, ref, g_prev=G1, g_now=G2)
```
`delta` beyond threshold → warning log (regression signal).


### Planned (Day 1+)

- SpikeDetectionMode (AND / OR / THRESHOLD) with presets
- False positive monitor & adaptive thresholds
- Divergence telemetry export

### Migration Notes
| Legacy Concept | New Equivalent / Status |
|----------------|-------------------------|
| `gedig = ged * ig` | Replaced by `structural_improvement - ig_value` |
| Separate GED / IG calculators | Unified in `GeDIGCore` |
| Manual normalization chains | Minimal conservation base only |
| Static thresholds | Incoming mode-based detection (Phase C) |

Refer also to `docs/development/gedig_refactoring_plan_unified_2025.md` for phased roadmap.


2. **Configure for Your Domain**
   - Low decay_factor (0.5) for local-focused analysis
   - High decay_factor (0.9) for global pattern detection
   - Adjust max_hops based on graph connectivity

3. **Monitor Performance**
   - Track computation times with agent.get_stats()
   - Watch for adaptive stopping patterns
   - Profile memory usage with large graphs

4. **Interpret Results**
   - Positive geDIG: Structural insight found
   - Negative geDIG: Simplification insight
   - Multi-hop patterns reveal insight depth

---

## Future Directions

1. **Learnable Parameters**: Automatically tune decay factors
2. **Graph Neural Networks**: Replace heuristics with learned representations  
3. **Hierarchical Analysis**: Multi-resolution graph decomposition
4. **Causal Discovery**: Use multi-hop to infer causal chains
5. **Explainable Insights**: Generate natural language explanations of hop patterns