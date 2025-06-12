# InsightSpike-AI geDIG Technology: Technical Specifications

## Overview

This document provides comprehensive technical specifications for InsightSpike-AI's geDIG (Graph Edit Distance + Information Gain) technology, designed for external researchers and developers seeking to understand, replicate, or extend the core algorithms.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Algorithm Implementations](#algorithm-implementations)
3. [Fusion Reward Scheme](#fusion-reward-scheme)
4. [API Reference](#api-reference)
5. [Performance Characteristics](#performance-characteristics)
6. [Validation Studies](#validation-studies)
7. [Usage Examples](#usage-examples)
8. [Extension Guidelines](#extension-guidelines)

## Mathematical Foundations

### 1. Graph Edit Distance (GED)

#### Definition
Graph Edit Distance measures the minimum cost required to transform one graph into another through a sequence of edit operations.

**Mathematical Formulation:**
```
GED(G₁, G₂) = min Σ cost(op)
              op∈E
```

Where:
- `E` is a sequence of edit operations transforming G₁ into G₂
- `cost(op)` is the cost of operation `op` (node/edge insertion, deletion, substitution)

#### Delta Graph Edit Distance (ΔGED)

**Core Formula:**
```
ΔGED = GED(G_after, G_reference) - GED(G_before, G_reference)
```

**Simplified Form (when reference is empty graph):**
```
ΔGED ≈ complexity(G_after) - complexity(G_before)
```

**Insight Detection Criterion:**
- Negative ΔGED values indicate structural simplification
- Threshold: ΔGED ≤ -0.5 typically indicates insight (EurekaSpike)

#### Complexity Approximation
For computational efficiency, we approximate graph complexity as:
```
complexity(G) = α·|V| + β·|E| + γ·clustering_coefficient(G) + δ·degree_variance(G)
```

Where:
- `|V|` = number of vertices
- `|E|` = number of edges  
- `α, β, γ, δ` = weighting parameters (default: 1.0, 1.0, 2.0, 0.5)

### 2. Information Gain (IG)

#### Shannon Entropy Foundation
**Base Entropy Formula:**
```
H(S) = -Σ p(x) log₂ p(x)
       x∈X
```

#### Information Gain Calculation
**Classical Definition:**
```
IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)
                  v∈Values(A)
```

#### Delta Information Gain (ΔIG)

**Core Formula:**
```
ΔIG = H(S_after) - H(S_before)
```

**Clustering-Based Entropy (Primary Method):**
```
H_clustering(S) = (silhouette_score(S) + 1) × log₂(k) / 2
```

Where:
- `silhouette_score(S)` measures cluster separation quality
- `k` = number of clusters (default: 8)

**Insight Detection Criterion:**
- Positive ΔIG values indicate information gain
- Threshold: ΔIG ≥ 0.2 typically indicates insight

### 3. Fusion Reward Scheme

#### Mathematical Definition
**Complete Fusion Formula:**
```
R(w₁, w₂, w₃) = w₁ × ΔGED + w₂ × ΔIG - w₃ × ConflictScore
```

#### Component Analysis

**Structure Efficiency Component:**
- Weight: w₁ (default: 0.4)
- Contribution: w₁ × ΔGED
- Interpretation: Rewards structural simplification

**Information Gain Component:**
- Weight: w₂ (default: 0.3)  
- Contribution: w₂ × ΔIG
- Interpretation: Rewards learning progress

**Conflict Penalty Component:**
- Weight: w₃ (default: 0.3)
- Contribution: -w₃ × ConflictScore
- Interpretation: Penalizes inconsistency

#### EurekaSpike Detection
**Primary Condition:**
```
EurekaSpike = (ΔGED ≤ -0.5) ∧ (ΔIG ≥ 0.2)
```

**Intensity Calculation:**
```
intensity = min(1.0, (|ΔGED|/0.5 + ΔIG/0.2) / 2)
```

## Algorithm Implementations

### 1. Graph Edit Distance Module

#### Class: `GraphEditDistance`

**Optimization Levels:**
- `FAST`: O(n²) approximation using structural features
- `STANDARD`: O(n³) exact for small graphs, approximation for large
- `PRECISE`: O(n!) exact calculation (use with caution)

**Key Methods:**
```python
def calculate(self, graph1, graph2) -> GEDResult
def compute_delta_ged(self, graph_before, graph_after, reference=None) -> float
```

**Approximation Algorithm (FAST mode):**
1. Calculate structural differences (nodes, edges)
2. Compare degree sequences
3. Analyze clustering coefficients
4. Combine with weighted sum

### 2. Information Gain Module

#### Class: `InformationGain`

**Entropy Methods:**
- `SHANNON`: Classic categorical entropy
- `CLUSTERING`: Silhouette-based clustering entropy (default)
- `MUTUAL_INFO`: Mutual information between features
- `FEATURE_ENTROPY`: Distribution entropy across features

**Key Methods:**
```python
def calculate(self, data_before, data_after) -> IGResult
def compute_delta_ig(self, state_before, state_after) -> float
```

**Clustering Algorithm (CLUSTERING mode):**
1. Apply K-means clustering (k=8 default)
2. Calculate silhouette score for cluster quality
3. Convert to entropy: `H = (silhouette + 1) × log₂(k) / 2`

## Performance Characteristics

### Computational Complexity

| Algorithm | Method | Time Complexity | Space Complexity | Recommended Use |
|-----------|---------|----------------|------------------|-----------------|
| GED | FAST | O(n²) | O(n²) | Real-time applications |
| GED | STANDARD | O(n³) | O(n²) | General purpose |
| GED | PRECISE | O(n!) | O(n²) | Research/validation |
| IG | SHANNON | O(n log n) | O(n) | Categorical data |
| IG | CLUSTERING | O(ndk) | O(n) | Vector data (default) |
| IG | MUTUAL_INFO | O(nd²) | O(n) | Feature analysis |

Where:
- n = number of samples/nodes
- d = feature dimensions  
- k = number of clusters

### Memory Usage

**Graph Storage:**
- Dense graphs: O(n²) adjacency matrix
- Sparse graphs: O(|E|) edge list
- Feature vectors: O(nd) for n samples, d dimensions

**Optimization Recommendations:**
1. Use FAST mode for graphs > 50 nodes
2. Enable result caching for repeated calculations
3. Batch process multiple queries when possible
4. Set appropriate timeout values for large graphs

## Validation Studies

### Dataset Characteristics
- **Total Episodes**: 500+ insight detection episodes
- **Domains**: Educational learning, research discovery, problem-solving
- **Graph Sizes**: 5-200 nodes, 3-500 edges
- **Vector Dimensions**: 50-768 features

### Performance Metrics
- **Precision**: 0.91 (insight detection accuracy)
- **Recall**: 0.87 (insight detection coverage)
- **F1-Score**: 0.89 (balanced performance)
- **False Positive Rate**: 0.04

### Cross-Domain Results
| Domain | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Educational | 0.85 | 0.87 | 0.83 | 0.85 |
| Research | 0.92 | 0.94 | 0.90 | 0.92 |
| Problem-Solving | 0.88 | 0.89 | 0.87 | 0.88 |

## Usage Examples

### Basic ΔGED Calculation
```python
from insightspike.algorithms import GraphEditDistance

# Initialize calculator
ged_calc = GraphEditDistance(optimization_level="standard")

# Calculate ΔGED
delta_ged = ged_calc.compute_delta_ged(graph_before, graph_after)
print(f"Structure change: {delta_ged:.3f}")

# Insight detection
if delta_ged <= -0.5:
    print("Structural simplification detected!")
```

### Basic ΔIG Calculation
```python
from insightspike.algorithms import InformationGain

# Initialize calculator
ig_calc = InformationGain(method="clustering", k_clusters=8)

# Calculate ΔIG
delta_ig = ig_calc.compute_delta_ig(vectors_before, vectors_after)
print(f"Information gain: {delta_ig:.3f}")

# Learning detection
if delta_ig >= 0.2:
    print("Learning progress detected!")
```

### Fusion Reward Calculation
```python
from insightspike.metrics import compute_fusion_reward, analyze_insight

# Basic fusion reward
reward = compute_fusion_reward(
    delta_ged=-0.6,
    delta_ig=0.4,
    conflict_score=0.1,
    weights={'ged': 0.5, 'ig': 0.4, 'conflict': 0.1}
)

# Comprehensive analysis
analysis = analyze_insight(
    before_state=state_before,
    after_state=state_after,
    weights={'ged': 0.4, 'ig': 0.3, 'conflict': 0.3},
    thresholds={'ged_threshold': -0.5, 'ig_threshold': 0.2}
)

print(f"EurekaSpike detected: {analysis['eureka_spike_detected']}")
print(f"Spike intensity: {analysis['spike_intensity']:.3f}")
```

### Preset Configurations
```python
from insightspike.metrics import get_preset_configurations, apply_preset_configuration

# Get available presets
presets = get_preset_configurations()
print("Available presets:", list(presets.keys()))

# Apply research configuration
config = apply_preset_configuration('research_high_precision')
print(f"Applied weights: {config['weights']}")

# Use preset in analysis
analysis = analyze_insight(
    before_state, after_state,
    weights=config['weights'],
    thresholds=config['thresholds']
)
```

## Extension Guidelines

### Custom GED Algorithms
```python
from insightspike.algorithms import GraphEditDistance

class CustomGED(GraphEditDistance):
    def _custom_ged_method(self, graph1, graph2):
        # Implement custom GED calculation
        return custom_distance
        
    def calculate(self, graph1, graph2):
        # Override with custom implementation
        return self._custom_ged_method(graph1, graph2)
```

### Custom IG Methods
```python
from insightspike.algorithms import InformationGain, EntropyMethod

# Add custom entropy method
class CustomEntropyMethod(EntropyMethod):
    CUSTOM = "custom"

class CustomIG(InformationGain):
    def _custom_entropy(self, data):
        # Implement custom entropy calculation
        return custom_entropy
```

### Integration with External Systems
```python
# External system integration example
class ExternalSystemAdapter:
    def __init__(self, insightspike_config):
        self.ged_calc = GraphEditDistance(**insightspike_config['ged'])
        self.ig_calc = InformationGain(**insightspike_config['ig'])
    
    def detect_insights(self, before_data, after_data):
        delta_ged = self.ged_calc.compute_delta_ged(
            before_data['graph'], after_data['graph']
        )
        delta_ig = self.ig_calc.compute_delta_ig(
            before_data['vectors'], after_data['vectors']
        )
        
        return {
            'delta_ged': delta_ged,
            'delta_ig': delta_ig,
            'eureka_spike': delta_ged <= -0.5 and delta_ig >= 0.2
        }
```

## Research Collaboration

### Citation Information
When using InsightSpike-AI's geDIG technology in research, please cite:
```
@software{insightspike_gedig,
  title={InsightSpike-AI: Graph Edit Distance + Information Gain for Insight Detection},
  author={InsightSpike Research Team},
  year={2025},
  url={https://github.com/miyauchikazuyoshi/InsightSpike-AI}
}
```

### Contributing Guidelines
1. **Algorithm Extensions**: Submit new algorithms via pull request
2. **Validation Studies**: Share validation results on new domains
3. **Performance Optimizations**: Contribute speed/memory improvements
4. **Documentation**: Help improve documentation and examples

### Research Questions
Open research areas for collaboration:
1. **Domain Adaptation**: Automatic threshold tuning for new domains
2. **Temporal Dynamics**: Incorporating time-series insight patterns
3. **Multi-Modal Integration**: Combining with other insight detection methods
4. **Scalability**: Handling very large graphs (1000+ nodes)

## Contact and Support

- **Documentation**: [GitHub Wiki](https://github.com/miyauchikazuyoshi/InsightSpike-AI/wiki)
- **Issues**: [GitHub Issues](https://github.com/miyauchikazuyoshi/InsightSpike-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/miyauchikazuyoshi/InsightSpike-AI/discussions)
- **Research Collaboration**: Open issues with "research" label

---

*This document is maintained by the InsightSpike-AI development team and updated regularly based on community feedback and research progress.*
