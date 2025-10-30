---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# GED/IG Algorithm Configuration Guide

## Overview

InsightSpike-AI now supports configurable GED (Graph Edit Distance) and IG (Information Gain) algorithms for spike detection. This allows you to balance accuracy and performance based on your specific use case.

## Configuration

### Via config.yaml

Edit the `config.yaml` file in the project root:

```yaml
graph:
  # GED/IG Algorithm Selection
  ged_algorithm: "networkx"   # Options: simple, advanced, networkx
  ig_algorithm: "entropy"     # Options: simple, advanced, entropy
```

### Via SimpleConfig (Python)

```python
from insightspike.config import SimpleConfig

config = SimpleConfig(
    ged_algorithm="networkx",
    ig_algorithm="entropy",
    spike_ged_threshold=0.5,
    spike_ig_threshold=0.2
)
```

## Algorithm Details

### GED Algorithms

1. **simple** (Fastest)
   - Complexity-based approximation
   - O(n) time complexity
   - Best for: Real-time applications, large graphs
   - Accuracy: Low

2. **advanced** (Balanced)
   - PyTorch Geometric-aware calculation
   - Combines structural and embedding distance
   - O(n²) time complexity
   - Best for: General use, medium-sized graphs
   - Accuracy: Medium

3. **networkx** (Most Accurate)
   - Uses NetworkX graph edit distance
   - Can detect graph simplification (negative ΔGED)
   - O(n³) to O(n!) time complexity
   - Best for: Research, small to medium graphs
   - Accuracy: High

### IG Algorithms

1. **simple** (Fastest)
   - Basic information content comparison
   - Uses graph size changes
   - Best for: Quick estimates
   - Accuracy: Low

2. **advanced** (Balanced)
   - Combined structural and embedding-based IG
   - Uses feature distributions
   - Best for: General use
   - Accuracy: Medium

3. **entropy** (Most Accurate)
   - Shannon entropy-based calculation
   - Measures actual information content changes
   - Best for: Research, detailed analysis
   - Accuracy: High

## Performance Comparison

Based on our experiments with 130-episode integrated dataset:

| Configuration | Spike Detection Rate | Avg. Processing Time |
|--------------|---------------------|---------------------|
| simple/simple | 2.0% | ~0.01s/episode |
| advanced/advanced | 2.0% | ~0.05s/episode |
| networkx/entropy | 6.0% | ~0.2s/episode |

## Recommendations

### For Production
```yaml
ged_algorithm: "advanced"
ig_algorithm: "advanced"
```
Good balance of speed and accuracy.

### For Research
```yaml
ged_algorithm: "networkx"
ig_algorithm: "entropy"
```
Maximum accuracy for insight detection.

### For Real-time Applications
```yaml
ged_algorithm: "simple"
ig_algorithm: "simple"
```
Fastest processing, suitable for streaming data.

## Threshold Tuning

Different algorithms require different thresholds:

- **simple**: Lower thresholds (0.1-0.3)
- **advanced**: Medium thresholds (0.3-0.5)
- **networkx**: Higher thresholds (0.5-1.0)

## Example: Detecting Graph Simplification

NetworkX algorithm can detect when a complex graph simplifies to reveal an insight:

```python
# Complex graph (20 nodes) → Simple graph (5 nodes)
# NetworkX GED: -22.5 (negative indicates simplification)
# This indicates a potential "Eureka moment"
```

## Implementation Details

The algorithm selection is handled by `MetricsSelector` class in `src/insightspike/utils/metrics_selector.py`.

Layer 3 (Graph Reasoner) automatically uses the configured algorithms when calculating ΔGED and ΔIG for spike detection.