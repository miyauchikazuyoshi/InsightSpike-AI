# Spectral GED Enhancement

## Overview

The spectral Graph Edit Distance (GED) enhancement adds Laplacian eigenvalue analysis to the geDIG calculation, enabling better detection of structural quality improvements in knowledge graphs.

## Mathematical Foundation

### Laplacian Matrix and Eigenvalues

The Laplacian matrix L of a graph G is defined as:
- L = D - A
  - D: Degree matrix (diagonal matrix with node degrees)
  - A: Adjacency matrix

The eigenvalues of L capture structural properties:
- λ₀ = 0 (always, for connected components)
- λ₁ > 0 (algebraic connectivity - measures graph connectivity)
- Higher eigenvalues relate to graph regularity and structure

### Spectral Score Calculation

```python
def _calculate_spectral_score(self, g: nx.Graph) -> float:
    """Calculate structural score using Laplacian eigenvalues."""
    L = nx.laplacian_matrix(g).toarray()
    eigvals = np.linalg.eigvalsh(L)
    return np.std(eigvals)  # Standard deviation as irregularity metric
```

The standard deviation of eigenvalues measures structural irregularity:
- Low std dev → More regular/organized structure
- High std dev → More irregular/chaotic structure

## Integration with geDIG

### Enhanced Structural Improvement

When spectral evaluation is enabled:

```python
if self.enable_spectral:
    spectral_before = self._calculate_spectral_score(g1)
    spectral_after = self._calculate_spectral_score(g2)
    
    # Improvement when structure becomes more regular
    spectral_improvement = (spectral_before - spectral_after) / (spectral_before + 1e-10)
    
    # Combine with existing structural improvement
    structural_improvement = (
        structural_improvement * (1 - self.spectral_weight) +
        np.tanh(spectral_improvement) * self.spectral_weight
    )
```

### Configuration

Enable spectral evaluation in `config.yaml`:

```yaml
metrics:
  spectral_evaluation:
    enabled: true     # Enable spectral GED
    weight: 0.3       # Weight for spectral component (0-1)
```

## Benefits

1. **Quality-aware Structure Changes**
   - Detects when added nodes/edges improve graph organization
   - Distinguishes between random growth and meaningful structure

2. **Mathematical Independence**
   - Spectral properties are orthogonal to information gain (IG)
   - Maintains theoretical soundness of geDIG = GED + IG

3. **Backward Compatibility**
   - Disabled by default
   - No impact on existing behavior when disabled
   - Configurable weight for gradual adoption

## Use Cases

### When to Enable Spectral Evaluation

1. **Knowledge Graph Construction**
   - Building structured ontologies
   - Organizing hierarchical information

2. **Insight Detection**
   - Identifying connections that improve overall understanding
   - Detecting "aha moments" that reorganize knowledge

3. **Research and Experiments**
   - Analyzing how different inputs affect graph structure
   - Measuring learning quality beyond simple growth

### Example Scenarios

**Scenario 1: Adding a Hub Node**
- Before: Disconnected clusters
- After: Hub connects clusters
- Result: Lower eigenvalue std dev → Negative GED (improvement)

**Scenario 2: Random Edge Addition**
- Before: Well-structured graph
- After: Random edges added
- Result: Higher eigenvalue std dev → Positive GED (degradation)

## Implementation Details

### Performance Considerations

- Eigenvalue computation: O(n³) for dense matrices
- Cached for repeated calculations
- Only computed when enabled

### Numerical Stability

- Uses `np.linalg.eigvalsh` for symmetric matrices
- Smoothing factor (1e-10) prevents division by zero
- `np.tanh` bounds improvement to [-1, 1]

## Future Enhancements

1. **Spectral Clustering Integration**
   - Use Fiedler vector for community detection
   - Measure inter-cluster connectivity improvements

2. **Adaptive Weighting**
   - Learn optimal spectral_weight from data
   - Different weights for different graph sizes

3. **GPU Acceleration**
   - Use CuPy for large-scale eigenvalue computation
   - Batch processing for multiple graphs

## References

- Chung, F. (1997). Spectral Graph Theory
- Von Luxburg, U. (2007). A Tutorial on Spectral Clustering
- Shuman, D. et al. (2013). The Emerging Field of Signal Processing on Graphs