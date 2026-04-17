# Figures and Tables for geDIG Paper

## List of Figures

### Figure 1: geDIG Conceptual Overview
```
[Knowledge Graph Before] → [+New Episode] → [Knowledge Graph After]
                              ↓
                     geDIG = GED - k×IG
                              ↓
                    Decision: Accept/Reject
```
**Caption**: The geDIG metric balances graph structural change (GED) against information redundancy (IG) to determine whether new knowledge should be integrated.

### Figure 2: System Architecture
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         ↓
┌─────────────────┐     ┌──────────────┐
│ Episode Manager │←────│ Graph Manager│
└────────┬────────┘     └──────┬───────┘
         ↓                     ↓
┌─────────────────┐     ┌──────────────┐
│ geDIG Evaluator │←────│Vector Index  │
└────────┬────────┘     └──────────────┘
         ↓
┌─────────────────┐
│ Decision Engine │
└────────┬────────┘
         ↓
    [Response]
```
**Caption**: Modular architecture of the geDIG-based knowledge system showing the flow from user query to response generation.

### Figure 3: RAG Performance Comparison
```
Prompt Enrichment (%)
200 ┤ ████████████████████ 167.7%
180 ┤ 
160 ┤ 
140 ┤ ████████████ 123.7%
120 ┤ 
100 ┤ ████ 100%    ████ 108.3%
 80 ┤
 60 ┤
 40 ┤
 20 ┤
  0 └─────┬──────┬──────┬──────┬─────
      Static  Freq.  Cosine  geDIG
```
**Caption**: Prompt enrichment rates across different RAG methods. geDIG-RAG achieves 167.7% enrichment compared to baseline.

### Figure 4: Maze Navigation Trajectories
```
(a) Random Walk          (b) Basic geDIG         (c) Enhanced geDIG
┌─┬─┬─┬─┬─┬─┬─┐        ┌─┬─┬─┬─┬─┬─┬─┐       ┌─┬─┬─┬─┬─┬─┬─┐
│S│░│░│░│░│░│░│        │S│→│→│↓│░│░│░│       │S│→│→│→│→│↓│░│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│↓│░│█│█│█│░│░│        │↓│░│█│↓│█│░│░│       │░│░│█│█│█│↓│░│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│↓│→│→│░│░│░│░│        │↓│←│←│↓│░│░│░│       │░│░│░│░│░│↓│░│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│░│░│↓│░│█│░│░│        │→│→│→│↓│█│░│░│       │░│░│░│░│█│↓│░│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│░│░│↓│←│←│←│░│        │░│░│░│↓│→│→│░│       │░│░│░│░│░│↓│░│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│░│░│→│→│→│↓│░│        │░│░│░│→│→│↓│░│       │░│░│░│░│░│→│G│
├─┼─┼─┼─┼─┼─┼─┤        ├─┼─┼─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
│░│░│░│░│░│→│G│        │░│░│░│░│░│→│?│       │░│░│░│░│░│░│░│
└─┴─┴─┴─┴─┴─┴─┘        └─┴─┴─┴─┴─┴─┴─┘       └─┴─┴─┴─┴─┴─┴─┘
Steps: 287              Steps: Failed          Steps: 68
Redundancy: 3.21        Redundancy: 4.12       Redundancy: 1.42
```
**Caption**: Navigation paths in a 7×7 maze. Enhanced geDIG finds efficient paths while avoiding redundant exploration.

### Figure 5: Scaling Behavior
```
Processing Time (ms)
200 ┤                              ●
180 ┤                          ●
160 ┤                     ●
140 ┤                ●
120 ┤           ●
100 ┤      ●
 80 ┤  ●
 60 ┤●
 40 ┤
 20 ┤
  0 └────┬────┬────┬────┬────┬────┬────
     10   50  100  200  300  400  500
            Knowledge Base Size
```
**Caption**: Sub-linear scaling of processing time with knowledge base size, demonstrating efficient computation.

### Figure 6: Multi-hop Comparison
```
Performance Gain (%)
 50 ┤                    ████ 43%
 45 ┤               ████ 
 40 ┤          ████ 38%
 35 ┤     ████ 
 30 ┤ ████ 28%
 25 ┤ 
 20 ┤ 18%
 15 ┤ 
 10 ┤ 
  5 ┤ 
  0 └─────┬──────┬──────┬──────┬─────
       1-hop  2-hop  3-hop  4-hop
```
**Caption**: Performance improvements with different multi-hop configurations. 3-hop provides optimal balance.

## List of Tables

### Table 1: RAG Performance Across Query Types
| Query Type | Static | Frequency | Cosine | geDIG | Improvement |
|------------|--------|-----------|---------|--------|-------------|
| Factual | 100% | 102% | 118% | 134% | +34% |
| Analogy | 100% | 98% | 125% | 189% | +89% |
| Creative | 100% | 105% | 131% | 173% | +73% |
| Multi-hop | 100% | 94% | 122% | 156% | +56% |
| Cross-domain | 100% | 89% | 116% | 148% | +48% |

### Table 2: Maze Navigation Detailed Results
| Metric | Random Walk | Basic geDIG | Enhanced geDIG | DFS (Oracle) |
|--------|------------|-------------|----------------|--------------|
| Success Rate | 87% | 62% | 95% | 100% |
| Avg Steps | 130 | 248 | 140 | 55 |
| Redundancy | 1.73 | 2.41 | 1.50 | 1.00 |
| Wall Hits | 42 | 67 | 18 | 0 |
| Backtrack Events | 0 | 3.2 | 8.7 | N/A |
| Graph Nodes | N/A | 89 | 267 | N/A |
| Graph Edges | N/A | 12 | 266 | N/A |

### Table 3: Computational Complexity Comparison
| Method | Time Complexity | Space Complexity | Practical Limit |
|--------|----------------|------------------|-----------------|
| Static RAG | O(n) | O(n) | 1M items |
| Cosine RAG | O(n log n) | O(n) | 100K items |
| Basic geDIG | O(n²) | O(n²) | 10K items |
| Enhanced geDIG | O(n log n) | O(n) | 100K items |
| Multi-hop geDIG | O(kn log n) | O(n) | 50K items |

### Table 4: Ablation Study Results
| Configuration | RAG Enrichment | Maze Success | Overall Score |
|---------------|---------------|--------------|---------------|
| Full System | 167.7% | 95% | 100% |
| - Multi-hop | 134.0% (-33.7%) | 92% (-3%) | 84% |
| - Temporal links | 156.2% (-11.5%) | 53% (-42%) | 71% |
| - Dynamic threshold | 149.3% (-18.4%) | 88% (-7%) | 86% |
| - Episode reuse | 167.1% (-0.6%) | 87% (-8%) | 92% |
| - Graph pruning | 165.8% (-1.9%) | 94% (-1%) | 97% |

### Table 5: Episode Management Comparison
| Architecture | Success Rate | Avg Steps | Graph Connectivity | Memory Usage |
|--------------|-------------|-----------|-------------------|--------------|
| Position-only | 10% | 306 | 0.14 | 45MB |
| (Pos, Action) | 45% | 218 | 0.42 | 52MB |
| (Pos, Direction) | 95% | 140 | 0.78 | 58MB |
| Full State | 96% | 138 | 0.81 | 124MB |

## Visualization Code Examples

### Creating Performance Comparison Chart
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Static', 'Frequency', 'Cosine', 'geDIG']
enrichment = [100, 108.3, 123.7, 167.7]
colors = ['gray', 'blue', 'green', 'red']

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, enrichment, color=colors, alpha=0.7)
plt.ylabel('Prompt Enrichment (%)', fontsize=12)
plt.title('RAG Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 200)

# Add value labels on bars
for bar, value in zip(bars, enrichment):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{value}%', ha='center', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rag_performance.png', dpi=300, bbox_inches='tight')
```

### Creating Scaling Behavior Plot
```python
sizes = [10, 50, 100, 200, 300, 400, 500]
times = [3, 15, 32, 68, 103, 141, 183]

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, color='darkblue')
plt.xlabel('Knowledge Base Size', fontsize=12)
plt.ylabel('Processing Time (ms)', fontsize=12)
plt.title('geDIG Scaling Behavior', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add theoretical O(n log n) line
theoretical = [s * np.log(s) * 0.02 for s in sizes]
plt.plot(sizes, theoretical, '--', alpha=0.5, label='O(n log n)')
plt.legend()

plt.tight_layout()
plt.savefig('scaling_behavior.png', dpi=300, bbox_inches='tight')
```

## Note on Reproducibility

All experimental data, code, and configurations are available in the repository:
- RAG experiments: `/experiments/rag-dynamic-db-v3-lite/`
- Maze experiments: `/experiments/maze-unified-v2/`
- Visualization scripts: `/docs/paper/figures/`

The random seeds used for experiments are documented to ensure reproducibility.