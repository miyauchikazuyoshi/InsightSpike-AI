# geDIG API Reference

**Version**: 2.0 (Refactored)
**Last Updated**: 2026-02-01

---

## Overview

geDIG (Graph Edit Distance and Information Gain) is the core metric system for InsightSpike. This document describes the modular API structure after the refactoring.

## Canonical API

New evaluation code uses the standalone unified core:

```python
from gedig.core import FEval
from gedig.adapters.transformer import TransformerFEval
```

For torch-native public use through InsightSpike:

```python
from insightspike.gedig import (
    compute_delta_f_score,
    compute_structural_profile,
)

# Canonical before/after delta. Lower F is better.
result = compute_delta_f_score(before_attention, after_attention, mask=mask)
print(result.F_mean, result.delta_epc, result.delta_h, result.delta_sp)

# Single-state diagnostic. This is not delta F and has no universal direction.
profile, metrics = compute_structural_profile(after_attention)
```

`compute_f_score(attention, ...)` remains as a compatibility wrapper for the
historical single-state Flash profile and returns historical `delta_*` key
names. `FlashGeDIGLoss(alpha=..., objective="minimize"|"maximize")` makes
the experiment-specific profile direction explicit; the default
`"maximize"` preserves its old `-profile` behavior.

The `insightspike.algorithms.gedig` API documented below is the existing
InsightSpike runtime/experiment compatibility layer. It is not the preferred
foundation for new domain adapters.

## Module Structure

```
insightspike.algorithms.gedig/
├── __init__.py       #  75行 - Public API exports (18 exports)
├── types.py          # 128行 - Data types (Enums, Dataclasses)
├── config.py         # 310行 - Configuration management
├── spike.py          # 114行 - Spike detection functions
├── graph_utils.py    # 416行 - Graph manipulation utilities
├── monitor.py        # 193行 - Runtime monitoring
├── logger.py         # 137行 - CSV logging
├── selector.py       # 270行 - Computation orchestration
├── linkset.py        # 218行 - Linkset metrics computation
├── multihop.py       # 370行 - Multi-hop geDIG calculation
└── ab_writer_helper.py
```

---

## Core Types

### `ProcessingMode`

```python
from insightspike.algorithms.gedig import ProcessingMode

class ProcessingMode(Enum):
    WAKE = "wake"   # Active processing mode
    SLEEP = "sleep" # Consolidation mode
```

### `SpikeDetectionMode`

```python
from insightspike.algorithms.gedig import SpikeDetectionMode

class SpikeDetectionMode(Enum):
    THRESHOLD = "threshold"  # Simple threshold comparison
    AND = "and"              # Both structural AND information conditions
    OR = "or"                # Either structural OR information condition
```

### `HopResult`

Per-hop result in multi-hop geDIG calculation.

```python
from insightspike.algorithms.gedig import HopResult

@dataclass
class HopResult:
    hop: int           # Hop number (0 = focal nodes)
    ged: float         # Normalized GED
    ig: float          # Information gain
    gedig: float       # geDIG value for this hop
    struct_cost: float # Structural cost
    node_count: int    # Nodes in subgraph
    edge_count: int    # Edges in subgraph
    sp: float = 0.0    # Shortest path contribution
    # ... additional diagnostic fields
```

### `GeDIGResult`

Complete result from geDIG calculation.

```python
from insightspike.algorithms.gedig import GeDIGResult

@dataclass
class GeDIGResult:
    gedig_value: float           # Final geDIG score
    ged_value: float             # GED component
    ig_value: float              # IG component
    structural_improvement: float # Structural improvement
    spike: bool                  # Spike detected flag
    hop_results: Dict[int, HopResult]  # Per-hop results
    # ... additional fields

    @property
    def has_spike(self) -> bool:
        """Backward compatibility alias."""
        return self.spike
```

---

## Configuration

### `GeDIGConfig`

Centralized configuration dataclass.

```python
from insightspike.algorithms.gedig import GeDIGConfig

# Default configuration
config = GeDIGConfig()

# From environment variables
config = GeDIGConfig.from_env()

# Using presets
config = GeDIGConfig.preset("balanced")  # balanced, conservative, aggressive, maze, rag

# Custom configuration
config = GeDIGConfig(
    lambda_weight=0.5,
    enable_multihop=True,
    max_hops=3,
    spike_detection_mode="and",
    tau_s=0.15,
    tau_i=0.25,
)

# Convert to dict for GeDIGCore kwargs
kwargs = config.to_dict()
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAZE_GEDIG_LAMBDA` | 1.0 | Lambda weight for IG term |
| `MAZE_GEDIG_NODE_COST` | 1.0 | Node edit cost |
| `MAZE_GEDIG_EDGE_COST` | 1.0 | Edge edit cost |
| `MAZE_GEDIG_EFF_WEIGHT` | 0.3 | Efficiency weight |
| `MAZE_GEDIG_IG_MODE` | raw | IG mode (raw, z, norm) |
| `INSIGHTSPIKE_ENTROPY_TAU` | 1.0 | Entropy temperature |

---

## Spike Detection

### `detect_spike`

Standalone spike detection function.

```python
from insightspike.algorithms.gedig import detect_spike

is_spike = detect_spike(
    result=gedig_result,
    mode="and",           # "threshold", "and", or "or"
    spike_threshold=-0.5,
    tau_s=0.15,           # Structural threshold
    tau_i=0.25,           # Information threshold
    ig_variance=0.0,      # Optional: current IG variance
)
```

### `compute_rewards`

Compute reward values for reinforcement learning.

```python
from insightspike.algorithms.gedig import compute_rewards

compute_rewards(
    result=gedig_result,
    lambda_weight=1.0,
    mu=0.5,
    decay_factor=0.7,
    warmup_steps=10,
    ig_count=current_step,
)
# result.hop0_reward and result.aggregate_reward are updated in-place
```

---

## Graph Utilities

### Core Functions

```python
from insightspike.algorithms.gedig import (
    graph_efficiency,
    spectral_score,
    avg_shortest_path_length_safe,
    compute_sp_gain_norm,
    extract_k_hop_subgraph,
    trim_terminal_edges,
    ensure_networkx,
    pyg_to_networkx,
    extract_features,
    filter_features,
    compute_ged_min_proxy,
)

# Graph efficiency metric
eff = graph_efficiency(graph)  # 0.7 * global_efficiency + 0.3 * clustering

# Spectral score (Laplacian eigenvalue std)
score = spectral_score(graph)

# Safe average shortest path length
avg_sp = avg_shortest_path_length_safe(graph, node_cap=200, pair_samples=400)

# Shortest path gain between graphs
gain = compute_sp_gain_norm(g_before, g_after, mode='relative')

# Extract k-hop subgraph
subgraph, nodes = extract_k_hop_subgraph(graph, focal_nodes={'a', 'b'}, k=2)

# Convert various formats to NetworkX
nx_graph = ensure_networkx(pyg_data)  # PyG, numpy array, or nx.Graph
```

---

## Linkset Metrics

### `compute_linkset_metrics`

Compute IG based on linkset (paper-aligned mode).

```python
from insightspike.algorithms.gedig import compute_linkset_metrics

metrics = compute_linkset_metrics(
    linkset_info=linkset_info,
    config=config,
    lambda_weight=1.0,
    entropy_tau=1.0,
)
# Returns LinksetMetrics with delta_ged_norm, delta_h_norm, gedig_value, etc.
```

---

## Multi-hop Calculation

### `calculate_multihop`

Standalone multi-hop geDIG calculation with callback support.

```python
from insightspike.algorithms.gedig import calculate_multihop

result = calculate_multihop(
    g1=graph_before,
    g2=graph_after,
    features_before=features_before,
    features_after=features_after,
    focal_nodes={'node_a', 'node_b'},
    start_time=time.time(),
    max_hops=3,
    decay_factor=0.7,
    adaptive_hops=True,
    use_multihop_sp_gain=True,
    # Optional: custom calculators
    ged_calculator=custom_ged_fn,
    ig_calculator=custom_ig_fn,
)
# Returns GeDIGResult with hop_results dict
```

---

## Monitoring

### `GeDIGMonitor`

Runtime monitoring with auto-threshold adjustment.

```python
from insightspike.algorithms.gedig import GeDIGMonitor

monitor = GeDIGMonitor(
    window_size=200,
    target_fp_rate=0.1,
    adjust_factor=1.1,
)

# Attach to core
core.attach_monitor(monitor)

# Manual usage
monitor.record_prediction(predicted_spike=True)
monitor.record_outcome(actual_spike=True)
monitor.auto_adjust_thresholds(core)

# Get metrics
metrics = monitor.get_metrics()
# {'spike_rate': 0.15, 'false_positive_rate': 0.05, ...}

# Export to file
monitor.export_metrics("metrics.json", core)
```

---

## Logging

### `GeDIGLogger`

CSV logging with rotation.

```python
from insightspike.algorithms.gedig import GeDIGLogger

logger = GeDIGLogger(
    output_path="gedig_log.csv",
    max_lines=50_000,
    max_bytes=50 * 1024 * 1024,
    compress_on_rotate=True,
)

# Attach to core
core.logger = logger

# Or log manually
logger.log(step=0, result=gedig_result)
logger.close()
```

---

## Legacy Compatibility

### GeDIGCore (gedig_core.py)

The main `GeDIGCore` class remains in `gedig_core.py` for backward compatibility.

```python
# These imports continue to work
from insightspike.algorithms.gedig_core import (
    GeDIGCore,
    GeDIGResult,
    GeDIGMonitor,
    GeDIGLogger,
    GeDIGPresets,
    ProcessingMode,
    SpikeDetectionMode,
    calculate_gedig,
    detect_insight_spike,
    delta_ged,
    delta_ig,
)
```

### GeDIGPresets (Legacy)

```python
from insightspike.algorithms.gedig import GeDIGPresets

# Class attributes with preset dictionaries
GeDIGPresets.CONSERVATIVE  # tau_s=0.2, tau_i=0.3, mode="and"
GeDIGPresets.BALANCED      # tau_s=0.15, tau_i=0.25, mode="and"
GeDIGPresets.AGGRESSIVE    # tau_s=0.08, tau_i=0.15, mode="or"
```

---

## Migration Guide

See [gedig_refactor_migration.md](../migration/gedig_refactor_migration.md) for detailed migration instructions.

### Quick Migration

```python
# Old import
from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGResult

# Existing InsightSpike runtime compatibility import
from insightspike.algorithms.gedig import GeDIGResult, GeDIGConfig
from insightspike.algorithms.gedig_core import GeDIGCore

# New adapters: use the standalone unified core
from gedig.core import FEval
from gedig.adapters.rag import RAGFEval
```
