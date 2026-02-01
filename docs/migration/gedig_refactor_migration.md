# geDIG Refactoring Migration Guide

**Version**: 2.0
**Date**: 2026-02-01

---

## Overview

The geDIG module has been refactored from a monolithic 2,159-line file (`gedig_core.py`) into a modular package structure. This guide helps you migrate existing code to use the new structure.

## Key Changes

### Before (Monolithic)

```
src/insightspike/algorithms/
└── gedig_core.py  # 2,159 lines - everything in one file
```

### After (Modular)

```
src/insightspike/algorithms/
├── gedig_core.py  # 779 lines - GeDIGCore orchestration (-64%)
└── gedig/
    ├── __init__.py       #  75 lines - Public API (18 exports)
    ├── types.py          # 128 lines - Enums, dataclasses
    ├── config.py         # 310 lines - GeDIGConfig
    ├── spike.py          # 114 lines - spike detection
    ├── graph_utils.py    # 416 lines - graph utilities
    ├── monitor.py        # 193 lines - GeDIGMonitor
    ├── logger.py         # 137 lines - GeDIGLogger
    ├── selector.py       # 270 lines - orchestration
    ├── linkset.py        # 218 lines - linkset metrics
    ├── multihop.py       # 370 lines - multi-hop calculation
    └── ab_writer_helper.py
```

---

## Backward Compatibility

**All existing imports continue to work.** The refactoring maintains full backward compatibility.

```python
# These imports still work exactly as before
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.gedig_core import GeDIGResult
from insightspike.algorithms.gedig_core import GeDIGMonitor
from insightspike.algorithms.gedig_core import GeDIGLogger
from insightspike.algorithms.gedig_core import GeDIGPresets
from insightspike.algorithms.gedig_core import calculate_gedig
```

---

## Recommended Migration

### 1. Import Types from `gedig` Package

```python
# Old
from insightspike.algorithms.gedig_core import (
    GeDIGResult,
    HopResult,
    ProcessingMode,
    SpikeDetectionMode,
)

# New (recommended)
from insightspike.algorithms.gedig import (
    GeDIGResult,
    HopResult,
    ProcessingMode,
    SpikeDetectionMode,
)
```

### 2. Use GeDIGConfig for Configuration

```python
# Old - scattered environment variable handling
core = GeDIGCore(
    lambda_weight=float(os.getenv("MAZE_GEDIG_LAMBDA", "1.0")),
    node_cost=float(os.getenv("MAZE_GEDIG_NODE_COST", "1.0")),
    # ... more env vars
)

# New - centralized configuration
from insightspike.algorithms.gedig import GeDIGConfig

config = GeDIGConfig.from_env()
core = GeDIGCore(**config.to_dict())

# Or use presets
config = GeDIGConfig.preset("balanced")
core = GeDIGCore(**config.to_dict())
```

### 3. Use Standalone Functions

```python
# Old - using private methods
result = core.calculate(g_before, g_after)
is_spike = core._detect_spike(result)

# New - standalone functions
from insightspike.algorithms.gedig import detect_spike

result = core.calculate(g_before, g_after)
is_spike = detect_spike(
    result,
    mode=core.spike_detection_mode,
    spike_threshold=core.spike_threshold,
    tau_s=core.tau_s,
    tau_i=core.tau_i,
)
```

### 4. Import Graph Utilities Directly

```python
# Old - through GeDIGCore instance
subgraph, nodes = core._extract_k_hop_subgraph(graph, focal, k=2)
eff = core._graph_efficiency(graph)

# New - standalone functions
from insightspike.algorithms.gedig import (
    extract_k_hop_subgraph,
    graph_efficiency,
)

subgraph, nodes = extract_k_hop_subgraph(graph, focal, k=2)
eff = graph_efficiency(graph)
```

---

## Module Reference

| Old Location | New Location | Description |
|--------------|--------------|-------------|
| `gedig_core.ProcessingMode` | `gedig.ProcessingMode` | Processing mode enum |
| `gedig_core.SpikeDetectionMode` | `gedig.SpikeDetectionMode` | Spike detection mode |
| `gedig_core.HopResult` | `gedig.HopResult` | Per-hop result |
| `gedig_core.GeDIGResult` | `gedig.GeDIGResult` | Complete result |
| `gedig_core.LinksetMetrics` | `gedig.LinksetMetrics` | Linkset metrics |
| `gedig_core.GeDIGPresets` | `gedig.GeDIGPresets` | Legacy presets |
| `gedig_core.GeDIGMonitor` | `gedig.GeDIGMonitor` | Runtime monitor |
| `gedig_core.GeDIGLogger` | `gedig.GeDIGLogger` | CSV logger |
| (new) | `gedig.GeDIGConfig` | Configuration class |
| (new) | `gedig.detect_spike` | Spike detection function |
| (new) | `gedig.compute_rewards` | Reward computation |
| (new) | `gedig.graph_efficiency` | Graph efficiency |
| (new) | `gedig.extract_k_hop_subgraph` | Subgraph extraction |
| (new) | `gedig.compute_sp_gain_norm` | SP gain calculation |
| (new) | `gedig.compute_linkset_metrics` | Linkset IG calculation |
| (new) | `gedig.calculate_multihop` | Multi-hop geDIG calculation |

---

## Testing Your Migration

After updating imports, run the geDIG tests:

```bash
# Run geDIG-specific tests
pytest tests/unit/test_gedig_*.py tests/gedig/ tests/unit/gedig/ -v

# Run full test suite
pytest tests/ -v
```

Expected: All tests should pass (673+ tests).

---

## Benefits of Migration

1. **Clearer Code Organization**: Each module has a single responsibility
2. **Easier Testing**: Standalone functions are easier to unit test
3. **Better IDE Support**: Smaller files = faster indexing and better autocomplete
4. **Reduced Cognitive Load**: ~400 lines per file instead of 2,159
5. **Centralized Configuration**: All settings in one place with `GeDIGConfig`

---

## Questions?

If you encounter issues during migration, check:
1. [API Reference](../api/gedig.md)
2. [Refactoring Plan](../design/refactoring_plan.md)
3. Test files in `tests/unit/gedig/` for usage examples
