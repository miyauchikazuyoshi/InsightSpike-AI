# geDIG Selector and Core Architecture (2026-02)

This document summarizes the canonical geDIG entrypoint, the modular package structure, and the supporting guardrails.

> **Last Updated**: 2026-02-01 (Post-Refactoring)

## Package Structure

The geDIG package has been refactored into 10 specialized modules:

```
src/insightspike/algorithms/gedig/
├── __init__.py       #  75行 - 公開API（18エクスポート）
├── types.py          # 128行 - 型定義（ProcessingMode, SpikeDetectionMode, HopResult, GeDIGResult, LinksetMetrics）
├── config.py         # 310行 - GeDIGConfig（from_env, from_kwargs, preset, to_dict）
├── spike.py          # 114行 - スパイク検出（detect_spike, compute_rewards）
├── graph_utils.py    # 416行 - グラフ操作（11関数）
├── monitor.py        # 193行 - GeDIGMonitor
├── logger.py         # 137行 - GeDIGLogger（Rotating CSV）
├── selector.py       # 270行 - TwoThresholdCandidateSelector, compute_gedig
├── linkset.py        # 218行 - compute_linkset_metrics
├── multihop.py       # 370行 - calculate_multihop
└── ab_writer_helper.py # A/B writer utilities
```

**Orchestration**: `algorithms/gedig_core.py` (779行, -64% from 2,159)
  - Uses `GeDIGConfig` for configuration
  - Delegates to modular functions: `calculate_multihop`, `compute_linkset_metrics`, etc.
  - No file I/O; monitoring/logging are optional hooks

## Canonical Entry (Linkset‑First)

- Single entry: `insightspike.algorithms.gedig.selector.compute_gedig(G_prev, G_curr, *, mode)`
  - `mode`: `pure | full | ab`
  - Side‑effect free; A/B logging is opt‑in via writer injection
  - For `mode='full'`, the selector supplies a minimal `linkset_info` to the Core

## Configuration

All geDIG parameters are centralized in `GeDIGConfig`:

```python
from insightspike.algorithms.gedig import GeDIGConfig

# From environment variables
config = GeDIGConfig.from_env()

# From kwargs with env overrides
config = GeDIGConfig.from_kwargs(lambda_weight=0.7, max_hops=3)

# From presets
config = GeDIGConfig.preset("maze")  # or "transformer", "rag", etc.

# Initialize GeDIGCore
from insightspike.algorithms.gedig_core import GeDIGCore
core = GeDIGCore(config=config)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAZE_GEDIG_LAMBDA` | Lambda weight | 1.0 |
| `MAZE_GEDIG_NODE_COST` | Node cost | 1.0 |
| `MAZE_GEDIG_EDGE_COST` | Edge cost | 1.0 |
| `MAZE_GEDIG_EFF_WEIGHT` | Efficiency weight | 0.3 |
| `MAZE_GEDIG_SPECTRAL` | Enable spectral | false |
| `MAZE_GEDIG_IG_MODE` | IG mode (raw/z/norm) | raw |
| `MAZE_GEDIG_ENTROPY_TAU` | Entropy tau | 1.0 |

## Core Composition

- Pure functions: `algorithms/core/metrics.py`
  - `normalized_ged(g1, g2, *, normalization, efficiency_weight, enable_spectral, spectral_weight)`
  - `entropy_ig(graph, features_before, features_after, *, smoothing, min_nodes)`
- Multi-hop: `algorithms/gedig/multihop.py`
  - `calculate_multihop(g1, g2, features_before, features_after, focal_nodes, ...)`
- Linkset metrics: `algorithms/gedig/linkset.py`
  - `compute_linkset_metrics(linkset_info, config, ...)`

## Guardrails

- CI selector enforcement
  - Forbids non‑selector `compute_gedig(...)` and direct `GeDIGCore/PureGeDIGCalculator` use (STRICT=1 fails)
- Public API usage in examples (top‑level)
  - `from insightspike.public import create_agent`
- Nightly KS regression
  - Detects distribution drift in core metrics (KS p‑value)

## A/B Logging Injection

- Use `algorithms/gedig_ab_logger.py` with `set_writer(file_like)`
- Helper: `algorithms/gedig/ab_writer_helper.create_csv_writer(path)`
- MainAgent no longer writes CSV directly; fallback header creation also uses writer injection

## Provider Strict Mode

- `INSIGHTSPIKE_STRICT_PROVIDER=1` forbids legacy/fallback provider initialization
- Direct Local/Ollama initializers are deprecated and scheduled for removal after two stable releases

## Refactoring Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| gedig_core.py lines | 2,159 | 779 | -64% |
| Modules (geDIG) | 1 | 10 | +9 |
| Test count | 53 | 227 | +174 |
| Test coverage | 54% | 84% | +30% |
