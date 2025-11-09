# geDIG Selector and Core Architecture (2025-09)

This document summarizes the canonical geDIG entrypoint, the refactored core, and the supporting guardrails.

## Canonical Entry (Linkset‑First)

- Single entry: `insightspike.algorithms.gedig.selector.compute_gedig(G_prev, G_curr, *, mode)`
  - `mode`: `pure | full | ab`
  - Side‑effect free; A/B logging is opt‑in via writer injection
  - For `mode='full'`, the selector supplies a minimal `linkset_info` to the Core. Callers that invoke the Core directly should pass `linkset_info` explicitly to avoid graph‑IG fallback (deprecated).

## Core Composition

- Pure functions: `algorithms/core/metrics.py`
  - `normalized_ged(g1, g2, *, normalization, efficiency_weight, enable_spectral, spectral_weight)`
  - `entropy_ig(graph, features_before, features_after, *, smoothing, min_nodes)`
- Orchestration: `algorithms/gedig_core.py`
  - Computes structural_improvement and IG, optional multihop aggregation
  - No file I/O; monitoring/logging are optional hooks

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
