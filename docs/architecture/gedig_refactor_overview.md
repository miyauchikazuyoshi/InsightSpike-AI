# GeDIG Refactor Overview (August 2025)

## Goals

- Unify structural (GED) & informational (IG) change computation
- Provide stable reward surface via warmup + z-score IG scaling
- Enable safe rollout with feature flags & dual evaluation
- Improve observability (rotating CSV metrics)

## Core Components

| Component | Responsibility |
|-----------|----------------|
| `GeDIGCore` | End-to-end calculation of GED, IG, rewards, optional multihop |
| `GeDIGFactory` | Feature flag instantiation (legacy vs refactored) |
| `dual_evaluate` | Parallel run & divergence check |
| `GeDIGLogger` | Rotating CSV export of key metrics |
| Welford Stats | Online IG mean/variance for z-score |

> Note (2025‑10): L3 Graph Reasoner uses GeDIGCore as the default metrics engine via `MetricsSelector` and applies query‑centric local evaluation (top‑K centers, r‑hop) by default. Configure with `graph.ged_algorithm/ig_algorithm` and `metrics.query_centric/*`.

> **Update (2025‑10).** The refactor now surfaces `structural_cost ≥ 0` directly; `structural_improvement` remains as a deprecated alias (`-structural_cost`) for readers that have not migrated yet.

> **Update (2025‑10b).** Candidate gating is now controlled by `metrics.theta_cand`, `metrics.theta_link`, `metrics.candidate_cap`, and optional `metrics.top_m`. These feed a `TwoThresholdCandidateSelector` that produces `k⋆ = min(|S_cand|, candidate_cap)`. When `metrics.ig_denominator = "fixed_kstar"`, L3 passes `k_star` and `log k⋆` to `GeDIGCore.calculate(...)`, enabling the fixed `log K⋆` denominator and the local normalization override (`use_local_normalization = True` ⇒ `Cmax = 1 + k⋆`).

## Data Flow

1. `calculate()` normalizes graphs & extracts features
2. Compute structural_cost (normalized GED ± efficiency/spectral blend)
3. Compute IG (entropy variance reduction)
4. Update IG running stats → z-score
5. Compute hop0 & (if enabled) aggregate rewards
6. Log metrics (rotation if thresholds exceeded)
7. Return `GeDIGResult`

## Key Differences vs Legacy

- Removed product `ged * ig`; now subtraction to decouple magnitude inflation
- Introduced minimal conservation base concept (internal guard; not user-facing yet)
- Reward separated from raw gedig_value; adds tunable weights & warmup
- Added multi-hop breakdown for future context-aware evaluation

## Configuration (excerpt)

```yaml
gedig:
  use_refactored_gedig: true
  use_refactored_reward: true
  lambda_weight: 0.5
  mu: 0.5
  warmup_steps: 10
  enable_multihop: false
  max_hops: 3
  decay_factor: 0.7
  spike_threshold: 0.45
  log_path: logs/gedig/gedig_metrics.csv
  max_log_lines: 50000
  max_log_bytes: 52428800
```

## Roadmap Snapshot

- Phase A (DONE): Core unification, reward refactor, logger, feature flag, smoke tests
- Phase B (DONE): Basic invariants
- Phase C: SpikeDetectionMode & presets
- Phase D: Navigator integration
- Phase E: Stability & reproducibility validation

## Usage Snippet (Linkset‑First)

```python
from insightspike.algorithms.linkset_adapter import build_linkset_info

core = GeDIGFactory.create({'use_refactored_gedig': True})
ls = build_linkset_info(
    s_link=[{"index": 1, "similarity": 1.0}],
    candidate_pool=[],
    decision={"index": 1, "similarity": 1.0},
    query_vector=[1.0],
    base_mode="link",
)
res = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)
print(res.structural_cost, res.ig_z_score, res.hop0_reward)
```

Note: Calling `calculate(...)` without `linkset_info` falls back to graph‑IG and now emits a one‑time deprecation warning. Prefer passing linkset info for paper‑aligned IG.

## Logging Columns

`step,raw_ged,ged_value,structural_cost,structural_improvement,ig_raw,ig_z_score,hop0_reward,aggregate_reward,reward,spike,version`

## Future Extensions

- Adaptive thresholding based on monitored false positive rate
- Divergence telemetry integration into experiment dashboards
- Expanded spectral & multi-scale embeddings coupling

## Navigator Integration (Day3 Update)

The `GeDIGNavigator` now exposes the most recent geDIG computation artifacts for downstream metrics & experimentation:

| Attribute | Type | Description |
|-----------|------|-------------|
| `last_result` | `GeDIGResult \| None` | Full result object from last evaluated action (ref core) |
| `last_reward` | `float \| None` | Reward (hop0) associated with `last_result` |
| `last_spike` | `bool` | Spike flag from `last_result` (False if unavailable) |
| `last_structural_improvement` | `float` | Structural improvement value for last action |

Action selection now captures geDIG results per candidate action when computing energy; the chosen action's result is cached. Metrics scripts and validation helpers transitioned from surrogate heuristics (e.g. memory node count) to these real values.

Implications:

1. Stability (E1) & Reproducibility (E2) tests will be tightened after true spike dynamics confirm adequate variance (threshold reversion: repro CV 0.35 → 0.25).
2. Structural simplification rate now reflects actual `structural_improvement>0` occurrences within a short random walk horizon.
3. False spike rate estimation uses condition: spike True & structural_improvement ≤ 0.

Next steps include surfacing an API hook for ground-truth spike labeling (goal proximity + Δstructural_improvement anomaly) feeding precision/recall metrics.


Last updated: 2025-08-23
