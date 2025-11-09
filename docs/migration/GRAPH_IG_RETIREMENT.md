# Migration: Retiring graph‑IG (Linkset‑First geDIG)

Status: In progress (staged)

## Why
- Graph‑IG (local entropy over node features) and Linkset‑IG (entropy over candidate weights) measure different phenomena.
- They often yield opposite ΔH signs for the same action (distribution spread vs local compactness), causing confusion and divergent knobs.
- We standardize on Linkset‑IG, which matches the paper definition (ΔH = H_after − H_before, norm by log K; IG = ΔH + γ·ΔSP).

## What changes for callers
- Always pass `linkset_info` to `GeDIGCore.calculate(...)`.
- Use `insightspike.algorithms.linkset_adapter.build_linkset_info(...)` to construct from your selection data:
  - Inputs: `s_link`, `candidate_pool`, `decision{index, similarity}`, optional `query_vector`, `base_mode` ('link'|'mem'|'pool').
- If you cannot form a pool yet, pass a minimal linkset (empty pool) to avoid falling back to graph‑IG (IG≈0 safe path):
  ```python
  from insightspike.algorithms.linkset_adapter import build_linkset_info
  ls = build_linkset_info(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link")
  res = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)
  ```
- Prefer `ig_hop_apply='all'` for paper parity; `hop0` remains for L3‑like behavior.

## Deprecation behavior
- Calling Core without `linkset_info` emits a one‑time WARNING that graph‑IG is in use and will be retired.
- A lightweight unit test asserts the WARNING (see `tests/unit/test_graph_ig_deprecation.py`).

## Thresholds
- Expect θ_AG/θ_DG to need retuning under Linkset‑IG due to scale/semantics changes.
- Experiments provide recommendations (e.g., `rec_theta_dg_p95`) and optional auto‑rerun.

## Rollout (summary)
- Phase 0/1: Warn + adapters + experiments/examples pass `linkset_info`.
- Phase 2: Deprecate Core graph‑IG (tests/docs/CI updated).
- Phase 3: Remove Core graph‑IG decision path; keep `entropy_ig()` as utility only.

