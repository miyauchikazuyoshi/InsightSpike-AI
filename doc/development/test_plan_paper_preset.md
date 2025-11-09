# Test Plan — Paper Preset Alignment

Scope: Validate that Layer3 (via adapter) and the experiment evaluator produce consistent geDIG components under paper settings.

## What we test

1) Configuration overlay sanity
- L3 receives `graph.sp_scope_mode='union'`, `graph.ig_source_mode='linkset'`, `graph.ged_norm_scheme='candidate_base'`, `graph.sp_eval_mode='fixed_before_pairs'`.
- L3 receives `metrics.ig_denominator='fixed_kstar'`, `metrics.use_local_normalization=True`.
- Context includes `candidate_selection={'k_star','log_k_star'}`.

2) Numerical consistency (hop0)
- For a fixed micro‑graph (e.g., 6–8 nodes), with `k_star` set to the linkset size, compare:
  - ΔGED_norm (candidate_base) between evaluator vs L3
  - ΔH (after-before with fixed log k★) between evaluator vs L3
  - ΔSP_rel under fixed-before-pairs
  - g0 (= ΔGED_norm − λ·(ΔH + γΔSP_rel)) and sign direction of minima

3) Regression on experiment logs
- Run a short episode (maze 11×11, max_steps=30, `--use-main-l3` and the evaluator path) and check that:
  - The difference |g0(L3) − g0(eval)| averaged across steps < ε (e.g., 1e-2…1e-1 depending on SP sampling)
  - Sign agreement for `dg_fire` decisions ≥ 95% for the run

## Current status

- Adapter overlay, selection_summary 注入、λ 同期: 実装済
- GraphConfig ノブ追加、`ConfigPresets.paper()` 登録、CLI preset=paper: 実装済
- L3 query-centric の candidate-base 正規化（hop0 Cmax 反映）: 実装済（k★提供時）
- Added:
  - L3 に `INSIGHTSPIKE_PRESET=paper` の正式分岐（adapter なし構成でも完結）
  - `experiments/maze-query-hub-prototype/tools/compare_eval_l3.py` で hop0 の一致度/DG一致率を数値化
  - ラン長対策: `--checkpoint-interval` による部分書き出し（`--step-log` に対して）

Pending:
  - 実験ロングラン後の不要経路（overlay など）の段階撤去

## How to run (locally or CI‑lite)

1) Adapter sanity (no heavy deps)
- Execute a smoke script that constructs an NX toy graph, loads `ConfigPresets.paper()` or sets equivalent dict overlay, calls `eval_query_centric_via_l3` with the selection summary, and asserts the config fields present in the returned metrics (h_scope=linkset, ig_source=linkset, ged_norm='candidate_base').

2) Experiment A/B (short)
- `PYTHONPATH=src INSIGHTSPIKE_LITE_MODE=1 python experiments/maze-query-hub-prototype/run_experiment_query.py --maze-size 11 --max-steps 30 --linkset-mode --norm-base link --use-main-l3 --output experiments/maze-query-hub-prototype/results/_tmp_summary.json --step-log experiments/maze-query-hub-prototype/results/_tmp_steps.json`
- Post‑process the resulting JSON to compute average |g0(L3) − g0(eval)| and DG firing agreement.
 - 比較ユーティリティ: `python experiments/maze-query-hub-prototype/tools/compare_eval_l3.py --eval-steps <eval_steps.json> --l3-steps <l3_steps.json>`

## Acceptance thresholds

- Config overlay visible in L3 metrics (`ig_source='linkset'`, `h_scope='linkset'`, GED norm candidate_base) for the adapter route.
- |ΔGED_norm|, |ΔH|, |ΔSP_rel| differences within expected tolerance (≤ 1e-2 … 1e-1) for the toy case; sign agreement on g0 minima for the episode ≥ 95%.

## Out of scope

- Full L3 multihop parity: current adapter path focuses on hop0 parity + union scope; per‑hop parity can be evaluated later by enabling `ig_recompute=True` in the evaluator to match L3 behavior.

## Notes

- 不要経路の削除は、迷路実験（paper preset）での安定再現を確認した後に実施する。
