# Maze 精密アブレーション実験計画（Paper 強化版）

本節は、迷路実験を「精密アブレーション」として再設計し、論文の4本柱（M‑Causal/M‑ROC/Mem‑Growth/Latency）を運用指標で実証するための計画です。既存の集計・図生成スクリプト（docs/paper/*.py）と、新規エクスポータ（experiments/maze-navigation-enhanced/analysis/*.py）に準拠してログ→CSV→図表のパスを確立します。

## 1. 目的と主張
- M‑Causal（NA→multi‑hop→BT のリード/ラグ）: NA発火後にBT/採否/エビクションが有意に立ち上がる（正のlead）。
- M‑ROC（偽橋抑制）: 集約minが低FPR域で偽橋を最も抑制（pAUC@FPR≤0.1↑、探索長の悪化は最小）。
- Mem‑Growth（圧縮）: geDIGはAll‑Add対照より冗長率が低く、ノード/エッジ成長が鈍化。
- Latency（追加レイテンシ）: (H,k)別のP50/P95/P99を提示し、H=3運用でP50≤200msを確認。

## 2. 運用定義（計測の前提）
- NA: 0‑hopの g0 > θNA（補助: L1類似エピソード不在）。
- multi‑hop: gmin = min_h g(h)。NAフレームでmulti‑hop評価を強制。
- BT: b(t) ≤ θBT を満たすとバックトラック発火（b(t)=min(g0,gmin)）。
- ROCのGT（真の橋）: ΔSPL_rel = (SPL_after − SPL_before)/SPL_before ≤ −0.02（相対閾値固定）。
- 追加レイテンシ: 候補列挙→g算出→ゲート判定までの測定時間（ms）。

## 3. 要因設計（アブレーション軸）
- A: NAトリガ定義 — A1: g0>θNA（基準）/ A2: A1+L1不在補助
- B: 集約 — B1: min / B2: soft‑min(τ∈{0.3,0.5,0.8}) / B3: sum
- C: multi‑hop — C0: H=0 / C1: H=1 / C2: H=2 / C3: H=3
- D: 圧縮 — D1: geDIG / D2: All‑Add（対照：全観測追加）
- E: 正規化/ゲート — E1: Cmax^local ON/OFF / E2: L1 gate τL1∈{0.65,0.75,0.85}
- F: 空間/時間ゲート — F1: 空間∈{3,4,5} / F2: 時間窓∈{4,6,8}
- G: 視野 — G1: 3×3 / G2: 5×5
- H: λ — H0: 0（GEDのみ） / Hbase / H×2

P0（最小コア）: A1 × B{1,2,3} × C{0,3} × D{1,2}（E/F/G/Hは基準）で 15×15, seeds=30。
P1（厚み）: 25×25, seeds=10＋G2（5×5）/E/F/Hの感度。

## 4. ログ仕様（CSVスキーマ）
※ 各runから以下の最小ログを保存し、エクスポータで `data/**` に統一出力します。

- M‑Causal: `data/maze_eval/event_alignment.csv`
  - 長形式: run_id, t_from_NA, event{BT|accept|evict}, value{0/1}
  - エクスポータ: `analysis/export_event_alignment.py`

- M‑ROC: `data/maze_eval/bridge_scores.csv`
  - run_id, step, aggregator, score（小さい=橋）, y_true（ΔSPL_rel≤−0.02）
  - エクスポータ: `analysis/export_bridge_scores.py`

- Mem‑Growth: （時系列）
  - 推奨: t, nodes, edges, redundant_ratio, accepted_new, pruned（geDIG/All‑Add両方）
  - 図の入力に合わせCSV整形（fig_mem_growth.pyに準拠）。

- Latency（追加レイテンシ）: `data/latency/maze_latency.csv`
  - run_id, H, k, latency_ms
  - エクスポータ: `analysis/export_latency.py`

- Steps CDF: `data/maze_eval/steps_distribution.csv`
  - run_id, method, steps, success
  - エクスポータ: `analysis/export_steps_distribution.py`

## 5. 実行計画（P0→P1）
- P0: 15×15, seeds=30
  - 条件: A1 × B{min,soft‑min,sum} × C{H=0,H=3} × D{geDIG,All‑Add}
  - ログ: scores.csv / events.csv / latency.csv / steps.csv / mem_growth.csv を各runに保存
  - エクスポート → 図/表生成（下記）
- P1: 25×25, seeds=10 + G2/E/F/Hの感度

## 6. エクスポート → 図/表生成
- エクスポート（既定パス・自動スキャン）
  - `python experiments/maze-navigation-enhanced/analysis/export_bridge_scores.py`
  - `python experiments/maze-navigation-enhanced/analysis/export_event_alignment.py`
  - `python experiments/maze-navigation-enhanced/analysis/export_latency.py`
  - `python experiments/maze-navigation-enhanced/analysis/export_steps_distribution.py`
- 図/表
  - M‑ROC: `python docs/paper/fig_maze_bridge_roc.py`
  - M‑Causal: `python docs/paper/fig_maze_event_alignment.py`
  - Mem‑Growth: `python docs/paper/fig_mem_growth.py`
  - Steps CDF: `python docs/paper/fig_steps_cdf.py`
  - Latency表: `python docs/paper/tab_latency_summary.py`

## 7. 受け入れ基準（運用定義に基づく）
- M‑Causal: NA(t=0)後、BT/accept/evictの条件付き発火確率が正のリードを示す（例: BTピーク t∈[+2,+6]）。
- M‑ROC: 低FPR域（FPR≤0.1）の部分AUCで min ≥ soft‑min ≥ sum。PR/AUPRとTop‑K FDRを補助。
- Mem‑Growth: geDIGがAll‑Add対照より冗長率↓、ノード/エッジ成長鈍化。
- Latency: (H,k)別P50/P95/P99を提示し、H=3運用でP50≤200ms。

---
# Maze Navigation (NA→DA, L1-Frontier) — Paper Results Outline

Note: A Japanese outline is included at the end of this file.

This document proposes what to include in the paper and how to reproduce it from this repo. It matches the current experimental stack (NA→DA gating, L1=global recall with walls included, frontier‐biased forced L1, hop=min wiring) and the live viewer outputs.

## 0. TL;DR of the Method

- If‑less control by continuous scores and thresholds only:
  - Wiring acceptance: geDIG ≤ τ_edge (no forced acceptance in PoC).
  - Backtrack trigger: (geDIG0 > τ_NA) ∧ (bt_eval ≤ τ_BT), where bt_eval aggregates DA only when NA (agg=na_min).
  - Target choice: argmin_x ||q − v(x)|| (weighted L2) with index‑only option (no frontier pre‑definition).
- Two optional target sources:
  - frontier: visited cells that border unexplored (legacy, still supported).
  - index (recommended PoC): vector index Top‑K → nearest by distance; representative at a position chosen by best‑match episode (not “most recent”).
- Optional global ΔSP (pos‑level average shortest path) is off in PoC and can be enabled for braided suites.

## 1. Datasets / Mazes

- Synthetic mazes (15×15, 25×25).
- Generator options (env):
  - `MAZE_BRAID_PROB` (default 0.0; e.g., 0.05) adds loops (braided mazes) to make shortcuts measurable.
  - `MAZE_GOAL_FARTHEST=1` makes the goal the farthest reachable cell from start.

Report which seeds are used. Example sets: 

- 15×15: {5, 7, 12, 21, 33, 57, 77}
- 25×25: {33, 57, 88, 101}

## 2. Configurations (Ablations)

Include these four primary configurations for each maze/seed:

1) Full (ours)
   - Query‑Hub: ON; L1=global recall (walls included); forced‑L1(frontier)
   - Wiring Top‑K=3; hop=min; BT aggregation=na_min; SP(pos‑level), NA‑forced SP

2) −forced‑L1 (remove frontier bias)

3) −DA (0‑hop only; `MAZE_BT_AGG=base` and/or `MAZE_USE_HOP_DECISION=0`)

4) −QHUB (no query hub injection)

Optional baselines: greedy DFS‑like (no geDIG), BFS‑like backtracking, random walk.

## 2.5 Current Status Snapshot (2025‑09‑22)

- Latest aggregates
  - CSV: `results/fast_results_all.csv`, `results/pure_ge_bt_l1tau_summary.csv`
  - JSON: `results/final_gedig_test/summary_*.json`
  - Gallery: `scripts/build_gallery.py` renders `index.html` from folders with `run_summary.json`.
- Suites covered
  - 15×15, 25×25: multiple seeds with NA→DA gating; “Full” uses forced‑L1/hop=min; PoC uses geDIG‑only acceptance and index‑only targeting.
  - 50×50: quick runs available via `scripts/run_50x50_quick.sh` (semantic BT, index‑only target, memory_graph path, ΔSP off).
- Open items
  - Finalize θNA/θBT per size (narrow bands identified; 50×50 PoC: NA≈−0.005, BT≈+0.004〜0.005).
  - Consolidate ΔSP capture budgets by size; confirm impact on braided vs non‑braided mazes.
  - Lock “Full (ours)” vs ablations for the paper table (F4) and freeze seeds.

## 3. Metrics to Report

- Success, steps to goal
- Path optimality ratio (steps / shortest‐path length) — compute via maze BFS
- Backtrack statistics: triggers, executed steps
- Coverage (%) vs steps
- Wiring statistics: total edges, per‑type counts (geDIG / l1_force / query / trajectory / hub / spatial / other)
- L1 recall stats: candidate count per step (min/mean/max), forced‑L1 uses (% of edges)
- geDIG series: mean, min, AUC(<0), NA step ratio
- ΔSP (if enabled): count of negative events, min/mean negative magnitude, AUC(negative)
- BT suppression markers (viewer only): steps where NA/BT conditions hold but trigger is suppressed (plan active / cooldown). Helps explain trigger vs execution.

## 4. Figures / Tables

- F1: Episode graph + exploration map (one run):
  - Left: episode graph (color=timestamp; edge color by strategy; width=|geDIG|)
  - Right: maze with path (coverage annotated)

- F2: Time series:
  - geDIG(t) with NA threshold line, bt events
  - ΔSP(t) (negatives highlighted)
  - Growth (nodes/edges)

- F3: Wiring composition: stacked bar (geDIG / l1_force / …)

- F4: Ablation table:
  - success, steps, optimality, bt count, ΔSP<0 count/mean, composition

- F5: Hyper sweeps (brief): forced‑L1 τ vs performance, L1 τ, BT threshold

## 5. Repro Scripts

Seed runners (viewer同梱):

- 15×15 seeds: `./scripts/run_15x15_seeds.sh 5 7 12 21 33 57 77`
- 25×25 seeds: `./scripts/run_25x25_seeds.sh 33 57 88 101`

Env knobs (sane defaults set in scripts):

- Generator: `MAZE_BRAID_PROB`, `MAZE_GOAL_FARTHEST`
- L1: `MAZE_L1_WEIGHTS=1,1,0,0,6,4,0,0`, `MAZE_L1_NORM_TAU`, `MAZE_L1_CAND_TOPK`
- Forced‑L1: `MAZE_WIRING_FORCE_L1`, `_TAU`, `_TOPK`
- Wiring: `MAZE_WIRING_TOPK`, `MAZE_WIRING_MIN_ACCEPT`, `MAZE_USE_HOP_DECISION=1`, `MAZE_HOP_DECISION_LEVEL=min`
- NA/BT: `MAZE_NA_GE_THRESH`, `MAZE_BT_AGG=na_min`, `MAZE_BACKTRACK_THRESHOLD`
- SP: `MAZE_SP_GLOBAL_ENABLE`, `MAZE_SP_POSLEVEL`, `MAZE_SP_GLOBAL_SAMPLES/BUDGET_MS`, `MAZE_SP_FORCE_ON_NA` and `SP_FORCE_*`

Quick 50×50 (PoC defaults):

- `./scripts/run_50x50_quick.sh <seed ...>`
  - Semantic BT with index‑only target: `MAZE_BT_SEM_SOURCE=index`, representative=`best`.
  - Path planning on actual trajectory: `MAZE_BT_PATH_MODE=memory_graph`, strict memory on.
  - Acceptance: geDIG‑only (no forced‑L1/trajectory edges), ΔSP off.
  - Thresholds: keep NA (−0.005); adjust BT via `BT_TAU` (e.g., 0.0045). Both saved into `run_summary.json`.

## 6. Metrics Aggregation (CSV)

Use the summarizer below to convert `run_summary.json` into a CSV line per run. The script extracts: size, seed, steps, goal, edges, l1_force, bt_triggers, geDIG mean/min, ΔSP negatives, growth totals, coverage.

```bash
python experiments/maze-navigation-enhanced/scripts/summarize_runs.py \
  docs/images/gedegkaisetsu/threshold_dfs_live_s33 \
  docs/images/gedegkaisetsu/threshold_dfs_live_s57 \
  > results_25x25.csv
```

## 7. Narrative Checkpoints

1) NA→DA two‑stage gating works: NA域で min‑hop を合成した bt_eval がしきい値でBT発火（図F2）
2) Frontier‑biased forced‑L1 recall connects nearby unexplored branches (マゼンタの配線; 図F1/F3)
3) ΔSP<0イベント（SP利得）がショートカット形成と一致（図F2）
4) Ablations show each component’s contribution（表F4）

---

# 迷路ナビゲーション（NA→DA, L1-Frontier）— 論文化に向けた結果アウトライン（日本語）

本ドキュメントは、現在の実験構成（NA→DA 二段ゲート、L1=全記憶＋壁も候補、frontier 優先の強制 L1、hop=min 配線、位置レベル ΔSP、Query Hub）に合わせて、論文に掲載すべき結果と再現手順を整理したものです。

## 0. 方法の概要（TL;DR）
- 単一スカラー geDIG（≈ ΔGED_norm − λ·IG）で 0-hop（NA）と multi-hop（DA）を駆動。
- NA（geDIG>τ）で DA 合成（bt_eval=min(0-hop, min-hop)）と「frontier 近傍候補の強制配線（L1 距離）」をトリガ。
- L1 検索は全メモリ（エピソード）＋壁も候補。重み付き距離で τ 以下の frontier 候補は geDIG を待たず配線。
- 位置レベル ΔSP（平均最短距離の差 after−before）が負になると、ショートカット/ループの利得を示す。

## 1. 迷路 / データセット
- 合成迷路（15×15, 25×25）。
- 生成オプション（環境変数）:
  - `MAZE_BRAID_PROB`（例 0.05）: 壁を“一部編む”ことでループを作り ΔSP を出やすくする。
  - `MAZE_GOAL_FARTHEST=1`: 開始点から最も遠い開通セルをゴールに設定（長尺の行動を引き出す）。
- 推奨 seed セット: 15×15={5,7,12,21,33,57,77}, 25×25={33,57,88,101}。

## 2. 構成とアブレーション
- 本命（Ours）:
  - Query Hub=ON / L1=全記憶＋壁 / 強制 L1（frontier 優先）
  - 配線 Top‑K=3 / hop=min / BT 合成=na_min / ΔSP(pos-level) + NA 強制 SP
- −forced‑L1（frontier バイアスを除く）
- −DA（0-hop のみ: `MAZE_BT_AGG=base` かつ/または `MAZE_USE_HOP_DECISION=0`）
- −QHUB（ハブ無し）
- 参考ベースライン: Greedy DFS, BFS 的バックトラック, ランダムウォーク

## 3. 報告すべき指標
- 成功率 / 到達ステップ / 最短路比（steps / 最短距離）
- バックトラック統計（トリガ回数・実行ステップ）
- カバレッジ（到達ユニークセル / 開通セル）
- 配線統計（総エッジ・種別内訳: geDIG / l1_force / query / trajectory / hub / spatial / other）
- L1 リコール統計（候補数 min/mean/max、強制 L1 の割合）
- geDIG 時系列（平均/最小/AUC(<0)、NA ステップ比）
- ΔSP（有効時）：負イベント数 / 最小 / 負部分の AUC

## 4. 図表
- F1: エピソードグラフ＋探索マップ（代表 run）
  - 左: グラフ（ノード色=時刻、エッジ色=戦略、太さ=|geDIG|）
  - 右: 迷路＋軌跡（カバレッジ併記）
- F2: 時系列（geDIG, ΔSP, growth）＋NA/BT マーカー
- F3: 配線内訳の積み上げ棒（geDIG / l1_force / …）
- F4: アブレーション表（成功・最短比・BT・ΔSP<0・配線構成など）
- F5: ハイパスイープ（L1 τ / 強制 L1 τ / BT しきい値 vs 指標）

## 5. 再現スクリプト
- 15×15 複数 seed:
  - `./experiments/maze-navigation-enhanced/scripts/run_15x15_seeds.sh 5 7 12 21 33 57 77`
- 25×25 複数 seed:
  - `./experiments/maze-navigation-enhanced/scripts/run_25x25_seeds.sh 33 57 88 101`
- 各 seed 出力（viewer 同梱）: `docs/images/gedegkaisetsu/threshold_dfs_live_s{seed}/`
  - `index.html`（ビューワ）, `run_summary.json/js`

主な環境変数:
- 生成: `MAZE_BRAID_PROB`, `MAZE_GOAL_FARTHEST`
- L1: `MAZE_L1_WEIGHTS=1,1,0,0,6,4,0,0`, `MAZE_L1_NORM_TAU`, `MAZE_L1_CAND_TOPK`
- 強制 L1: `MAZE_WIRING_FORCE_L1`, `_TAU`, `_TOPK`
- 配線: `MAZE_WIRING_TOPK`, `MAZE_WIRING_MIN_ACCEPT`, `MAZE_USE_HOP_DECISION=1`, `MAZE_HOP_DECISION_LEVEL=min`
- NA/BT: `MAZE_NA_GE_THRESH`, `MAZE_BT_AGG=na_min`, `MAZE_BACKTRACK_THRESHOLD`
- SP: `MAZE_SP_GLOBAL_ENABLE`, `MAZE_SP_POSLEVEL`, `MAZE_SP_GLOBAL_SAMPLES/BUDGET_MS`, `MAZE_SP_FORCE_ON_NA` など

## 6. 集計（CSV）
- `experiments/maze-navigation-enhanced/scripts/summarize_runs.py`
- 例: 
  - `python experiments/maze-navigation-enhanced/scripts/summarize_runs.py docs/images/gedegkaisetsu/threshold_dfs_live_s33 docs/images/gedegkaisetsu/threshold_dfs_live_s57 > results_25x25.csv`
- 出力列: size,seed,steps,goal,edges,l1_force_edges,bt_triggers,gedig_mean,gedig_min,sp_neg_count,nodes_total,edges_total,coverage

## 7. ストーリーのチェックポイント
1) NA→DA の二段ゲートが機能し、NA 域で bt_eval=min(0-hop,min-hop) による BT が自然発火する（F2）
2) 強制 L1（frontier バイアス）によって、近距離の未探索分岐へ配線が入りやすい（F1/F3）
3) ΔSP<0 のイベントがショートカット/ループ形成と一致する（F2）
4) アブレーションで各要素の寄与が定量化される（F4）
5) PoC（ifレス）: 配線=geDIG、発火=NA/BT閾値、戻り先=距離最小（index）。frontierや手続きifに依存せず、分位点校正で再現性を確保（F2/F5）。

## 8. Parameters & Status (Spec)

This section lists parameters to decide, current status, and our recommended defaults for 15×15 and 25×25 suites.

- NA/BT
  - `MAZE_NA_GE_THRESH`: current ≈ −0.005; decision pending; recommended −0.004 (15×15) / −0.006 (25×25)
  - `MAZE_BT_AGG`: current `na_min`; decision pending; recommended `na_min`
  - `MAZE_BACKTRACK_THRESHOLD`: suite‑dependent; PoC tuning by percentiles recommended（例: 50×50で p90(bt_eval)≈0.004→ BT≈0.0045）

- Multi‑hop decision
  - `MAZE_USE_HOP_DECISION`: current 1; recommended 1
  - `MAZE_HOP_DECISION_LEVEL`: current `min`; recommended `min`
  - `MAZE_HOP_DECISION_MAX`: current unset; recommended 3 (15×15) / 4 (25×25)

- L1 recall/search
  - `MAZE_L1_WEIGHTS`: current [1,1,0,0,6,4,0,0]; recommended keep
  - `MAZE_L1_NORM_TAU`: current 1.05–1.2; recommended 1.10 (15×15) / 1.15 (25×25)
  - `MAZE_L1_CAND_TOPK`: current 6–10; recommended 8 (15×15) / 10 (25×25)
  - `MAZE_L1_UNIT_NORM`: current mixed; recommended 1 (on)
  - `MAZE_L1_NORM_SEARCH` / `MAZE_L1_WEIGHTED`: current weighted; recommended both on
  - `MAZE_L1_FILTER_UNVISITED`: current off (include walls); recommended off

- Forced‑L1 (frontier‑biased immediate wiring)
  - `MAZE_WIRING_FORCE_L1`: current on; recommended on
  - `MAZE_WIRING_FORCE_L1_TAU`: current 0.6–0.8; recommended 0.70 (15×15) / 0.66 (25×25)
  - `MAZE_WIRING_FORCE_L1_TOPK`: current 1–2; recommended 1 (15×15) / 2 (25×25)
  - Frontier heuristic: current visit_count==0 (position/episode); recommended keep

- Wiring acceptance (geDIG / Top‑K)
  - `MAZE_WIRING_TOPK`: current 3; recommended 3
  - `MAZE_WIRING_MIN_ACCEPT`: current 1; recommended 1
  - `MAZE_GEDIG_THRESHOLD`: current −0.01..0.02; recommended 0.0 (both) with Top‑K gating

- SP (shortest‑path) measurement
  - `MAZE_SP_GLOBAL_ENABLE`: current suite‑dependent; recommended on for braided suites
  - `MAZE_SP_POSLEVEL`: current on; recommended on
  - `MAZE_SP_GLOBAL_SAMPLES` / `MAZE_SP_GLOBAL_BUDGET_MS`: recommended 250/35 (15×15) / 400/45 (25×25)
  - `MAZE_SP_FORCE_ON_NA`: recommended 1 (on)
  - `MAZE_SP_FORCE_SAMPLES` / `MAZE_SP_FORCE_BUDGET_MS`: recommended 120/12 (both)

- Query Hub (QHUB)
  - `MAZE_USE_QUERY_HUB`: current toggle; recommended on for “Full (ours)”, off for ablation
  - `MAZE_QUERY_HUB_PERSIST`: current off; recommended off
  - `MAZE_QUERY_HUB_CONNECT_CURRENT`: current on; recommended on

- Maze generator
  - `MAZE_BRAID_PROB`: decision pending; recommended 0.05 (15×15) / 0.08 (25×25)
  - `MAZE_GOAL_FARTHEST`: recommended 1 (on)

- Candidate gates (currently off)
  - `MAZE_SPATIAL_GATE`, `MAZE_ESCALATE_REEVAL`, `MAZE_ESCALATE_RING`, `MAZE_WIRING_WINDOW`: recommended keep off (documented rationale: avoid double‑gating, rely on geDIG/QHUB/L1)

- Experiment presets / reproducibility
  - Seeds: 15×15={5,7,12,21,33,57,77}; 25×25={33,57,88,101}
  - `MAX_STEPS`: recommended 800 (15×15) / 1500 (25×25)
  - Viewer copy: `MAZE_COPY_VIEWER=1` (keep per‑seed standalone viewer)

### Observed trends (to inform defaults)
- geDIG≈0 plateaus can halt growth; forced‑L1 at τ≈0.66–0.70 injects frontier edges to break stagnation without excessive noise.
- In braided mazes, ΔSP<0 correlates with shortcut creation; enabling global SP (sampling) captures these gains with modest overhead at the above budgets.
- Slightly stricter NA gating on 25×25 (−0.006) and a milder BT threshold (−0.012) reduce backtrack ping‑pong while preserving exploration.
- In 50×50 PoC with index‑only target, BT≈0.004〜0.005 captures cul‑de‑sacs reliably while avoiding excessive triggers.
- L1 weights emphasizing semantic/context channels ([…,6,4,…]) improved recall stability; `UNIT_NORM=1` reduces scale drift across episodes.
- `WIRING_TOPK=3` balances acceptance and noise; larger values increased spurious wiring in 25×25.

---

## 8-J. パラメータ仕様と現状（日本語）

---

## 9. Applicability & Scope (non‑overclaiming)

What carries over beyond grid mazes, and what doesn’t.

- Core decision law (if‑less):
  - Wiring acceptance: geDIG ≤ τ_edge
  - Backtrack trigger: (geDIG0 > τ_NA) and (bt_eval ≤ τ_BT)
  - Target choice: argmin_x ||q − v(x)|| (weighted L2)
  - These three thresholds allow the policy to be written without hand‑crafted rules/frontier definitions.
- Where it helps (incremental value over pure carrot/nearest‑frontier):
  - Loop‑closure gating: geDIG suppresses spurious map closures under aliasing; improves map consistency.
  - Recovery trigger: NA→BT provides rule‑free stagnation/abnormality detection and automatic back‑off.
  - Map scalability: periodic pruning by small |geDIG| edges keeps graphs compact (optional).
  - Domain‑agnostic: same control law applies if an embedding q,v(·) exists (spatial, knowledge, UI flows).
- Equivalence and limits:
  - Efficiency can approach nearest‑frontier/carrot when embeddings correlate with geodesic distance.
  - Not claiming SOTA path optimality; we rely on embedding coherence and feasibility masks.
  - Thresholds need data‑driven calibration (e.g., τ_NA=p85(geDIG_t), τ_BT=p10(bt_eval_t)).
  - Safety/feasibility handled outside the rule by masks (collisions, no‑go zones, dynamics constraints).
- Practical knobs (already in scripts):
  - `MAZE_BT_SEM_SOURCE=index` (vector‑only target), `MAZE_BT_SEM_POS_REP=best`, `MAZE_BT_SEM_DIST=l2_w`.
  - `MAZE_L1_INDEX_SEARCH=1`, `MAZE_SPATIAL_GATE=…` to bound Top‑K and locality.
- Suggested evaluation tasks (to demonstrate generality without over‑claiming):
  1) Dynamic obstacle/closure: measure recovery latency, success, map growth.
  2) Sensor noise/dropout: geDIG‑triggered recovery vs. baseline; spurious closure rate.
  3) Knowledge/graph domain: accept/reject rate vs. contradictions; retrieval quality under τ tuning.

## 9-J. 応用可能性と範囲（控えめな主張）

- 基本方針（ifレス）:
  - 配線受理: geDIG ≤ τ_edge／BT発火: geDIG0>τ_NA かつ bt_eval≤τ_BT／戻り先: 距離最小。
  - frontier定義や手続き的ルールに依存せず、しきい値と距離のみで記述。
- 付加価値（carrot/nearest‑frontier比）:
  - ループ閉路の抑制（geDIGゲート）、異常時の自動回復（NA→BT）、地図のスパース化（任意）。
  - 埋め込みが定義できれば、空間以外（知識/ワークフロー等）にも同一則で適用可能。
- 前提と制約:
  - 効率は埋め込みと地理的距離の整合に依存（SOTA最適性は主張しない）。
  - しきい値は分位点等で校正（例: τ_NA=p85, τ_BT=p10）。
  - 安全/可到達性は外部マスクで担保（衝突・進入禁止・動力学制約）。
- 検証タスク（小規模で示す）:
  - 動的障害物/封鎖、センサノイズ/欠落、知識領域での受理/棄却の比較。


決定すべきパラメータ、現状、推奨（15×15 / 25×25）のまとめ。

- NA/BT
  - `MAZE_NA_GE_THRESH`: 現状≈−0.005 → 推奨 −0.004 / −0.006
  - `MAZE_BT_AGG`: 現状 `na_min` → 推奨 `na_min`
  - `MAZE_BACKTRACK_THRESHOLD`: 現状≈−0.02/−0.011 → 推奨 −0.018 / −0.012

- マルチホップ意思決定
  - `MAZE_USE_HOP_DECISION`: 現状1 → 推奨1
  - `MAZE_HOP_DECISION_LEVEL`: 現状`min` → 推奨`min`
  - `MAZE_HOP_DECISION_MAX`: 現状未設定 → 推奨 3 / 4

- L1 リコール/検索
  - `MAZE_L1_WEIGHTS`: 現状[1,1,0,0,6,4,0,0] → 推奨維持
  - `MAZE_L1_NORM_TAU`: 現状1.05–1.2 → 推奨 1.10 / 1.15
  - `MAZE_L1_CAND_TOPK`: 現状6–10 → 推奨 8 / 10
  - `MAZE_L1_UNIT_NORM`: 現状混在 → 推奨1（ON）
  - `MAZE_L1_NORM_SEARCH` / `MAZE_L1_WEIGHTED`: 現状重み付き → 推奨 ON
  - `MAZE_L1_FILTER_UNVISITED`: 現状OFF（壁含む） → 推奨OFF

- 強制 L1（frontier 優先）
  - `MAZE_WIRING_FORCE_L1`: 現状ON → 推奨ON
  - `MAZE_WIRING_FORCE_L1_TAU`: 現状0.6–0.8 → 推奨 0.70 / 0.66
  - `MAZE_WIRING_FORCE_L1_TOPK`: 現状1–2 → 推奨 1 / 2
  - フロンティア定義: 現状 visit_count==0 → 推奨維持

- 配線受理（geDIG / Top‑K）
  - `MAZE_WIRING_TOPK`: 現状3 → 推奨3
  - `MAZE_WIRING_MIN_ACCEPT`: 現状1 → 推奨1
  - `MAZE_GEDIG_THRESHOLD`: 現状−0.01..0.02 → 推奨 0.0（Top‑K 併用）

- SP（最短路）
  - `MAZE_SP_GLOBAL_ENABLE`: スイート依存 → 編み込み（braid）有りで ON を推奨
  - `MAZE_SP_POSLEVEL`: 現状ON → 推奨ON
  - `MAZE_SP_GLOBAL_SAMPLES/BUDGET_MS`: 推奨 250/35（15×15） / 400/45（25×25）
  - `MAZE_SP_FORCE_ON_NA`: 推奨1（ON）
  - `MAZE_SP_FORCE_SAMPLES/BUDGET_MS`: 推奨 120/12（共通）

- クエリハブ（QHUB）
  - `MAZE_USE_QUERY_HUB`: 本命で ON、アブレーションで OFF
  - `MAZE_QUERY_HUB_PERSIST`: 現状OFF → 推奨OFF
  - `MAZE_QUERY_HUB_CONNECT_CURRENT`: 現状ON → 推奨ON

- 迷路生成
  - `MAZE_BRAID_PROB`: 未確定 → 推奨 0.05 / 0.08
  - `MAZE_GOAL_FARTHEST`: 推奨1（ON）

- 候補ゲート（現状OFF）
  - `MAZE_SPATIAL_GATE` / `MAZE_ESCALATE_REEVAL` / `MAZE_ESCALATE_RING` / `MAZE_WIRING_WINDOW`: 当面 OFF（二重ゲート回避、geDIG/QHUB/L1 に委ねる）

- 実験プリセット/再現性
  - Seeds: 15×15={5,7,12,21,33,57,77} / 25×25={33,57,88,101}
  - `MAX_STEPS`: 推奨 800 / 1500
  - Viewer 同梱: `MAZE_COPY_VIEWER=1`（seed ごとに独立閲覧）

### 傾向（推奨値の根拠）
- geDIG が 0 付近で停滞する run を、強制 L1（τ≈0.66–0.70）で健全に解消。τ を下げ過ぎるとノイズ増。
- 編み込み迷路で ΔSP<0 とショートカット形成が整合。上記サンプル/予算で測定安定。
- 25×25 は探索が長く、NA をやや厳しめ、BT はやや緩めにすることで BT の往復を抑制。
- L1 の重みベクトル（特に後半次元の強調）で recall の安定性が向上。`UNIT_NORM=1` でエピソード間スケール差を抑制。
- `WIRING_TOPK=3` が受理率とノイズのバランス良好。大きくし過ぎると 25×25 で誤配線が増加。
- 50×50 PoC では target=“indexのみ”（frontier不要）で ifレスの設計が有効であることを確認。BT しきい値は分位点（例: p90）ベースでの調整が安定。

---

## 10. Recent implementation updates (2025‑09‑22)

- Semantic BT (index‑only): `MAZE_BT_SEM_SOURCE=index`, representative at position=`best` episode, distance=`l2_w`.
- Backtrack path planning: `memory_graph`（実踏破の軌跡のみ、strict）
- Pure geDIG acceptance in PoC: no forced‑L1/trajectory edges; ΔSP disabled by default.
- Quick script knobs: `BT_TAU` (default 0.0045), `NA_TAU` (default −0.0050 kept), `SPATIAL_GATE`, `ENABLE_INDEX`.
- Viewer: analysis event `bt_suppressed` marks steps where NA/BT conditions hold but trigger is suppressed（計画実行/クールダウン）。

## 10-J. 実装アップデート（2025‑09‑22）

- セマンティックBT（indexのみ）: `MAZE_BT_SEM_SOURCE=index`／代表は“最も近い”エピソード（`best`）／距離は`l2_w`。
- 戻り経路: `memory_graph`（実踏破のみ・strict）。
- PoCでは配線は geDIG のみ（強制L1や軌跡エッジなし）。ΔSPは既定オフ。
- クイック実行ノブ: `BT_TAU`（既定0.0045）, `NA_TAU`（既定−0.0050）, `SPATIAL_GATE`, `ENABLE_INDEX`。
- ビューワ: `bt_suppressed` 分析イベントで、NA/BT条件成立でも trigger が抑制された箇所を可視化。
