# Maze Query‑Hub 実験の改善計画と全体改善ポイント

本ドキュメントは、迷路PoC（query‑hubプロトタイプ）を論文のPhase 1（One‑Gauge制御）に沿って再実験・拡充するための改善項目と、リポジトリ全体の改善ポイントをまとめたものです。

## 対象範囲（現行実装）
- 迷路実験ドライバ: `experiments/maze-query-hub-prototype/run_experiment_query.py`
  - geDIGCore 初期化と評価経路: `experiments/maze-query-hub-prototype/run_experiment_query.py:452`, `experiments/maze-query-hub-prototype/run_experiment_query.py:475`
  - 0‑hop/多ホップ評価とログ: `experiments/maze-query-hub-prototype/qhlib/evaluator.py:1`
- 可視化/レポート: `experiments/maze-query-hub-prototype/build_reports.py`, `experiments/maze-query-hub-prototype/query_interactive_template.html`
- 参考（RAG v3‑lite, PSZ実装）: `experiments/rag-dynamic-db-v3-lite/`

---

## 迷路実験（Query‑Hub）の改善ポイント

### 1) 理論整合・評価安定化
- Linkset IG と候補台正規化の既定化
  - 既定運用: `--linkset-mode --norm-base link`（IG=linkset, GED分母=1+|S_link|）
  - 実装箇所確認: `experiments/maze-query-hub-prototype/run_experiment_query.py:452`
- 二段ゲート（AG/DG）の運用明確化
  - 推奨: 動的AG（分位）+ 静的DG（しきい値固定）。例: `--ag-auto --ag-quantile 0.9 --theta-dg 0.6 --top-link 1 --commit-budget 1`
  - g0/gmin 記録: `StepRecord.g0/gmin`（HTMLに時系列プロットあり）
- SP評価の厳密化と境界処理
  - 既定: `--sp-scope union --sp-boundary trim`、固定ペア評価（`qhlib/evaluator.py` 参照）
- 0‑hop配線数の制御
  - 既定は `--link-autowire-all`（S_link全本配線）。Top‑Lのみで評価したいケース用にOFF切替フラグを明示（CLIに `--no-link-autowire-all` を追加予定）。

### 2) メトリクス/ログ拡充（迷路用PSZに相当）
- 実装追加（提案）
  - ステップ効率: 平均/中央値ステップ、到達率、バックトラック率
  - 冗長エッジ圧縮率: 追加エッジに対する不要辺の割合（近似でOK）
  - レイテンシ: `eval`時間の P50/P95（既に `avg_time_ms_eval`/`p95_time_ms_eval` を出力済）
  - ゲート統計: AG/DG発火率、`best_hop` 分布、`k_star` 分布
- 表示/出力
  - summary.json に集計を追加、HTMLに g0/gmin 分布ヒストグラムを追加（既存タイムラインに加える）

### 3) 性能最適化
- 距離計算キャッシュ/固定ペアサンプリング
  - 既定: `--sp-cache --sp-cache-mode cached_incr --sp-pair-samples 200`
- Dead‑end 早期打切り
  - 既定: `--skip-mh-on-deadend`（行き止まり/バックトラック時は hop0 のみ）
- 大規模迷路（≥50×50）
  - `--max-hops 10` から段階的に増やす。P95/P99 監視を推奨。

### 4) 再現レシピ（推奨コマンド）
- クイック（25×25, 60 ステップ, λ=1, γ=1, H=10）
  - 実行:  
    `python experiments/maze-query-hub-prototype/run_experiment_query.py --maze-size 25 --max-steps 60 --linkset-mode --norm-base link --ag-auto --ag-quantile 0.9 --theta-dg 0.6 --top-link 1 --commit-budget 1 --output experiments/maze-query-hub-prototype/results/quick25_summary.json --step-log experiments/maze-query-hub-prototype/results/quick25_steps.json`
  - 可視化:  
    `python experiments/maze-query-hub-prototype/build_reports.py --summary experiments/maze-query-hub-prototype/results/quick25_summary.json --steps experiments/maze-query-hub-prototype/results/quick25_steps.json --out experiments/maze-query-hub-prototype/results/quick25_interactive.html`
- 長め（168/500 ステップ）
  - `--max-steps 168` または `--max-steps 500`、性能オプションを付与

---

## 迷路実験の残タスク（優先度順）

1) AG/DG 既定値の校正とサマリ表の確定
- グリッドサーチ（AG:分位×窓, DG:固定）で success/steps/P50 を同時最適。結果を summary に反映。

2) 0‑hop配線数の切替フラグの追加
- CLIに `--no-link-autowire-all` を追加（Top‑Lのみベース配線）。READMEに記載。

3) アブレーション一式（equal‑resources相当）
- `w/o ΔSP`, `w/o AG`, `w/o multi‑hop`, `w/o ΔEPC`, `w/o ΔIG` を固定バジェットで比較。

4) 迷路タイプ/サイズ別の安定性確認
- `--maze-type`（dfs等）× サイズ 15/25/50 の横断・P95/P99測定。

5) HTML強化
- g0/gmin ヒストグラム・AG/DG発火率・best_hop分布の可視化ブロック追加。

---

## リポジトリ全体の改善ポイント

### A) ゲーティングのライブラリ化
- 目的: 実験側にあるAG/DG判定（g0/gmin, θAG/θDG）を共通化。
- 施策: `src/insightspike/algorithms/gating.py`（仮）に `decide_gates(g0, gmin, theta_ag, theta_dg) -> {ag, dg, b}` を追加。
- 適用箇所: 
  - L3グラフ推論: `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
  - RAG v3‑lite: `experiments/rag-dynamic-db-v3-lite/src/gedig_scoring.py`

### B) PSZ/equal‑resources ユーティリティの公開
- 目的: 受容率/FMR/Latency P50 計算とバジェットの標準化。
- 施策: `src/insightspike/metrics/psz.py`（仮）に `compute_psz(stats, thresholds)` 等を実装し、RAG/迷路双方で流用。
- 参考実装: `experiments/rag-dynamic-db-v3-lite/src/metrics.py`

### C) ΔGED/ΔIG のAPI注記強化
- `GraphAnalyzer.calculate_metrics` のdocstringに、`delta_ged`（改善で負）と `delta_ged_norm`（正規化コスト, 常に正）の使い分けを明記。
  - 対象: `src/insightspike/features/graph_reasoning/graph_analyzer.py:39`

### D) 例の追加（学習コスト低）
- `examples/hello_gating.py`（仮）: 小グラフで g0/gmin と AG/DG を表示（`examples/hello_insight.py` の拡張）。

### E) 迷路レガシー経路の一本化
- 旧ナビゲータの簡易ヒューリスティクを geDIGCore 経由に置換（Phase 1 同一計器の原則）。
  - 主に: `src/insightspike/maze_experimental/navigators/` 配下の選択ロジック

### F) 設定正規化の拡張
- `NormalizedConfig` に linkset/norm‑base/SP 設定を取り込み、L3/L4 から一貫アクセス。
  - 対象: `src/insightspike/config/normalized.py`

### G) 小規模テストの追加
- `GateDecider` と `PSZ` ユーティリティの単体テスト（極小データ）。

---

## 実行計画（提案）

1) 速攻タスク（~0.5日）
- `--no-link-autowire-all` 追加（CLI）
- AG/DGプリセットのコマンドテンプレをREADMEへ追記

2) 小規模実装（~1–2日）
- GateDecider（共通化）実装 → L3/RAGで採用
- ΔGED/ΔIGのdocstring追記

3) 並行評価（~1–3日）
- しきい値掃引（AG分位×DG固定）で summary 確定、HTML更新
- 迷路サイズ/タイプ横断、アブレーション最小版

4) 共有ユーティリティ（~1–2日）
- PSZユーティリティ公開、RAG v3‑lite/迷路で共通使用

---

## 参考コマンド（再掲）
- クイック実行:  
  `python experiments/maze-query-hub-prototype/run_experiment_query.py --maze-size 25 --max-steps 60 --linkset-mode --norm-base link --ag-auto --ag-quantile 0.9 --theta-dg 0.6 --top-link 1 --commit-budget 1 --output experiments/maze-query-hub-prototype/results/quick25_summary.json --step-log experiments/maze-query-hub-prototype/results/quick25_steps.json`
- レポート生成:  
  `python experiments/maze-query-hub-prototype/build_reports.py --summary experiments/maze-query-hub-prototype/results/quick25_summary.json --steps experiments/maze-query-hub-prototype/results/quick25_steps.json --out experiments/maze-query-hub-prototype/results/quick25_interactive.html`

