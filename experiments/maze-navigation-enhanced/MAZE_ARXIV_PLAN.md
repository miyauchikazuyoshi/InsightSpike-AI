# Maze Experiment Plan for arXiv (geDIG)

Purpose
- Reach “discussion-worthy” evidence for geDIG on controlled mazes with statistical strength and reproducibility.
- Produce figures/tables directly reusable in the paper.

Success criteria (discussion-ready)
- Sizes 15/25/50 with ≥30 seeds each, confidence intervals reported.
- Success rate: 25x25 ≥ 0.8 (95% CI shown).
- Efficiency: average steps −15% vs baselines; revisit/redundancy reduced with effect sizes.
- Compression: graph edges −90% 近辺（CI付き）または明確な削減傾向。
- Causal evidence: geDIG ON/OFF および NA/BT/MH/SP-gain のON/OFFで勝率>0.6、ΔgeDIGと下流改善の有意な相関。
- Full reproducibility: preset→calibration→fixed-eval の手順と seeds を公開。

Current state (what we have)
- Presets + loader
  - configs/: default, 15x15, 25x25, 50x50
  - src/utils/preset_loader.py (ENV < preset < CLI, apply_env exports MAZE_* vars)
- Calibration + stats
  - src/analysis/calibrate_ktau.py (grid: k, τ, τ_bt; success→avg steps)
  - src/analysis/stats_summary.py (success率・平均歩数±95%CI・edges±95%CI・win-rate)
  - src/analysis/run_preset_suite.py (一括: preset→calibrate→stats)
  - Makefile: maze-preset / maze-calibrate / maze-stats / maze-suite
- Baselines & utilities
  - experiments/baseline_vs_simple_plot.py に random/DFS baseline 実装あり（baseline_explorers.py を利用）
  - analysis/clean_maze_run.py で simple vs geDIG の集計と BFS 最短路計算（スモーク）
- Structural logging and visualization
  - 多数のテスト・可視化（test_*, visualize_*）があり、イベントログ・gedig_structural等を取得可能

Gaps (what’s missing now)
1) Unified, paper-ready runners
- clean_maze_run.py は simple/geDIG の二者比較。DFS/Random を同一フォーマットで回す軽量 runner が無い。
- 対策: clean_maze_run に baseline hooks を追加するか、analysis/ 下に paper_runner.py を新設（simple/geDIG/DFS/Random の同予算対決、JSON 集約）。

2) Yoked control (候補集合固定で意思決定のみ差替)
- 現 logger で観測系列や候補集合の完全固定化は未整備。
- 対策: MazeNavigator に “decision tap” を追加（候補スコア・採否ログを保存）、replay モードで simple/geDIG を切替比較。
  - 参考: navigation/maze_navigator.py（decision_engine/graph_manager 周辺）に hook。

3) Ablation one-shot scripts
- NA/BT/MH/SP-gain の ON/OFF アブレーションを一括実行するスクリプトが未統合。
- 対策: src/analysis/ablation_suite.py を追加し、環境変数/引数で以下の組み合わせを回す。
  - NA off（θ_NA 無効化相当）/ BT off / multihop off / SP-gain off
  - λ・τ の感度（小グリッド）

4) Aggregation for paper figures
- JSON の集約（サイズ×種の平均±CI、効果量、勝率）と図表の自動生成のひとまとめがない。
- 対策: src/analysis/aggregate_for_paper.py を作成
  - 入力: calibration.json, grid_results.json, stats_summary.json, baseline比較結果
  - 出力: figs/ 配下にスケーリング曲線・アブレーション棒グラフ・相関散布図、tables/ に Markdown テーブル

5) Reproducibility glue
- seeds の固定と run metadata の保存先が散在。
- 対策: results/ 以下に size/seed/day で階層化し、run_config_seeds.json / calibration.json / summary.json を必ず保存。

Concrete tasks (files to touch)
- [NEW] src/analysis/paper_runner.py
  - Run: sizes=(15,25,50), seeds=N（≥30）、algos=[simple,gedig,dfs,random]、予算=同一（max_steps=α·size^2）
  - Save: results/paper/{size}/seed_*.json + summary_{size}.json
- [NEW] src/analysis/ablation_suite.py
  - Flags: --na-off --bt-off --mh-off --sp-off --lambda-grid ... --tau-grid ...
  - Output: results/ablation/{size}/summary.json
- [MOD] src/analysis/clean_maze_run.py
  - Option to include DFS/Random; write per-algo JSON in a common schema
- [MOD] navigation/maze_navigator.py
  - (Optional) decision tap: store candidate ranking + accept/reject per step (minimal fields) for yoked control
- [NEW] src/analysis/aggregate_for_paper.py
  - Build scaling curves, ablation bars, ΔgeDIG vs 改善相関（Spearman ρ, p）

Execution plan (1–2 weeks)
- Day 1–2: presets 適用→ calibrate → fixed → paper_runner で 15/25/50×≥30 seeds
- Day 3–4: ablation_suite（NA/BT/MH/SP）と λ/τ 感度（小グリッド）を回し、aggregate_for_paper で図表化
- Day 5: yoked control の skeleton（decision tap 追加→ 簡易リプレイ比較を 25x25 で実施）
- Day 6–7: 論文図表差替（スケーリング、アブレーション、相関、ON/OFF可視化1例）と再現手順整理

Paper-ready deliverables
- Figures: scaling (success/steps/edges by size), ablation bars (ON/OFF), ΔF vs 改善散布図（ρ,p）, ON/OFF比較1例
- Tables: success率/平均±CI/効果量（simple/DFS/Random対比）, 勝率（geDIGがより短い割合）
- Repro: seeds.json, calibration.json, scripts/commands（Makefile タスク含む）

Commands (quick)
- 25x25 一括（プリセット→校正→要約）
  - `make maze-suite PRESET=25x25 SIZE=25 SEEDS=32`
- 25x25 統計のみ
  - `make maze-stats SIZE=25 SEEDS=32`
- 15/50 も同様に（SEEDS≥30）

Risks & mitigations
- 計算量: 多ホップ×多seedで時間がかかる → `MAZE_FAST_MODE=1`, スナップショット間引き, SP-gain サンプリング、分散実行
- 過度なチューニング依存: calibrate→fixed→本番の運用を徹底（calibration.json を保存）
- 再現性: 乱数seedとENVを必ず保存（run_config_seeds.json / env_dump.json）

Owner/Status
- Owner: Miyauchi
- Status: presets/calibration/stats は実装済。baseline比較・ablation・aggregate・yoked は最小の追加で到達可能。

