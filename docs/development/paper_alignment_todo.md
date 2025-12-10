# Paper Alignment TODO (lambda scan & phase transition)

Context
- Current codebase is aligned with the v4 One-Gauge spec; v5 paper text is ahead (lambda scan, phase-transition analysis, global rewiring via GED_min).
- Goal: add the missing experiments/diagnostics with minimal disruption, keep v4 reproducibility intact.

Targets (deliverables)
- Reproducible lambda sweep for Maze (query-hub) with aggregated metrics and plots.
- Reproducible lambda sweep for RAG lite (exp23) with aggregated metrics and plots.
- Non-destructive global rewiring diagnostic (GED_min or proxy) emitted alongside geDIG results.
- Updated figure generation that covers any new v5 plots derived from the above.

Immediate steps (order)
1) Paper-to-code diff (timebox 1h)
   - Extract v5-specific figures/tables and note required metrics.
   - Map each required metric to existing outputs or mark as new.
   - Produce a short checklist (scope/owner/ETA) under docs/development once done.

2) Maze lambda sweep (query-hub runner)
   - Add CLI option to experiments/maze-query-hub-prototype/run_experiment_query.py to accept a list of lambda values and emit lambda into summaries/step logs.
   - Extend scripts/aggregate_maze_batch.py (or a sibling) to group by lambda and compute PSZ shortfall, F distribution stats, SP stats.
   - Add a lightweight plotting script (matplotlib) for lambda vs PSZ/Regret and lambda vs structural metrics.
   - Smoke: run a short sweep (e.g., size 15, steps 150, lambdas [0.2, 0.5, 0.8]) and stash results under results/lambda_sweep_maze/.

3) RAG lite lambda sweep (exp23)
   - Surface lambda as a CLI arg/env override in experiments/exp2to4_lite/src/run_suite and downstream configs.
   - Add an aggregation helper to merge exp23_* JSONs by lambda and compute Acc/FMR/PSZ shortfall.
   - Add plotting for lambda vs Acc/FMR/P50 latency.
   - Smoke: 50-query subset sweep (e.g., lambdas [0.2, 0.5, 0.8]) into results/exp23_lambda_sweep/.

4) Global rewiring diagnostic (non-destructive)
   - In geDIGCore (algorithms/gedig_core.py), add optional computation of a GED_min-style proxy (e.g., normalized shortest-path compression or spectral gap delta) and expose it in the result/log.
   - Gate the feature behind a config flag; default off.
   - Add unit tests for the new metric and ensure existing outputs remain unchanged when disabled.

5) Figure rebuild path
   - Enumerate which v5 figures are reproducible from code (maze/RAG sweeps) and which remain paper-only.
   - Add/extend plotting scripts under scripts/paper_templates/ or experiments/*/src/viz.py to generate the new figures.
   - Document the exact commands (inputs → outputs) in docs/paper/README or a new appendix.

Guardrails
- Keep default behavior backward compatible (no change when lambda sweep flags are absent).
- Prefer lite mode for smoke runs; mark heavyweight runs as optional.
- No external API calls; keep mock LLM as default.
- Avoid touching legacy presets unless required; add new presets if needed.

Quick commands (maze lambda sweep, interim)
- Run sweep (paper preset example):  
  `python experiments/maze-query-hub-prototype/run_experiment_query.py --preset paper --maze-size 15 --max-steps 150 --lambda-sweep 0.2 0.5 0.8 --output results/maze-lambda/maze15_s150_summary.json --step-log results/maze-lambda/maze15_s150_steps.json`
- Aggregate by lambda + plot (optional):  
  `python scripts/aggregate_maze_lambda_sweep.py --dir results/maze-lambda --glob \"*summary.json\" --out results/maze-lambda/maze_lambda_agg.json --plot results/maze-lambda/maze_lambda_plot.png`

Quick commands (RAG lite lambda sweep, interim)
- Run sweep (exp23_paper, no calibration):  
  `python experiments/exp2to4_lite/src/run_suite.py --config experiments/exp2to4_lite/configs/exp23_paper.yaml --lambda-sweep 0.2 0.5 0.8`  
  (結果ファイル名に `lambda{val}` サフィックスが付与され、timestamp付きで output_dir に保存)
- Aggregate by lambda + plot (baseline=gedig_ag_dg):  
  `python scripts/aggregate_rag_lambda_sweep.py --dir experiments/exp2to4_lite/results --glob \"exp23_paper_lambda*.json\" --baseline gedig_ag_dg --out results/rag-lambda/agg.json --plot results/rag-lambda/plot.png`

GED_min proxy diagnostic (opt-in)
- 計算内容: ASP_before/after の相対短縮を proxy として `GeDIGResult.ged_min_proxy` に格納
- 有効化: Core init で `enable_ged_min_diag=True` もしくは `INSIGHTSPIKE_GED_MIN_DIAG=1`
- デフォルト: 無効（既存の挙動は変わらない）
- RAG gate_logs にもged_min_proxyが追加済み。Maze steps/summaryにもged_min_proxy, avg_ged_min_proxyを記録。
- SP/ged_minを確実に出したいとき: `--force-sp-gain-eval` を付与（診断用）。大きめ迷路＋`--dg-bfs-shortcut`と組み合わせて試す。
- 迷路短絡を狙う診断コマンド例（タイムアウトする場合はサイズ/stepsを落とす）  
  ```
  INSIGHTSPIKE_GED_MIN_DIAG=1 KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE KMP_CREATE_SHM=0 OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU \
    .venv/bin/python3 experiments/maze-query-hub-prototype/run_experiment_query.py \
    --preset paper --maze-size 15 --max-steps 200 --seeds 1 --lambda-sweep 0.5 \
    --sp-allpairs-exact --sp-report-best-hop --dg-bfs-shortcut --force-sp-gain-eval \
    --output results/maze-lambda/maze15_force_summary.json --step-log results/maze-lambda/maze15_force_steps.json
  ```

Status snapshot (2025-12-02)
- Maze λスイープ smoke: 15x15/80steps/seed2/λ=0.2,0.5,0.8 → `results/maze-lambda/maze15_s80_agg.json`, `maze15_s80_plot.png`
- RAG λスイープ smoke: exp23, λ=0.2/0.5/0.8 → `results/rag-lambda/agg.json`, `results/rag-lambda/plot.png`
- GED_min proxy: 実装・テスト済み（デフォルトOFF）。短ステップでは0.0に寄るので、より長いステップ/追加エッジでの確認が必要。
- 図テンプレ: `scripts/paper_templates/fig_lambda_phase_template.py` 追加（maze/ragのagg.jsonからλ図を生成）。

Figure regeneration (lambda)
- Maze: `python scripts/paper_templates/fig_lambda_phase_template.py --agg results/maze-lambda/maze15_s80_agg.json --kind maze --out docs/paper/figures/fig_lambda_maze.png`
- RAG:  `python scripts/paper_templates/fig_lambda_phase_template.py --agg results/rag-lambda/agg.json --kind rag --out docs/paper/figures/fig_lambda_rag.png`
