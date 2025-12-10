# Paper v5 → Code Alignment Checklist (Step 1)

Purpose
- Capture where the v5 paper assets (figures/tables/claims) rely on metrics or experiments that are not yet implemented in code or scripts.
- Keep this list short and actionable; use it as the driver for the lambda-scan/phase-transition work and any new diagnostics.

Artifacts referenced in v5 (EN draft includes):
- Figures: `fig5_maze_scaling.pdf`, `fig1_rag_performance.pdf`, `fig_r_operating_curves.pdf`, `fig8_ablation.pdf` (no lambda-scan figure yet in repo).
- Tables: exp23 (RAG) alignment/resources/ablation tables under `docs/paper/figures/exp23_*_table.tex`.
- Text claims: lambda-scan / phase-transition outlook; offline global rewiring via GED_min; PSZ shortfall narrative.

Status snapshot (2025-11, this pass)
- v4 reproducibility scripts exist (Maze query-hub, RAG exp23-lite). They generate PSZ/FMR/latency and ablations, but do not sweep lambda or emit GED_min-style diagnostics.
- No figure or script currently outputs lambda-scan or phase-transition plots.
- Global rewiring/GED_min is design-only (no code path).

Checklist (fill owner/ETA when assigned)
- [ ] Figure: lambda-scan (Maze) — metrics: PSZ shortfall, F distribution stats, SP stats by lambda. Owner: _TBD_. ETA: _TBD_.  
      Progress: sweep ran (15x15/80steps/seed2, λ=0.2/0.5/0.8) → `results/maze-lambda/maze15_s80_agg.json`, `maze15_s80_plot.png`
- [ ] Figure: lambda-scan (RAG lite) — metrics: Acc/FMR/P50 (or PSZ shortfall) by lambda. Owner: _TBD_. ETA: _TBD_.  
      Progress: sweep ran (exp23, λ=0.2/0.5/0.8) → `results/rag-lambda/agg.json`, `plot.png`
- [x] Diagnostic: GED_min or proxy — emitted in geDIG results/logs (config-gated). Owner: _done_. Tests: `tests/gedig/test_ged_min_diag.py`
- [x] Script update: aggregation helper for lambda sweeps (Maze) — scripts/aggregate_maze_lambda_sweep.py added.
- [x] Script update: aggregation helper for lambda sweeps (RAG lite) — scripts/aggregate_rag_lambda_sweep.py added.
- [ ] Script update: plotting templates for new figures (lambda/phase-transition). Owner: _TBD_. ETA: _TBD_.
- [ ] Documentation: commands to regenerate v5 figures (including new ones) in docs/paper README/appendix. Owner: _TBD_. ETA: _TBD_.  
      Progress: λ図テンプレ追加＋再生成コマンドを docs/development/paper_alignment_todo.md に記載。

Notes / mapping hints
- Maze: use `experiments/maze-query-hub-prototype/run_experiment_query.py` and `scripts/aggregate_maze_batch.py` as insertion points for lambda sweep and grouped aggregation.
- RAG lite: use `experiments/exp2to4_lite/src/run_suite` and existing viz modules; exp23_* JSON outputs already contain Acc/FMR/P50 but need lambda surfaced/propagated.
- GED_min proxy can be attached in `src/insightspike/algorithms/gedig_core.py` as an optional field in the result/log (default off to preserve backward compatibility).
