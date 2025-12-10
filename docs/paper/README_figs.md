Paper figure generation cheatsheet (A-series additions)

This repository includes small, self‑contained scripts to generate the additional figures/tables suggested in the review. They are designed to render sensible placeholders when raw CSVs are absent, so the paper can compile end‑to‑end. Replace the placeholders by providing CSVs as described below.

Where outputs go
- Figures: docs/paper/figures/*.pdf (paper already searches here)
- Tables: docs/paper/templates/*.tex (paper \input{}’s these if present)

## Lambda/phase figures (v5 outlook)

- Template: `scripts/paper_templates/fig_lambda_phase_template.py`
- Inputs: aggregated JSONs from lambda sweeps
  - Maze example: `results/maze-lambda/maze15_s80_agg.json`
  - RAG example: `results/rag-lambda/agg.json`
- Commands:
  ```
  # Maze
  python scripts/paper_templates/fig_lambda_phase_template.py \
    --agg results/maze-lambda/maze15_s80_agg.json \
    --kind maze \
    --out docs/paper/figures/fig_lambda_maze.png

  # RAG
  python scripts/paper_templates/fig_lambda_phase_template.py \
    --agg results/rag-lambda/agg.json \
    --kind rag \
    --out docs/paper/figures/fig_lambda_rag.png
  ```
- Requires matplotlib in your (venv) environment.

## Phase-transition placeholder
- Template: `scripts/paper_templates/fig_phase_transition_template.py`
- Input: CSV or JSON with columns (lambda, success_rate, fmr, psz_shortfall, ged_min_proxy). If absent, synthetic data are used.
- Command:
  ```
  python scripts/paper_templates/fig_phase_transition_template.py \
    --input results/phase_transition.csv \
    --out docs/paper/figures/fig_phase_transition.png
  ```
  (Use `--input` pointing to your sweep CSV/JSON; omit to get a synthetic placeholder.)

Scripts and expected inputs
- A7 (R‑PSZ‑Scatter): docs/paper/fig_scripts/fig_rag_psz_scatter.py
  - Input (optional): data/rag_eval/psz_points.csv
  - Columns: run_id, query_id, H, k, PER, acceptance, FMR, latency_ms
  - Env override: PSZ_INPUT=/path/to/your.csv
  - Output: figures/fig7_psz_scatter.{pdf,png}

- A2 (M‑ROC): docs/paper/fig_scripts/fig_maze_bridge_roc.py
  - Input (optional): data/maze_eval/bridge_scores.csv
  - Columns: run_id, step, aggregator, score, y_true
    - aggregator ∈ {min, softmin:τ, sum} (e.g., "min", "softmin:0.5")
    - score: lower is better (negative spike more confident)
    - y_true: 1 if true shortcut (ΔSPL ≤ −ε), else 0
  - Env override: BRIDGE_SCORES=/path/to/your.csv
  - Output: figures/fig_m_roc.pdf

- A8 (Latency Summary): docs/paper/fig_scripts/tab_latency_summary.py
  - Inputs (optional):
    - data/latency/maze_latency.csv
    - data/latency/rag_latency.csv
  - Columns: run_id, H, k, latency_ms
  - Env override: LAT_MAZE=..., LAT_RAG=...
  - Output: templates/tab_latency_summary.tex (auto \input{} in paper)

- A1 (Event Alignment / M‑Causal): docs/paper/fig_scripts/fig_maze_event_alignment.py
  - Input (optional): data/maze_eval/event_alignment.csv
    - Wide form: run_id, t_from_NA, BT, accept, evict (0/1)
    - Long form: run_id, t_from_NA, event, value (0/1)
  - Env override: EVENT_ALIGN=/path/to/your.csv
  - Output: figures/fig_m_causal.{pdf,png}

- A3 (Memory Growth / Mem‑Growth): docs/paper/fig_scripts/fig_mem_growth.py
  - Input (optional): data/maze_eval/memory_growth.csv
    - Columns: run_id, size, step, nodes, edges, redundant_edge_ratio
  - Env override: MEM_GROWTH=/path/to/your.csv
  - Output: figures/fig_mem_growth.{pdf,png}

- A5 (Steps CDF + effect size): docs/paper/fig_scripts/fig_steps_cdf.py
  - Input (optional): data/maze_eval/steps_distribution.csv
    - Columns: run_id, method, steps, success
  - Env override: STEPS_DIST=/path/to/your.csv
  - Output: figures/fig_steps_cdf.{pdf,png}

- A6 (Observation radius sensitivity): docs/paper/fig_scripts/fig_obs_radius_sensitivity.py
  - Input (optional): data/maze_eval/obs_radius_sensitivity.csv
    - Columns: radius, H, k, lambda, success_rate, steps_mean, bt_precision
  - Env override: OBS_SENS=/path/to/your.csv
  - Output: figures/fig_m_obs_sensitivity.{pdf,png}

- A9 (IG robustness panel): docs/paper/fig_scripts/fig_r_ig_robust.py
  - Input (optional): data/rag_eval/ig_robustness.csv
    - Columns: run_id, ig_def, H, k, PER, acceptance, FMR, latency_ms, rank_pos (optional)
  - Env override: IG_ROBUST=/path/to/your.csv
  - Output: figures/fig_r_ig_robust.{pdf,png}

- A10 (Multi‑hop by type): docs/paper/fig_scripts/fig_r_multihop_by_type.py
  - Input (optional): data/rag_eval/multihop_by_type.csv
  - Env override: HOP_BY_TYPE=/path/to/your.csv
  - Output: figures/fig_r_multihop_by_type.{pdf,png}

- A11 (Operating curves τ×λ): docs/paper/fig_scripts/fig_r_operating_curves.py
  - Input (optional): data/rag_eval/tau_lambda_grid.csv
  - Env override: TAU_LAM=/path/to/your.csv
  - Output: figures/fig_r_operating_curves.{pdf,png}

- A12 (Human acceptance reliability κ): docs/paper/fig_scripts/fig_r_human_kappa.py
  - Input (optional): data/rag_eval/human_acceptance.csv
  - Env override: HUMAN_ACC=/path/to/your.csv
  - Output: figures/fig_r_human_kappa.{pdf,png}

- A13 (Baseline comparison): docs/paper/fig_scripts/fig_r_baseline_comparison.py
  - Input (optional): data/rag_eval/baseline_summary.csv
  - Env override: BASE_SUM=/path/to/your.csv
  - Output: figures/fig_r_baseline_comparison.{pdf,png}

Usage
1) Drop your CSVs under data/* as above, or point the scripts via env vars.
2) Run:
   - python3 docs/paper/fig_scripts/fig_rag_psz_scatter.py
   - python3 docs/paper/fig_scripts/fig_maze_bridge_roc.py
   - python3 docs/paper/fig_scripts/tab_latency_summary.py
3) Rebuild the paper (XeLaTeX build recommended):
   - cd docs/paper && latexmk -xelatex -interaction=nonstopmode geDIG_paper_restructured_draft_xe.tex

Notes
- Without CSVs, the scripts emit placeholders (synthetic but plausible) so LaTeX doesn’t break.
- The paper uses \IfFileExists to include these figures/tables only when present.
- For strict reproducibility, add your calibration JSON or seed lists next to CSVs and reference them in captions if desired.
