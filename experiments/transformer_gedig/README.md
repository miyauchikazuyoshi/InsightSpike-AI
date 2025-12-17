# Transformer geDIG Validation (λ-scan & Phase-Transition Outlook)

This directory contains a lightweight framework to validate whether Transformer attention patterns align with the geDIG gauge.

## Hypotheses
- H1: Real Transformer attention produces lower geDIG F than random/uniform/local baselines.
- H2: As λ varies, structural sparsity and F show predictable changes; a critical λ_c may appear where structural metrics shift sharply.
- H3: Forced edits to attention (sparsify vs noise) change downstream performance in line with ΔF.

## geDIG (attention) definition
- Build a directed graph from a single head: edge i→j if attention weight > threshold τ (default: percentile, e.g., top 10%). Pad tokens are excluded.
- ΔEPC: edge density (|E| / L^2 for directed). 
- ΔSP: relative average shortest-path gain (Lb−La)/Lb on the largest weakly connected component (sampling allowed for long sequences).
- ΔH: normalized entropy of flattened attention (H / log(L^2)).
- F = (ΔEPC − λ·γ·ΔSP) − λ·ΔH, with γ ≈ 0.5 (configurable). Computed per layer/head.
- Log extras: density, num_edges, seq_len, threshold mode (abs/perc), directed/undirected.

## Plan (minimal runs, default = percentile τ)
1) Phase 1 (descriptive, percentile τ only by default):
   - Models: bert-base-uncased, gpt2 (extend to gpt2-medium if budget allows).
   - Data: Wikitext short samples (~200), max length 256–512.
   - Thresholds: τ_pct ≈ top10% (default); τ_abs はオプション扱い（τ=0.01では符号逆転が起こるためデフォルトでは使用しない）。
   - Baselines: random, uniform, local window (w=5), diagonal.
   - Success: F_real < F_random (p<0.001, d>0.5); deeper layers show lower F (Spearman ρ<−0.3).

2) Phase 2 (λ-scan / phase, percentile τ):
   - λ grid: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0] (extend if needed).
   - Metrics per λ: F_mean, E_eff_mean, sparsity=1−density, ΔSP_mean. 
   - Detect λ_c via d²F/dλ² peak and structural shifts (sparsity/clustering).
   - Plot: λ vs F, sparsity, d²F/dλ² (use fig_lambda/phase templates).
   - If sparsity is flat, allow λ-dependent thresholds (e.g., percentile += k·λ or abs *= λ) to induce structural change for diagnostics.

3) Phase 3 (causal, percentile τ):
   - Interventions: sparsify top-k (lower F), add noise (raise F), and measure downstream task accuracy (e.g., SST2 batches). Record ΔF vs Δaccuracy.
   - Optional: F-regularized fine-tuning (small budget) to see if lower F correlates with accuracy.

4) Phase 4 (learning-dynamics / checkpoints)
   - Capture model checkpoints across training/fine-tuning steps (e.g., 0/100/500/1k/2k steps on a small task).
   - For each checkpoint, run Phase 1 stats (F_mean, sparsity, ΔSP) and λ-scan (or a small λ grid).
   - Track performance (perplexity/accuracy) alongside geDIG metrics; look for inflection points (d²/d(step)² peaks) aligned with performance jumps.
   - Optionally add geDIG regularization during training to see whether structural changes align with improved accuracy.

5) Phase 5 (causal verification via F-regularization) **NEW**
   - **Goal**: Demonstrate causality, not just correlation. If minimizing F improves performance, F is causally relevant.
   - **Method**: L_total = L_CE + α·F_mean during fine-tuning
   - **α sweep**: [0, 0.001, 0.01, 0.1, 1.0] with multiple seeds
   - **Success criteria**:
     - α > 0 outperforms baseline (α=0)
     - Optimal α exists (not monotonic)
     - Final F is lower for regularized models
   - **Implication**: If successful, this proves F is not just a post-hoc descriptor but a trainable objective.

## Implementation notes
- Pad/causal masks must be applied before graph building. 
- For long sequences, sample SP pairs (e.g., 200) to bound cost. Largest component for ASP.
- Thresholding: support absolute τ and percentile. Prefer logging both modes for stability analysis.
- Force SP evaluation: if SP is skipped in some paths, offer a `force_sp_gain_eval`-like flag and record a separate `delta_sp_forced` if needed.

## Outputs
- Per layer/head CSV/JSON with: F, ΔEPC, ΔSP, ΔH, density, num_edges, seq_len, λ, τ mode, directed flag, baselines’ F.
- λ-scan summary JSON for plotting (λ, F_mean, sparsity_mean, dF/dλ, d²F/dλ²).

## Scripts
- `extract_and_score.py`: extract attention, compute geDIG per layer/head, compare to baselines. Default: percentile threshold only (top10%), bert-base-uncased + gpt2, 32 short texts, 6 layers.
- `lambda_scan.py`: λ sweep on a model/dataset with percentile threshold; emits `lambda_scan.json` (+ `lambda_phase.{json,csv}` helper).
- `intervene_eval.py`: apply attention edits and log ΔF/精度（SST2ミニ）. 強介入・単一層介入のサマリ: `intervene_summary.json`, `fig_intervention_summary.png`, `report_interventions.md`.
- `train_f_regularized.py`: **Phase 5** F-regularized fine-tuning experiment. Implements differentiable geDIG and custom loss. Usage:
  ```bash
  # Quick smoke test
  bash run_f_reg_smoke.sh

  # Full α sweep (recommended)
  python train_f_regularized.py --alpha-sweep --alphas "0,0.001,0.01,0.1,1.0" --seeds "42,123,456"

  # Single α run
  python train_f_regularized.py --alpha 0.1 --train-samples 2000 --epochs 5
  ```
- `plot_f_reg_results.py`: Generate plots and summary table from F-reg experiment results.
- Plotting: reuse `scripts/paper_templates/fig_lambda_phase_template.py` and `fig_phase_transition_template.py` with produced agg JSON/CSV.

## Smoke status (current)
- Phase 1: pct閾値で `extract_and_score.py` → `results/transformer_gedig/score_smoke.json` (4,608 rows). Aggregation: ΔF≈+0.092 vs random, d≈+2.32（BERT d≈+2.47, GPT-2 d≈+2.20）。固定閾値τ=0.01は逆符号になるためデフォルト外。デバッグメモ: `report_phase1_debug.md`.
- Phase 2 (BERT): `lambda_scan_mixed_dyn.json` with λ-dependent thresholds (pct=0.8+0.02·λ, abs=0.01×λ). Phase slices: `lambda_phase_p80_dyn.{json,csv}`, `lambda_phase_abs_dyn.{json,csv}`; figures `fig_phase_transition.png`, `fig_phase_transition_abs.png` show sparsity change with λ.
- Phase 2 (GPT-2 medium, cached locally): `lambda_scan_gpt2m.json` (rows=12,288, agg=16) with pct=0.8/0.9; phase exports `lambda_phase_gpt2m.{json,csv}` and figures `fig_phase_transition_gpt2m_pct80.png`, `fig_phase_transition_gpt2m_pct90.png`（強オフセット版も別途あり）。
- Phase 3: 強介入（全層）・単一層中強度の評価済み。精度は最大で ~53–56% まで低下、F_mean も大幅低下。サマリ: `intervene_summary.json`, `fig_intervention_summary.png`, `report_interventions.md`.
- Phase 4: DistilBERTミニ学習のチェックポイント解析。精度は0.54→0.77と上昇、F_meanはほぼ一定。CLSアンカーのサブグラフFは hop1 で精度と強い正相関（corr≈0.97）。図: `fig_phase4_checkpoints.png`, `fig_phase4_hopcorr.png`.
- Grokking toy (mod加算): 現行ハイパラでは汎化せず、val Acc ≈0.09–0.27。`results/transformer_gedig/grokking_mod_add/run_summary.json`。
- Phase 5 (F-regularization): **READY TO RUN**. Script: `train_f_regularized.py`.
  - Differentiable geDIG implementation complete
  - α sweep over [0, 0.001, 0.01, 0.1, 1.0] with 3 seeds
  - Expected runtime: ~30min on GPU for full sweep
  - Quick smoke test: `bash run_f_reg_smoke.sh` (~5min)
