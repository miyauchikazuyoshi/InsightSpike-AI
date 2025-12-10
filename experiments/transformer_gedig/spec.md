# Transformer geDIG Validation — Detailed Spec (λ-scan & Phase Transition)

## Goal
Test whether Transformer attention patterns align with the geDIG gauge and exhibit a λ-driven structural transition.

## Hypotheses
- H1: Real attention yields lower geDIG F than random/uniform/local baselines.
- H2: λ controls sparsity/structure in a predictable way; a critical λ_c may exist where structure metrics shift sharply (peak in d²F/dλ²).
- H3: Manipulating attention to lower/raise F changes downstream performance accordingly.

## Gauge on Attention
- Graph: nodes = tokens (pad removed). Directed edge i→j if A_ij > τ (default: percentile, e.g., top10%). Optionally test undirected.
- ΔEPC: edge density (|E| / L² for directed).
- ΔSP: relative avg shortest-path gain (Lb−La)/Lb on the largest weakly connected component; allow sampling pairs for long seqs.
- ΔH: normalized entropy of flattened attention (H / log(L²)).
- F = (ΔEPC − λ·γ·ΔSP) − λ·ΔH, with γ ~ 0.5 (config).
- Log extras: density, num_edges, seq_len, τ mode (abs/perc), directed flag.
- Optional: `force_sp_gain_eval` path to always compute SP (record `delta_sp_forced` if needed).

## Experiments
### Phase 1 (Descriptive, default percentile τ)
- Models: bert-base-uncased, gpt2 (extend to gpt2-medium if budget allows).
- Data: Wikitext (~200 short samples), max_len 256–512.
- Thresholds: τ_pct ≈ top10% (default); optionally add τ_abs ∈ {0.01, 0.05} for sensitivity. Directed/undirected both if needed.
- Baselines: random, uniform, local window (w=5), diagonal.
- Checks: F_real < F_random (p<0.001, d>0.5); deeper layers have lower F (Spearman ρ<−0.3); head variance large.

### Phase 2 (λ-scan / Phase, percentile τ)
- λ grid: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0] (extend if needed).
- Metrics per λ: F_mean, E_eff_mean, sparsity=1−density, ΔSP_mean (and forced variant).
- Detect λ_c: peak of |d²F/dλ²| and structural shift (sparsity/clustering).
- Plot: λ vs F, sparsity, d²F/dλ² (use fig_lambda_phase/phase_transition templates).

### Phase 3 (Causal)
- Interventions: sparsify top-k (lower F), add noise (raise F), measure downstream accuracy (e.g., SST2 batches). Record ΔF vs Δaccuracy.
- Optional: F-regularized fine-tuning (small budget) to see if lower F correlates with accuracy.

## Implementation Notes
- Apply pad/causal masks before graph build. Remove pads from seq_len and attention.
- SP: for long seqs, sample ~200 pairs; use largest component ASP; allow `force_sp_gain_eval`.
- Thresholding: support absolute τ and percentile; log both for stability analysis.
- Outputs: per head/layer JSON/CSV with F, ΔEPC, ΔSP, ΔH, density, num_edges, seq_len, λ, τ mode, directed, baselines’ F. λ-scan summary JSON for plotting.

## Scripts to add
- `extract_and_score.py`: extract attention, compute geDIG per layer/head, compare to baselines.
- `lambda_scan.py`: run λ sweep on a model/dataset, emit summary JSON.
- `intervene_eval.py`: apply attention edits and evaluate on a small downstream task.
- Plotting: reuse `scripts/paper_templates/fig_lambda_phase_template.py` and `fig_phase_transition_template.py`.
