# Transformer Inference geDIG v2

Fixed-model inference experiment for validating geDIG v2 definitions.

## Scope

This subdirectory is for the inference-only experiment described in:

- `experiments/transformer/inference_gedig_v2/experiment_design_transformer_inference_gedig_v2.md`

The model weights stay fixed. The script analyzes layer-wise hidden-state transitions.

## Implemented Definitions

- `H(l)`: vocab entropy from `hidden(l) -> unembedding -> softmax`
- `EPC(l)`: normalized Frobenius change of pairwise distance matrices
- `SP(l)`: Spearman correlation of depth predictions across layers
- `B1(l)`: first Betti number of the layer-wise distance graph
- `F(l)`: `delta_EPC(l) - lambda * (delta_H(l) + gamma * delta_structural(l))`

Where:

- `delta_H(l) = H(l) - H(l-1)`
- `delta_EPC(l) = EPC(l) - EPC(l-1)`
- `delta_SP(l) = SP(l) - SP(l-1)`
- `delta_B1(l) = B1(l) - B1(l-1)`
- `delta_structural(l)` is chosen by `--f-structural-term` (`sp` or `betti1`)

## Files

- `run_inference_gedig_v2.py`: experiment runner
- `run_inference_gedig_v2.sh`: batch run shell (multi-model + auto-plot)
- `visualize_inference_gedig_v2.py`: visualization from `run_*.json`
- `summarize_multi_model_results.py`: aggregate multi-model runs with `delta_r2_learn` / `delta_r2_struct`
- `metrics.py`: metric definitions and fitting/grid-search utilities
- `experiment_design_transformer_inference_gedig_v2.md`: design doc

## Quick Start

Run baseline only:

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --max-samples 8 \
  --sp-mode both \
  --f-structural-term betti1 \
  --device auto
```

Run with controls + lambda/gamma grid search:

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --max-samples 16 \
  --sp-mode both \
  --f-structural-term betti1 \
  --grid-search \
  --shuffle-control \
  --random-control \
  --save-samples
```

Batch run with shell (default: `bert-base-uncased,gpt2`):

```bash
bash experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.sh
```

The shell sets safe defaults for constrained environments:
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `KMP_USE_SHM=0`
- `MPLCONFIGDIR` / `XDG_CACHE_HOME` under the results directory

Environment overrides for shell:

```bash
MODELS="bert-base-uncased,gpt2" \
MAX_SAMPLES=24 \
RANDOM_CONTROL=1 \
GRID_SEARCH=1 \
LOCAL_FILES_ONLY=1 \
PREFER_SAFETENSORS=0 \
bash experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.sh
```

Cache-only execution is also available from Python:

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --local-files-only
```

If you want safetensors-first behavior, add `--prefer-safetensors`
(or `PREFER_SAFETENSORS=1` in shell mode).

Use external structural-probe matrices (`.npy`):

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --b-dist path/to/B_dist.npy \
  --b-depth path/to/B_depth.npy \
  --proj-dim 128
```

Key structural options:

- `--sp-mode spearman|betti1|both` (default: `both`)
- `--f-structural-term sp|betti1` (default: `betti1`)
- `--betti-k-neighbors 5` (k-NN graph for Betti-1)
- `--betti-threshold <float>` (used when `--betti-k-neighbors <= 0`)

Visualize a specific run:

```bash
python experiments/transformer/inference_gedig_v2/visualize_inference_gedig_v2.py \
  --input experiments/transformer/inference_gedig_v2/results/bert-base-uncased/run_YYYYMMDDTHHMMSSZ.json
```

Visualize latest run in a results dir:

```bash
python experiments/transformer/inference_gedig_v2/visualize_inference_gedig_v2.py \
  --results-dir experiments/transformer/inference_gedig_v2/results/bert-base-uncased \
  --latest
```

Summarize multi-model results in one table:

```bash
python experiments/transformer/inference_gedig_v2/summarize_multi_model_results.py \
  --results-dir experiments/transformer/inference_gedig_v2/results/transfer_beta1_multi64_6models_20260210T194406
```

This writes:
- `multi_model_metrics.csv`
- `multi_model_metrics.md`

Primary (token-LM only) summary:

```bash
python experiments/transformer/inference_gedig_v2/summarize_multi_model_results.py \
  --results-dir experiments/transformer/inference_gedig_v2/results/transfer_beta1_multi64_6models_20260210T194406 \
  --track token_lm
```

This writes:
- `multi_model_metrics_token_lm.csv`
- `multi_model_metrics_token_lm.md`

Recommended reporting policy:
- Primary claims: `--track token_lm`
- Sentence-transformers: exploratory/secondary track only

Local deepening analysis (z-score refit + sequence checks):

```bash
python experiments/transformer/inference_gedig_v2/analyze_local_deepening.py \
  --results-dir experiments/transformer/inference_gedig_v2/results/transfer_beta1_multi64_6models_20260210T194406 \
  --track token_lm
```

This writes:
- `local_deepening_metrics_token_lm.csv`
- `local_deepening_summary_token_lm.md`

LLV-lite (causal-only) for Oyama-style distance check:

```bash
python experiments/transformer/inference_gedig_v2/compute_llv_lite.py \
  --models "distilgpt2,gpt2" \
  --text-file experiments/transformer/inference_gedig_v2/data/hotpotqa_questions_128.txt \
  --max-samples 128 \
  --output-dir experiments/transformer/inference_gedig_v2/results/transfer_beta1_gpt128_20260211T120137 \
  --metrics-csv experiments/transformer/inference_gedig_v2/results/transfer_beta1_gpt128_20260211T120137/local_deepening_metrics_token_lm.csv \
  --reference-model distilgpt2 \
  --prefer-safetensors
```

This writes:
- `llv_lite_summary.csv`
- `llv_lite_pairwise.csv`
- `llv_lite_vs_f.csv`
- `llv_lite_summary.md`

## Current Signal (2026-02-11)

From:
- `experiments/transformer/inference_gedig_v2/results/transfer_beta1_gpt128_20260211T120137`

Token-LM GPT series (`distilgpt2, gpt2, gpt2-medium`, `max-samples=128`) shows an
**initial signal** (preliminary):

- `delta_r2_struct`: `-0.7300 -> -0.1779 -> -0.1599` (nondecreasing)
- `random_r2`: `0.7490 -> 0.2542 -> 0.2315` (nonincreasing)

Metric glossary (for first-time readers):
- `baseline_r2`: linear-fit R² of the baseline mean `F(l)` curve.
- `shuffle_r2`: same R² under shuffled-input control.
- `random_r2`: same R² under random-initialized model control.
- `delta_r2_learn = baseline_r2 - random_r2`:
  positive means learned weights outperform random init in linearity.
- `delta_r2_struct = baseline_r2 - max(shuffle_r2, random_r2)`:
  positive means baseline beats the strongest control (stricter criterion).
- `best_lambda`, `best_gamma`:
  grid-search coefficients maximizing baseline linearity; not assumed universal.

Interpretation policy:
- This is enough to report as "initial signal" for the GPT family.
- It is not enough for a strong claim yet, because absolute `delta_r2_struct` values are still negative and cross-family verification is pending.

Results are saved under:

- `experiments/transformer/inference_gedig_v2/results/`
