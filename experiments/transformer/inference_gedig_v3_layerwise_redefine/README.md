# Transformer Inference geDIG v3 (Layerwise Redefine)

Active workspace for redefined layer-wise observation experiments.

Base frozen snapshot:
- `experiments/transformer/archive/inference_gedig_v2_freeze_20260211T142929`

Archive policy:
- Past experiments are frozen under `experiments/transformer/archive/`.
- Do not edit snapshot directories.

## Scope

This directory is the working area for re-running transformer layerwise observation with redefined EPC/H/B settings.

Design document:
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/experiment_design_transformer_inference_gedig_v3_layerwise_redefine.md`

## Files

- `run_inference_gedig_v2.py`: main runner (v3 options added)
- `run_inference_gedig_v2.sh`: batch runner
- `metrics.py`: core metric definitions
- `visualize_inference_gedig_v2.py`: plots from `run_*.json`
- `summarize_multi_model_results.py`: aggregate model-level metrics
- `analyze_local_deepening.py`: z-score refit + sequence checks
- `compute_llv_lite.py`: LLV-lite distance analysis
- `fit_probe_b_lightweight.py`: local lightweight trainer for `B_dist` / `B_depth`

## v3 Redefinition Controls

- `--require-probe-b`
  - claim-grade mode: requires both `--b-dist` and `--b-depth`
- `--entropy-token-mode auto|all|content|tail_k`
  - selects token subset used for entropy observation
- `--entropy-tail-k`
  - tail token count for `tail_k` / `auto(causal)`
- `--epc-distance-norm none|median`
  - normalizes pairwise distance matrix scale per layer before EPC

## Quick Start

Baseline run:

```bash
python experiments/transformer/inference_gedig_v3_layerwise_redefine/run_inference_gedig_v2.py \
  --model gpt2 \
  --max-samples 8 \
  --sp-mode both \
  --f-structural-term betti1 \
  --entropy-token-mode auto \
  --epc-distance-norm median \
  --local-files-only \
  --prefer-safetensors
```

Claim-grade run (probe B required):

```bash
python experiments/transformer/inference_gedig_v3_layerwise_redefine/run_inference_gedig_v2.py \
  --model gpt2 \
  --max-samples 32 \
  --sp-mode both \
  --f-structural-term betti1 \
  --entropy-token-mode auto \
  --entropy-tail-k 8 \
  --epc-distance-norm median \
  --require-probe-b \
  --b-dist path/to/B_dist.npy \
  --b-depth path/to/B_depth.npy \
  --grid-search \
  --shuffle-control \
  --random-control
```

Build probe B locally (spaCy dependency-tree supervision):

```bash
python experiments/transformer/inference_gedig_v3_layerwise_redefine/fit_probe_b_lightweight.py \
  --model gpt2 \
  --max-samples 64 \
  --max-length 128 \
  --layer-index -1 \
  --proj-dim 128 \
  --epochs-dist 80 \
  --epochs-depth 80 \
  --prefer-safetensors
```

Outputs:
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/artifacts/probe_b_lightweight/<model>/.../B_dist.npy`
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/artifacts/probe_b_lightweight/<model>/.../B_depth.npy`

Batch run:

```bash
MODELS="gpt2,gpt2-medium" \
MAX_SAMPLES=64 \
RANDOM_CONTROL=1 \
GRID_SEARCH=1 \
ENTROPY_TOKEN_MODE=auto \
ENTROPY_TAIL_K=8 \
EPC_DISTANCE_NORM=median \
bash experiments/transformer/inference_gedig_v3_layerwise_redefine/run_inference_gedig_v2.sh
```

## Output

Default output root:
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/results/`

Each run writes `run_*.json`; visualization writes PNGs in the same model directory.

## Current Status

Exploratory pilot (no probe-B requirement):
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/results/pilot_redefine_16_20260211T144602`

Claim-grade trials (`--require-probe-b`):
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/results/claim_probeb_distilgpt2_64/`
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/results/claim_probeb_gpt2_64/`
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/results/claim_probeb_gpt2-medium_layer12_64/`

Fixed-parameter control margin (`delta_r2_struct_fixed = baseline_r2 - max(shuffle_r2, random_r2)`):
- `distilgpt2` (64): `+0.0937`
- `gpt2` (64): `+0.0787`
- `gpt2-medium` layer12-B (64): `+0.0102`

Interpretation:
- Pipeline and artifact generation are stable (`run_*.json` + PNG outputs confirmed).
- Initial positive signal appears on `distilgpt2` and `gpt2`.
- `gpt2-medium` remains unstable/weak; treat current claim as early-stage evidence.
