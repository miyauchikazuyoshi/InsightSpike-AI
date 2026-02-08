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
- `F(l)`: `delta_EPC(l) - lambda * (delta_H(l) + gamma * delta_SP(l))`

Where:

- `delta_H(l) = H(l) - H(l-1)`
- `delta_EPC(l) = EPC(l) - EPC(l-1)`
- `delta_SP(l) = SP(l) - SP(l-1)`

## Files

- `run_inference_gedig_v2.py`: experiment runner
- `run_inference_gedig_v2.sh`: batch run shell (multi-model + auto-plot)
- `visualize_inference_gedig_v2.py`: visualization from `run_*.json`
- `metrics.py`: metric definitions and fitting/grid-search utilities
- `experiment_design_transformer_inference_gedig_v2.md`: design doc

## Quick Start

Run baseline only:

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --max-samples 8 \
  --device auto
```

Run with controls + lambda/gamma grid search:

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --max-samples 16 \
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
bash experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.sh
```

Use external structural-probe matrices (`.npy`):

```bash
python experiments/transformer/inference_gedig_v2/run_inference_gedig_v2.py \
  --model bert-base-uncased \
  --b-dist path/to/B_dist.npy \
  --b-depth path/to/B_depth.npy \
  --proj-dim 128
```

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

Results are saved under:

- `experiments/transformer/inference_gedig_v2/results/`
