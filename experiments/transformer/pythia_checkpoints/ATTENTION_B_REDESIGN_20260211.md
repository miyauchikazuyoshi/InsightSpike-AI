# Pythia Attention-B Redesign (2026-02-11)

## Goal

Keep the training-dynamics experiment **attention-based**, and add explicit `B` definition.

## B Definition (attention-based)

In this redesign, `B` is not probe matrix.  
`B` is defined as topological complexity from attention graph:

- `B(l) := beta1(l)` (first Betti number) of layer-`l` attention graph.
- Graph is built from symmetric attention:
  - `W = (A + A^T) / 2`
  - then k-NN graph (`k=5` default) or threshold graph.

So structural term can be either:

- `delta_SP` (legacy shortcut purity), or
- `delta_B1` (new B term).

## F Formula

`F = delta_EPC - lambda * (entropy_sign * delta_H + gamma * delta_structural)`

where:

- `delta_structural = delta_SP` if `--structural-term sp`
- `delta_structural = delta_B1` if `--structural-term betti1`

## Implementation

Updated file:

- `experiments/transformer/pythia_checkpoints/analyze_training_dynamics.py`

Key additions:

- `--structural-term sp|betti1`
- `--betti-k-neighbors`
- `--betti-threshold`
- `B1_mean`, `delta_B1_mean` in JSON output
- structure plot switches between SP and B1 by `--structural-term`

## Example Runs

Legacy SP mode:

```bash
python experiments/transformer/pythia_checkpoints/analyze_training_dynamics.py \
  --light \
  --samples 10 \
  --structural-term sp
```

Attention-B mode (recommended):

```bash
python experiments/transformer/pythia_checkpoints/analyze_training_dynamics.py \
  --light \
  --samples 10 \
  --structural-term betti1 \
  --betti-k-neighbors 5 \
  --lambda-param 1.0 \
  --gamma 0.5
```

## Interpretation Note

This redesign stays fully in attention space.  
If you later need probe-matrix `B_dist/B_depth`, that is a separate hidden-state experiment track.
