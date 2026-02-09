#!/usr/bin/env bash
# β₁ smoke test: small maze, --sp-mode both
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$ROOT_DIR/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

PYTHON="${ROOT_DIR}/.venv/bin/python3"

echo "=== β₁ smoke test ==="
echo "maze-size: 9, max-steps: 100, seeds: 3, sp-mode: both"
echo ""

for SEED in 0 1 2; do
    OUTFILE="$RESULTS_DIR/betti1_smoke_seed${SEED}.json"
    echo "[seed $SEED] running..."
    $PYTHON "$MAZE_DIR/run_experiment_query.py" \
        --maze-size 9 \
        --max-steps 100 \
        --seeds 1 --seed-start "$SEED" \
        --max-hops 10 \
        --sp-cand-topk 5 \
        --curriculum-warmup-steps 100 \
        --lambda-weight 1.0 \
        --theta-ag 0.4 \
        --vector-mode extended \
        --steps-ultra-light \
        --sp-mode both \
        --output "$OUTFILE" \
        2>&1 | tail -3
    echo "[seed $SEED] done -> $OUTFILE"
    echo ""
done

echo "=== all seeds done ==="
echo "results in: $RESULTS_DIR/"
