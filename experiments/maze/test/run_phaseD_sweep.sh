#!/usr/bin/env bash
# Phase D: propagated-alpha sweep — 0, 0.1, 0.3, 0.5, 1.0
# WSW mode, 25x25 maze, 500 steps, 3 seeds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD_sweep"

mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$ROOT_DIR/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

PYTHON="${ROOT_DIR}/.venv/bin/python3"

COMMON_ARGS=(
    --maze-size 25 --max-steps 500 --seeds 1
    --max-hops 10 --sp-cand-topk 5
    --curriculum-warmup-steps 200
    --lambda-weight 1.0 --theta-ag 0.4
    --steps-ultra-light --sp-mode both
    --vector-mode extended --search-mode threelayer
    --sleep-propagate-gamma 0.9 --sleep-propagate-iters 5
    --action-policy softmax --action-temp 1.0
)

ALPHAS="0.0 0.1 0.3 0.5 1.0"
PIDS=()

echo "=== Phase D: Alpha Sweep ==="
echo "alphas: $ALPHAS, seeds: 0-2"

for ALPHA in $ALPHAS; do
    LABEL=$(echo "$ALPHA" | tr '.' 'p')
    for SEED in 0 1 2; do
        "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
            "${COMMON_ARGS[@]}" \
            --propagated-alpha "$ALPHA" \
            --seed-start "$SEED" \
            --output "$RESULTS_DIR/a${LABEL}_s${SEED}.json" \
            --step-log "$RESULTS_DIR/a${LABEL}_s${SEED}_steps.json" \
            > "$RESULTS_DIR/a${LABEL}_s${SEED}.log" 2>&1 &
        PIDS+=($!)
    done
done

echo "Launched ${#PIDS[@]} processes"
echo "Waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAILED=$((FAILED + 1)); fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All done ==="
else
    echo "=== $FAILED failed ==="
fi
