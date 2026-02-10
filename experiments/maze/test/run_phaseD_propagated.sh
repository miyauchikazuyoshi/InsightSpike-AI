#!/usr/bin/env bash
# Phase D: propagated bias comparison — ON (α=1.0) vs OFF (α=0.0)
# WSW mode, extended vectors, 25x25 maze, 500 steps, 3 seeds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD_propagated"

mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$ROOT_DIR/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

PYTHON="${ROOT_DIR}/.venv/bin/python3"

MAZE_SIZE=25
MAX_STEPS=500
MAX_HOPS=10
SP_CAND_TOPK=5
WARMUP=200
THETA_AG=0.4

COMMON_ARGS=(
    --maze-size "$MAZE_SIZE"
    --max-steps "$MAX_STEPS"
    --seeds 1
    --max-hops "$MAX_HOPS"
    --sp-cand-topk "$SP_CAND_TOPK"
    --curriculum-warmup-steps "$WARMUP"
    --lambda-weight 1.0
    --theta-ag "$THETA_AG"
    --steps-ultra-light
    --sp-mode both
    --vector-mode extended
    --search-mode threelayer
    --sleep-propagate-gamma 0.9
    --sleep-propagate-iters 5
    --action-policy softmax
    --action-temp 1.0
)

echo "=== Phase D: Propagated Bias Comparison ==="
echo "maze: ${MAZE_SIZE}x${MAZE_SIZE}, steps: $MAX_STEPS"
echo "Conditions: A(OFF α=0), B(ON α=1.0), C(ON α=2.0)"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # Condition A: propagated bias OFF (α=0)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condA_seed${SEED}.json" \
        --step-log "$RESULTS_DIR/condA_seed${SEED}_steps.json" \
        > "$RESULTS_DIR/condA_seed${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition B: propagated bias ON (α=1.0)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --propagated-alpha 1.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condB_seed${SEED}.json" \
        --step-log "$RESULTS_DIR/condB_seed${SEED}_steps.json" \
        > "$RESULTS_DIR/condB_seed${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition C: propagated bias strong (α=2.0)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --propagated-alpha 2.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condC_seed${SEED}.json" \
        --step-log "$RESULTS_DIR/condC_seed${SEED}_steps.json" \
        > "$RESULTS_DIR/condC_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes"
echo "Waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All 9 runs completed ==="
else
    echo "=== WARNING: $FAILED runs failed ==="
fi
echo "Results in: $RESULTS_DIR"
