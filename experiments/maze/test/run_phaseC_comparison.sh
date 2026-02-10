#!/usr/bin/env bash
# Phase C-2: Comparison experiment — legacy vs 3att-always vs 3att-after-warmup
# 3 conditions × 3 seeds, WSW mode, 25x25 maze, 500 steps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseC_comparison"

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
)

echo "=== Phase C-2: Score Mode Comparison ==="
echo "maze: ${MAZE_SIZE}x${MAZE_SIZE}, steps: $MAX_STEPS"
echo "Conditions: A(legacy), B(3att-always), C(3att-after-200)"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # Condition A: legacy scoring throughout
    OUT="$RESULTS_DIR/condA_seed${SEED}.json"
    STEP="$RESULTS_DIR/condA_seed${SEED}_steps.json"
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --l1-score-mode legacy \
        --seed-start "$SEED" \
        --output "$OUT" --step-log "$STEP" \
        > "$RESULTS_DIR/condA_seed${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition B: 3att scoring from step 0
    OUT="$RESULTS_DIR/condB_seed${SEED}.json"
    STEP="$RESULTS_DIR/condB_seed${SEED}_steps.json"
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --l1-score-mode 3att \
        --seed-start "$SEED" \
        --output "$OUT" --step-log "$STEP" \
        > "$RESULTS_DIR/condB_seed${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition C: legacy until step 200, then 3att
    OUT="$RESULTS_DIR/condC_seed${SEED}.json"
    STEP="$RESULTS_DIR/condC_seed${SEED}_steps.json"
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --l1-score-mode 3att \
        --l1-score-switch-step 200 \
        --seed-start "$SEED" \
        --output "$OUT" --step-log "$STEP" \
        > "$RESULTS_DIR/condC_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes (3 conditions × 3 seeds)"
echo "Waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All 9 runs completed successfully ==="
else
    echo "=== WARNING: $FAILED runs failed ==="
fi
echo "Results in: $RESULTS_DIR"
