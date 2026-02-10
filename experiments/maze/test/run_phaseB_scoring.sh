#!/usr/bin/env bash
# Phase B-3: Parallel scoring experiment (legacy + 3att logged in parallel)
# WSW mode, threelayer search, 25x25 maze, 500 steps, 3 seeds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseB_scoring"

mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$ROOT_DIR/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

PYTHON="${ROOT_DIR}/.venv/bin/python3"

MAZE_SIZE=25
MAX_STEPS=500
SEEDS=3
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
    --l1-score-mode legacy
)

echo "=== Phase B-3: Parallel Scoring Experiment ==="
echo "maze: ${MAZE_SIZE}x${MAZE_SIZE}, steps: $MAX_STEPS, seeds: $SEEDS"
echo "mode: WSW + threelayer (both legacy & 3att scores logged)"
echo ""

for SEED in $(seq 0 $((SEEDS - 1))); do
    echo "--- seed $SEED ---"
    OUT="$RESULTS_DIR/phaseB_seed${SEED}.json"
    STEP="$RESULTS_DIR/phaseB_seed${SEED}_steps.json"
    LOG="$RESULTS_DIR/phaseB_seed${SEED}.log"

    echo "  Running WSW+threelayer seed=$SEED ..."
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --seed-start "$SEED" \
        --output "$OUT" \
        --step-log "$STEP" \
        2>&1 | tee "$LOG" &
done

echo ""
echo "Waiting for all seeds to complete..."
wait
echo ""
echo "=== All seeds done ==="
echo "Results in: $RESULTS_DIR"
