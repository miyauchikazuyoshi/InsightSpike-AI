#!/usr/bin/env bash
# Day 3 ablation: A(baseline/standard) vs B(DG-only/extended) vs C(DG+threelayer)
# 25x25 maze, 500 steps, 5 seeds (max_hops=10, sp-cand-topk=5 for speed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/ablation_day3_25x25"

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
)

echo "=== Day 3 Ablation Experiment ==="
echo "maze: ${MAZE_SIZE}x${MAZE_SIZE}, steps: $MAX_STEPS, seeds: $SEEDS"
echo "conditions: A(baseline/standard), B(DG-only/extended), C(DG+threelayer)"
echo ""

for SEED in $(seq 0 $((SEEDS - 1))); do
    echo "--- seed $SEED ---"

    # Condition A: baseline (standard 8D, legacy search)
    OUTA="$RESULTS_DIR/condA_seed${SEED}.json"
    STEPA="$RESULTS_DIR/condA_seed${SEED}_steps.json"
    echo "  [A] baseline (standard, legacy)..."
    $PYTHON "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --seed-start "$SEED" \
        --vector-mode standard \
        --search-mode legacy \
        --output "$OUTA" \
        --step-log "$STEPA" \
        2>&1 | tail -1
    echo "  [A] done -> $OUTA"

    # Condition B: DG-only (extended 10D, legacy search)
    OUTB="$RESULTS_DIR/condB_seed${SEED}.json"
    STEPB="$RESULTS_DIR/condB_seed${SEED}_steps.json"
    echo "  [B] DG-only (extended, legacy)..."
    $PYTHON "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --seed-start "$SEED" \
        --vector-mode extended \
        --search-mode legacy \
        --output "$OUTB" \
        --step-log "$STEPB" \
        2>&1 | tail -1
    echo "  [B] done -> $OUTB"

    # Condition C: DG + threelayer (extended 10D, threelayer search)
    OUTC="$RESULTS_DIR/condC_seed${SEED}.json"
    STEPC="$RESULTS_DIR/condC_seed${SEED}_steps.json"
    echo "  [C] DG+threelayer (extended, threelayer)..."
    $PYTHON "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --seed-start "$SEED" \
        --vector-mode extended \
        --search-mode threelayer \
        --output "$OUTC" \
        --step-log "$STEPC" \
        2>&1 | tail -1
    echo "  [C] done -> $OUTC"

    echo ""
done

echo "=== all conditions done ==="
echo "results in: $RESULTS_DIR/"
echo ""
echo "Run analysis: python $SCRIPT_DIR/analyze_ablation_day3.py $RESULTS_DIR"
