#!/usr/bin/env bash
# Phase D-5: W-S-W-S-W triple cycle ablation
# 5 conditions × 3 seeds = 15 runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD5_wswsw"

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

echo "=== Phase D-5: W-S-W-S-W Triple Cycle ==="
echo "A: W-S-W override  B: W-S-W prefer+Q4  C: W-S-W-S-W Q4  D: W-S-W-S-W Q4+grad  E: W-S-W-S-W grad"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # A: W-S-W override baseline
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --wsw-cycles 1 \
        --sleep-guide override \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condA_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condA_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condA_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # B: W-S-W prefer+Q4 (D-3 best)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --wsw-cycles 1 \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condB_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condB_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condB_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # C: W-S-W-S-W prefer+Q4 (2 cycles, no propagated)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --wsw-cycles 2 \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condC_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condC_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condC_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # D: W-S-W-S-W prefer+Q4+gradient (2 cycles + propagated)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --wsw-cycles 2 \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 1.0 \
        --propagated-mode gradient \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condD_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condD_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condD_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # E: W-S-W-S-W prefer+gradient only (no Q-bias)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --wsw-cycles 2 \
        --sleep-guide prefer \
        --sleep-q-beta 0.0 \
        --propagated-alpha 1.0 \
        --propagated-mode gradient \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condE_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condE_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condE_s${SEED}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes"
echo "Waiting..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAILED=$((FAILED + 1)); fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All ${#PIDS[@]} runs completed ==="
else
    echo "=== WARNING: $FAILED runs failed ==="
fi
echo "Results in: $RESULTS_DIR"
