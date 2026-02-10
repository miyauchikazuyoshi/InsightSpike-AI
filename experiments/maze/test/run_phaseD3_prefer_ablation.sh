#!/usr/bin/env bash
# Phase D-3: --sleep-guide prefer ablation
# 5 conditions × 3 seeds = 15 runs
# A: override baseline  B: prefer+Q  C: prefer+Q+prop0.5  D: prefer+Q+prop1.0  E: prefer+prop1.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD3_prefer"

mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$ROOT_DIR/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1

PYTHON="${ROOT_DIR}/.venv/bin/python3"

# Shared params (same as Phase D sweep)
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

echo "=== Phase D-3: Prefer Mode Ablation ==="
echo "A: override(baseline)  B: prefer+Q4  C: prefer+Q4+p0.5  D: prefer+Q4+p1.0  E: prefer+p1.0"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # Condition A: override baseline (α=0)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide override \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condA_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condA_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condA_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition B: prefer + Q-bias 4.0, no propagated
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condB_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condB_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condB_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition C: prefer + Q-bias 4.0 + propagated 0.5
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 0.5 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condC_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condC_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condC_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition D: prefer + Q-bias 4.0 + propagated 1.0
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 1.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condD_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condD_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condD_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # Condition E: prefer + propagated 1.0 only (no Q-bias)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 0.0 \
        --propagated-alpha 1.0 \
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
