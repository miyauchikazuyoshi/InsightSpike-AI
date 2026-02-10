#!/usr/bin/env bash
# Phase D-4: gradient propagated bias ablation
# 5 conditions × 3 seeds = 15 runs
# A: override baseline  B: prefer+Q4  C: prefer+Q4+grad1  D: prefer+grad1  E: prefer+grad2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD4_gradient"

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

echo "=== Phase D-4: Gradient Propagated Bias ==="
echo "A: override  B: prefer+Q4  C: prefer+Q4+grad1  D: prefer+grad1  E: prefer+grad2"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # A: override baseline (α=0)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide override \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condA_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condA_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condA_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # B: prefer + Q-bias 4.0, no propagated (D-3 best)
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

    # C: prefer + Q-bias 4.0 + gradient α=1.0
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 4.0 \
        --propagated-alpha 1.0 \
        --propagated-mode gradient \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condC_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condC_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condC_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # D: prefer + gradient α=1.0 only (no Q-bias)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 0.0 \
        --propagated-alpha 1.0 \
        --propagated-mode gradient \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condD_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condD_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condD_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # E: prefer + gradient α=2.0 only (stronger, no Q-bias)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-guide prefer \
        --sleep-q-beta 0.0 \
        --propagated-alpha 2.0 \
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
