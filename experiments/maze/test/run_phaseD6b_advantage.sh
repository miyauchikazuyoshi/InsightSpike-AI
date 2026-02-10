#!/usr/bin/env bash
# Phase D-6b: advantage-gated action selection
# 5 conditions × 3 seeds = 15 runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/phaseD6b_advantage"

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
    --sleep-guide prefer --wsw-cycles 1
)

echo "=== Phase D-6b: Advantage-Gated Selection ==="
echo "A: q8 no-gate  B: q8 gate1.5  C: q8 gate2.0  D: q8 gate3.0  E: q4 gate2.0"
echo ""

PIDS=()

for SEED in 0 1 2; do
    # A: q-beta=8, no gate (D-6 best)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-q-beta 8.0 --advantage-commit 0.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condA_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condA_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condA_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # B: q-beta=8, gate=1.5 (mild commit)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-q-beta 8.0 --advantage-commit 1.5 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condB_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condB_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condB_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # C: q-beta=8, gate=2.0 (moderate commit)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-q-beta 8.0 --advantage-commit 2.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condC_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condC_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condC_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # D: q-beta=8, gate=3.0 (strong commit)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-q-beta 8.0 --advantage-commit 3.0 \
        --propagated-alpha 0.0 \
        --seed-start "$SEED" \
        --output "$RESULTS_DIR/condD_s${SEED}.json" \
        --step-log "$RESULTS_DIR/condD_s${SEED}_steps.json" \
        > "$RESULTS_DIR/condD_s${SEED}.log" 2>&1 &
    PIDS+=($!)

    # E: q-beta=4, gate=2.0 (lower Q + moderate commit)
    "$PYTHON" "$MAZE_DIR/run_experiment_query.py" \
        "${COMMON_ARGS[@]}" \
        --sleep-q-beta 4.0 --advantage-commit 2.0 \
        --propagated-alpha 0.0 \
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
