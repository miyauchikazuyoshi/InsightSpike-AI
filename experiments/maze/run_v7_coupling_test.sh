#!/bin/bash
# v7 exploration: DG-readout coupling test.
# Tonight's link-radius sweep found β₁ structure is behaviorally SILENT in
# the winning legacy config (guide off): eval invariant to link-radius.
# Question: does turning on a DG-aware readout (--l1-score-mode 3att, which
# wires σ(-dg/τ) via dg_attention=g_min) make eval RESPOND to link-radius?
# If 3att eval differs ring2 vs ring5 (while legacy was flat) -> the DG
# readout is the structure->behavior coupling lever -> green light for the
# full σ(dg) build. If flat -> silent even under 3att.
# 3att is the config that LOST to legacy in 2026-02; we test COUPLING
# (does eval respond to the knob), not whether 3att wins.
# Exploration only (isolated, NOTES.md). Usage: bash run_v7_coupling_test.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
BASE="results/graph_persistent_dg/_exploratory_v7_coupling"
SEEDS="129 120 118"
declare -a LRS=("0.05:ring2" "0.20:ring5")

echo "=== v7 coupling test: 3att × {ring2,ring5} × seeds $SEEDS  start $(date) ==="
for PAIR in "${LRS[@]}"; do
  for SEED in $SEEDS; do
    LR="${PAIR%%:*}"; TAG="${PAIR##*:}"
    DIR="$BASE/3att_$TAG"; mkdir -p "$DIR"
    OUT="$DIR/seed${SEED}.json"
    [ -f "$OUT" ] && { echo "[3att $TAG seed=$SEED] skip"; continue; }
    T0=$(date +%s)
    PYTHONPATH="$REPO/src" INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 \
    "$PY" run_experiment_query.py \
      --maze-size 25 --max-steps 500 --seeds 1 --seed-start "$SEED" \
      --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 \
      --vector-mode extended --sleep-propagate replay --sleep-guide off --steps-ultra-light \
      --link-radius "$LR" --search-mode threelayer --l1-score-mode 3att \
      --output "$OUT" > "$DIR/log_seed${SEED}.txt" 2>&1
    T1=$(date +%s)
    EV=$("$PY" -c "
import json
d=json.load(open('$OUT'))['curriculum']['per_seed']['$SEED']; e=d['eval']
print(f\"eval={'OK' if e['success'] else 'FAIL'}({e['steps']}) warmup_goal={d.get('warmup_any_goal')}\")" 2>/dev/null || echo "parse err")
    echo "[3att $TAG seed=$SEED lr=$LR] $((T1-T0))s | $EV  (legacy baseline was ring-invariant)"
  done
done
echo "=== done $(date) ==="
