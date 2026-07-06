#!/bin/bash
# v7 exploration: link-radius sweep. The premise-check found recall edges
# are capped at Manhattan-2 (ring = ceil(link_radius*25) = 2 at lr=0.05).
# Question: does raising link-radius let recall reach farther branches and
# enrich the β₁ cycle structure (more shortcuts, larger reach/sizes)?
# Rr = ceil(lr*25): lr 0.05->ring2, 0.10->ring3, 0.20->ring5.
# Exploration only (isolated, NOTES.md). Records graph dump + eval steps.
# Usage: bash run_v7_linkradius_sweep.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
BASE="results/graph_persistent_dg/_exploratory_v7_linkradius"
SEEDS="129 120 118"
# lr:ringtag pairs
declare -a LRS=("0.05:ring2" "0.10:ring3" "0.20:ring5")

echo "=== v7 link-radius sweep (seeds $SEEDS) start $(date) ==="
# radii OUTER so fast rings (2,3) finish for all seeds before the slow ring 5
for PAIR in "${LRS[@]}"; do
  for SEED in $SEEDS; do
    LR="${PAIR%%:*}"; TAG="${PAIR##*:}"
    DUMP="$BASE/$TAG"; mkdir -p "$DUMP"
    OUT="$DUMP/seed${SEED}.json"
    [ -f "$DUMP/graph_seed${SEED}.json" ] && { echo "[seed=$SEED $TAG] skip"; continue; }
    T0=$(date +%s)
    INSIGHTSPIKE_DUMP_GRAPH="$DUMP" PYTHONPATH="$REPO/src" \
    INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 \
    "$PY" run_experiment_query.py \
      --maze-size 25 --max-steps 500 --seeds 1 --seed-start "$SEED" \
      --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 \
      --vector-mode extended --sleep-propagate replay --sleep-guide off --steps-ultra-light \
      --link-radius "$LR" \
      --output "$OUT" > "$DUMP/log_seed${SEED}.txt" 2>&1
    T1=$(date +%s)
    EV=$("$PY" -c "
import json
d=json.load(open('$OUT'))['curriculum']['per_seed']['$SEED']
e=d['eval']; print(f\"eval={'OK' if e['success'] else 'FAIL'}({e['steps']}) warmup_goal={d.get('warmup_any_goal')}\")" 2>/dev/null || echo "parse err")
    echo "[seed=$SEED $TAG lr=$LR] $((T1-T0))s | $EV"
  done
done
echo "=== done $(date) ==="
