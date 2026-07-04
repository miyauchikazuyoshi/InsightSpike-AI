#!/bin/bash
# v4: deadend carving (pre-registered: docs/prereg/maze_sleep_v4_deadend_carving.md)
# carving (deadend/blocked -1.0, unified with dim8) vs current (defaults 0.0),
# both arms replay + guide off, on FRESH seeds (default 60-89).
# Usage: bash run_sleep_v4_carving.sh [seed_start] [seed_end]
#   (extension rule: 90-119, once, only if <5 failed warmups — prereg section 4)

set -e

SEED_START="${1:-60}"
SEED_END="${2:-89}"

OUTDIR="results/graph_persistent_dg/sleep_v4_carving"
mkdir -p "$OUTDIR"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended --sleep-propagate replay --sleep-guide off"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep v4 carving: carving vs current (replay, guide off), seeds $SEED_START-$SEED_END ==="
echo "Start: $(date)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for COND in carving current; do
        OUTFILE="$OUTDIR/${COND}_seed${SEED}.json"

        if [ -f "$OUTFILE" ]; then
            echo "[seed=$SEED $COND ] skip (already done)"
            continue
        fi

        if [ "$COND" = "carving" ]; then
            EXTRA="--sleep-q-deadend-penalty -1.0 --sleep-q-blocked-penalty -1.0"
        else
            EXTRA=""
        fi

        T0=$(date +%s)
        echo -n "[seed=$SEED $COND ] running... "

        PYTHONPATH="$REPO_ROOT/src" \
        INSIGHTSPIKE_MIN_IMPORT=1 \
        INSIGHTSPIKE_LITE_MODE=1 \
        "$PY" run_experiment_query.py \
            --seeds 1 --seed-start "$SEED" \
            $COMMON_ARGS $EXTRA \
            --output "$OUTFILE" \
            > /dev/null 2>&1

        T1=$(date +%s)
        ELAPSED=$((T1 - T0))

        RESULT=$("$PY" -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
c = d.get('curriculum',{}).get('per_seed',{}).get('$SEED',{})
w, e = c.get('warmup',{}), c.get('eval',{})
ev = (d.get('runs') or [{}])[0]
sq = c.get('sleep_q') or {}
qmin = sq.get('q_min') if isinstance(sq, dict) else None
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) ev_deadends={ev.get(\"dead_end_steps\",\"?\")} q_min={qmin if qmin is None else round(float(qmin),3)}')
" 2>/dev/null || echo "parse error")

        echo "${ELAPSED}s | $RESULT"
    done
done

echo "Done: $(date)"
