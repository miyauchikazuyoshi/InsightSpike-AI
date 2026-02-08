#!/bin/bash
# Run experiments 1 seed at a time to avoid memory accumulation.
# Usage: bash run_v6_perseed.sh [baseline|extended] [seed_start] [seed_end]

set -e

MODE="${1:-extended}"       # baseline or extended
SEED_START="${2:-0}"
SEED_END="${3:-59}"

OUTDIR="results/graph_persistent_dg/v6_perseed"
SQLITEDIR="$OUTDIR/sqlite"
mkdir -p "$OUTDIR" "$SQLITEDIR"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light"

if [ "$MODE" = "extended" ]; then
    EXTRA_ARGS="--vector-mode extended"
else
    EXTRA_ARGS="--vector-mode standard"
fi

echo "=== $MODE mode: seeds $SEED_START to $SEED_END ==="
echo "Start: $(date)"

for SEED in $(seq $SEED_START $SEED_END); do
    OUTFILE="$OUTDIR/${MODE}_seed${SEED}.json"
    SQLITE="$SQLITEDIR/${MODE}_seed${SEED}.db"

    if [ -f "$OUTFILE" ]; then
        echo "[seed=$SEED] skip (already done)"
        continue
    fi

    T0=$(date +%s)
    echo -n "[seed=$SEED] running... "

    PYTHONPATH=/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/src \
    INSIGHTSPIKE_MIN_IMPORT=1 \
    INSIGHTSPIKE_LITE_MODE=1 \
    /Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/.venv/bin/python3 \
        run_experiment_query.py \
        --seeds 1 --seed-start "$SEED" \
        $COMMON_ARGS $EXTRA_ARGS \
        --persist-graph-sqlite "$SQLITE" \
        --output "$OUTFILE" \
        > /dev/null 2>&1

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    # Extract key metrics from JSON
    RESULT=$(/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/.venv/bin/python3 -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
s = d.get('summary',{})
ws = d.get('warmup_summary',{})
c = d.get('curriculum',{}).get('per_seed',{}).get('$SEED',{})
w = c.get('warmup',{})
e = c.get('eval',{})
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) edges={s.get(\"avg_edges\",\"?\")}')
" 2>/dev/null || echo "parse error")

    echo "${ELAPSED}s | $RESULT"
done

echo "Done: $(date)"
