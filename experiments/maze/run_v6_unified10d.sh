#!/bin/bash
# Run unified 10D experiments: 1 seed at a time.
# Key changes from v6_perseed: lambda=1.0, theta_ag=0.4, max_hops=10, linkset ON (default).
# Usage: bash run_v6_unified10d.sh [seed_start] [seed_end]

set -e

BASEDIR="/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/maze"
cd "$BASEDIR"

SEED_START="${1:-0}"
SEED_END="${2:-59}"

OUTDIR="results/graph_persistent_dg/v6_perseed/unified10d"
SQLITEDIR="$OUTDIR/sqlite"
LOGFILE="$OUTDIR/progress.log"
mkdir -p "$OUTDIR" "$SQLITEDIR"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 10 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 1.0 --theta-ag 0.4 --vector-mode extended --steps-ultra-light"

echo "=== unified10d: seeds $SEED_START to $SEED_END ===" | tee -a "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"

for SEED in $(seq $SEED_START $SEED_END); do
    OUTFILE="$OUTDIR/seed${SEED}.json"
    SQLITE="$SQLITEDIR/seed${SEED}.db"

    if [ -f "$OUTFILE" ]; then
        echo "[seed=$SEED] skip (already done)" | tee -a "$LOGFILE"
        continue
    fi

    T0=$(date +%s)
    echo -n "[seed=$SEED] running... " | tee -a "$LOGFILE"

    PYTHONPATH=/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/src \
    INSIGHTSPIKE_MIN_IMPORT=1 \
    INSIGHTSPIKE_LITE_MODE=1 \
    /Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/.venv/bin/python3 \
        run_experiment_query.py \
        --seeds 1 --seed-start "$SEED" \
        $COMMON_ARGS \
        --persist-graph-sqlite "$SQLITE" \
        --output "$OUTFILE" \
        > /dev/null 2>&1

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    RESULT=$(/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/.venv/bin/python3 -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
s = d.get('summary',{})
c = d.get('curriculum',{}).get('per_seed',{}).get('$SEED',{})
w = c.get('warmup',{})
e = c.get('eval',{})
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) edges={s.get(\"avg_edges\",\"?\")}')
" 2>/dev/null || echo "parse error")

    echo "${ELAPSED}s | $RESULT" | tee -a "$LOGFILE"
done

echo "Done: $(date)" | tee -a "$LOGFILE"
