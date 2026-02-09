#!/bin/bash
# β₁ comparison experiment: unified10d + --sp-mode both
# Same params as run_v6_unified10d.sh, but records both ASP and β₁.
# Usage: bash run_betti1_25x25.sh [seed_start] [seed_end]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAZE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MAZE_DIR/../.." && pwd)"

SEED_START="${1:-0}"
SEED_END="${2:-4}"

OUTDIR="$SCRIPT_DIR/results"
SQLITEDIR="$OUTDIR/sqlite"
LOGFILE="$OUTDIR/progress.log"
mkdir -p "$OUTDIR" "$SQLITEDIR"

PYTHON="$ROOT_DIR/.venv/bin/python3"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 10 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 1.0 --theta-ag 0.4 --vector-mode extended --steps-ultra-light --sp-mode both"

echo "=== β₁ comparison 25x25: seeds $SEED_START to $SEED_END ===" | tee -a "$LOGFILE"
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

    PYTHONPATH="$ROOT_DIR/src" \
    INSIGHTSPIKE_MIN_IMPORT=1 \
    INSIGHTSPIKE_LITE_MODE=1 \
    $PYTHON "$MAZE_DIR/run_experiment_query.py" \
        --seeds 1 --seed-start "$SEED" \
        $COMMON_ARGS \
        --persist-graph-sqlite "$SQLITE" \
        --output "$OUTFILE" \
        > /dev/null 2>&1

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    RESULT=$($PYTHON -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
r = d.get('runs',[{}])[0]
b1 = r.get('betti1_series',[])
nc = r.get('node_count_series',[])
max_b = max(b1) if b1 else 0
final_b = b1[-1] if b1 else 0
final_v = nc[-1] if nc else 0
c = d.get('curriculum',{}).get('per_seed',{}).get('$SEED',{})
w = c.get('warmup',{})
e = c.get('eval',{})
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) V={final_v} β₁={final_b}(max={max_b})')
" 2>/dev/null || echo "parse error")

    echo "${ELAPSED}s | $RESULT" | tee -a "$LOGFILE"
done

echo "Done: $(date)" | tee -a "$LOGFILE"
echo ""
echo "Run analysis: $PYTHON $SCRIPT_DIR/analyze_betti1.py"
