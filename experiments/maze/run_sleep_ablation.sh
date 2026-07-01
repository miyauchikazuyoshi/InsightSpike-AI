#!/bin/bash
# Sleep-only ablation (pre-registered: docs/prereg/maze_sleep_ablation.md)
# Runs sleep-on and sleep-off per seed (paired), 1 seed at a time to avoid
# memory accumulation. Both conditions share ALL flags except --sleep-propagate.
# Usage: bash run_sleep_ablation.sh [seed_start] [seed_end]

set -e

SEED_START="${1:-0}"
SEED_END="${2:-29}"

OUTDIR="results/graph_persistent_dg/sleep_ablation"
mkdir -p "$OUTDIR"

# v6_perseed COMMON_ARGS + extended, held fixed for BOTH groups (see prereg §4)
COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep ablation: seeds $SEED_START to $SEED_END (on/off paired) ==="
echo "Start: $(date)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for COND in on off; do
        OUTFILE="$OUTDIR/${COND}_seed${SEED}.json"

        if [ -f "$OUTFILE" ]; then
            echo "[seed=$SEED $COND ] skip (already done)"
            continue
        fi

        T0=$(date +%s)
        echo -n "[seed=$SEED $COND ] running... "

        PYTHONPATH="$REPO_ROOT/src" \
        INSIGHTSPIKE_MIN_IMPORT=1 \
        INSIGHTSPIKE_LITE_MODE=1 \
        "$PY" run_experiment_query.py \
            --seeds 1 --seed-start "$SEED" \
            $COMMON_ARGS \
            --sleep-propagate "$COND" \
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
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) prop_nodes={c.get(\"inherited_propagated_nodes\",\"?\")}')
" 2>/dev/null || echo "parse error")

        echo "${ELAPSED}s | $RESULT"
    done
done

echo "Done: $(date)"
