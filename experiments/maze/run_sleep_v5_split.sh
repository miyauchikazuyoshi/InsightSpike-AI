#!/bin/bash
# v5: budget-split warmup (pre-registered: docs/prereg/maze_sleep_v5_budget_split.md)
# cyc2 (250+250 with intermediate sleep) vs cyc1 (500 single), total budget fixed,
# both arms replay + guide off + default penalties, on FRESH seeds (default 90-119).
# Usage: bash run_sleep_v5_split.sh [seed_start] [seed_end]
#   (extension rule: 120-149, once, only if <5 failed-cyc1-warmup seeds)

set -e

SEED_START="${1:-90}"
SEED_END="${2:-119}"

OUTDIR="results/graph_persistent_dg/sleep_v5_split"
mkdir -p "$OUTDIR"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended --sleep-propagate replay --sleep-guide off"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep v5 budget-split: cyc2 vs cyc1 (replay, guide off), seeds $SEED_START-$SEED_END ==="
echo "Start: $(date)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for NC in 2 1; do
        OUTFILE="$OUTDIR/cyc${NC}_seed${SEED}.json"

        if [ -f "$OUTFILE" ]; then
            echo "[seed=$SEED cyc=$NC ] skip (already done)"
            continue
        fi

        T0=$(date +%s)
        echo -n "[seed=$SEED cyc=$NC ] running... "

        PYTHONPATH="$REPO_ROOT/src" \
        INSIGHTSPIKE_MIN_IMPORT=1 \
        INSIGHTSPIKE_LITE_MODE=1 \
        "$PY" run_experiment_query.py \
            --seeds 1 --seed-start "$SEED" \
            $COMMON_ARGS \
            --wsw-cycles "$NC" \
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
wc = c.get('warmup_cycles') or []
cyc_str = '+'.join(('OK' if x['success'] else 'F')+str(x['steps']) for x in wc)
print(f'warmup[{cyc_str}] any_goal={c.get(\"warmup_any_goal\")} eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) ev_deadends={ev.get(\"dead_end_steps\",\"?\")}')
" 2>/dev/null || echo "parse error")

        echo "${ELAPSED}s | $RESULT"
    done
done

echo "Done: $(date)"
