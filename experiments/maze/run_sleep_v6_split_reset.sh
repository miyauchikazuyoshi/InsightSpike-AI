#!/bin/bash
# v6: budget-split warmup WITH episode-boundary reset (pre-registered:
# docs/prereg/maze_sleep_v6_split_reset.md).
# cyc2reset (250+250, intermediate sleep, --sleep-q-episode-reset) vs
# cyc1 (500 single; the flag is a structural no-op there and is not passed),
# total budget fixed, both arms replay + guide off + default penalties,
# FRESH seeds (default 120-149; v5 used 90-119, exploration used 5 of those).
# Usage: bash run_sleep_v6_split_reset.sh [seed_start] [seed_end]

set -e

SEED_START="${1:-120}"
SEED_END="${2:-149}"

OUTDIR="results/graph_persistent_dg/sleep_v6_split_reset"
mkdir -p "$OUTDIR"

COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended --sleep-propagate replay --sleep-guide off"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep v6 split+reset: cyc2reset vs cyc1 (replay, guide off), seeds $SEED_START-$SEED_END ==="
echo "Start: $(date)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for ARM in cyc2reset cyc1; do
        OUTFILE="$OUTDIR/${ARM}_seed${SEED}.json"

        if [ -f "$OUTFILE" ]; then
            echo "[seed=$SEED $ARM ] skip (already done)"
            continue
        fi

        if [ "$ARM" = "cyc2reset" ]; then
            ARM_ARGS="--wsw-cycles 2 --sleep-q-episode-reset"
        else
            ARM_ARGS="--wsw-cycles 1"
        fi

        T0=$(date +%s)
        echo -n "[seed=$SEED $ARM ] running... "

        PYTHONPATH="$REPO_ROOT/src" \
        INSIGHTSPIKE_MIN_IMPORT=1 \
        INSIGHTSPIKE_LITE_MODE=1 \
        "$PY" run_experiment_query.py \
            --seeds 1 --seed-start "$SEED" \
            $COMMON_ARGS \
            $ARM_ARGS \
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
bnd = (c.get('sleep_q') or {}).get('episode_boundaries_applied')
print(f'warmup[{cyc_str}] any_goal={c.get(\"warmup_any_goal\")} eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) ev_deadends={ev.get(\"dead_end_steps\",\"?\")} bounds={bnd}')
" 2>/dev/null || echo "parse error")

        echo "${ELAPSED}s | $RESULT"
    done
done

echo "Done: $(date)"
