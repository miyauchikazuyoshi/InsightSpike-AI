#!/bin/bash
# v3: lift of failed warmups (pre-registered: docs/prereg/maze_sleep_ablation_v3.md)
# replay vs off, both arms --sleep-guide off, on FRESH seeds (default 30-59).
# All seeds are run; stratification by warmup outcome happens at analysis time.
# Usage: bash run_sleep_ablation_v3.sh [seed_start] [seed_end]
#   (extension rule of prereg section 4: only 60-89, only once, only if <5 failed warmups)

set -e

SEED_START="${1:-30}"
SEED_END="${2:-59}"

OUTDIR="results/graph_persistent_dg/sleep_ablation_v3"
mkdir -p "$OUTDIR"

# identical to v2: v6_perseed COMMON_ARGS + extended + guide off; negative-example params at defaults
COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended --sleep-guide off"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep ablation v3: replay vs off (guide off), seeds $SEED_START-$SEED_END ==="
echo "Start: $(date)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for COND in replay off; do
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
ev = (d.get('runs') or [{}])[0]
print(f'warmup={\"OK\" if w.get(\"success\") else \"FAIL\"}({w.get(\"steps\",\"?\")}) eval={\"OK\" if e.get(\"success\") else \"FAIL\"}({e.get(\"steps\",\"?\")}) ev_deadends={ev.get(\"dead_end_steps\",\"?\")} prop_nodes={c.get(\"inherited_propagated_nodes\",\"?\")}')
" 2>/dev/null || echo "parse error")

        echo "${ELAPSED}s | $RESULT"
    done
done

echo "Done: $(date)"
