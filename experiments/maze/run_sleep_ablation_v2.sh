#!/bin/bash
# Redesign variant #1 (pre-registered: docs/prereg/maze_sleep_ablation_v2.md)
# replay propagation vs off, BOTH arms with --sleep-guide off (self-navigation).
# Subjects: the 23 warmup-success seeds from the v1 ablation (deterministic
# stratification; warmup is guidance-independent).
# Usage: bash run_sleep_ablation_v2.sh

set -e

SEEDS="0 1 2 3 5 6 7 8 10 11 12 13 14 15 17 18 19 20 21 24 25 26 29"

OUTDIR="results/graph_persistent_dg/sleep_ablation_v2"
mkdir -p "$OUTDIR"

# v6_perseed COMMON_ARGS + extended, PLUS --sleep-guide off (R1 of the design audit)
COMMON_ARGS="--maze-size 25 --max-steps 500 --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 --steps-ultra-light --vector-mode extended --sleep-guide off"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"

echo "=== sleep ablation v2: replay vs off (guide off), ${SEEDS// /,} ==="
echo "Start: $(date)"

for SEED in $SEEDS; do
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
