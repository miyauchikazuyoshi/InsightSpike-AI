#!/bin/bash
# Grid search for AG threshold and MAX_K parameters
# Spec Q.1 parameter optimization
#
# Parameters to search:
#   AG threshold: 0.1, 0.15, 0.2, 0.3, 0.4
#   AG MAX_K: 5, 10, 20, 50, None (unlimited)
#
# Fixed:
#   ag_min_k=5, beta1_weight=0.3, domains=biology, limit=50

set -e
cd "$(dirname "$0")/.."

PYTHON=../../.venv/bin/python3
SCRIPT=scripts/run_bright.py
RESULT_BASE=results/v21_specq1_grid

# Common flags
COMMON="--mode cot_retrieval --domains biology --limit 50 \
  --scoring-mode gedig_refine --rerank-alpha 0.1 --graph-top-k 50 \
  --token-graph --token-graph-walk-score \
  --ria-loop --ria-max-rounds 3 \
  --entity-feval --entity-feval-version v2"

# Grid
THRESHOLDS="0.1 0.15 0.2 0.3 0.4"
MAX_KS="5 10 20 50 0"  # 0 = unlimited (no --ag-max-k flag)

echo "=== Spec Q.1 Grid Search: AG Parameters ==="
echo "Thresholds: $THRESHOLDS"
echo "MAX_Ks: $MAX_KS (0=unlimited)"
echo ""

for thresh in $THRESHOLDS; do
  for maxk in $MAX_KS; do
    if [ "$maxk" = "0" ]; then
      label="t${thresh}_kinf"
      maxk_flag=""
    else
      label="t${thresh}_k${maxk}"
      maxk_flag="--ag-max-k $maxk"
    fi

    outdir="${RESULT_BASE}/${label}"

    # Skip if already done
    result_file="${outdir}/results.jsonl/biology_results.jsonl"
    if [ -f "$result_file" ]; then
      n=$(wc -l < "$result_file")
      if [ "$n" -ge "50" ]; then
        echo "SKIP $label (already done: $n queries)"
        continue
      fi
    fi

    echo "RUN  $label: threshold=$thresh, max_k=$maxk"
    PYTHONPATH=src $PYTHON $SCRIPT $COMMON \
      --ag-threshold $thresh $maxk_flag \
      --output "${outdir}/results.jsonl" \
      2>&1 | tail -3
    echo ""
  done
done

echo "=== Grid Search Complete ==="
echo ""
echo "Analyzing results..."

# Summarize
$PYTHON -c "
import json, os, glob

print(f'{'Config':20s} {'nDCG@10':>10s} {'Recall@10':>10s} {'MRR':>10s} {'n_edges':>10s} {'Δβ₁':>10s}')
print('-' * 70)

for d in sorted(glob.glob('${RESULT_BASE}/*/results.jsonl/biology_results.jsonl')):
    label = d.split('/')[-3]
    ndcgs, recalls, mrrs, edges, db1s = [], [], [], [], []
    with open(d) as f:
        for line in f:
            r = json.loads(line)
            ndcgs.append(r.get('ndcg_10', 0))
            recalls.append(r.get('recall_10', 0))
            mrrs.append(r.get('mrr', 0))
            edges.append(r.get('entity_feval_n_bridge', 0))
            db1s.append(r.get('entity_feval_beta1_global', 0))
    n = len(ndcgs)
    if n == 0: continue
    avg = lambda x: sum(x)/len(x)
    print(f'{label:20s} {avg(ndcgs):10.4f} {avg(recalls):10.4f} {avg(mrrs):10.4f} {avg(edges):10.1f} {avg(db1s):10.1f} (n={n})')
"
