#!/usr/bin/env bash
set -euo pipefail

# Paper v4 — 51x51 / 250 steps / 40 seeds (Evaluator vs L3)
# - Ultra-light steps + post-diagnostics OFF for speed
# - DS persistence enabled for HTML reconstruction

ROOT="experiments/maze-query-hub-prototype/results/paper_grid"
NS="paper_v4_51x51_s250"
SQLITE="$ROOT/mq_51x51.sqlite"
SUMMARY_EVAL="$ROOT/_51x51_s250_eval_summary.json"
STEPS_EVAL="$ROOT/_51x51_s250_eval_steps.json"
SUMMARY_L3="$ROOT/_51x51_s250_l3_summary.json"
STEPS_L3="$ROOT/_51x51_s250_l3_steps.json"

mkdir -p "$ROOT" docs/paper/data results/logs results/mpl

export PYTHONPATH=src
export INSIGHTSPIKE_PRESET=paper
export INSIGHTSPIKE_LOG_DIR=${INSIGHTSPIKE_LOG_DIR:-results/logs}
export MPLCONFIGDIR=${MPLCONFIGDIR:-results/mpl}

echo "[run] 51x51 s=250 seeds=40 (ultra-light)"
python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size 51 --max-steps 250 --seeds 40 \
  --out-root "$ROOT" \
  --namespace "$NS" \
  --ultra-light \
  --maze-snapshot-out docs/paper/data/maze_51x51.json

echo "[export] Eval/L3 metrics and A/B diffs are written under docs/paper/data/"
echo "  Eval summary:   $SUMMARY_EVAL"
echo "  Eval steps:     $STEPS_EVAL"
echo "  L3 summary:     $SUMMARY_L3"
echo "  L3 steps:       $STEPS_L3"
echo "  DS (SQLite):    $SQLITE"

echo "[hint] Build HTML (strict DS):"
echo "  python experiments/maze-query-hub-prototype/build_reports.py \\
    --summary $SUMMARY_L3 --steps $STEPS_L3 \\
    --sqlite $SQLITE --namespace ${NS}_l3 --light-steps --present-mode strict \\
    --out $ROOT/_51x51_s250_l3_interactive.html"

