#!/usr/bin/env bash
set -euo pipefail

# Paper-level 25x25 run (Evaluator + L3) with exports and HTML.
# Defaults: SEEDS=32, STEPS=500 (override via env).

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT_DIR"

PY="python"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
fi

SEEDS="${SEEDS:-32}"
STEPS="${STEPS:-500}"
SIZE=25
OUT_ROOT="experiments/maze-query-hub-prototype/results/paper_grid"
NS="paper_v4_${SIZE}x${SIZE}_s${STEPS}"
SQLITE="${OUT_ROOT}/mq_${SIZE}x${SIZE}.sqlite"
MAZE_JSON="docs/paper/data/maze_${SIZE}x${SIZE}.json"

export PYTHONPATH="${ROOT_DIR}/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1
export INSIGHTSPIKE_PRESET=paper
export INSIGHTSPIKE_LOG_DIR="results/logs"
export MPLCONFIGDIR="results/mpl"

mkdir -p "$OUT_ROOT" "$(dirname "$MAZE_JSON")"

echo "[info] Running paper comparison for ${SIZE}x${SIZE}, steps=${STEPS}, seeds=${SEEDS}"
"$PY" experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
  --maze-size "$SIZE" \
  --max-steps "$STEPS" \
  --seeds "$SEEDS" \
  --out-root "$OUT_ROOT" \
  --namespace "$NS" \
  --sqlite "$SQLITE" \
  --maze-snapshot-out "$MAZE_JSON"

L3_SUM="${OUT_ROOT}/_${SIZE}x${SIZE}_s${STEPS}_l3_summary.json"
L3_STEPS="${OUT_ROOT}/_${SIZE}x${SIZE}_s${STEPS}_l3_steps.json"
HTML_OUT="${OUT_ROOT}/_${SIZE}x${SIZE}_s${STEPS}_l3_interactive.html"

echo "[info] Building interactive HTML: ${HTML_OUT}"
"$PY" experiments/maze-query-hub-prototype/build_reports.py \
  --summary "$L3_SUM" \
  --steps   "$L3_STEPS" \
  --sqlite  "$SQLITE" \
  --present-mode strict --strict --light-steps \
  --out     "$HTML_OUT"

echo "\n[done] 25x25 paper bundle created"
echo "  Eval summary : ${OUT_ROOT}/_${SIZE}x${SIZE}_s${STEPS}_eval_summary.json"
echo "  L3 summary   : ${OUT_ROOT}/_${SIZE}x${SIZE}_s${STEPS}_l3_summary.json"
echo "  SQLite DS    : ${SQLITE} (namespace: ${NS}_eval / ${NS}_l3)"
echo "  HTML (L3)    : ${HTML_OUT}"
