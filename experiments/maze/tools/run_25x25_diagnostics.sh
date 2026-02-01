#!/usr/bin/env bash
set -euo pipefail

# Quick 25x25 short-run with per-step diagnostics and HTML (no DS by default).

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT_DIR"

PY="python"; [[ -x ".venv/bin/python" ]] && PY=".venv/bin/python" || true

OUT="experiments/maze-query-hub-prototype/results/25x25_diag"
mkdir -p "$OUT" results/mpl

export PYTHONPATH="${ROOT_DIR}/src"
export INSIGHTSPIKE_MIN_IMPORT=1
export INSIGHTSPIKE_LITE_MODE=1
export MPLCONFIGDIR="results/mpl"

echo "[run] 25x25 short diagnostics (180 steps)"
"$PY" experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 180 \
  --linkset-mode --norm-base link \
  --theta-ag -0.40 --eval-per-hop-on-ag --skip-mh-on-deadend \
  --sp-cache --sp-cache-mode cached_incr --sp-pair-samples 200 \
  --layer1-prefilter --l1-cap 16 --log-minimal \
  --steps-ultra-light --no-post-sp-diagnostics \
  --use-main-l3 \
  --output  "$OUT/summary.json" \
  --step-log "$OUT/steps.json"

echo "[diag] exporting per-step diagnostics"
"$PY" experiments/maze-query-hub-prototype/tools/export_step_diagnostics.py \
  --steps "$OUT/steps.json" \
  --out-csv "$OUT/diag.csv" \
  --plot "$OUT/diag.png" --logy

echo "[html] building quick HTML (no DS)"
"$PY" experiments/maze-query-hub-prototype/build_reports.py \
  --summary "$OUT/summary.json" \
  --steps   "$OUT/steps.json" \
  --out     "$OUT/interactive.html" \
  --present-mode none --relaxed --light-steps

echo "[done] see: $OUT/interactive.html, $OUT/diag.csv, $OUT/diag.png"
