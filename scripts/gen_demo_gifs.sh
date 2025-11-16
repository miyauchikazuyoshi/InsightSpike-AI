#!/usr/bin/env bash
set -euo pipefail

# Generate demo PNGs and optional GIFs for gating timeseries and PSZ curves.
# Requires: experiments/exp2to4_lite/src/viz.py utilities and (optional) ImageMagick `convert`.

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR="$ROOT/docs/figures/demo"
mkdir -p "$OUT_DIR"

RESULT_JSON=${1:-$ROOT/experiments/exp2to4_lite/results/$(ls -1t $ROOT/experiments/exp2to4_lite/results/exp23_paper_*.json 2>/dev/null | head -n1 | xargs -I{} basename {})}
RESULT_PATH="$ROOT/experiments/exp2to4_lite/results/${RESULT_JSON##*/}"

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "[demo] Result JSON not found: $RESULT_PATH" >&2
  echo "Usage: $0 <path/to/exp23_paper_*.json>" >&2
  exit 1
fi

echo "[demo] Using result: $RESULT_PATH"

python - <<PY
from experiments.exp2to4_lite.src.viz import plot_psz_curves, plot_gating_timeseries
from pathlib import Path
res = Path(r"$RESULT_PATH")
out = Path(r"$OUT_DIR")
out.mkdir(parents=True, exist_ok=True)
plot_psz_curves(res, out)
plot_gating_timeseries(res, out)
print("[demo] PNGs written to", out)
PY

if command -v convert >/dev/null 2>&1; then
  echo "[demo] Converting PNGs to 60fps GIFs (if multi-frame)"
  for f in "$OUT_DIR"/*.png; do
    base="${f%.png}"
    convert -delay 3 -loop 0 "$f" "${base}.gif" || true
  done
else
  echo "[demo] 'convert' not found; PNGs generated only."
fi

echo "[demo] Done: $OUT_DIR"

