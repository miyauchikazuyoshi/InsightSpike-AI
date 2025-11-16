#!/usr/bin/env bash
set -euo pipefail

# Package JA/EN LaTeX sources and figures for arXiv submission.
# Usage: scripts/pack_arxiv.sh [en|ja|both]

mode=${1:-both}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PAPER_DIR="$ROOT/docs/paper"
OUT_DIR="$PAPER_DIR/arxiv_v4_en"

mkdir -p "$OUT_DIR"

package_one() {
  local lang=$1
  local main
  if [[ "$lang" == "en" ]]; then
    main=geDIG_onegauge_improved_v4_en.tex
  else
    main=geDIG_onegauge_improved_v4.tex
  fi

  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local zip="$PAPER_DIR/arxiv_${lang}_$stamp.zip"

  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT

  rsync -a --include '*/' \
    --include "$main" \
    --include 'sections/***' \
    --include 'figures/***' \
    --include 'references.bib' \
    --exclude '*' "$PAPER_DIR/" "$tmpdir/"

  (cd "$tmpdir" && zip -r "$zip" .)
  mv "$zip" "$PAPER_DIR/"
  echo "[arxiv] Created: $zip"
}

case "$mode" in
  en) package_one en ;;
  ja) package_one ja ;;
  both) package_one en; package_one ja ;;
  *) echo "Usage: $0 [en|ja|both]" ; exit 1 ;;
esac

