#!/usr/bin/env bash
set -euo pipefail

# Package v5 JA/EN LaTeX sources and figures for arXiv submission.
# Usage: scripts/pack_arxiv_v5.sh [en|ja|both]
#
# This script assumes:
#   - EN main: docs/paper/arxiv_v5_en/geDIG_onegauge_improved_v5_en.tex
#   - JA main: docs/paper/arxiv_v5_ja/geDIG_onegauge_improved_v5_short_ja.tex
#   - Figures: docs/paper/figures/*
#   - References: docs/paper/arxiv_v5_en/references.bib

mode=${1:-en}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PAPER_DIR="$ROOT/docs/paper"

package_v5_en() {
  local main="geDIG_onegauge_improved_v5_en.tex"
  local src_dir="$PAPER_DIR/arxiv_v5_en"

  if [[ ! -f "$src_dir/$main" ]]; then
    echo "[arxiv v5] EN main TeX not found at $src_dir/$main" >&2
    return 1
  fi

  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local zip="$PAPER_DIR/arxiv_v5_en_${stamp}.zip"

  local tmpdir
  tmpdir=$(mktemp -d)

  # Copy main TeX and references to root of archive
  cp "$src_dir/$main" "$tmpdir/"
  if [[ -f "$src_dir/references.bib" ]]; then
    cp "$src_dir/references.bib" "$tmpdir/"
  elif [[ -f "$PAPER_DIR/references.bib" ]]; then
    cp "$PAPER_DIR/references.bib" "$tmpdir/"
  fi

  # Copy figures (full set; TeX will pick what it needs)
  if [[ -d "$PAPER_DIR/figures" ]]; then
    mkdir -p "$tmpdir/figures"
    rsync -a "$PAPER_DIR/figures/" "$tmpdir/figures/"
  fi

  (cd "$tmpdir" && zip -r "$zip" .)
  mv "$zip" "$PAPER_DIR/"
  rm -rf "$tmpdir"
  echo "[arxiv v5] Created EN package: $zip"
}

package_v5_ja() {
  local main="geDIG_onegauge_improved_v5_short_ja.tex"
  local src_dir="$PAPER_DIR/arxiv_v5_ja"

  if [[ ! -f "$src_dir/$main" ]]; then
    echo "[arxiv v5] JA main TeX not found at $src_dir/$main" >&2
    return 1
  fi

  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local zip="$PAPER_DIR/arxiv_v5_ja_${stamp}.zip"

  local tmpdir
  tmpdir=$(mktemp -d)

  cp "$src_dir/$main" "$tmpdir/"
  # Japanese short version bib lives one level up
  if [[ -f "$PAPER_DIR/references.bib" ]]; then
    cp "$PAPER_DIR/references.bib" "$tmpdir/references.bib"
  fi

  if [[ -d "$PAPER_DIR/figures" ]]; then
    mkdir -p "$tmpdir/figures"
    rsync -a "$PAPER_DIR/figures/" "$tmpdir/figures/"
  fi

  (cd "$tmpdir" && zip -r "$zip" .)
  mv "$zip" "$PAPER_DIR/"
  rm -rf "$tmpdir"
  echo "[arxiv v5] Created JA package: $zip"
}

case "$mode" in
  en) package_v5_en ;;
  ja) package_v5_ja ;;
  both) package_v5_en; package_v5_ja ;;
  *) echo "Usage: $0 [en|ja|both]" ; exit 1 ;;
esac

