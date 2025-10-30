#!/usr/bin/env bash
set -euo pipefail

# Build geDIG paper PDF with local writable caches (robust on macOS/Linux)
# Usage: bash scripts/build_paper.sh

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
paper_dir="$root/docs/paper"

mkdir -p "$root/results/tex-cache" "$root/results/texmf-var" \
         "$root/results/texmf-cache" "$root/results/luaotfload-cache" \
         "$root/results/mpl" "$root/results/tex-home" "$root/results/texmf-home"

cd "$paper_dir"

export HOME="$root/results/tex-home"
export TEXMFHOME="$root/results/texmf-home"
export XDG_CACHE_HOME="$root/results/tex-cache"
export TEXMFVAR="$root/results/texmf-var"
export TEXMFCACHE="$root/results/texmf-cache"
export LUAOTFLOAD_DIR="$root/results/luaotfload-cache"
export LUAOTFLOAD_CACHE="$root/results/luaotfload-cache"
export MPLCONFIGDIR="$root/results/mpl"
export TEXINPUTS=.://

echo "[build] Cleaning aux..."
latexmk -C geDIG_paper_restructured_draft.tex >/dev/null 2>&1 || true

echo "[build] Compiling with lualatex..."
if latexmk -g -f -lualatex -interaction=nonstopmode -halt-on-error geDIG_paper_restructured_draft.tex; then
  ls -lh geDIG_paper_restructured_draft.pdf || true
  command -v md5 >/dev/null 2>&1 && md5 geDIG_paper_restructured_draft.pdf || true
  command -v md5sum >/dev/null 2>&1 && md5sum geDIG_paper_restructured_draft.pdf || true
  echo "[build] Done."
else
  echo "[build] Failed. Showing last 50 lines of log:" >&2
  tail -n 50 geDIG_paper_restructured_draft.log || true
  exit 1
fi

