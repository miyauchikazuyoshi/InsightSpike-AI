#!/usr/bin/env bash
set -euo pipefail

# Quick notation audit for ΔSP_rel and \ignorm usage in LaTeX/Markdown.
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "[audit] Checking for ΔSP without _rel suffix (LaTeX/Markdown)"
rg -n "\\Delta\\s*SP(?![_\\}]|\\s*\\_rel)" docs | sed 's/^/[ΔSP]/'

echo "[audit] Checking for ignorm usage around F definition"
rg -n "\\ignorm|\\gednorm|ΔEPC|ΔIG|SP_rel" docs | sed 's/^/[F]/' | head -n 200

echo "[audit] Done. Review lines above for inconsistent notation."

