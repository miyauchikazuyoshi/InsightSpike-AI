#!/usr/bin/env bash
set -euo pipefail

# Check LaTeX logs for undefined refs/citations and cross-ref issues.
# Usage: scripts/check_latex_logs.sh docs/paper/geDIG_onegauge_improved_v4{_en}.log

fail=0
for log in "$@"; do
  echo "[latex-check] Scanning $log"
  if rg -n 'LaTeX Warning: (There were undefined references|Citation `|Reference `|Rerun to get cross-references right\.)' "$log"; then
    echo "[latex-check] Issues found in $log" >&2
    fail=1
  fi

  # Overfull/Underfull boxes (warn-only)
  if rg -n 'Overfull \\hbox|Underfull \\hbox' "$log"; then
    echo "[latex-check] Note: Overfull/Underfull boxes reported above (warn-only)" >&2
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[latex-check] Failing due to LaTeX reference/citation issues" >&2
  exit 1
else
  echo "[latex-check] No critical LaTeX reference/citation issues detected"
fi
