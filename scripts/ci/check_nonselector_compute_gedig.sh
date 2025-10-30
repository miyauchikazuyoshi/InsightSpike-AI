#!/usr/bin/env bash
set -euo pipefail

# Simple CI check: ensure geDIG computations go through the selector entrypoint only.
# Enable hard-fail with STRICT_GEDIG_SELECTOR=1

ROOT_DIR="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT_DIR" || exit 1

STRICT=${STRICT_GEDIG_SELECTOR:-0}

have_rg=0
if command -v rg >/dev/null 2>&1; then
  have_rg=1
fi

echo "[check_nonselector_compute_gedig] Scanning for non-selector geDIG usage..."

# Collect violations
violations=()

# 1) Direct import of GeDIGCore outside its module is not allowed
if [[ $have_rg -eq 1 ]]; then
  mapfile -t hits1 < <(rg -n "from\s+insightspike\.algorithms\.gedig_core\s+import" \
    --glob '!tests/**' --glob '!src/insightspike/algorithms/gedig_core.py' || true)
else
  mapfile -t hits1 < <(grep -RIn "from\s\+insightspike\.algorithms\.gedig_core\s\+import" . \
    | grep -v "tests/" | grep -v "src/insightspike/algorithms/gedig_core.py" || true)
fi
if [[ ${#hits1[@]} -gt 0 ]]; then
  violations+=("Direct import of GeDIGCore: ${hits1[*]}")
fi

# 2) Direct references to GeDIGCore( outside allowed files
if [[ $have_rg -eq 1 ]]; then
  mapfile -t hits2 < <(rg -n "\bGeDIGCore\s*\(" \
    --glob '!tests/**' --glob '!src/insightspike/algorithms/gedig_core.py' --glob '!src/insightspike/algorithms/gedig/selector.py' || true)
else
  mapfile -t hits2 < <(grep -RIn "GeDIGCore\s*\(" . \
    | grep -v "tests/" | grep -v "src/insightspike/algorithms/gedig_core.py" | grep -v "src/insightspike/algorithms/gedig/selector.py" || true)
fi
if [[ ${#hits2[@]} -gt 0 ]]; then
  violations+=("Direct construction/use of GeDIGCore: ${hits2[*]}")
fi

# 3) Direct import of gedig_pure (bypass selector)
if [[ $have_rg -eq 1 ]]; then
  mapfile -t hits3 < <(rg -n "from\s+insightspike\.algorithms\.gedig_pure\s+import" \
    --glob '!tests/**' --glob '!src/insightspike/algorithms/gedig/selector.py' || true)
else
  mapfile -t hits3 < <(grep -RIn "from\s\+insightspike\.algorithms\.gedig_pure\s\+import" . \
    | grep -v "tests/" | grep -v "src/insightspike/algorithms/gedig/selector.py" || true)
fi
if [[ ${#hits3[@]} -gt 0 ]]; then
  violations+=("Direct import of gedig_pure: ${hits3[*]}")
fi

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "Found potential non-selector geDIG usage:" >&2
  for v in "${violations[@]}"; do
    echo " - $v" >&2
  done
  if [[ "$STRICT" == "1" ]]; then
    echo "Failing due to STRICT_GEDIG_SELECTOR=1" >&2
    exit 2
  else
    echo "WARN: Set STRICT_GEDIG_SELECTOR=1 to fail CI on these." >&2
  fi
else
  echo "OK: No non-selector geDIG usage found."
fi

