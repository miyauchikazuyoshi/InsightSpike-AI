#!/usr/bin/env bash
set -euo pipefail

# Simple CI check: ensure example scripts import from the public API surface.
# Enable hard-fail with INSIGHTSPIKE_STRICT_PUBLIC_IMPORTS=1

ROOT_DIR="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT_DIR" || exit 1

STRICT=${INSIGHTSPIKE_STRICT_PUBLIC_IMPORTS:-0}

have_rg=0
if command -v rg >/dev/null 2>&1; then
  have_rg=1
fi

echo "[check_public_api_imports] Scanning examples for non-public imports..."

targets=(examples)

violations=()
for t in "${targets[@]}"; do
  if [[ ! -d "$t" ]]; then
    continue
  fi
  if [[ $have_rg -eq 1 ]]; then
    mapfile -t hits < <(rg -n "from\s+insightspike\s+import\s+" "$t" --glob "**/*.py" || true)
  else
    mapfile -t hits < <(grep -RIn "from\s\+insightspike\s\+import\s\+" "$t" --include "*.py" || true)
  fi
  # Filter out the allowed public path
  filtered=()
  for h in "${hits[@]:-}"; do
    # Ignore matches that already import from insightspike.public
    if echo "$h" | grep -q "from insightspike\.public import"; then
      continue
    fi
    filtered+=("$h")
  done
  if [[ ${#filtered[@]} -gt 0 ]]; then
    violations+=("$t: ${filtered[*]}")
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "Found example imports bypassing the public API surface:" >&2
  for v in "${violations[@]}"; do
    echo " - $v" >&2
  done
  if [[ "$STRICT" == "1" ]]; then
    echo "Failing due to INSIGHTSPIKE_STRICT_PUBLIC_IMPORTS=1" >&2
    exit 2
  else
    echo "WARN: Set INSIGHTSPIKE_STRICT_PUBLIC_IMPORTS=1 to fail CI on these." >&2
  fi
else
  echo "OK: Example imports use insightspike.public"
fi

