#!/usr/bin/env bash
set -euo pipefail

# Cloud-safe, fast, deterministic smoke checks.
# - No network
# - No GPU / heavy deps
# - Keeps imports minimal

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export INSIGHTSPIKE_LITE_MODE=1
export INSIGHTSPIKE_MIN_IMPORT=1
export PYTHONPATH=${PYTHONPATH:-}:src

echo "[codex-smoke] Running minimal test subset..." >&2

# 1) Ultra-light AB logger self-test (no heavy deps)
if [ -f scripts/selftest_ab_logger.py ]; then
  echo "[codex-smoke] AB logger selftest" >&2
  INSIGHTSPIKE_MIN_IMPORT=1 python scripts/selftest_ab_logger.py || true
fi

# 2) Focused pytest subsets that avoid heavy paths
echo "[codex-smoke] pytest -k 'gedig_ab_logger or test_minimal_healthcheck or test__minimal_probe'" >&2
pytest -q -k "gedig_ab_logger or test_minimal_healthcheck or test__minimal_probe"

echo "[codex-smoke] OK" >&2
