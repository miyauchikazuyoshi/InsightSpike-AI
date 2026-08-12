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
# Cloud-safe logging/cache locations
export INSIGHTSPIKE_LOG_DIR=${INSIGHTSPIKE_LOG_DIR:-results/logs}
export MPLCONFIGDIR=${MPLCONFIGDIR:-results/mpl}

echo "[codex-smoke] Running minimal test subset..." >&2

# 1) Ultra-light AB logger self-test (no heavy deps)
# NOTE: no "|| true" here — a failing selftest must fail the smoke run.
if [ -f scripts/selftest_ab_logger.py ]; then
  echo "[codex-smoke] AB logger selftest" >&2
  INSIGHTSPIKE_MIN_IMPORT=1 python scripts/selftest_ab_logger.py
fi

# 2) Focused pytest subset (explicit files to avoid importing heavy modules)
# "python -m pytest" (not bare "pytest") so the test runner is guaranteed to
# share the interpreter that has this project's dependencies installed.
echo "[codex-smoke] pytest minimal subset (explicit files)" >&2
python -m pytest -q \
  tests/test_minimal_healthcheck.py \
  tests/unit/test_core_metrics.py \
  tests/unit/test_gedig_ab_logger.py \
  tests/unit/test_gedig_ab_logger_alerts_csv.py \
  tests/unit/test_message_passing_aggregation_contract.py \
  tests/unit/test_quick_start_overrides.py

echo "[codex-smoke] OK" >&2
