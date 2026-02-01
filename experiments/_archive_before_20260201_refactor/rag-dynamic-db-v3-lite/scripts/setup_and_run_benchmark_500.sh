#!/usr/bin/env bash

# Convenience script:
# 1. Install required Python packages inside Poetry env
# 2. Generate 500-sample dataset (if needed)
# 3. Run geDIG vs. baselines benchmark

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

echo "[setup] Installing Python dependencies (numpy, matplotlib, sentence-transformers)"
poetry run pip install --quiet numpy matplotlib sentence-transformers

echo "[setup] Installing PyTorch (CPU build) for GeDIGCore"
poetry run pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "[run] Launching benchmark on 500-sample dataset"
poetry run bash "${ROOT_DIR}/scripts/run_benchmark_500.sh"

