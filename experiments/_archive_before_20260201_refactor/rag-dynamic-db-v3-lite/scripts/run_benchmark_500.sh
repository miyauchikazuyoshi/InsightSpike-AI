#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts"
DATA_DIR="${ROOT_DIR}/data"
CONFIG="${ROOT_DIR}/configs/experiment_geDIG_vs_baselines.yaml"
DATASET="${DATA_DIR}/sample_queries_500.jsonl"

cd "${ROOT_DIR}"

if [[ ! -f "${DATASET}" ]]; then
    echo "[info] generating 500-sample dataset at ${DATASET}"
    python "${SCRIPT_DIR}/generate_dataset.py" --num-queries 500 --output "${DATASET}"
fi

python "${SCRIPT_DIR}/run_benchmark_suite.py" \
    --config "${CONFIG}" \
    --datasets "${DATASET}"
