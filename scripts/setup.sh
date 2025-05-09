#!/usr/bin/env bash
set -euo pipefail

echo "1) Poetry-managed deps (pure-Python)……"
poetry install --no-root

echo "2) Pin NumPy to <2.0……"
poetry run pip uninstall -y numpy
poetry run pip install "numpy<2.0"

echo "3) Install PyTorch & friends……"
if [[ "$(uname)" == "Darwin" ]]; then
  poetry run pip install torch torch-geometric vector-quantize-pytorch
else
  poetry run pip install \
    torch==2.6.1+cpu \
    torch-geometric==2.6.1 \
    vector-quantize-pytorch==1.22.16 \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "✅ Done."
