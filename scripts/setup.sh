#!/usr/bin/env bash
set -euo pipefail

echo "1) Install PyTorch & friends via pip……"
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS 版
  pip install \
    torch==2.6.1 \
    torch-geometric==2.6.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu
elif [[ "$(uname)" == "Linux" ]]; then
  # Linux（CPU）版
  pip install \
    torch==2.6.1+cpu \
    torch-geometric==2.6.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu
else
  echo "Unsupported OS: $(uname)"
  exit 1
fi

echo "2) Poetry-managed deps (pure-Python)……"
poetry install --no-root

echo "3) Pin NumPy to <2.0……"
poetry run pip uninstall -y numpy
poetry run pip install "numpy<2.0"

echo "✅ Done."
