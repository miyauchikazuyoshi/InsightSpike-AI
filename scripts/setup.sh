#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) Ensure poetry is installed and venv is active ==="
echo "    (If not: pip install poetry)"
echo

echo "=== 2) Install poetry dependencies and create venv ==="
poetry config virtualenvs.in-project true 
poetry lock --no-cache --regenerate
poetry install 

echo "=== 3) Install pip dependencies inside poetry venv ==="
poetry run pip install --upgrade pip
poetry run pip install -r requirements-torch.txt
poetry run pip install -r requirements-PyG.txt
