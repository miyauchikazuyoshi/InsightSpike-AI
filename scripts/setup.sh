#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) 仮想環境の有効化を確認してください ==="
echo "    (.venvが有効でなければ python3.11 -m venv .venv && source .venv/bin/activate )"
echo "    (Poetry が未インストールなら: pip install poetry)"
echo

echo "=== 2) pip, setuptools, wheel, torch, PyG関連をインストール ==="
pip install --upgrade pip
pip install -r requirements-pip.txt

echo "=== 3) Poetry依存をインストール ==="
poetry lock --no-cache --regenerate
poetry install --no-root
