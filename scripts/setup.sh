#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) 仮想環境の有効化を確認してください ==="
echo "    (.venvが有効でなければ python3.11 -m venv .venv && source .venv/bin/activate )"
echo "    (Poetry が未インストールなら: pip install poetry)"
echo

echo "=== 2) pip, setuptools, wheel を最新化 ==="
pip install --upgrade pip setuptools wheel

echo "=== 3) torch を先にインストール ==="
pip install torch==2.2.2

echo "=== 4) PyG関連を1つずつインストール（CPU版）==="
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
pip install torch-geometric==2.5.2

echo "=== 5) Poetry依存をインストール ==="
poetry lock --no-cache --regenerate
poetry install --no-root

echo "✅ 環境構築 完了"

