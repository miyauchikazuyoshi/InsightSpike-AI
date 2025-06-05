# Agent Contribution Guidelines

> ⚠️ Codex等のオフライン環境では、ネットワークが使えるうちに全依存パッケージのwheelファイルをダウンロードし、  
> セットアップ時は `pip install --no-index --find-links=./wheels ...` でインストールしてください。

- **Testing**: Before committing changes, run `python -m pytest -q`.
  If `pytest` is missing or tests fail, mention this in the PR.

- **Setup**:
#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) Ensure poetry is installed and venv is active ==="
echo "    (If not: pip install poetry)"
echo

echo "=== 2) Install poetry dependencies and create venv ==="
poetry config virtualenvs.in-project true 
poetry lock --no-cache --regenerate
poetry install --with dev

echo "=== 3) Install pip dependencies inside poetry venv ==="
poetry run pip install --upgrade pip
poetry run pip install -r requirements-torch.txt
poetry run pip install -r requirements-PyG.txt

- **Committing**: Use clear commit messages describing the change.

- **PR Summary**: Include a short summary of changes and test results.
