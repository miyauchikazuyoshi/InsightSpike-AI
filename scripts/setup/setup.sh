#!/usr/bin/env bash
set -euo pipefail

echo "=== InsightSpike-AI Local Development Setup ==="
echo "🎯 Optimized for local development with Poetry"
echo ""

echo "=== 1) Ensure poetry is installed and venv is active ==="
# Check if poetry is installed, if not install it
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    pip install poetry
else
    echo "✅ Poetry is already installed"
fi
echo

echo "=== 2) Install poetry dependencies and create venv ==="
poetry config virtualenvs.in-project true 
poetry lock --no-cache
poetry install --with dev  # localグループは空のため除外

echo "=== 3) Install environment-specific packages ==="
poetry run pip install --upgrade pip

# Note: NumPy is managed by pyproject.toml (>=1.24.0,<2.0.0)
# No need to install separately as Poetry handles it

# Install PyTorch CPU version for local development
echo "🔥 Installing PyTorch 2.2.2 (CPU version for local)..."
poetry run pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install FAISS CPU version for local development
echo "📦 Installing FAISS (CPU version)..."
poetry run pip install faiss-cpu

# Install PyTorch Geometric for local development (optional)
echo "🌐 Installing PyTorch Geometric (local CPU)..."
poetry run pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cpu.html || {
    echo "⚠️ PyTorch Geometric installation failed - continuing without PyG"
    echo "   (This is OK for basic functionality)"
}

echo ""
echo "✅ Local development environment setup complete!"
echo "🚀 Ready for development:"
echo "   • Run tests: poetry run pytest"
echo "   • Start CLI: poetry run insightspike --help"
echo "   • Activate env: poetry shell"
