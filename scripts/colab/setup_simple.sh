#!/bin/bash
# Simple Colab Setup for InsightSpike-AI
# Poetry-first approach with minimal dependencies

set -e

echo "🚀 InsightSpike-AI Simple Colab Setup"
echo "====================================="

# Install Poetry if not available
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone repository if needed (for Colab)
if [ ! -d "InsightSpike-AI" ]; then
    echo "📥 Cloning InsightSpike-AI repository..."
    git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
    cd InsightSpike-AI
else
    cd InsightSpike-AI
fi

# Configure Poetry for Colab environment
echo "⚙️ Configuring Poetry for Colab..."
poetry config virtualenvs.create false  # Use system Python in Colab
poetry config installer.max-workers 1   # Prevent memory issues

# Install core dependencies with Poetry
echo "📦 Installing core dependencies..."
poetry install --only main --no-dev

# Install ML dependencies with proven versions
echo "🧠 Installing ML dependencies (CPU-optimized for Colab)..."
poetry run pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install FAISS CPU (lightweight for Colab)
echo "🔍 Installing FAISS-CPU..."
poetry run pip install faiss-cpu==1.8.0

# Install torch-geometric
echo "🕸️ Installing torch-geometric..."
poetry run pip install torch-geometric==2.4.0

# Verify installation
echo "✅ Verifying installation..."
poetry run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'✗ PyTorch: {e}')

try:
    import faiss
    print(f'✓ FAISS: {faiss.__version__}')
except ImportError as e:
    print(f'✗ FAISS: {e}')

try:
    import torch_geometric
    print(f'✓ Torch-Geometric: {torch_geometric.__version__}')
except ImportError as e:
    print(f'✗ Torch-Geometric: {e}')

try:
    from insightspike.core.agent import InsightSpikeAgent
    print('✓ InsightSpike-AI: Ready!')
except ImportError as e:
    print(f'✗ InsightSpike-AI: {e}')
"

echo ""
echo "🎉 Colab setup complete!"
echo "Quick test: poetry run python -c 'from insightspike.core.agent import InsightSpikeAgent; print(\"Ready!\")'"
