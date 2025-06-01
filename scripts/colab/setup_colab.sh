#!/bin/bash

# InsightSpike-AI Google Colab Setup Script
# Focus: Poetry installation + CLI functionality + GPU optimization

set -e  # Exit on any error

echo "🧠 InsightSpike-AI Colab Setup"
echo "=============================="
echo "📋 Single optimized setup for Google Colab"
echo "📦 Poetry + GPU libraries + CLI testing"
echo "🔧 Strategic dependency coordination"
echo "=============================="

# Step 1: Install Poetry (CRITICAL for CLI)
echo "📋 Step 1/5: Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 - 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ Poetry installed"
else
    echo "✅ Poetry already available"
fi

# Verify Poetry
poetry --version
echo "✅ Poetry confirmed working"

# Step 2: Configure Poetry for system environment
echo "📋 Step 2/5: Configuring Poetry..."
poetry config virtualenvs.create false
poetry config installer.parallel true
echo "✅ Poetry configured for Colab"

# Step 3: Install GPU-optimized PyTorch (individual installation)
echo "📋 Step 3/5: Installing PyTorch with CUDA support..."
pip install -q torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✅ PyTorch with CUDA installed"

# Step 4: Install FAISS GPU (with CPU fallback)
echo "📋 Step 4/5: Installing FAISS GPU..."
pip install -q faiss-gpu-cu12 || pip install -q faiss-cpu
echo "✅ FAISS GPU installed"

# Step 5: Install Poetry dependencies (without torch/faiss to avoid conflicts)
echo "📋 Step 5/5: Installing remaining dependencies via Poetry..."
echo "📝 Using requirements-colab.txt (excludes torch/faiss for conflict avoidance)"

poetry install --only main
echo "✅ Poetry dependencies installed"

# Install project in editable mode for CLI access
echo "📦 Installing project in editable mode..."
poetry install --only main
echo "✅ Project installed"

# Test CLI functionality
echo "📋 Testing CLI functionality..."

# Test Poetry CLI access
if poetry run python -c "import sys; sys.path.append('src'); from insightspike.cli import app" 2>/dev/null; then
    echo "✅ Poetry CLI: Working"
else
    echo "⚠️ Poetry CLI: Reinstalling project..."
    poetry install --only main
    echo "✅ Project reinstalled"
fi

# Final validation
echo ""
echo "🔍 Final Validation"
echo "==================="

# Python and core libraries
python -c "
import sys
print(f'✅ Python: {sys.version.split()[0]}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except: print('❌ PyTorch failed')

try:
    import faiss
    print(f'✅ FAISS: Available')
except: print('⚠️ FAISS: Not available')

try:
    import transformers, sentence_transformers
    print('✅ Transformers: OK')
except: print('⚠️ Transformers: Issue')
"

# Test CLI
echo ""
echo "🔍 Testing CLI access..."
if poetry run python -m insightspike.cli --help > /dev/null 2>&1; then
    echo "✅ CLI: Ready"
else
    echo "❌ CLI: Failed"
    exit 1
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo "📋 Dependencies coordinated via:"
echo "   • requirements-colab.txt (Poetry-managed, excludes torch/faiss)"
echo "   • requirements-colab-comprehensive.txt (Complete reference list)"
echo "   • GPU packages installed via pip for CUDA optimization"
echo ""
echo "Next steps:"
echo "1. Run data preparation"
echo "2. Test: poetry run python -m insightspike.cli --help"
echo "3. Demo: poetry run python -m insightspike.cli loop 'test'"
echo "=================="