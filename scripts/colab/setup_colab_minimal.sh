#!/usr/bin/env bash
# InsightSpike-AI Minimal Setup for Google Colab - Essential Dependencies Only
# Ultra-fast setup for core functionality without PyTorch Geometric

set -e

echo "⚡ InsightSpike-AI Minimal Setup for Google Colab"
echo "🎯 Essential dependencies only - 60 second setup"
echo "⚠️ PyTorch Geometric skipped for speed"

# Timer for setup
start_time=$(date +%s)

# Essential Python updates
echo ""
echo "🐍 Updating Python environment..."
pip install -q --upgrade pip setuptools wheel
pip install -q "numpy<2.0"

# Core ML framework (fast prebuilt)
echo ""
echo "🔥 Installing PyTorch (CUDA 12.1)..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vector search (critical for performance)
echo ""
echo "🔍 Installing FAISS with GPU support..."
pip install -q faiss-gpu-cu12 || {
    echo "🔄 Fallback to CPU FAISS..."
    pip install -q faiss-cpu
}

# Hugging Face ecosystem (essential for datasets)
echo ""
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers datasets tokenizers sentence-transformers

# Core scientific libraries
echo ""
echo "📊 Installing scientific libraries..."
pip install -q pandas matplotlib scikit-learn networkx

# InsightSpike dependencies
echo ""
echo "🎯 Installing InsightSpike essentials..."
pip install -q typer rich click pyyaml

# Install project
echo ""
echo "🚀 Installing InsightSpike-AI..."
pip install -q -e .

# Essential directories
mkdir -p experiment_results logs data/processed data/raw

# Quick validation
echo ""
echo "✅ Minimal validation..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import faiss
print(f'✅ FAISS {faiss.__version__}')

import transformers, datasets
print(f'✅ HuggingFace ready')

try:
    from insightspike.core.config import get_config
    print('✅ InsightSpike-AI ready')
except:
    print('⚠️ Use PYTHONPATH=src for InsightSpike commands')
"

# Calculate setup time
end_time=$(date +%s)
setup_time=$((end_time - start_time))

echo ""
echo "⚡ Minimal setup complete in ${setup_time}s!"
echo ""
echo "🚀 Ready for:"
echo "   • Basic insight detection"
echo "   • Vector search operations"
echo "   • Hugging Face dataset processing"
echo "   • CLI commands (with PYTHONPATH=src)"
echo ""
echo "❌ Not available:"
echo "   • Graph neural networks (PyTorch Geometric)"
echo "   • Advanced graph analysis"
echo ""
echo "📝 Quick test:"
echo "   PYTHONPATH=src python -c 'from insightspike.core.config import get_config; print(\"Ready!\")'"
