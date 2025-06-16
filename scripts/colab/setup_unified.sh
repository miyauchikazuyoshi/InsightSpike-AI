#!/bin/bash
# InsightSpike-AI Unified Colab Setup Script
# ==========================================
# Single script for seamless Colab/Local/CI compatibility

echo "🚀 InsightSpike-AI Unified Setup Starting..."
echo "============================================"

# Check if running in Colab
if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_TPU_ADDR" ]; then
    echo "📱 Google Colab environment detected"
    IN_COLAB=true
else
    echo "💻 Local environment detected"
    IN_COLAB=false
fi

# Install dependencies using pyproject.toml
echo "📦 Installing dependencies from pyproject.toml..."
pip install -e .

# Additional Colab-specific optimizations
if [ "$IN_COLAB" = true ]; then
    echo "⚡ Applying Colab optimizations..."
    
    # Enable GPU if available
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected - enabling CUDA acceleration"
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    # Colab-specific directory setup
    mkdir -p /content/data
    mkdir -p /content/models
    mkdir -p /content/outputs
    
    echo "📁 Colab directories created"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import numpy as np
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ NumPy: {np.__version__}')

try:
    import torch_geometric
    print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
except ImportError:
    print('⚠️  PyTorch Geometric: Not available (optional)')

try:
    from insightspike.core.agents.main_agent import MainAgent
    print('✅ InsightSpike-AI: Core modules loaded successfully')
except ImportError as e:
    print(f'❌ InsightSpike-AI: Import failed - {e}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
if [ "$IN_COLAB" = true ]; then
    echo "💡 Ready to use InsightSpike-AI in Google Colab with GPU acceleration!"
else
    echo "💡 Ready to use InsightSpike-AI in local environment!"
fi
echo ""
echo "Quick start:"
echo "  from insightspike.core.agents.main_agent import MainAgent"
echo "  agent = MainAgent()  # Auto-optimized for your environment!"
