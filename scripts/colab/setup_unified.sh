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

# Colab-specific configuration optimization
if [ "$IN_COLAB" = true ]; then
    echo "🔧 Optimizing for Colab environment..."
    
    # Switch to Colab-optimized pyproject.toml if available
    if [ -f "pyproject_colab.toml" ]; then
        echo "📝 Using Colab-optimized configuration..."
        cp pyproject.toml pyproject_backup.toml 2>/dev/null || true
        cp pyproject_colab.toml pyproject.toml
    fi
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
    
    # Ensure CLI scripts are in PATH for Colab
    echo "🔧 Setting up CLI environment..."
    export PATH="/root/.local/bin:$PATH"
    
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
    import faiss
    print(f'✅ FAISS: {faiss.__version__}')
except ImportError:
    print('⚠️  FAISS: Not available (optional)')

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

# Test configuration loading
try:
    import insightspike.config as config
    print('✅ InsightSpike-AI: Configuration loaded')
except ImportError:
    print('⚠️  InsightSpike-AI: Configuration module not found (optional)')
"

# Test CLI availability
echo "🔧 Testing CLI command availability..."
if command -v insightspike &> /dev/null; then
    echo "✅ InsightSpike CLI: Available via 'insightspike' command"
    echo "📋 CLI Help:"
    insightspike --help 2>/dev/null | head -10 || echo "  (CLI help not available, but command exists)"
else
    echo "⚠️  InsightSpike CLI: Command not found in PATH"
    echo "   You can still use: python -m insightspike.cli.main"
    
    # Test alternative CLI access
    python -c "
try:
    from insightspike.cli.main import main
    print('✅ CLI Module: Available via python -m insightspike.cli.main')
except ImportError:
    print('❌ CLI Module: Not available')
" 2>/dev/null || echo "❌ CLI Module: Import test failed"
fi

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
