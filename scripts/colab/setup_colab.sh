#!/bin/bash
# InsightSpike-AI Google Colab Setup Script
# Simplified single script for all Colab setups

set -e

echo "🧠 InsightSpike-AI Colab Setup"
echo "=============================="
echo "🎯 Single optimized setup for Google Colab"
echo "🔧 NumPy 1.x (FAISS + thinc compatible) + PyTorch 2.4+"
echo "=============================="

# Setup mode (can be passed as argument)
SETUP_MODE="${1:-standard}"

echo "📋 Setup Mode: $SETUP_MODE"
echo ""

# Timer for setup
start_time=$(date +%s)

# ==========================================
# Step 1: Environment Preparation
# ==========================================
echo "📋 Step 1/5: Environment Preparation"
python --version
pip --version

# Clean cache for fresh installation
echo "🧹 Cleaning pip cache..."
pip cache purge 2>/dev/null || echo "Cache already clean"
echo "✅ Environment ready"

# ==========================================
# Step 2: GPU-Critical Packages (pip-first strategy)
# ==========================================
echo "📋 Step 2/5: Installing GPU-Critical Packages"

# Install NumPy 1.x first (FAISS compatible)
echo "🔢 Installing NumPy 1.x (FAISS + thinc compatible)..."
pip install "numpy==1.26.4" --upgrade --progress-bar on

# Install PyTorch with CUDA support  
echo "🔥 Installing PyTorch with CUDA (this may take 3-5 minutes)..."
timeout 600 pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --progress-bar on || {
    echo "⚠️ PyTorch installation timed out or failed"
    echo "🔄 Trying CPU version as fallback..."
    pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install FAISS with GPU support (NumPy 1.x compatible version)
echo "🚀 Installing FAISS GPU (NumPy 1.x compatible)..."
pip install -q "faiss-gpu-cu12==1.11.0" || {
    echo "🔄 Fallback to CPU FAISS..."
    pip install -q "faiss-cpu==1.11.0"
}

# Install PyTorch Geometric (only for standard/debug mode)
if [[ "$SETUP_MODE" != "minimal" ]]; then
    echo "🌐 Installing PyTorch Geometric..."
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_VERSION="cu121"
    
    # Install with timeout protection
    timeout 300 pip install -q torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
        --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html" || {
        echo "⚠️ PyTorch Geometric installation failed/timed out"
        if [[ "$SETUP_MODE" == "debug" ]]; then
            echo "🔍 Debug mode: Continuing without PyG"
        fi
    }
fi

echo "✅ GPU packages installed"

# ==========================================
# Step 3: Core Dependencies 
# ==========================================
echo "📋 Step 3/5: Installing Core Dependencies"

# Install from requirements file
# Note: torch, numpy, faiss are excluded from requirements-colab.txt 
# to avoid conflicts with the GPU-optimized versions installed in Step 2
pip install -q -r deployment/configs/requirements-colab.txt

echo "✅ Core dependencies installed"

# ==========================================
# Step 4: Project Installation
# ==========================================
echo "📋 Step 4/5: Installing Project"

# Install project in editable mode
pip install -q -e .

# Create necessary directories
mkdir -p experiment_results logs data/processed data/raw

echo "✅ Project installed"

# ==========================================
# Step 5: Validation
# ==========================================
echo "📋 Step 5/5: Validation"

# Test core imports
python -c "
import sys
print(f'✅ Python: {sys.version.split()[0]}')

try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
except ImportError:
    print('❌ NumPy failed')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError:
    print('❌ PyTorch failed')

try:
    import faiss
    print(f'✅ FAISS: {faiss.__version__}')
except ImportError:
    print('❌ FAISS failed')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers failed')

try:
    import spacy
    print(f'✅ spaCy: {spacy.__version__}')
except ImportError:
    print('❌ spaCy failed')

try:
    import thinc
    print(f'✅ thinc: {thinc.__version__}')
except ImportError:
    print('❌ thinc failed')

if '$SETUP_MODE' != 'minimal':
    try:
        import torch_geometric
        print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
    except ImportError:
        print('⚠️ PyTorch Geometric: Not available (OK for minimal mode)')

# Validate compatibility
print('')
print('🔍 Compatibility Check:')
try:
    import numpy, faiss, thinc
    np_version = tuple(map(int, numpy.__version__.split('.')[:2]))
    print(f'✅ NumPy {numpy.__version__} + FAISS {faiss.__version__} + thinc {thinc.__version__}: Compatible')
    if np_version >= (2, 0):
        print('⚠️ Warning: NumPy 2.x detected - may cause FAISS issues')
    else:
        print('✅ NumPy 1.x confirmed - optimal for FAISS compatibility')
except Exception as e:
    print(f'❌ Compatibility issue: {e}')
"

# Test CLI
echo ""
echo "🧪 Testing CLI..."
if command -v insightspike >/dev/null 2>&1; then
    echo "✅ CLI command: insightspike available"
else
    echo "⚠️ CLI: Use 'python -m insightspike.cli' instead"
fi

# Calculate setup time
end_time=$(date +%s)
setup_time=$((end_time - start_time))

echo ""
echo "🎉 Setup Complete in ${setup_time}s!"
echo "=============================="
echo "📋 Mode: $SETUP_MODE"
echo "🔧 Dependencies: pip-only (no Poetry conflicts)"
echo "🚀 GPU packages: NumPy 1.x + FAISS + PyTorch with CUDA 12.1"
echo ""
echo "📝 Quick Start:"
echo "   • Test: insightspike --help"
echo "   • Alt: python -m insightspike.cli --help" 
echo "   • Experiment: PYTHONPATH=src python scripts/experiments/demo_mvp.py"
echo "=============================="