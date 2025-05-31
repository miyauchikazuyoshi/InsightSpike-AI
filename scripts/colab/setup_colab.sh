#!/bin/bash

# InsightSpike-AI Google Colab Setup Script
# Comprehensive setup with multiple speed options and error handling

set -e  # Exit on any error

echo "🧠 InsightSpike-AI Google Colab Setup Script"
echo "============================================="

# Parse command line arguments
SETUP_TYPE=${1:-"standard"}  # standard, fast, minimal, debug

echo "📋 Setup Type: $SETUP_TYPE"
echo "⏱️ Estimated Time:"
case $SETUP_TYPE in
    "fast")
        echo "   ⚡ Fast Setup: 3-5 minutes (with timeout protection)"
        ;;
    "minimal") 
        echo "   🚀 Minimal Setup: Under 1 minute (basic features only)"
        ;;
    "debug")
        echo "   🔍 Debug Setup: 15-20 minutes (with detailed logging)"
        ;;
    *)
        echo "   📋 Standard Setup: 8-12 minutes (full installation)"
        ;;
esac
echo "============================================="

# Function to install with timeout and error handling
install_with_timeout() {
    local package="$1"
    local timeout="${2:-300}"  # Default 5 minutes
    local description="$3"
    
    echo "📦 Installing $description..."
    
    timeout $timeout pip install -q "$package" || {
        echo "⚠️ $description installation failed or timed out"
        return 1
    }
    
    echo "✅ $description installed successfully"
    return 0
}

# Function to check GPU availability
check_gpu() {
    echo "📋 Checking GPU availability..."
    if python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"; then
        echo "✅ GPU check completed"
    else
        echo "⚠️ GPU check failed - continuing with CPU mode"
    fi
}

# Core dependencies (always installed)
echo "📋 Step 1/5: Installing core dependencies..."
install_with_timeout "torch torchvision" 180 "PyTorch"
install_with_timeout "numpy pandas" 60 "NumPy and Pandas"
echo "✅ Core dependencies completed"

# Visualization dependencies
echo "📋 Step 2/5: Installing visualization libraries..."
install_with_timeout "matplotlib seaborn plotly" 120 "Visualization libraries"
echo "✅ Visualization libraries completed"

# FAISS installation with GPU/CPU fallback
echo "📋 Step 3/5: Installing FAISS..."
if [ "$SETUP_TYPE" = "minimal" ]; then
    install_with_timeout "faiss-cpu" 60 "FAISS-CPU (minimal mode)"
elif [ "$SETUP_TYPE" = "debug" ]; then
    echo "🔍 Debug mode: Attempting FAISS-GPU with detailed logging..."
    if timeout 300 pip install -v faiss-gpu-cu12; then
        echo "✅ FAISS-GPU installed successfully"
    else
        echo "⚠️ FAISS-GPU failed, installing CPU version..."
        install_with_timeout "faiss-cpu" 120 "FAISS-CPU (fallback)"
    fi
else
    # Fast and standard modes
    if timeout 180 pip install -q faiss-gpu-cu12; then
        echo "✅ FAISS-GPU installed successfully"
    else
        echo "⚠️ FAISS-GPU failed, installing CPU version..."
        install_with_timeout "faiss-cpu" 120 "FAISS-CPU (fallback)"
    fi
fi
echo "✅ FAISS installation completed"

# PyTorch Geometric (optional, skip in minimal mode)
echo "📋 Step 4/5: Installing PyTorch Geometric..."
if [ "$SETUP_TYPE" = "minimal" ]; then
    echo "⚠️ Skipping PyTorch Geometric in minimal mode"
elif [ "$SETUP_TYPE" = "debug" ]; then
    echo "🔍 Debug mode: Installing PyTorch Geometric with logging..."
    if timeout 300 pip install -v torch-geometric; then
        echo "✅ PyTorch Geometric installed successfully"
    else
        echo "⚠️ PyTorch Geometric installation failed (optional)"
    fi
else
    # Fast and standard modes with timeout protection
    if timeout 120 pip install -q torch-geometric; then
        echo "✅ PyTorch Geometric installed successfully"
    else
        echo "⚠️ PyTorch Geometric installation failed (optional)"
    fi
fi
echo "✅ PyTorch Geometric step completed"

# Additional ML libraries
echo "📋 Step 5/5: Installing additional dependencies..."
if [ "$SETUP_TYPE" = "minimal" ]; then
    install_with_timeout "scikit-learn transformers" 120 "Basic ML libraries"
else
    install_with_timeout "networkx scikit-learn transformers datasets" 180 "ML libraries"
fi
echo "✅ Additional dependencies completed"

# System validation
echo "📋 Running system validation..."
check_gpu

# Test basic imports
echo "📋 Testing basic imports..."
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ Core libraries: OK')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ PyTorch: {e}')

try:
    import faiss
    print(f'✅ FAISS: Available')
    gpu_count = faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0
    print(f'   GPU count: {gpu_count}')
except Exception as e:
    print(f'❌ FAISS: {e}')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers: {e}')

try:
    import torch_geometric
    print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
except Exception as e:
    print(f'⚠️ PyTorch Geometric: Not available (OK for basic setup)')
"

echo ""
echo "🎉 InsightSpike-AI Colab Setup Complete!"
echo "============================================="
echo "Setup Type: $SETUP_TYPE"
echo "Next Steps:"
echo "1. Run data preparation cell to create sample data"
echo "2. Test the validation cell to confirm setup"
echo "3. Try the demo execution cells"
echo "============================================="

# Return appropriate exit code
exit 0