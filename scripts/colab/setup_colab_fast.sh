#!/usr/bin/env bash
# InsightSpike-AI Fast Setup for Google Colab - Optimized for Quick Testing
# Resolves PyTorch Geometric hanging issues with prebuilt wheels and timeouts

set -e

echo "⚡ InsightSpike-AI Fast Setup for Google Colab"
echo "🎯 Optimized for quick testing and development"
echo "📦 Using prebuilt wheels and timeout handling"
echo "🔧 Coordinated dependency strategy"

# Function to install with timeout
install_with_timeout() {
    local package="$1"
    local timeout="$2"
    echo "📦 Installing $package with ${timeout}s timeout..."
    timeout "$timeout" pip install -q "$package" || {
        echo "⚠️ $package installation timed out, skipping..."
        return 1
    }
    return 0
}

# Function to verify installation
verify_package() {
    local package="$1"
    local import_name="${2:-$package}"
    python -c "import $import_name; print(f'✅ $package verified')" 2>/dev/null || {
        echo "❌ $package verification failed"
        return 1
    }
}

# GPU Detection
echo ""
echo "🔍 Detecting hardware..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️ No GPU detected"

# 1. Essential Python packages (quick)
echo ""
echo "🐍 Installing essential Python packages..."
pip install -q --upgrade pip setuptools wheel
pip install -q "numpy>=1.24.0,<3.0.0" pandas matplotlib

# 2. PyTorch with CUDA (fast prebuilt wheels)
echo ""
echo "🔥 Installing PyTorch (CUDA 12.1 optimized)..."
install_with_timeout "torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" 300

verify_package "torch" "torch"

# 3. PyTorch Geometric (with prebuilt wheels and fallback)
echo ""
echo "🌐 Installing PyTorch Geometric (optimized)..."

# Use specific PyG wheel repository for faster installation
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d'+' -f1)
CUDA_VERSION="cu121"

echo "📦 Using PyG wheels for torch-${TORCH_VERSION}+${CUDA_VERSION}"

# Install PyG components with timeout and fallback
install_with_timeout "torch-scatter --find-links https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html" 180 || {
    echo "🔄 Fallback: Installing torch-scatter from PyPI..."
    pip install -q torch-scatter
}

install_with_timeout "torch-sparse --find-links https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html" 180 || {
    echo "🔄 Fallback: Installing torch-sparse from PyPI..."
    pip install -q torch-sparse
}

install_with_timeout "torch-cluster --find-links https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html" 180 || {
    echo "🔄 Fallback: Skipping torch-cluster (optional)..."
}

install_with_timeout "torch-geometric" 120 || {
    echo "❌ PyTorch Geometric installation failed"
    echo "ℹ️ Continuing without PyG - basic functionality will work"
}

# 4. FAISS with GPU support (critical for performance) - Improved dependency handling
echo ""
echo "🔍 Installing FAISS with GPU support..."

# First ensure CUDA runtime libraries are installed
echo "📦 Installing CUDA runtime libraries..."
pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# Install FAISS-GPU with explicit version and dependency checking
install_with_timeout "faiss-gpu-cu12>=1.11.0" 180 || {
    echo "⚠️ FAISS-GPU installation failed, trying without version constraint..."
    install_with_timeout "faiss-gpu-cu12" 120 || {
        echo "🔄 Fallback: Installing faiss-cpu..."
        pip install -q faiss-cpu
        echo "ℹ️ Using CPU-only FAISS - GPU acceleration unavailable"
    }
}

# Verify FAISS installation with detailed error reporting
python -c "
try:
    import faiss
    print(f'✅ FAISS {faiss.__version__} installed successfully')
    
    # Test GPU availability
    try:
        res = faiss.StandardGpuResources()
        print('🚀 FAISS GPU support available')
    except Exception as e:
        print(f'⚠️ FAISS GPU unavailable (using CPU): {e}')
except ImportError as e:
    print(f'❌ FAISS import failed: {e}')
    exit(1)
"

# 5. Hugging Face (essential for datasets)
echo ""
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers datasets tokenizers sentence-transformers

verify_package "transformers" "transformers"
verify_package "datasets" "datasets"

# 6. Core dependencies for InsightSpike
echo ""
echo "🎯 Installing InsightSpike core dependencies..."
pip install -q typer rich click pyyaml networkx scikit-learn

# 7. Quick poetry setup (lightweight)
echo ""
echo "📦 Setting up Poetry (lightweight)..."
curl -sSL https://install.python-poetry.org | python3 - &
POETRY_PID=$!

# Don't wait for Poetry, continue with essentials
echo "⏩ Continuing while Poetry installs in background..."

# 8. Install project in development mode
echo ""
echo "🚀 Installing InsightSpike-AI..."
pip install -q -e .

# 9. Create necessary directories
echo ""
echo "📁 Creating project structure..."
mkdir -p experiment_results logs data/processed data/raw

# 10. Download minimal NLTK data
echo ""
echo "📝 Downloading essential NLTK data..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for corpus in ['punkt', 'stopwords']:
    try:
        nltk.download(corpus, quiet=True)
        print(f'✅ {corpus} downloaded')
    except:
        print(f'⚠️ {corpus} download failed')
"

# 11. Wait for Poetry if still running
if kill -0 $POETRY_PID 2>/dev/null; then
    echo "⏳ Waiting for Poetry installation to complete..."
    wait $POETRY_PID
fi

export PATH="/root/.local/bin:$PATH"
if command -v poetry &> /dev/null; then
    echo "✅ Poetry available"
    
    # Clear Poetry cache for clean environment
    echo "🧹 Clearing Poetry cache..."
    rm -rf ~/.cache/pypoetry || true
    rm -f poetry.lock || true
    
    poetry config virtualenvs.create false
    
    echo "📦 Installing remaining dependencies with Poetry..."
    poetry install --only main
else
    echo "⚠️ Poetry not available - using pip only"
fi

# 12. Final validation
echo ""
echo "🔬 Running fast validation..."
python -c "
import sys
print(f'Python: {sys.version}')

# Core imports
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except:
    print('❌ PyTorch failed')

try:
    import torch_geometric
    print(f'✅ PyTorch Geometric {torch_geometric.__version__}')
except:
    print('⚠️ PyTorch Geometric not available (fallback mode)')

try:
    import faiss
    print(f'✅ FAISS {faiss.__version__}')
    if hasattr(faiss, 'get_num_gpus'):
        print(f'   GPU support: {faiss.get_num_gpus()} GPUs')
except:
    print('❌ FAISS failed')

try:
    import transformers, datasets
    print(f'✅ HuggingFace: transformers-{transformers.__version__}, datasets-{datasets.__version__}')
except:
    print('❌ HuggingFace libraries failed')

try:
    from insightspike.core.config import get_config
    print('✅ InsightSpike-AI project available')
except:
    print('⚠️ InsightSpike-AI import issues (may work with PYTHONPATH)')
"

echo ""
echo "⚡ Fast setup complete!"
echo "🎯 Total setup time: ~3-5 minutes"
echo ""
echo "📋 Dependencies coordinated via:"
echo "   • GPU packages installed first via pip (torch, faiss)"
echo "   • Remaining dependencies via Poetry (when available)"
echo "   • Strategic conflict avoidance"
echo ""
echo "📝 Quick start commands:"
echo "   🔬 Test basic functionality:"
echo "     PYTHONPATH=src python scripts/colab/test_colab_env.py"
echo ""
echo "   🧪 Run minimal experiment:"
echo "     PYTHONPATH=src python scripts/colab/colab_large_scale_experiment.py --mode quick"
echo ""
echo "   🚀 CLI test:"
echo "     PYTHONPATH=src python -m insightspike.cli --help"

# Save setup log
echo "$(date): Fast setup completed" >> logs/colab_setup.log
