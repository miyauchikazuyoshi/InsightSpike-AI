#!/usr/bin/env bash
# InsightSpike-AI Debug Setup for Google Colab - Diagnostic Version
# Identifies PyTorch Geometric installation bottlenecks with detailed logging

set -e

echo "🔍 InsightSpike-AI Debug Setup for Google Colab"
echo "🛠️ Diagnostic mode with detailed logging"
echo "📊 Identifying PyTorch Geometric installation issues"
echo "🔧 Strategic dependency coordination analysis"

# Create log file
LOG_FILE="colab_debug_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "📝 Logging to: $LOG_FILE"

# Function to log with timestamp
log_info() {
    echo "[$(date '+%H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $1"
}

log_warning() {
    echo "[$(date '+%H:%M:%S')] WARNING: $1"
}

# System Information Collection
log_info "Collecting system information..."
echo "=== SYSTEM INFO ==="
uname -a
cat /etc/os-release
python --version
pip --version

echo ""
echo "=== GPU INFO ==="
nvidia-smi || log_warning "No NVIDIA GPU detected"

echo ""
echo "=== MEMORY INFO ==="
free -h
df -h

echo ""
echo "=== PYTHON ENVIRONMENT ==="
which python
which pip
pip list | head -20

echo ""
echo "=== DEPENDENCY COORDINATION STRATEGY ==="
log_info "1. GPU-critical packages installed first via pip"
log_info "2. Poetry dependencies from requirements-colab.txt (excludes torch/faiss)"
log_info "3. Complete reference: requirements-colab-comprehensive.txt"

# 1. Environment Preparation
log_info "Preparing environment..."
pip install -q --upgrade pip setuptools wheel
pip install -q "numpy>=1.24.0,<3.0.0"

# 2. PyTorch Installation with Debug
log_info "Installing PyTorch with CUDA support..."
echo "=== PYTORCH INSTALLATION START ==="
start_time=$(date +%s)

pip install -q torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

end_time=$(date +%s)
pytorch_time=$((end_time - start_time))
log_info "PyTorch installation completed in ${pytorch_time}s"

# PyTorch validation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# 3. PyTorch Geometric Installation with Detailed Monitoring
log_info "Starting PyTorch Geometric installation..."
echo "=== PYTORCH GEOMETRIC INSTALLATION START ==="

# Get PyTorch version for wheel selection
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION="cu121"
log_info "Using wheels for torch-${TORCH_VERSION}+${CUDA_VERSION}"

# Install each PyG component separately with monitoring
components=("torch-scatter" "torch-sparse" "torch-cluster" "torch-spline-conv" "torch-geometric")

for component in "${components[@]}"; do
    log_info "Installing $component..."
    start_time=$(date +%s)
    
    # Use timeout and detailed output
    if [[ "$component" == "torch-geometric" ]]; then
        # Main package installation
        timeout 300 pip install -v "$component" || {
            log_error "$component installation timed out after 300s"
            continue
        }
    else
        # Extension packages with wheel lookup
        timeout 600 pip install -v "$component" \
            --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html" || {
            log_warning "$component installation failed/timed out, trying fallback..."
            timeout 180 pip install -v "$component" || {
                log_error "$component fallback also failed"
                continue
            }
        }
    fi
    
    end_time=$(date +%s)
    install_time=$((end_time - start_time))
    log_info "$component installation completed in ${install_time}s"
    
    # Verify installation
    python -c "
try:
    import ${component/_/}
    print(f'✅ $component verified')
except ImportError as e:
    print(f'❌ $component verification failed: {e}')
" || log_warning "$component verification failed"
done

echo "=== PYTORCH GEOMETRIC INSTALLATION END ==="

# 4. System Resource Monitoring During Installation
log_info "Collecting post-installation resource usage..."
echo "=== POST-INSTALLATION RESOURCES ==="
free -h
df -h
pip list | grep -E "(torch|pyg|geometric)"

# 5. FAISS Installation with Debug
log_info "Installing FAISS with GPU support (debug mode)..."
echo "=== FAISS INSTALLATION START ==="

# Install CUDA dependencies first
pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# Install FAISS-GPU with detailed monitoring
timeout 300 pip install -v faiss-gpu-cu12 || {
    log_warning "FAISS-GPU installation failed, trying CPU fallback..."
    pip install -v faiss-cpu
}

# Test FAISS functionality
python -c "
import faiss
print(f'FAISS version: {faiss.__version__}')
try:
    # Test GPU
    res = faiss.StandardGpuResources()
    print('✅ FAISS GPU support working')
except Exception as e:
    print(f'⚠️ FAISS GPU not available: {e}')
"

echo "=== FAISS INSTALLATION END ==="

# 6. Installation Validation
log_info "Running comprehensive validation..."
echo "=== VALIDATION RESULTS ==="

python -c "
import sys
import time

# Test imports with timing
test_modules = [
    ('torch', 'PyTorch'),
    ('torch_geometric', 'PyTorch Geometric'),
    ('torch_scatter', 'PyTorch Scatter'),
    ('torch_sparse', 'PyTorch Sparse'),
    ('torch_cluster', 'PyTorch Cluster'),
]

for module, name in test_modules:
    try:
        start = time.time()
        exec(f'import {module}')
        import_time = time.time() - start
        version = eval(f'{module}.__version__')
        print(f'✅ {name} {version} (import: {import_time:.3f}s)')
    except ImportError as e:
        print(f'❌ {name} failed: {e}')
    except Exception as e:
        print(f'⚠️ {name} import issue: {e}')

# Test GPU functionality
print('')
print('GPU Functionality Test:')
try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x.t())
        print('✅ GPU tensor operations working')
    else:
        print('⚠️ No GPU available for testing')
except Exception as e:
    print(f'❌ GPU test failed: {e}')
"

# 7. Detailed Build Information
log_info "Collecting build environment information..."
echo "=== BUILD ENVIRONMENT ==="
gcc --version 2>/dev/null || log_warning "GCC not available"
nvcc --version 2>/dev/null || log_warning "NVCC not available"

echo ""
echo "=== PYTHON BUILD INFO ==="
python -c "
import sysconfig
print('Python build info:')
for key, value in sysconfig.get_config_vars().items():
    if any(keyword in key.lower() for keyword in ['cc', 'compiler', 'cuda', 'gpu']):
        print(f'  {key}: {value}')
"

# 8. Network and Download Speed Test
log_info "Testing network connectivity to PyG servers..."
echo "=== NETWORK DIAGNOSTICS ==="
curl -I "https://data.pyg.org/whl/" 2>/dev/null | head -5 || log_warning "PyG server unreachable"

# Download speed test
log_info "Testing download speed..."
time_start=$(date +%s)
curl -s "https://data.pyg.org/whl/torch-2.2.2+cu121.html" > /dev/null || log_warning "PyG download test failed"
time_end=$(date +%s)
download_time=$((time_end - time_start))
log_info "PyG server response time: ${download_time}s"

# 9. Install remaining dependencies
log_info "Installing remaining dependencies..."
pip install -q transformers datasets tokenizers sentence-transformers
pip install -q typer rich click pyyaml networkx scikit-learn matplotlib

# 9. Install Poetry and remaining dependencies
log_info "Installing Poetry and coordinated dependencies..."
echo "=== POETRY INSTALLATION START ==="

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

if command -v poetry &> /dev/null; then
    log_info "Poetry installation successful"
    poetry config virtualenvs.create false
    
    # Install remaining dependencies via Poetry
    log_info "Installing coordinated dependencies via Poetry..."
    poetry install --only main
else
    log_warning "Poetry installation failed - falling back to pip"
fi

echo "=== POETRY INSTALLATION END ==="

# 10. Install project and test
pip install -q -e .
mkdir -p experiment_results logs data/processed data/raw

# Final summary
echo ""
echo "=== DIAGNOSTIC SUMMARY ==="
log_info "Debug setup completed"
echo "📝 Detailed logs saved to: $LOG_FILE"
echo ""
echo "🔍 Key findings:"
echo "   • PyTorch installation time: ${pytorch_time}s"
echo "   • PyG server response time: ${download_time}s"
echo "   • Log file contains detailed bottleneck analysis"
echo ""
echo "💡 Recommendations:"
echo "   • Use fast setup script for development: setup_colab_fast.sh"
echo "   • For production: Consider prebaked Docker image"
echo "   • Specific issues identified in: $LOG_FILE"

echo ""
echo "🔗 Next steps:"
echo "   • Review log file for specific bottlenecks"
echo "   • Use timeout-based installation strategy"
echo "   • Consider PyTorch Geometric alternatives if needed"

echo ""
echo "📋 Dependencies coordinated via:"
echo "   • GPU packages installed first via pip (torch, faiss)"
echo "   • Strategic timing analysis for optimization"
echo "   • Complete diagnostic logging for troubleshooting"
