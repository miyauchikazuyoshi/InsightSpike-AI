#!/usr/bin/env bash
# InsightSpike-AI Debug Setup for Google Colab - Diagnostic Version
# Identifies PyTorch Geometric installation bottlenecks with detailed logging

set -e

echo "🔍 InsightSpike-AI Debug Setup for Google Colab"
echo "🛠️ Diagnostic mode with detailed logging"
echo "📊 Identifying PyTorch Geometric installation issues"

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

# 1. Environment Preparation
log_info "Preparing environment..."
pip install -q --upgrade pip setuptools wheel
pip install -q "numpy<2.0"

# 2. PyTorch Installation with Debug
log_info "Installing PyTorch with CUDA support..."
echo "=== PYTORCH INSTALLATION START ==="
start_time=$(date +%s)

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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

# 5. Installation Validation
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

# 6. Detailed Build Information
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

# 7. Network and Download Speed Test
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

# 8. Summary Report
echo ""
echo "=== INSTALLATION SUMMARY ==="
log_info "Setup debug analysis complete"
log_info "Total PyTorch installation time: ${pytorch_time}s"
log_info "Log file saved as: $LOG_FILE"

echo ""
echo "📋 Key Findings:"
echo "   • PyTorch installation: ${pytorch_time}s"
echo "   • GPU availability: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   • PyG component status:"

for component in torch_geometric torch_scatter torch_sparse torch_cluster; do
    status=$(python -c "
try:
    import $component
    print('✅ AVAILABLE')
except:
    print('❌ MISSING')
" 2>/dev/null)
    echo "     - $component: $status"
done

echo ""
echo "📝 Recommendations:"
echo "   • Use fast setup script for development: setup_colab_fast.sh"
echo "   • For production: Consider prebaked Docker image"
echo "   • Specific issues identified in: $LOG_FILE"

echo ""
echo "🔗 Next steps:"
echo "   • Review log file for specific bottlenecks"
echo "   • Use timeout-based installation strategy"
echo "   • Consider PyTorch Geometric alternatives if needed"
