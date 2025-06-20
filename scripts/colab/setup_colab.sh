#!/bin/bash

# ---
# Colab-Optimized Setup Script for InsightSpike-AI (v2: Lockfile Regeneration)
# ---
# This script forces regeneration of poetry.lock for the Colab environment
# to avoid dependency conflicts from the local (e.g., macOS) lock file.
#
# Exit on any error
set -e

# ---
# 1. Install and Configure Poetry
# ---
echo "Installing and configuring Poetry..."
pip install poetry > /dev/null
poetry config virtualenvs.create false

# ---
# 2. **CRITICAL STEP: Switch to Colab-optimized configuration**
# ---
# Use pyproject_colab.toml which has Colab-specific dependency versions
# and remove any existing lock file to force fresh resolution
echo "Switching to Colab-optimized configuration..."
rm -f poetry.lock

# Backup original pyproject.toml and use Colab version
echo "Backing up original pyproject.toml and switching to Colab configuration..."
cp pyproject.toml pyproject_local_backup.toml
cp pyproject_colab.toml pyproject.toml

# ---
# 3. Resolve and Install All Dependencies with Colab-optimized Poetry
# ---
# With Colab-specific pyproject.toml, Poetry will:
# 1. Use Colab-optimized package versions (faiss-gpu, specific torch versions, etc.)
# 2. Resolve all dependencies for the Colab Linux CUDA environment
# 3. Generate a BRAND NEW, Colab-specific poetry.lock file
# 4. Install packages optimized for Colab's hardware and software stack
echo "Installing Colab-optimized dependencies and generating new lock file..."
# Install with Colab-specific extras, including package本体
poetry install --extras "full"

# NumPy互換性問題を解決
echo "Fixing NumPy compatibility..."
poetry run pip install "numpy<2.0" --force-reinstall

# ---
# 4. Install Difficult GPU Libraries with pip FIRST
# ---
# We handle the most problematic, pre-compiled libraries manually.
# This ensures compatibility with Colab's CUDA and Python environment.
#echo "Installing GPU-accelerated libraries with pip..."
#pip install faiss-gpu-cu12 > /dev/null

# Install PyTorch Geometric and its optimized dependencies
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION_SHORT=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
echo "Detected PyTorch ${PYTORCH_VERSION} and CUDA ${CUDA_VERSION_SHORT}. Installing PyG..."
poetry run pip install torch_geometric > /dev/null
poetry run pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_VERSION_SHORT}.html > /dev/null


# ---
# 5. Verification
# ---
echo "Installation complete. Verifying libraries..."
python -c " \
import torch; \
import faiss; \
import torch_geometric; \
print('✅ PyTorch version:', torch.__version__); \
print('✅ CUDA available for PyTorch:', torch.cuda.is_available()); \
print('✅ FAISS GPU enabled:', hasattr(faiss, 'GpuIndexIVFFlat')); \
print('✅ PyG version:', torch_geometric.__version__); \
print('\n🎉 Colab environment setup successful!'); \
print('📦 Using Colab-optimized dependency versions with NumPy 2.x'); \
print('🚀 CLI command \"insightspike\" is now available!'); \
print('⚡ GPU-optimized packages (faiss-gpu, PyTorch CUDA) ready'); \
print('🔬 NumPy 2.x compatibility enabled for latest ML features')"

# ---
# Cleanup function for restoring original configuration
# ---
cleanup_colab_config() {
    echo "Restoring original configuration..."
    if [ -f "pyproject_local_backup.toml" ]; then
        cp pyproject_local_backup.toml pyproject.toml
        rm -f pyproject_local_backup.toml
        echo "✅ Original pyproject.toml restored"
    fi
}

# Set trap to cleanup on exit (optional)
# trap cleanup_colab_config EXIT