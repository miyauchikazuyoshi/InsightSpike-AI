#!/bin/bash

# ---
# Colab-Optimized Setup Script for InsightSpike-AI
# ---
# This script installs dependencies in the correct order for Colab's GPU environment,
# bypassing the poetry dependency resolution issues with pre-compiled binaries like faiss-gpu.
#
# Exit on any error
set -e

# ---
# 1. Install Poetry
# ---
echo "Installing Poetry..."
pip install poetry

# ---
# 2. Configure Poetry to use the system's Python environment
# ---
# This is crucial. We tell Poetry not to create its own virtual environment,
# but to install packages directly into the Colab runtime.
echo "Configuring Poetry to use system python..."
poetry config virtualenvs.create false

# ---
# 3. Install the "Difficult" GPU Libraries with pip
# ---
# We handle the most problematic, pre-compiled libraries manually first.
# This ensures they are compatible with Colab's CUDA and Python environment.
# We target CUDA 12.1 which is common in modern Colab runtimes.
echo "Installing GPU-accelerated libraries with pip..."

# Install FAISS for CUDA 12.1. This will automatically pull a compatible numpy version.
#pip install faiss-gpu-cu12

# Install PyTorch Geometric and its optimized dependencies.
# The URL must match the PyTorch and CUDA version of the Colab environment.
# First, get the exact PyTorch version from Colab's environment to build the URL.
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION_SHORT=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
echo "Detected PyTorch ${PYTORCH_VERSION} and CUDA ${CUDA_VERSION_SHORT}"

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_VERSION_SHORT}.html

# ---
# 4. Install Remaining Dependencies with Poetry
# ---
# Now that the hard parts are done, we let Poetry install the rest of the pure-python
# dependencies. It will see that torch, faiss, etc. are already installed and skip them.
echo "Installing remaining application dependencies with Poetry..."
poetry install --no-root --without dev,ci --extras "full"

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
print('\n🎉 Environment setup successful! You are ready to run InsightSpike-AI on Colab with GPU acceleration.')"
