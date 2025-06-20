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
# 2. **CRITICAL STEP: Remove Existing Lock File**
# ---
# The poetry.lock file generated on a local machine (like macOS) is not
# compatible with the Colab Linux GPU environment. We delete it to force
# Poetry to resolve dependencies specifically for Colab.
echo "Removing existing poetry.lock to ensure a fresh dependency resolution for Colab..."
rm -f poetry.lock


# ---
# 3. Resolve and Install All Other Dependencies with Poetry
# ---
# With no lock file, Poetry will now:
# 1. Read pyproject.toml.
# 2. See that torch, faiss-gpu, pyg are already installed and accept them.
# 3. Resolve all other dependencies to be compatible with the Colab environment.
# 4. Generate a BRAND NEW, Colab-specific poetry.lock file.
# 5. Install the remaining packages.
echo "Installing dependencies with Poetry and generating a new lock file..."
# パッケージ本体も含めて全依存関係をインストール（--no-rootを削除）
poetry install --without dev,ci --extras "full"

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
print('\n🎉 Environment setup successful! A new, Colab-specific poetry.lock has been generated.'); \
print('📦 InsightSpike-AI package and all dependencies installed.'); \
print('🚀 CLI command \"insightspike\" is now available!')"