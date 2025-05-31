#!/usr/bin/env bash
# Enhanced Colab Setup for InsightSpike-AI Large-Scale Experiments
# Compatible with GPU acceleration and production testing

set -e

echo "🚀 Setting up InsightSpike-AI for Google Colab (Enhanced)..."
echo "📊 Optimized for large-scale experiments with GPU acceleration"

# GPU Detection and Configuration
echo ""
echo "🔍 Detecting hardware configuration..."
nvidia-smi || echo "⚠️ No NVIDIA GPU detected - will use CPU mode"
echo "💾 Available memory:"
free -h | head -2

# 1. System Updates and Core Dependencies
echo ""
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential git curl

# 2. Python Environment Setup with NumPy fix
echo ""
echo "🐍 Setting up Python environment..."
pip install -q --upgrade pip setuptools wheel
# NumPy 2.0 問題の回避
pip install -q "numpy<2.0"

# 3. Poetry Installation for CLI support
echo ""
echo "📦 Installing Poetry for CLI commands..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
poetry --version

# 4. PyTorch with GPU Support (with NumPy constraint)
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
pip install -q torch==2.2.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed, CUDA: {torch.cuda.is_available()}')"

# 5. PyTorch Geometric for Graph Neural Networks
echo ""
echo "🌐 Installing PyTorch Geometric..."
pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
python -c "import torch_geometric; print(f'✅ PyTorch Geometric {torch_geometric.__version__} installed')"

# 6. Hugging Face Ecosystem
echo ""
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers datasets tokenizers accelerate evaluate
python -c "import transformers; print(f'✅ Transformers {transformers.__version__} installed')"

# 7. Vector Database and Search (GPU optimized)
echo ""
echo "🔍 Installing vector search libraries..."
# Colabでfaiss (GPU機能統合版) とsentence-transformersを先にインストール
echo "📦 Installing Faiss with GPU support for CUDA 12.x..."
pip install -q faiss sentence-transformers
echo "🔍 Verifying Faiss GPU functionality..."
python -c "
import faiss
print(f'✅ Faiss {faiss.__version__} installed')
try:
    # GPU機能の確認
    if hasattr(faiss, 'get_num_gpus'):
        num_gpus = faiss.get_num_gpus()
        print(f'✅ Number of GPUs detected by Faiss: {num_gpus}')
    else:
        print('⚠️  GPU detection method not available in this Faiss version')
    
    # StandardGpuResourcesの確認
    if hasattr(faiss, 'StandardGpuResources'):
        print('✅ GPU resources class available')
    else:
        print('⚠️  GPU resources not available - CPU-only version')
except Exception as e:
    print(f'⚠️  Faiss GPU test error: {e}')
"

# 8. Scientific Computing and Visualization
echo ""
echo "📊 Installing scientific libraries..."
pip install -q pandas matplotlib seaborn plotly scikit-learn networkx
pip install -q jupyter ipywidgets tqdm

# 9. InsightSpike-AI Core Dependencies (Poetry環境設定も含む)
echo ""
echo "🎯 Installing InsightSpike-AI dependencies..."
# Poetry設定: Colabの既存環境を使用
poetry config virtualenvs.create false
# 直接必要なパッケージをpipでインストール
pip install -q typer rich click pyyaml psutil

# プロジェクトを開発モードでインストール
pip install -q -e .

# Poetry環境でも同様に利用可能になるよう、poetry installを実行（依存関係競合を避けるため--no-deps）
poetry install --no-deps

# 10. Environment Validation
echo ""
echo "✅ Comprehensive environment validation..."

# GPU Validation
echo "🚀 GPU Configuration:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
    print(f'  CUDA Version: {torch.version.cuda}')
else:
    print('  CPU mode - no GPU available')
"

# Library Validation
echo ""
echo "📚 Library Validation:"
python -c "
try:
    import torch_geometric; print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
    import transformers; print(f'✅ Transformers: {transformers.__version__}')
    import datasets; print(f'✅ Datasets: {datasets.__version__}')
    import faiss
    print(f'✅ Faiss: {faiss.__version__}')
    # GPU対応テスト
    try:
        index = faiss.IndexFlatL2(128)
        if hasattr(faiss, 'StandardGpuResources'):
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            print('✅ Faiss-GPU: GPU acceleration available')
        else:
            print('⚠️ Faiss-GPU: GPU functions not available')
    except Exception as e:
        print(f'⚠️ Faiss GPU test failed: {e}')
    import networkx; print(f'✅ NetworkX: {networkx.__version__}')
    import numpy; print(f'✅ NumPy: {numpy.__version__}')
    print('✅ All core libraries validated')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# 11. Initialize Project Structure
echo ""
echo "📁 Initializing project structure..."
mkdir -p experiment_results logs data/processed data/raw

# Download NLTK Data
echo ""
echo "📝 Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print('✅ NLTK data downloaded')
"

echo ""
echo "🎉 Enhanced Colab setup complete!"
echo "🚀 Ready for large-scale experiments with CLI support!"

# 環境テスト実行
echo ""
echo "🔬 Running comprehensive environment validation..."
python scripts/colab/test_colab_env.py

echo ""
echo "📝 Next steps:"
echo "   🔬 Run system validation:"
echo "     PYTHONPATH=src python scripts/production/system_validation.py"
echo ""
echo "   🧪 Use CLI commands:"
echo "     PYTHONPATH=src python -m insightspike.cli loop 'What is quantum entanglement?'"