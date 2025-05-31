#!/usr/bin/env bash
# Enhanced Colab Setup for InsightSpike-AI Large-Scale Experiments
# Compatible with GPU acceleration and production testing

set -e  # エラー時に停止

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

# 2. Python Environment Setup
echo ""
echo "🐍 Setting up Python environment..."
pip install -q --upgrade pip setuptools wheel

# 3. PyTorch with GPU Support (Latest Stable)
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
pip install -q torch==2.2.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed, CUDA: {torch.cuda.is_available()}')"

# 4. PyTorch Geometric for Graph Neural Networks
echo ""
echo "🌐 Installing PyTorch Geometric..."
pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
python -c "import torch_geometric; print(f'✅ PyTorch Geometric {torch_geometric.__version__} installed')"

# 5. Hugging Face Ecosystem for Datasets
echo ""
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers datasets tokenizers accelerate evaluate
python -c "import transformers; print(f'✅ Transformers {transformers.__version__} installed')"

# 6. Vector Database and Search
echo ""
echo "🔍 Installing vector search libraries..."
pip install -q faiss-gpu sentence-transformers
python -c "import faiss; print(f'✅ Faiss-GPU {faiss.__version__} installed')"

# 7. Scientific Computing and Visualization
echo ""
echo "📊 Installing scientific libraries..."
pip install -q numpy pandas matplotlib seaborn plotly scikit-learn networkx
pip install -q jupyter ipywidgets tqdm

# 8. InsightSpike-AI Core Dependencies
echo ""
echo "🎯 Installing InsightSpike-AI dependencies..."
pip install -q rich typer sqlalchemy nltk pydantic

# 9. Environment Validation
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
    import faiss; print(f'✅ Faiss: {faiss.__version__}')
    import networkx; print(f'✅ NetworkX: {networkx.__version__}')
    print('✅ All core libraries validated')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# 10. Initialize Project Structure
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
echo "🚀 Ready for large-scale experiments!"
echo ""
echo "📝 Next steps:"
echo "   🔬 Run system validation:"
echo "     PYTHONPATH=src python scripts/production/system_validation.py"
echo ""
echo "   🧪 Run large-scale experiments:"
echo "     PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode quick"
echo "     PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode standard"
echo "     PYTHONPATH=src python scripts/colab_large_scale_experiment.py --mode comprehensive"
echo ""
echo "   📊 Environment diagnostics:"
echo "     PYTHONPATH=src python scripts/colab_diagnostic.py"