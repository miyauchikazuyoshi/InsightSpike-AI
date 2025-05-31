# InsightSpike-AI Environment Setup Guide

This guide explains the three-environment strategy for InsightSpike-AI and provides detailed setup instructions for each environment.

## 🎯 Three-Environment Strategy Overview

InsightSpike-AI employs a sophisticated dependency management strategy that addresses the faiss-gpu vs faiss-cpu compatibility challenge across different environments:

### Environment Types

| Environment | Primary Use | faiss Package | GPU Support | Key Features |
|-------------|-------------|---------------|-------------|--------------|
| **Local Development** | Development, Testing | faiss-cpu | No | Full Poetry environment, comprehensive testing |
| **Google Colab** | Research, Large-scale experiments | faiss-gpu | Yes | GPU acceleration, CLI access, performance optimization |
| **CI/CD** | Automated testing | faiss-cpu | No | Minimal dependencies, fast execution, LITE_MODE |

## 🏠 Local Development Environment

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip
- Git

### Installation
```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Install with Poetry (recommended)
poetry install --with dev

# Alternative: Install with pip
pip install -e .
pip install -r requirements-dev.txt
```

### Usage
```bash
# Run CLI commands
poetry run insightspike loop "What is quantum entanglement?"

# Run tests
poetry run pytest development/tests/unit/ -v

# Interactive Python
poetry shell
python -c "from insightspike import InsightSpikeAI; ai = InsightSpikeAI()"
```

### Features
- ✅ Full development environment
- ✅ Comprehensive testing suite
- ✅ faiss-cpu for cross-platform compatibility
- ✅ All visualization and analysis tools
- ✅ Poetry dependency management

## ☁️ Google Colab Environment

### Key Innovation: faiss-gpu Priority Installation

Our Colab setup script (`scripts/colab/setup_colab.sh`) implements a critical optimization:

1. **faiss-gpu installed FIRST** via pip (ensures GPU support)
2. **Poetry configured** for global environment usage
3. **Project dependencies** installed without conflicts
4. **Comprehensive validation** of GPU functionality

### Installation Method 1: Enhanced Setup Script
```bash
# Clone in Colab
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI

# Run enhanced setup
!chmod +x scripts/colab/setup_colab.sh
!bash scripts/colab/setup_colab.sh
```

### Installation Method 2: Pre-configured Notebook
Open `InsightSpike_Colab_Demo.ipynb` which includes:
- Automated environment setup
- faiss-gpu validation tests
- CLI access verification
- Performance comparison (CPU vs GPU)

### Usage in Colab
```bash
# CLI commands (both methods work)
!PYTHONPATH=src python -m insightspike.cli loop "Your question here"
!poetry run insightspike loop "Your question here"

# Interactive Python
import sys
sys.path.append('src')
from insightspike import InsightSpikeAI
ai = InsightSpikeAI()
```

### Features
- ✅ GPU-accelerated faiss operations
- ✅ Full CLI access via Poetry
- ✅ Performance optimization for large datasets
- ✅ Comprehensive environment validation
- ✅ Automatic dependency conflict resolution

### Validation
The setup includes automatic validation:
```bash
# Comprehensive environment test
!python scripts/colab/test_colab_env.py
```

Tests:
- PyTorch GPU functionality
- faiss-gpu acceleration
- SentenceTransformers compatibility
- CLI command access

## 🔧 CI/CD Environment

### Strategy: LITE_MODE for Fast Testing

CI environment uses `INSIGHTSPIKE_LITE_MODE=1` to enable:
- Mock SentenceTransformer (no heavy model downloads)
- Minimal dependency installation
- Fast test execution

### Installation (in .github/workflows/ci.yml)
```bash
# Install minimal dependencies
pip install pytest numpy pyyaml networkx scikit-learn psutil faiss-cpu typer rich click
pip install -e .

# Set environment
export INSIGHTSPIKE_LITE_MODE=1
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Run tests
python -m pytest development/tests/unit/ -v
```

### Features
- ✅ Fast dependency installation
- ✅ Minimal resource usage
- ✅ Mock models for testing
- ✅ Parallel test execution
- ✅ Environment isolation

## 🚨 Common Issues and Solutions

### Issue 1: faiss-gpu Installation Conflicts
**Problem**: Poetry tries to install both faiss-cpu and faiss-gpu
**Solution**: Use our Colab setup script which installs faiss-gpu first

### Issue 2: CLI Commands Not Found in Colab
**Problem**: Poetry not properly installed or configured
**Solution**: Use `poetry config virtualenvs.create false` and verify with `!poetry --version`

### Issue 3: LITE_MODE Not Working in Tests
**Problem**: Environment variable not properly set or module import conflicts
**Solution**: Set `INSIGHTSPIKE_LITE_MODE=1` before imports and clear module cache

### Issue 4: GPU Not Detected in Colab
**Problem**: Colab runtime not set to GPU
**Solution**: Runtime → Change runtime type → Hardware accelerator → GPU

## 🔍 Environment Validation

### Local Development
```bash
poetry run python -c "
import faiss
print(f'Faiss version: {faiss.__version__}')
print('✅ Local environment ready')
"
```

### Google Colab
```python
# Run comprehensive validation
!python scripts/colab/test_colab_env.py

# Quick GPU check
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

### CI/CD
```bash
# Check LITE_MODE
python -c "
import os
print(f'LITE_MODE: {os.getenv(\"INSIGHTSPIKE_LITE_MODE\")}')
"
```

## 📊 Performance Comparison

| Environment | faiss Package | Search Speed | Memory Usage | Setup Time |
|-------------|---------------|--------------|--------------|------------|
| Local | faiss-cpu | 1x | Low | Fast |
| Colab | faiss-gpu | 10-50x | High | Medium |
| CI | faiss-cpu (mock) | N/A | Minimal | Very Fast |

## 🔄 Migration Between Environments

### Local → Colab
1. Commit changes locally
2. Clone in Colab: `!git clone <your-fork>`
3. Run Colab setup script
4. Use CLI commands with `!poetry run` prefix

### Colab → Local
1. Download modified files from Colab
2. Commit to your fork
3. Pull changes locally: `git pull origin main`
4. Re-run `poetry install --with dev`

## 📚 Additional Resources

- [Colab Demo Notebook](../../InsightSpike_Colab_Demo.ipynb)
- [Environment Validation Script](../../scripts/colab/test_colab_env.py)
- [CI Configuration](.github/workflows/ci.yml)
- [Poetry Configuration](../../pyproject.toml)

## 🔮 Future Enhancements

- **Auto-detection**: Automatic environment detection and setup
- **Container support**: Docker images for each environment
- **Cloud deployment**: AWS/GCP deployment configurations
- **Hybrid mode**: Local development with cloud GPU resources
