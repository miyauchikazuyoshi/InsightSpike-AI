# 🛠️ InsightSpike-AI: Environment Setup & Troubleshooting Guide

## 🚀 Quick Start Checklist

### Local Development (Mac/Intel)

**Recommended Setup: Using Conda for Core ML Dependencies**

For robust management of complex machine learning dependencies like PyTorch, PyTorch Geometric, and FAISS, we highly recommend using Conda.

```bash
# 1. Install Miniconda (if not already installed)
#    Download from: https://docs.conda.io/en/latest/miniconda.html
#    Follow installation prompts. Restart terminal after installation.

# 2. Create and activate a new Conda environment
conda create -n insightspike_env python=3.11  # Use Python 3.11
conda activate insightspike_env

# 3. Install core ML dependencies via Conda (recommended channels)
#    FAISS (from conda-forge for better compatibility)
conda install -c conda-forge faiss-cpu

#    PyTorch Geometric (from pyg channel)
conda install pyg -c pyg

#    PyTorch (ensure compatibility with PyTorch Geometric)
#    Check PyTorch website for specific version compatibility with your system and PyG
#    Example for PyTorch 2.2.2 (compatible with PyG 2.6.1)
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -c pytorch

# 4. Install other project dependencies via pip (from pyproject.toml)
#    First, ensure you are in the project root and the Conda environment is active.
#    Export poetry dependencies to a requirements file (excluding local paths)
poetry run pip freeze | grep "==" > poetry_requirements.txt
pip install -r poetry_requirements.txt

# 5. Set PYTHONPATH for module imports
#    Add this to your shell profile (~/.zshrc or ~/.bash_profile)
echo 'export PYTHONPATH="/path/to/InsightSpike-AI/src:$PYTHONPATH"' >> ~/.zshrc # or ~/.bash_profile
source ~/.zshrc # or ~/.bash_profile

# 6. Verify installation
python -c "from src.insightspike.core.system import InsightSpikeSystem; print('✅ Ready!')"

# 7. Run validation
python scripts/pre_push_validation.py
```

---

## 🔧 Common Issues & Quick Fixes

### Segmentation Faults with FAISS/PyTorch Geometric (macOS Intel)

**Problem**: Encountering `Segmentation fault` errors, especially during graph building or similarity search operations involving FAISS and PyTorch Geometric. This is often due to low-level library conflicts (e.g., OpenMP runtimes) or subtle version incompatibilities between these highly optimized C/C++-backed libraries on macOS Intel.

**Solution**:
1.  **Prioritize Conda Installation**: As detailed in the "Recommended Setup" above, installing FAISS and PyTorch Geometric via Conda (specifically from `conda-forge` and `pyg` channels) is crucial. Conda provides a more controlled environment, ensuring compatible binaries and managing underlying C/C++ dependencies.
2.  **Strict Version Alignment**: Ensure the following versions are used, as they have shown better compatibility:
    *   `torch`: `2.2.2`
    *   `torchvision`: `0.17.2`
    *   `torchaudio`: `2.2.2`
    *   `faiss-cpu`: `1.8.0` (from `conda-forge`)
    *   `torch-geometric`: `2.5.2` (from `pyg` channel, which installs `pyg`)
    *   `numpy`: `1.26.4`
3.  **Import Order**: While less common with Conda-managed environments, if issues persist, ensure `import faiss` occurs *before* `import torch` in your Python scripts where both are used. (Note: This is often handled by the package manager, but can be a manual workaround).
4.  **Debugging**: If a segfault occurs, try to isolate the problematic code section. You can temporarily add `import sys; sys.exit(1)` before the suspected line to see if the code reaches that point. Inspect input data for `NaN` or `Inf` values (`np.isnan().any()`, `np.isinf().any()`).

---

### "InsightSpike-AI not available" CLI Warning

**Problem**: Editable install not working
**Solution**:
```bash
# Check environment
which python
which pip
pip list | grep -i insight

# Reinstall
pip install -e .
# OR with Poetry
poetry install
```

### Package Version Conflicts

**Problem**: NumPy/PyTorch compatibility issues
**Solution**:
```bash
# Clean reinstall with compatible versions
pip uninstall numpy torch sentence-transformers
pip install numpy==1.26.4 torch==2.2.2 sentence-transformers==2.7.0
```

### Colab Runtime Issues

**Problem**: Meta tensor errors, import failures
**Solution**:
```python
# 1. Ensure runtime restart after package installation
# 2. Verify versions:
import torch, numpy as np
print(f"PyTorch: {torch.__version__}, NumPy: {np.__version__}")

# 3. Force CPU mode if GPU issues:
device = "cpu"
torch.cuda.empty_cache()
```

### Poetry Installation Issues

**Problem**: Poetry install fails, or `poetry export` command is not found.

**Solution**:
1.  **Update Poetry**: Ensure your Poetry installation is up-to-date.
    ```bash
    pip install --upgrade pip poetry poetry-core
    ```
2.  **Conda Integration**: If you are using Conda, ensure Poetry is configured to use your Conda environment's Python interpreter.
    ```bash
    # First, activate your Conda environment: conda activate your_env_name
    poetry env use $(which python) # This tells Poetry to use the active Conda env's python
    poetry install # Now Poetry should install dependencies into the Conda env
    ```
3.  **Manual Dependency Export (if `poetry export` is unavailable)**: If `poetry export` is not available (e.g., older Poetry versions), you can manually extract dependencies for `pip` installation:
    ```bash
    # Activate your Poetry environment (if using one): poetry shell
    # Or just run directly if Poetry is configured to use Conda env:
    poetry run pip freeze | grep "==" > poetry_requirements.txt
    # Then, in your Conda environment:
    pip install -r poetry_requirements.txt
    ```

---

### GitHub Token Setup (Colab)

**Problem**: "GitHub token not found"
**Solution**:
... (既存の内容を維持) ...

---

### Environment Path Issues

**Problem**: `ModuleNotFoundError` for `insightspike` or other project modules.

**Solution**:
Ensure your project's `src` directory is correctly added to your `PYTHONPATH`. This tells Python where to find your custom modules.

```bash
# Add this to your shell profile (~/.zshrc or ~/.bash_profile)
# Replace /path/to/InsightSpike-AI with your actual project root path
echo 'export PYTHONPATH="/path/to/InsightSpike-AI/src:$PYTHONPATH"' >> ~/.zshrc # or ~/.bash_profile
source ~/.zshrc # or ~/.bash_profile
```

---

## 🎯 Local ↔ Colab Best Practices

### Development Workflow

... (既存の内容を維持) ...

### Version Synchronization

**Critical Versions (for macOS Intel with Conda)**:
- Python: 3.11
- PyTorch: 2.2.2
- NumPy: 1.26.4
- FAISS-CPU: 1.8.0 (installed via `conda install -c conda-forge faiss-cpu`)
- PyTorch Geometric: 2.5.2 (installed via `conda install pyg -c pyg`)
- sentence-transformers: 2.7.0

**Check Compatibility**:
... (既存の内容を維持) ...

---

## 🚨 Emergency Recovery

### Complete Environment Reset

```bash
# Local (Conda-based setup)
conda deactivate # Deactivate current Conda env
conda env remove -n insightspike_env # Remove the Conda environment
# Then, follow "Recommended Setup" from scratch.

# Colab
# Runtime → Factory reset runtime
# Re-run setup cells
```

### Fallback Installation

... (既存の内容を維持) ...

### Data Corruption Recovery

... (既存の内容を維持) ...

---

## 📞 Getting Help

... (既存の内容を維持) ...

---

## ✅ Success Indicators

... (既存の内容を維持) ...

---
*Last updated: July 2025*
