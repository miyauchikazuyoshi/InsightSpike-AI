# 🛠️ InsightSpike-AI: Environment Setup & Troubleshooting Guide

## 🚀 Quick Start Checklist

### Local Development (Mac/Intel)

```bash
# 1. Clone and install
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
poetry install  # OR: pip install -e .

# 2. Verify installation
python -c "from src.insightspike.core.system import InsightSpikeSystem; print('✅ Ready!')"

# 3. Run validation
python scripts/pre_push_validation.py
```

### Google Colab

```python
# 1. Set up GitHub token in secrets (🔑 sidebar)
#    Name: GITHUB_TOKEN, Value: your_token

# 2. Run first cell (package installation)
# 3. RESTART RUNTIME when prompted
# 4. Run second cell (repository setup)
# 5. Continue with experiment
```

## 🔧 Common Issues & Quick Fixes

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

**Problem**: Poetry install fails
**Solution**:
```bash
# Update Poetry and pip
pip install --upgrade pip poetry poetry-core

# Clear cache
poetry cache clear --all pypi

# Alternative: Direct pip install
pip install torch torchvision torchaudio faiss-cpu typer click pydantic
pip install -e .
```

### GitHub Token Setup (Colab)

**Problem**: "GitHub token not found"
**Solution**:
1. Go to https://github.com/settings/tokens
2. Generate token with `repo` scope
3. In Colab: Click 🔑 → Add secret → Name: `GITHUB_TOKEN`
4. Paste token value → Save

### Environment Path Issues

**Problem**: Import errors in local development
**Solution**:
```bash
# Add to shell profile (.zshrc, .bashrc)
export PYTHONPATH="${PYTHONPATH}:/path/to/InsightSpike-AI"

# OR in Python
import sys
sys.path.append('/path/to/InsightSpike-AI')
```

## 🎯 Local ↔ Colab Best Practices

### Development Workflow

1. **Local Development**:
   - Use Poetry for dependency management
   - Test with `python scripts/pre_push_validation.py`
   - Commit to main branch

2. **Colab Execution**:
   - Notebooks pull latest code automatically
   - Pinned dependency versions for stability
   - Results saved to experiment directories

### Version Synchronization

**Critical Versions**:
- Python: 3.10+
- PyTorch: 2.2.2
- NumPy: 1.26.4
- sentence-transformers: 2.7.0

**Check Compatibility**:
```python
# Save this snippet for environment verification
import sys, torch, numpy as np
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

## 🚨 Emergency Recovery

### Complete Environment Reset

```bash
# Local
rm -rf .venv/
poetry install --no-cache

# Colab
# Runtime → Factory reset runtime
# Re-run setup cells
```

### Fallback Installation

```bash
# Minimal working setup
pip install torch==2.2.2
pip install numpy==1.26.4  
pip install faiss-cpu typer click pydantic
pip install -e .
```

### Data Corruption Recovery

```bash
# Restore clean data state
python scripts/utilities/restore_clean_data.py

# Verify data integrity
python scripts/validation/data_integrity_check.py
```

## 📞 Getting Help

1. **Check Logs**: `monitoring/production_monitor.py`
2. **Run Diagnostics**: `scripts/debugging/debug_experiment_state.py`
3. **GitHub Issues**: Create issue with error logs
4. **Documentation**: See `docs/` and experiment READMEs

## ✅ Success Indicators

**Local Setup Complete**:
```bash
✅ poetry install succeeded
✅ python -c "from src.insightspike..." works
✅ scripts/pre_push_validation.py passes
✅ CLI warning "InsightSpike-AI not available" gone
```

**Colab Setup Complete**:
```python
✅ Runtime restarted after package installation
✅ GitHub token found in secrets
✅ Repository cloned successfully
✅ Import experiment modules works
✅ Environment verification shows correct versions
```

---
*Last updated: January 2025*
