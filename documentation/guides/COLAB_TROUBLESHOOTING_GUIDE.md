# Google Colab Setup Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. PyTorch Geometric Hanging (Most Common)

**Problem**: Setup script hangs during PyTorch Geometric installation
**Root Cause**: CUDA extension compilation takes too long
**Solutions**:

#### Option A: Use Fast Setup (Recommended)
```bash
!chmod +x scripts/colab/setup_colab_fast.sh
!./scripts/colab/setup_colab_fast.sh
```
- Skips PyTorch Geometric with timeouts
- Core functionality preserved
- Setup time: 3-5 minutes

#### Option B: Use Minimal Setup (Ultra-Fast)
```bash
!chmod +x scripts/colab/setup_colab_minimal.sh
!./scripts/colab/setup_colab_minimal.sh
```
- Essential dependencies only
- No graph neural networks
- Setup time: <60 seconds

#### Option C: Debug Mode
```bash
!chmod +x scripts/colab/setup_colab_debug.sh
!./scripts/colab/setup_colab_debug.sh
```
- Detailed logging
- Identifies specific bottlenecks
- Creates diagnostic log file

### 2. GPU Not Available

**Problem**: `torch.cuda.is_available()` returns False
**Solutions**:
1. Check Runtime settings: Runtime → Change runtime type → GPU
2. Restart runtime if needed
3. Verify with: `!nvidia-smi`

### 3. Memory Issues

**Problem**: Out of memory errors
**Solutions**:
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()

# Check memory usage
!free -h
```

### 4. Import Errors

**Problem**: `ModuleNotFoundError` for InsightSpike modules
**Solutions**:
```python
# Add project to Python path
import sys
sys.path.insert(0, 'src')

# Or use PYTHONPATH in commands
!PYTHONPATH=src python -m insightspike.cli --help
```

### 5. FAISS GPU Issues

**Problem**: FAISS GPU acceleration not working
**Solutions**:
```python
# Test FAISS GPU support
import faiss
print(f"FAISS version: {faiss.__version__}")
print(f"GPU count: {faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 'Unknown'}")

# Fallback to CPU if needed
try:
    # Try GPU first
    import faiss
    gpu_res = faiss.StandardGpuResources()
    print("✅ GPU FAISS available")
except:
    print("⚠️ Using CPU FAISS")
```

## 🔧 Setup Strategy Decision Tree

```
Start
  ├─ Need full PyTorch Geometric? 
  │   ├─ Yes → Try debug setup first
  │   └─ No → Use fast setup ✅
  │
  ├─ Setup hanging >10 minutes?
  │   ├─ Yes → Stop, use fast setup
  │   └─ No → Continue waiting
  │
  ├─ Just need basic testing?
  │   └─ Yes → Use minimal setup ✅
  │
  └─ Need detailed diagnostics?
      └─ Yes → Use debug setup
```

## 📊 Setup Comparison

| Setup Type | Time | PyTorch Geometric | FAISS GPU | HuggingFace | Use Case |
|------------|------|------------------|-----------|-------------|----------|
| **Fast** | 3-5 min | ⚠️ Timeout fallback | ✅ | ✅ | Development & testing |
| **Minimal** | <60 sec | ❌ | ✅ | ✅ | Quick validation |
| **Debug** | 15-20 min | 🔍 Detailed logs | ✅ | ✅ | Troubleshooting |
| **Full** | 8-15 min | ✅ (may hang) | ✅ | ✅ | Production (if stable) |

## 🚀 Quick Validation Script

```python
# Run this after any setup to validate installation
def validate_setup():
    """Comprehensive setup validation"""
    results = {}
    
    # Core ML framework
    try:
        import torch
        results['pytorch'] = f"✅ {torch.__version__} (CUDA: {torch.cuda.is_available()})"
    except ImportError:
        results['pytorch'] = "❌ Not available"
    
    # Graph neural networks
    try:
        import torch_geometric
        results['pyg'] = f"✅ {torch_geometric.__version__}"
    except ImportError:
        results['pyg'] = "⚠️ Not available (OK for fast/minimal setup)"
    
    # Vector search
    try:
        import faiss
        gpu_count = faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0
        results['faiss'] = f"✅ {faiss.__version__} ({gpu_count} GPUs)"
    except ImportError:
        results['faiss'] = "❌ Not available"
    
    # NLP libraries
    try:
        import transformers, datasets
        results['huggingface'] = "✅ Ready"
    except ImportError:
        results['huggingface'] = "❌ Not available"
    
    # Project modules
    try:
        import sys
        sys.path.insert(0, 'src')
        from insightspike.core.config import get_config
        results['insightspike'] = "✅ Ready"
    except ImportError:
        results['insightspike'] = "⚠️ Use PYTHONPATH=src"
    
    # Print results
    print("🔍 Setup Validation Results:")
    for component, status in results.items():
        print(f"   {component}: {status}")
    
    return results

# Run validation
validate_setup()
```

## 🆘 When All Else Fails

### 1. Runtime Reset
```bash
# Nuclear option - restart everything
# Runtime → Restart runtime
# Then re-run setup
```

### 2. Use Colab Pro
- More stable GPU access
- Better compilation performance
- Longer timeout limits

### 3. Container-based Alternative
```python
# Use pre-built Docker image (if available)
# docker run -it insightspike/colab:latest
```

### 4. CPU-only Mode
```bash
# Force CPU mode for development
export CUDA_VISIBLE_DEVICES=""
./scripts/colab/setup_colab_minimal.sh
```

## 📞 Getting Help

### Debug Information to Collect
```bash
# System info
!uname -a
!nvidia-smi
!python --version
!pip list | grep -E "(torch|faiss|transformers)"

# Setup logs
!cat colab_debug_*.log  # if using debug setup

# Error messages
# Copy full error output when reporting issues
```

### Reporting Issues
Include:
1. Setup script used
2. GPU model (`nvidia-smi`)
3. Error message (full traceback)
4. Setup log file (if available)
5. Python version and key package versions

### Community Resources
- GitHub Issues: Report setup problems
- Documentation: Check latest guides
- Discord/Slack: Real-time help (if available)

---

*This guide is updated based on real Colab usage patterns and common issues encountered during InsightSpike-AI setup.*
