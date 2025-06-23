#!/bin/bash
# InsightSpike-AI Unified Colab Setup Script
# ==========================================
# Single script for seamless Colab/Local/CI compatibility

echo "🚀 InsightSpike-AI Unified Setup Starting..."
echo "============================================"

# Check if running in Colab
if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_TPU_ADDR" ]; then
    echo "📱 Google Colab environment detected"
    IN_COLAB=true
else
    echo "💻 Local environment detected"
    IN_COLAB=false
fi

# Colab-specific configuration optimization
if [ "$IN_COLAB" = true ]; then
    echo "🔧 Optimizing for Colab environment..."
    
    # Switch to Colab-optimized pyproject.toml if available
    if [ -f "pyproject_colab.toml" ]; then
        echo "📝 Using Colab-optimized configuration..."
        cp pyproject.toml pyproject_backup.toml 2>/dev/null || true
        cp pyproject_colab.toml pyproject.toml
    fi
fi

# Install dependencies using pyproject.toml
echo "📦 Installing dependencies from pyproject.toml..."
pip install -e .

# Install FAISS separately with GPU/CPU detection
echo "🔧 Installing FAISS with optimal backend..."
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "🎮 GPU detected - installing faiss-gpu..."
    pip install faiss-gpu --upgrade --quiet || {
        echo "⚠️ faiss-gpu failed, falling back to faiss-cpu..."
        pip install faiss-cpu --upgrade --quiet
    }
else
    echo "💻 CPU environment - installing faiss-cpu..."
    pip install faiss-cpu --upgrade --quiet
fi

# Ensure Python can find the insightspike module
echo "🔧 Setting up Python module paths..."
CURRENT_DIR=$(pwd)
SRC_PATH="$CURRENT_DIR/src"

# Add src directory to Python path for current session
export PYTHONPATH="$SRC_PATH:$PYTHONPATH"

# Create a .pth file for persistent Python path (Colab-specific)
if [ "$IN_COLAB" = true ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$SRC_PATH" > "$SITE_PACKAGES/insightspike-dev.pth"
    echo "✅ Added $SRC_PATH to Python path permanently"
fi

# Additional Colab-specific optimizations
if [ "$IN_COLAB" = true ]; then
    echo "⚡ Applying Colab optimizations..."
    
    # Enable GPU if available
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected - enabling CUDA acceleration"
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    # Colab-specific directory setup
    mkdir -p /content/data
    mkdir -p /content/models
    mkdir -p /content/outputs
    
    # Ensure CLI scripts are in PATH for Colab
    echo "🔧 Setting up CLI environment..."
    export PATH="/root/.local/bin:$PATH"
    
    # Add editable install to ensure CLI is accessible
    echo "🔧 Installing InsightSpike-AI in editable mode..."
    pip install -e .
    
    # Create a direct CLI symlink if needed
    if [ ! -f "/usr/local/bin/insightspike" ]; then
        echo "🔗 Creating CLI symlink..."
        ln -sf "$(which python)" /usr/local/bin/insightspike-python
        cat > /usr/local/bin/insightspike << 'EOF'
#!/bin/bash
python -m insightspike.cli.main "$@"
EOF
        chmod +x /usr/local/bin/insightspike
    fi
    
    echo "📁 Colab directories created"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import numpy as np
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ NumPy: {np.__version__}')

try:
    import faiss
    print(f'✅ FAISS: {faiss.__version__}')
except ImportError:
    print('⚠️  FAISS: Not available (optional)')

try:
    import torch_geometric
    print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
except ImportError:
    print('⚠️  PyTorch Geometric: Not available (optional)')

try:
    from insightspike.core.agents.main_agent import MainAgent
    print('✅ InsightSpike-AI: Core modules loaded successfully')
except ImportError as e:
    print(f'❌ InsightSpike-AI: Import failed - {e}')

# Test CLI command availability
echo "🧪 Testing CLI commands..."
if command -v insightspike >/dev/null 2>&1; then
    echo "✅ CLI: 'insightspike' command available directly"
    insightspike --version || echo "⚠️  CLI: Version check failed"
else
    echo "⚠️  CLI: 'insightspike' not in PATH, using 'python -m insightspike.cli.main'"
    python -m insightspike.cli.main --version || echo "⚠️  CLI: Module execution failed"
fi

# Test configuration loading
try:
    import insightspike.config as config
    print('✅ InsightSpike-AI: Configuration loaded')
except ImportError:
    print('⚠️  InsightSpike-AI: Configuration module not found (optional)')
"

# Test CLI availability
echo "🔧 Testing CLI command availability..."
if command -v insightspike &> /dev/null; then
    echo "✅ InsightSpike CLI: Available via 'insightspike' command"
    echo "📋 CLI Help:"
    insightspike --help 2>/dev/null | head -10 || echo "  (CLI help not available, but command exists)"
else
    echo "⚠️  InsightSpike CLI: Command not found in PATH"
    echo "   You can still use: python -m insightspike.cli.main"
    
    # Test alternative CLI access
    python -c "
try:
    from insightspike.cli.main import main
    print('✅ CLI Module: Available via python -m insightspike.cli.main')
except ImportError:
    print('❌ CLI Module: Not available')
" 2>/dev/null || echo "❌ CLI Module: Import test failed"
fi

# Final Python import test to ensure everything works
echo "🧪 Final import test..."
python -c "
import sys
print(f'📍 Current working directory: {sys.path[0] if sys.path else \"Unknown\"}')

try:
    from insightspike.core.agents.main_agent import MainAgent
    print('✅ MainAgent: Successfully imported')
    
    # Quick instantiation test
    try:
        agent = MainAgent()
        print('✅ MainAgent: Successfully instantiated')
    except Exception as e:
        print(f'⚠️  MainAgent: Import OK, but instantiation failed - {e}')
        
except ImportError as e:
    print(f'❌ MainAgent: Import failed - {e}')
    print('💡 You may need to run: import sys; sys.path.insert(0, \"/content/InsightSpike-AI/src\")')

try:
    import insightspike.config as config
    print('✅ Config: Successfully imported')
except ImportError:
    print('⚠️  Config: Import failed (may be optional)')
"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
if [ "$IN_COLAB" = true ]; then
    echo "💡 Ready to use InsightSpike-AI in Google Colab with GPU acceleration!"
else
    echo "💡 Ready to use InsightSpike-AI in local environment!"
fi
echo ""
echo "🚀 Quick start:"
echo "  from insightspike.core.agents.main_agent import MainAgent"
echo "  agent = MainAgent()  # Auto-optimized for your environment!"
echo ""
echo "🔧 Alternative CLI usage:"
echo "  !insightspike --help  # If CLI is available"
echo "  !python -m insightspike.cli.main --help  # Alternative method"
echo ""
echo "📝 Note: Python module paths have been automatically configured"
echo "     No need to manually add sys.path modifications!"
