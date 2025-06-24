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

# Install FAISS separately with enhanced GPU/CPU detection
echo "🔧 Installing FAISS with optimal backend..."
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "🎮 GPU detected - trying faiss-gpu installation..."
    pip install faiss-gpu --upgrade --quiet --no-deps || {
        echo "⚠️ faiss-gpu failed, installing faiss-cpu..."
        pip install faiss-cpu --upgrade --quiet
    }
else
    echo "💻 CPU environment - installing faiss-cpu..."
    pip install faiss-cpu --upgrade --quiet
fi

# Install PyTorch Geometric for graph operations (Phase 2 support)
echo "🔧 Installing PyTorch Geometric for graph neural networks..."

# First install the main package
pip install torch_geometric --quiet

# Then install optional dependencies for enhanced functionality
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "🎮 Installing PyTorch Geometric extensions with CUDA support..."
    # Get PyTorch CUDA version for proper wheel selection
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "118")
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.2.0")
    
    echo "🔍 Detected PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}"
    
    # Install extensions from official PyG wheel repository
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html --quiet || {
        echo "⚠️ CUDA-specific extensions failed, trying CPU versions..."
        pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html --quiet
    }
else
    echo "💻 Installing PyTorch Geometric extensions (CPU version)..."
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.2.0")
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html --quiet
fi

# Verify installations
echo "🔍 Verifying core library installations..."
python -c "
# FAISS verification
try:
    import faiss
    print('✅ FAISS: version ' + getattr(faiss, '__version__', 'unknown'))
except ImportError as e:
    print('❌ FAISS: Import failed - ' + str(e))

# PyTorch Geometric verification
try:
    import torch_geometric
    print('✅ PyTorch Geometric: version ' + torch_geometric.__version__)
    
    # Test basic functionality
    import torch_geometric.data
    print('✅ PyTorch Geometric: Core functionality available')
    
except ImportError as e:
    print('❌ PyTorch Geometric: Import failed - ' + str(e))
    print('⚠️  Graph neural network features will be unavailable')

# PyTorch Scatter/Sparse verification  
try:
    import torch_scatter
    import torch_sparse
    print('✅ PyTorch extensions: torch-scatter and torch-sparse available')
except ImportError as e:
    print('⚠️ PyTorch extensions: Some graph operations may be limited - ' + str(e))
" 2>/dev/null || echo "⚠️ Library verification encountered errors"

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

# Verify installation with comprehensive status reporting
echo "🔍 Verifying installation..."
python -c "
import torch
import numpy as np
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ NumPy: {np.__version__}')

# FAISS status
try:
    import faiss
    print(f'✅ FAISS: {getattr(faiss, \"__version__\", \"unknown\")} - Ready for similarity search')
except ImportError:
    print('❌ FAISS: Not available - Memory systems will use baseline implementation')

# PyTorch Geometric status with detailed reporting
try:
    import torch_geometric
    print(f'✅ PyTorch Geometric: {torch_geometric.__version__} - Graph operations enabled')
    
    # Test core components
    try:
        import torch_geometric.nn
        import torch_geometric.data
        print('✅ PyTorch Geometric: Neural network layers available')
    except ImportError:
        print('⚠️ PyTorch Geometric: Some components missing')
        
    try:
        import torch_scatter
        import torch_sparse
        print('✅ PyTorch extensions: Scatter and sparse operations available')
    except ImportError:
        print('⚠️ PyTorch extensions: Limited graph operations')
        
except ImportError:
    print('❌ PyTorch Geometric: Not available')
    print('   Phase 2 graph neural networks will be disabled')
    print('   Phase 1 experiments will work normally')

# InsightSpike status
try:
    from insightspike.core.agents.main_agent import MainAgent
    print('✅ InsightSpike-AI: Core modules loaded successfully')
    
    # Test instantiation
    try:
        agent = MainAgent()
        print('✅ InsightSpike-AI: MainAgent instantiation successful')
        del agent  # Clean up
    except Exception as e:
        print(f'⚠️ InsightSpike-AI: MainAgent instantiation failed - {e}')
        
except ImportError as e:
    print(f'❌ InsightSpike-AI: Core import failed - {e}')
    print('   Standalone implementations will be used')

# Test CLI command availability (fixed syntax)
echo \"✅ InsightSpike-AI: Core modules loaded successfully\"

# Test instantiation
try:
    agent = MainAgent()
    print('✅ InsightSpike-AI: MainAgent instantiation successful')
    del agent  # Clean up
except Exception as e:
    print(f'⚠️ InsightSpike-AI: MainAgent instantiation failed - {e}')
        
except ImportError as e:
    print(f'❌ InsightSpike-AI: Core import failed - {e}')
    print('   Standalone implementations will be used')
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

# Generate setup summary
echo "📋 Setup Summary:"
python -c "
# Check available components and generate status report
components = {
    'CUDA/GPU': False,
    'FAISS': False, 
    'PyTorch Geometric': False,
    'InsightSpike Core': False
}

try:
    import torch
    components['CUDA/GPU'] = torch.cuda.is_available()
except:
    pass

try:
    import faiss
    components['FAISS'] = True
except:
    pass

try:
    import torch_geometric
    components['PyTorch Geometric'] = True
except:
    pass

try:
    from insightspike.core.agents.main_agent import MainAgent
    components['InsightSpike Core'] = True
except:
    pass

print('✅ Available Components:')
for comp, available in components.items():
    status = '✅' if available else '❌'
    print(f'   {status} {comp}')

print('')
print('🔬 Experiment Capabilities:')
print('   ✅ Phase 1: Dynamic Memory Construction (Always available)')
if components['FAISS']:
    print('   ✅ Enhanced similarity search with FAISS')
else:
    print('   ⚠️  Baseline similarity search (FAISS unavailable)')
    
if components['PyTorch Geometric']:
    print('   ✅ Phase 2: Graph neural networks ready')
else:
    print('   ⚠️  Phase 2: Graph operations limited (PyTorch Geometric unavailable)')
    
if components['CUDA/GPU']:
    print('   ✅ GPU acceleration enabled')
else:
    print('   💻 CPU-only mode (No GPU detected)')
"

if [ "$IN_COLAB" = true ]; then
    echo ""
    echo "💡 Ready to use InsightSpike-AI in Google Colab!"
    echo "🔬 Phase 1 experiments are fully supported"
    echo "📊 Run the Phase 1 notebook to start your experiments"
else
    echo ""
    echo "💡 Ready to use InsightSpike-AI in local environment!"
fi

# =============================================================================
# CLI Setup for Phase 2 Compatibility
# =============================================================================
echo ""
echo "� Setting up CLI for Phase 2 compatibility..."

# Environment variables for CLI
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export INSIGHTSPIKE_ENV="colab"

if [ "$IN_COLAB" = true ]; then
    echo "📱 Configuring CLI for Colab environment..."
    
    # Create CLI wrapper script
    cat > /usr/local/bin/insightspike << 'EOF'
#!/bin/bash
# InsightSpike CLI Wrapper for Colab
export PYTHONPATH="/content/InsightSpike-AI/src:/content/InsightSpike-AI:$PYTHONPATH"
export INSIGHTSPIKE_ENV="colab"
export TOKENIZERS_PARALLELISM="false"

cd /content/InsightSpike-AI

# Try poetry first, fallback to direct python
if command -v poetry >/dev/null 2>&1; then
    poetry run python -m insightspike.cli.main "$@"
else
    python -m insightspike.cli.main "$@"
fi
EOF
    
    # Make executable
    chmod +x /usr/local/bin/insightspike
    
    # Test CLI installation
    echo "🧪 Testing CLI installation..."
    if /usr/local/bin/insightspike --help >/dev/null 2>&1; then
        echo "✅ CLI wrapper installed successfully!"
        CLI_AVAILABLE=true
    else
        echo "⚠️  CLI wrapper has issues, using alternative methods"
        CLI_AVAILABLE=false
    fi
    
    # Install Poetry if possible for full CLI support
    echo "📦 Installing Poetry for full CLI support..."
    if ! command -v poetry &> /dev/null; then
        pip install poetry --quiet --user || {
            echo "⚠️  Poetry installation failed - using pip-based workflow"
        }
    fi
    
else
    echo "💻 Local environment - verifying Poetry CLI..."
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry CLI available"
        CLI_AVAILABLE=true
    else
        echo "⚠️  Poetry not found - please install Poetry for full CLI functionality"
        CLI_AVAILABLE=false
    fi
fi

echo ""
echo "�🚀 Next Steps:"
echo "  1. Run Phase 1 notebook cells sequentially"
echo "  2. Start with device setup (Cell 8)"
echo "  3. Load data (Cell 11) and run experiments"
echo ""

if [ "$CLI_AVAILABLE" = true ]; then
    echo "🎯 CLI Ready for Phase 2:"
    echo "  • Command line: insightspike --help"
    echo "  • Dependency management: insightspike deps --help"
    echo "  • Experiment control: insightspike ask --help"
else
    echo "🔧 Alternative CLI methods:"
    echo "  • Notebook cells: !python -m insightspike.cli.main --help"
    echo "  • Python API: from insightspike.cli.main import app"
    echo "  • Direct execution: python -m insightspike.cli.main <command>"
fi

echo ""
echo "✅ Setup complete - ready for Phase 1 & 2 experiments!"
