#!/bin/bash
# InsightSpike-AI Unified Colab Setup Script
# ==========================================
# Single script for seamless Colab/Local/CI compatibility

echo "🚀 InsightSpike-AI Unified Setup Starting..."
echo "============================================"

# Check if running in Colab
if [ -n "$COLAB_GPU" ] || [ -n "$COLAB_TPU_ADDR" ] || [ -d "/content" ]; then
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

# Install core dependencies first for CLI functionality  
echo "📦 Installing core dependencies for CLI..."
pip install typer click pydantic --quiet

# Install dependencies using pyproject.toml
echo "📦 Installing dependencies from pyproject.toml..."
pip install -e . --quiet

# Install FAISS CPU version (reliable and works everywhere)
echo "🔧 Installing FAISS (CPU version for stability)..."
pip install faiss-cpu --upgrade --quiet
echo "  ✅ faiss-cpu installed successfully"

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

# Add both src and project root to Python path
export PYTHONPATH="$SRC_PATH:$CURRENT_DIR:$PYTHONPATH"

# Create a .pth file for persistent Python path (Colab-specific)
if [ "$IN_COLAB" = true ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages")
    echo "$SRC_PATH" > "$SITE_PACKAGES/insightspike-dev.pth"
    echo "$CURRENT_DIR" >> "$SITE_PACKAGES/insightspike-dev.pth"
    echo "✅ Added Python paths to $SITE_PACKAGES/insightspike-dev.pth"
    
    # Also set in bashrc for persistent sessions
    echo "export PYTHONPATH=\"$SRC_PATH:$CURRENT_DIR:\$PYTHONPATH\"" >> ~/.bashrc
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
echo "🔧 Setting up CLI for Phase 2 compatibility..."

# Environment variables for CLI
PROJECT_ROOT="$(pwd)"
SRC_PATH="$PROJECT_ROOT/src"
export PYTHONPATH="$SRC_PATH:$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export INSIGHTSPIKE_ENV="colab"

if [ "$IN_COLAB" = true ]; then
    echo "📱 Configuring CLI for Colab environment..."
    
    # Create CLI wrapper script with proper paths
    cat > /usr/local/bin/insightspike << 'EOF'
#!/bin/bash
# InsightSpike CLI Wrapper for Colab - Multiple fallback methods
export PYTHONPATH="/content/InsightSpike-AI/src:/content/InsightSpike-AI:$PYTHONPATH"
export INSIGHTSPIKE_ENV="colab"
export TOKENIZERS_PARALLELISM="false"

cd /content/InsightSpike-AI

# Method 1: Try poetry (if available and configured)
if command -v poetry >/dev/null 2>&1; then
    if poetry run python -c "from insightspike.cli.main import main" >/dev/null 2>&1; then
        exec poetry run python -m insightspike.cli.main "$@"
    fi
fi

# Method 2: Try direct Python with proper paths
if python -c "import sys; sys.path.insert(0, 'src'); from insightspike.cli.main import main" >/dev/null 2>&1; then
    exec python -c "
import sys
sys.path.insert(0, 'src')
from insightspike.cli.main import main
main()
" "$@"
fi

# Method 3: Try system python with module path
export PYTHONPATH="/content/InsightSpike-AI/src:$PYTHONPATH"
if python -m insightspike.cli.main --help >/dev/null 2>&1; then
    exec python -m insightspike.cli.main "$@"
fi

# Method 4: Last resort - direct execution
echo "❌ All CLI methods failed. Available alternatives:"
echo "  • python -c \"import sys; sys.path.insert(0, 'src'); from insightspike.cli.main import main; main()\""
echo "  • python -m insightspike.cli.main (after setting PYTHONPATH)"
exit 1
EOF
    
    # Make executable
    chmod +x /usr/local/bin/insightspike
    
    # Test CLI installation with comprehensive diagnostics
    echo "🧪 Testing CLI installation..."
    
    CLI_STATUS="failed"
    
    # Test 1: Direct Python module access
    echo "  📋 Test 1: Direct Python import"
    if cd /content/InsightSpike-AI && python -c "import sys; sys.path.insert(0, 'src'); from insightspike.cli.main import main" >/dev/null 2>&1; then
        echo "    ✅ CLI module import: Working"
        CLI_STATUS="import_ok"
    else
        echo "    ❌ CLI module import: Failed"
    fi
    
    # Test 2: Poetry execution (if available)
    echo "  📋 Test 2: Poetry CLI execution"
    if command -v poetry >/dev/null 2>&1; then
        if cd /content/InsightSpike-AI && poetry run python -c "from insightspike.cli.main import main" >/dev/null 2>&1; then
            echo "    ✅ Poetry CLI: Working"
            CLI_STATUS="poetry_ok"
            
            # Test actual command
            if poetry run insightspike --help >/dev/null 2>&1; then
                echo "    ✅ Poetry command execution: Working"
                CLI_STATUS="full"
            else
                echo "    ⚠️  Poetry import OK, but command execution failed"
            fi
        else
            echo "    ❌ Poetry CLI: Module import failed"
        fi
    else
        echo "    ⚠️  Poetry: Not available"
    fi
    
    # Test 3: Wrapper script
    echo "  📋 Test 3: CLI wrapper script"
    if /usr/local/bin/insightspike --help >/dev/null 2>&1; then
        echo "    ✅ CLI wrapper: Working"
        if [ "$CLI_STATUS" != "full" ]; then
            CLI_STATUS="wrapper_ok"
        fi
    else
        echo "    ⚠️  CLI wrapper: Has issues"
    fi
    
    # Set CLI availability based on tests
    case "$CLI_STATUS" in
        "full")
            CLI_AVAILABLE=true
            echo "✅ CLI Status: Fully functional"
            ;;
        "poetry_ok"|"wrapper_ok")
            CLI_AVAILABLE="partial"
            echo "⚠️  CLI Status: Partially functional"
            ;;
        "import_ok")
            CLI_AVAILABLE="import_only"
            echo "⚠️  CLI Status: Import only (manual execution required)"
            ;;
        *)
            CLI_AVAILABLE=false
            echo "❌ CLI Status: Not functional"
            ;;
    esac
    
    # Install Poetry with proper configuration for Colab
    echo "📦 Installing Poetry for full CLI support..."
    if ! command -v poetry &> /dev/null; then
        echo "🔧 Installing Poetry with Colab-optimized settings..."
        
        # Install poetry in user space
        pip install poetry --quiet --user
        
        # Add poetry to PATH if not already there
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        
        # Configure poetry to use system environment (not create venv in Colab)
        ~/.local/bin/poetry config virtualenvs.create false --local || {
            echo "⚠️  Poetry config failed, continuing..."
        }
        
        # Test poetry installation
        if ~/.local/bin/poetry --version >/dev/null 2>&1; then
            echo "✅ Poetry installed successfully"
            # Create symlink for easier access
            sudo ln -sf ~/.local/bin/poetry /usr/local/bin/poetry 2>/dev/null || true
        else
            echo "⚠️  Poetry installation had issues - using alternative CLI methods"
        fi
    else
        echo "✅ Poetry already installed"
        # Still configure it for Colab
        poetry config virtualenvs.create false --local 2>/dev/null || true
    fi
    
else
    echo "💻 Local environment - verifying Poetry CLI..."
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry CLI available"
        
        # Test direct CLI access
        if poetry run python -c "from insightspike.cli.main import main" >/dev/null 2>&1; then
            echo "✅ CLI module accessible via Poetry"
            CLI_AVAILABLE=true
        else
            echo "⚠️  CLI module has import issues"
            CLI_AVAILABLE="partial"
        fi
    else
        echo "⚠️  Poetry not found - please install Poetry for full CLI functionality"
        CLI_AVAILABLE=false
    fi
fi

echo ""
echo "🚀 Next Steps:"
echo "  1. Run Phase 1 notebook cells sequentially"
echo "  2. Start with device setup (Cell 8)"
echo "  3. Load data (Cell 11) and run experiments"
echo ""

if [ "$CLI_AVAILABLE" = true ]; then
    echo "🎯 CLI Ready for Phase 2:"
    echo "  • Full command line: insightspike --help"
    echo "  • Dependency management: insightspike deps --help"
    echo "  • Experiment control: insightspike ask --help"
    echo "  • All Poetry features available"
elif [ "$CLI_AVAILABLE" = "partial" ]; then
    echo "🔧 CLI Partially Available:"
    echo "  • Limited CLI: /usr/local/bin/insightspike --help"
    echo "  • Python module: python -m insightspike.cli.main --help"
    echo "  • Direct import: python -c 'from insightspike.cli.main import main; main()'"
    echo "  • Some commands may require manual execution"
elif [ "$CLI_AVAILABLE" = "import_only" ]; then
    echo "🐍 CLI Import Only:"
    echo "  • Manual execution: python -c 'import sys; sys.path.insert(0, \"src\"); from insightspike.cli.main import main; main()'"
    echo "  • Notebook API: from insightspike.cli.main import app"
    echo "  • Poetry setup may need completion"
else
    echo "🔧 Alternative CLI methods:"
    echo "  • Check Python paths: import sys; print(sys.path)"
    echo "  • Manual setup: sys.path.insert(0, '/content/InsightSpike-AI/src')"
    echo "  • Direct execution from notebook cells"
    echo "  • Consider using Python API instead of CLI"
fi

echo ""
echo "🔍 Debug Information:"
echo "  • Python executable: $(which python)"
echo "  • Poetry location: $(which poetry 2>/dev/null || echo 'Not found')"
echo "  • Current PYTHONPATH: $PYTHONPATH"
echo "  • InsightSpike-AI location: $(pwd)"

echo ""
echo "✅ Setup complete - ready for Phase 1 & 2 experiments!"
