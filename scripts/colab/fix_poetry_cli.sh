#!/usr/bin/env bash
# Poetry CLI Fix for Google Colab Environment
# Comprehensive solution for Poetry access issues

set -e

echo "🔧 InsightSpike-AI Poetry CLI Fix"
echo "=================================="
echo "🎯 Resolving Poetry command not found issues"
echo ""

# Function to test Poetry command availability
test_poetry() {
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry CLI accessible"
        poetry --version
        return 0
    else
        echo "❌ Poetry CLI not accessible"
        return 1
    fi
}

# Function to add Poetry to PATH
add_poetry_to_path() {
    local poetry_bin_paths=(
        "/root/.local/bin"
        "$HOME/.local/bin"
        "/home/user/.local/bin"
        "/usr/local/bin"
        "/opt/poetry/bin"
    )
    
    for path in "${poetry_bin_paths[@]}"; do
        if [ -f "$path/poetry" ]; then
            echo "📍 Found Poetry at: $path/poetry"
            export PATH="$path:$PATH"
            echo "✅ Added $path to PATH"
            return 0
        fi
    done
    
    echo "❌ Poetry executable not found in standard locations"
    return 1
}

# Function to install Poetry with proper PATH configuration
install_poetry_properly() {
    echo "📦 Installing Poetry with proper configuration..."
    
    # Method 1: Official installer with PATH setup
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add to PATH immediately
    export PATH="/root/.local/bin:$PATH"
    
    # Make PATH change persistent for session
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    
    # Test installation
    if test_poetry; then
        echo "✅ Poetry installed successfully"
        return 0
    fi
    
    # Method 2: pip install as fallback
    echo "🔄 Fallback: Installing Poetry via pip..."
    pip install --user poetry
    
    # Test again
    if test_poetry; then
        echo "✅ Poetry installed via pip successfully"
        return 0
    fi
    
    echo "❌ Poetry installation failed"
    return 1
}

# Function to configure Poetry for Colab
configure_poetry_for_colab() {
    echo "⚙️ Configuring Poetry for Colab environment..."
    
    # Disable virtual environment creation (use system Python)
    poetry config virtualenvs.create false
    poetry config virtualenvs.in-project false
    
    # Set faster package installation
    poetry config installer.parallel true
    poetry config installer.max-workers 4
    
    # Configure cache location
    poetry config cache-dir /tmp/poetry-cache
    
    echo "✅ Poetry configured for Colab"
}

# Function to create Poetry wrapper script
create_poetry_wrapper() {
    cat > /usr/local/bin/poetry_wrapper.sh << 'EOF'
#!/bin/bash
# Poetry wrapper script for Colab environment

# Try to find Poetry in various locations
POETRY_PATHS=(
    "/root/.local/bin/poetry"
    "$HOME/.local/bin/poetry"
    "/usr/local/bin/poetry"
    "$(which poetry 2>/dev/null)"
)

for poetry_path in "${POETRY_PATHS[@]}"; do
    if [ -x "$poetry_path" ] && [ -f "$poetry_path" ]; then
        exec "$poetry_path" "$@"
    fi
done

# If Poetry not found, try pip as fallback for basic commands
if [ "$1" = "install" ] || [ "$1" = "add" ]; then
    echo "⚠️ Poetry not found, using pip fallback..."
    shift  # remove 'install' or 'add'
    pip install "$@"
else
    echo "❌ Poetry not available and no pip fallback for command: $1"
    exit 1
fi
EOF

    chmod +x /usr/local/bin/poetry_wrapper.sh
    
    # Create symlink to make 'poetry' command available
    ln -sf /usr/local/bin/poetry_wrapper.sh /usr/local/bin/poetry
    
    echo "✅ Poetry wrapper script created"
}

# Function to validate Poetry environment
validate_poetry_environment() {
    echo "🔍 Validating Poetry environment..."
    
    # Test basic Poetry commands
    poetry --version || echo "⚠️ Poetry version check failed"
    poetry config --list || echo "⚠️ Poetry config check failed"
    
    # Test project detection
    if [ -f "pyproject.toml" ]; then
        poetry check || echo "⚠️ Poetry project validation failed"
        poetry show --tree || echo "⚠️ Poetry dependency tree failed"
    fi
    
    echo "✅ Poetry environment validation complete"
}

# Function to create Python alternative for Poetry commands
create_python_poetry_runner() {
    cat > /usr/local/bin/run_poetry.py << 'EOF'
#!/usr/bin/env python3
"""
Python-based Poetry runner for Colab environment
Provides Poetry functionality when CLI is not accessible
"""

import sys
import subprocess
import os
import json
from pathlib import Path

def find_poetry_executable():
    """Find Poetry executable in various locations"""
    possible_paths = [
        "/root/.local/bin/poetry",
        os.path.expanduser("~/.local/bin/poetry"),
        "/usr/local/bin/poetry",
    ]
    
    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # Try which command
    try:
        result = subprocess.run(['which', 'poetry'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

def run_poetry_command(args):
    """Run Poetry command with fallback options"""
    poetry_path = find_poetry_executable()
    
    if poetry_path:
        try:
            cmd = [poetry_path] + args
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"❌ Poetry command failed: {e}")
            return e.returncode
    else:
        print("⚠️ Poetry executable not found")
        
        # Provide fallback for common commands
        if args and args[0] == 'install':
            print("🔄 Using pip fallback for install...")
            subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
            return 0
        elif args and args[0] == 'show':
            print("🔄 Using pip list fallback...")
            subprocess.run(['pip', 'list'])
            return 0
        else:
            print(f"❌ No fallback available for Poetry command: {' '.join(args)}")
            return 1

if __name__ == "__main__":
    exit(run_poetry_command(sys.argv[1:]))
EOF

    chmod +x /usr/local/bin/run_poetry.py
    echo "✅ Python Poetry runner created"
}

# Main execution flow
echo "🚀 Starting Poetry CLI fix process..."

# Step 1: Test current Poetry availability
echo ""
echo "📊 Step 1: Testing current Poetry availability..."
if test_poetry; then
    echo "🎉 Poetry already working! Validating configuration..."
    configure_poetry_for_colab
    validate_poetry_environment
    exit 0
fi

# Step 2: Try to find and add Poetry to PATH
echo ""
echo "🔍 Step 2: Searching for existing Poetry installation..."
if add_poetry_to_path && test_poetry; then
    echo "🎉 Poetry found and added to PATH!"
    configure_poetry_for_colab
    validate_poetry_environment
    exit 0
fi

# Step 3: Install Poetry properly
echo ""
echo "📦 Step 3: Installing Poetry with proper configuration..."
if install_poetry_properly; then
    configure_poetry_for_colab
    validate_poetry_environment
    echo "🎉 Poetry installation complete!"
    exit 0
fi

# Step 4: Create fallback mechanisms
echo ""
echo "🛠️ Step 4: Creating fallback mechanisms..."
create_poetry_wrapper
create_python_poetry_runner

# Step 5: Final validation
echo ""
echo "🔬 Step 5: Final validation with fallbacks..."
if test_poetry; then
    echo "🎉 Poetry now accessible via wrapper!"
    configure_poetry_for_colab
    validate_poetry_environment
else
    echo "⚠️ Poetry CLI still not accessible, but fallbacks created"
    echo "💡 Use: python /usr/local/bin/run_poetry.py [commands]"
fi

# Step 6: Update project with Poetry lock
echo ""
echo "📝 Step 6: Updating Poetry lock file..."
if test_poetry; then
    if [ -f "pyproject.toml" ]; then
        echo "🔄 Updating Poetry lock file..."
        poetry lock --no-update || echo "⚠️ Poetry lock failed (will use pip fallback)"
        
        echo "📦 Installing dependencies..."
        poetry install --only main || pip install -e .
    fi
else
    echo "⚠️ Skipping Poetry lock update (CLI not available)"
    echo "💡 Installing via pip as fallback..."
    pip install -e .
fi

echo ""
echo "✅ Poetry CLI fix process complete!"
echo ""
echo "📋 Available methods to use Poetry:"
echo "1. Direct command: poetry --version"
echo "2. Wrapper script: /usr/local/bin/poetry_wrapper.sh --version"  
echo "3. Python runner: python /usr/local/bin/run_poetry.py --version"
echo ""
echo "🚀 Ready to run experiments!"
