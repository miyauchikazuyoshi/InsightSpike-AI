#!/bin/bash
# Clean Installation Test Script for InsightSpike-AI

echo "🧪 Testing clean installation of InsightSpike-AI..."
echo "================================================"

# Create temporary directory for testing
TEMP_DIR="/tmp/insightspike_test_$(date +%s)"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "📁 Working directory: $TEMP_DIR"
echo ""

# Test 1: Git clone
echo "1️⃣ Testing git clone..."
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
if [ $? -ne 0 ]; then
    echo "❌ Git clone failed"
    exit 1
fi
echo "✅ Git clone successful"
echo ""

cd InsightSpike-AI

# Test 2: Poetry installation
echo "2️⃣ Testing poetry installation..."
if ! command -v poetry &> /dev/null; then
    echo "⚠️  Poetry not found. Please install poetry first."
    echo "   Visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

poetry install
if [ $? -ne 0 ]; then
    echo "❌ Poetry install failed"
    exit 1
fi
echo "✅ Poetry install successful"
echo ""

# Test 3: Run basic import test
echo "3️⃣ Testing basic imports..."
poetry run python -c "
from src.insightspike.core.system import InsightSpikeSystem
from src.insightspike.core.agents.main_agent import MainAgent
print('✅ Core imports successful')
"
if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi
echo ""

# Test 4: Run minimal unit tests
echo "4️⃣ Running minimal test suite..."
poetry run pytest tests/unit/test_core.py -v
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed"
    exit 1
fi
echo "✅ Unit tests passed"
echo ""

# Test 5: Test CLI commands
echo "5️⃣ Testing CLI commands..."
poetry run spike --help > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ CLI command failed"
    exit 1
fi
echo "✅ CLI commands working"
echo ""

# Test 6: Test model download
echo "6️⃣ Testing model setup..."
poetry run python scripts/setup_models.py
if [ $? -ne 0 ]; then
    echo "❌ Model setup failed"
    exit 1
fi
echo "✅ Model setup successful"
echo ""

# Summary
echo "🎉 All installation tests passed!"
echo "================================="
echo "✅ Git clone"
echo "✅ Poetry install"
echo "✅ Core imports"
echo "✅ Unit tests"
echo "✅ CLI commands"
echo "✅ Model setup"
echo ""
echo "📁 Test directory: $TEMP_DIR"
echo "   (You can safely delete this directory)"

# Cleanup option
echo ""
read -p "🗑️  Delete test directory? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /
    rm -rf "$TEMP_DIR"
    echo "✅ Test directory cleaned up"
fi