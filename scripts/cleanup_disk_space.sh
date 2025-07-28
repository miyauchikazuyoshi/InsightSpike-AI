#!/bin/bash
# Disk space cleanup script for InsightSpike-AI development

echo "=== Disk Space Cleanup Script ==="
echo "This script will help free up disk space"
echo ""

# 1. Claude CLI cache (約1GB)
CLAUDE_CACHE=~/Library/Caches/claude-cli-nodejs
if [ -d "$CLAUDE_CACHE" ]; then
    SIZE=$(du -sh "$CLAUDE_CACHE" | cut -f1)
    echo "Found Claude CLI cache: $SIZE"
    read -p "Delete Claude CLI cache? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CLAUDE_CACHE"
        echo "✓ Deleted Claude CLI cache"
    fi
else
    echo "Claude CLI cache not found"
fi

# 2. mypy cache (約211MB)
MYPY_CACHE=.mypy_cache
if [ -d "$MYPY_CACHE" ]; then
    SIZE=$(du -sh "$MYPY_CACHE" | cut -f1)
    echo ""
    echo "Found mypy cache: $SIZE"
    read -p "Delete mypy cache? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$MYPY_CACHE"
        echo "✓ Deleted mypy cache"
    fi
fi

# 3. pytest cache
PYTEST_CACHE=.pytest_cache
if [ -d "$PYTEST_CACHE" ]; then
    SIZE=$(du -sh "$PYTEST_CACHE" 2>/dev/null | cut -f1)
    echo ""
    echo "Found pytest cache: $SIZE"
    read -p "Delete pytest cache? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PYTEST_CACHE"
        find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
        echo "✓ Deleted pytest caches"
    fi
fi

# 4. Python __pycache__
echo ""
read -p "Delete all Python __pycache__ directories? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    echo "✓ Deleted __pycache__ directories"
fi

# 5. Temporary test files
echo ""
echo "Checking for temporary test files..."
TEMP_FILES=$(find . -name "*.tmp" -o -name "*.temp" -o -name "test_*.txt" -o -name "test_*.json" -o -name "test_*.py" | grep -E "(continuous|baseline|results)" | wc -l)
if [ $TEMP_FILES -gt 0 ]; then
    echo "Found $TEMP_FILES temporary test files"
    read -p "Delete temporary test files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f test_continuous_*.py
        rm -f test_*_continuous.py
        rm -f *_results.txt
        rm -f continuous_test_results.json
        echo "✓ Deleted temporary test files"
    fi
fi

# 6. Clean data/temp and data/cache
echo ""
echo "Cleaning data directories..."
if [ -d "data/temp" ]; then
    rm -rf data/temp/*
    echo "✓ Cleaned data/temp"
fi
if [ -d "data/cache" ]; then
    rm -rf data/cache/*
    echo "✓ Cleaned data/cache"
fi

# 7. Show disk usage summary
echo ""
echo "=== Current disk usage ==="
echo "Project total: $(du -sh . 2>/dev/null | cut -f1)"
echo ""
echo "Largest directories:"
du -sh ./* 2>/dev/null | sort -rh | head -10

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 To prevent future disk usage:"
echo "1. Add .mypy_cache to .gitignore"
echo "2. Run 'rm -rf ~/.Library/Caches/claude-cli-nodejs' periodically"
echo "3. Use 'poetry cache clear' to clean Poetry cache if needed"