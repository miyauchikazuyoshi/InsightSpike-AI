#!/bin/bash

# run_local_app.sh
# One-click launcher for InsightSpike Local Knowledge App

echo "🧠 InsightSpike Local App Launcher"
echo "=================================="

# Check if python/venv is available
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    echo "✅ Using virtual environment (.venv)"
else
    PYTHON="python3"
    echo "⚠️ .venv not found, trying system python3"
fi

# Check for optional UI deps (avoid auto-install unless explicitly requested)
AUTO_INSTALL=${INSIGHTSPIKE_AUTO_INSTALL:-0}
missing=()
if ! $PYTHON -c "import streamlit" 2>/dev/null; then
    missing+=("streamlit")
fi
if ! $PYTHON -c "import pyvis" 2>/dev/null; then
    missing+=("pyvis")
fi
if [ ${#missing[@]} -gt 0 ]; then
    if [ "$AUTO_INSTALL" = "1" ]; then
        echo "Installing missing packages: ${missing[*]}"
        $PYTHON -m pip install "${missing[@]}"
    else
        echo "Missing packages: ${missing[*]}"
        echo "Install with: $PYTHON -m pip install ${missing[*]}"
        echo "Or rerun with INSIGHTSPIKE_AUTO_INSTALL=1"
        exit 1
    fi
fi

# Run the app
echo "🚀 Launching App..."
# Disable usage stats to prevent email registration prompt
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
$PYTHON -m streamlit run apps/knowledge_app.py
