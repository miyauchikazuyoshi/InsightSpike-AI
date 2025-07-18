#!/bin/bash

# Set up environment for InsightSpike experiment
echo "🚀 Starting Current Framework Comparison Experiment"

# Set environment variables to avoid multiprocessing issues
export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Navigate to experiment directory
cd "$(dirname "$0")"

# Run the experiment
echo "📊 Running experiment..."
python src/run_comparison_experiment.py

echo "✅ Experiment complete!"