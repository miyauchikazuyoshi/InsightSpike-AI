#!/bin/bash
# Run DistilGPT2 RAT experiment with proper cleanup

echo "🚀 Starting DistilGPT2 RAT Experiment..."
echo "=================================="

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the experiment with timeout
timeout 300 poetry run python src/distilgpt2_rat_experiment.py 2>&1 | tee experiment_log.txt

# Check exit status
if [ $? -eq 124 ]; then
    echo "⏰ Experiment timed out after 5 minutes"
else
    echo "✅ Experiment completed"
fi

# Extract summary from log
echo ""
echo "📊 Quick Summary:"
echo "================"
grep -E "(Accuracy:|Spike Rate:|Wrong:|Correct:)" experiment_log.txt | tail -20

echo ""
echo "💾 Check results in: results/distilgpt2_experiments/"