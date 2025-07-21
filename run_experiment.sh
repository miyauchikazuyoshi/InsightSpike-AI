#!/bin/bash
# Run experiment with proper output handling

cd /Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI

echo "Starting English Insight Reproduction Experiment..."
echo "================================================="

# Run with explicit output redirection
poetry run python experiments/english_insight_reproduction/src/quick_test.py 2>&1 | grep -v "Advanced metrics" | grep -v "Requested GED" | grep -v "Requested IG"

echo ""
echo "Experiment completed."