#!/bin/bash
# Run minimal solution experiment

echo "=========================================="
echo "Minimal Solution Experiment Runner"
echo "=========================================="

# Change to experiment directory
cd "$(dirname "$0")"

# Test with small dataset first
echo ""
echo "1. Running test with minimal dataset..."
echo "=========================================="
poetry run python src/test_minimal_solution.py

if [ $? -eq 0 ]; then
    echo ""
    echo "2. Test successful! Run full experiment? (y/n)"
    read -r response
    
    if [ "$response" = "y" ]; then
        echo ""
        echo "Running full experiment..."
        echo "=========================================="
        poetry run python src/run_minimal_solution_experiment.py \
            --config experiment_config_minimal.yaml \
            --output results/metrics
    fi
else
    echo ""
    echo "Test failed. Please check the error messages above."
fi

echo ""
echo "Experiment runner completed."