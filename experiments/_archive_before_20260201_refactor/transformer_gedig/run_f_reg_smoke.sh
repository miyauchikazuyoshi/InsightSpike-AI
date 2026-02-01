#!/bin/bash
# Quick smoke test for F-regularization experiment
# Runs a minimal experiment to verify the implementation works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "F-Regularization Smoke Test"
echo "=============================================="

# Minimal settings for quick validation
python experiments/transformer_gedig/train_f_regularized.py \
    --alpha-sweep \
    --alphas "0,0.1" \
    --seeds "42" \
    --train-samples 100 \
    --eval-samples 50 \
    --epochs 1 \
    --batch-size 8 \
    --output-dir results/transformer_gedig/f_reg_smoke

echo ""
echo "Smoke test complete! Check results in results/transformer_gedig/f_reg_smoke/"
