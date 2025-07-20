#!/bin/bash
# Complete Poetry setup for InsightSpike

echo "InsightSpike Poetry Setup"
echo "========================"

# 1. Install dependencies
echo "1. Installing dependencies..."
poetry install --no-interaction

# 2. Download models
echo -e "\n2. Pre-downloading models..."
poetry run python scripts/setup_models.py

# 3. Run tests to verify
echo -e "\n3. Running verification tests..."
poetry run python -c "
import torch
import transformers
import sentence_transformers
print('✓ PyTorch version:', torch.__version__)
print('✓ Transformers version:', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
"

# 4. Set environment variables
echo -e "\n4. Setting environment variables..."
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"' >> ~/.bashrc

echo -e "\n✓ Setup complete!"
echo "You can now run experiments with:"
echo "  poetry run python experiments/your_experiment.py"