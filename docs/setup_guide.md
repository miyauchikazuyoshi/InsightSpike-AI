# InsightSpike-AI Setup Guide

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/InsightSpike-AI.git
cd InsightSpike-AI

# Install with automatic model setup
make quickstart
```

This will:
1. Install InsightSpike-AI and dependencies
2. Download and cache required models:
   - Sentence Transformer (all-MiniLM-L6-v2) for embeddings
   - TinyLlama-1.1B-Chat for text generation

## Manual Model Setup

If you prefer to set up models separately:

```bash
# Install package first
pip install -e .

# Then download models
python scripts/setup_models.py
```

## Why Pre-download Models?

1. **First-run Experience**: Avoid long delays when first using InsightSpike
2. **Offline Usage**: Models are cached locally for offline use
3. **Consistent Performance**: Ensures the tested models are available
4. **Network Issues**: Prevents timeout issues during experiments

## Model Details

### Sentence Transformer (all-MiniLM-L6-v2)
- **Size**: ~90MB
- **Purpose**: Creating embeddings for semantic search
- **Performance**: Fast and accurate for general text

### TinyLlama-1.1B-Chat
- **Size**: ~1.1GB  
- **Purpose**: Text generation and question answering
- **Performance**: Lightweight but capable for RAG tasks

## Troubleshooting

### Model Download Fails
```bash
# Check your Python environment
python -c "import transformers; print(transformers.__version__)"
python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# Install missing dependencies
pip install transformers sentence-transformers torch
```

### Using Different Models

Edit `src/insightspike/core/config.py`:

```python
@dataclass
class LLMConfig:
    model_name: str = "microsoft/phi-2"  # Alternative model
    # or
    model_name: str = "google/flan-t5-base"  # Even smaller
```

### GPU Support

For GPU acceleration:
```python
@dataclass
class LLMConfig:
    device: str = "cuda"  # or "mps" for Mac M1/M2
    use_gpu: bool = True
```