# InsightSpike-AI Setup Guide

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/InsightSpike-AI.git
cd InsightSpike-AI

# Install with Poetry (recommended)
poetry install

# OR install with pip
pip install -e .
```

## Key Features

1. **NumPy-based Vector Search**: Fast and stable vector similarity search
2. **Flexible LLM Support**: Mock provider for testing, OpenAI/Anthropic for production
3. **DataStore Abstraction**: File-based or in-memory storage
4. **Performance Optimizations**: Message passing with configurable hop limits

## Configuration

### Basic Configuration (config.yaml)

```yaml
# LLM Provider (mock for testing)
llm:
  provider: mock  # Options: mock, openai, anthropic
  model: distilgpt2

# Vector Search
vector_search:
  backend: numpy  # Fast NumPy-based implementation
  optimize: true
  
# Message Passing (optimized)
graph:
  enable_message_passing: false  # Enable for advanced features
  message_passing:
    max_hops: 1  # Limited to 1-hop for performance
```

## Performance Considerations

### Message Passing Optimization (July 2025)

**Problem**: O(N²) complexity causing exponential slowdown
**Solution**: Hop limitation (default 1-hop)

```yaml
# Enable with caution
graph:
  enable_message_passing: true
  message_passing:
    max_hops: 1      # Limit to 1-hop neighbors
    iterations: 2    # Reduced from 3
    alpha: 0.3       # Question influence weight
```

**Performance Impact**:
- Baseline (no message passing): ~0.5s/question
- With 1-hop message passing: ~0.8s/question
- With 2-hop message passing: ~2s/question (grows exponentially)

## Troubleshooting

### Model Download Fails
```bash
# Check your Python environment
python -c "import transformers; print(transformers.__version__)"
python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# Install missing dependencies
pip install transformers sentence-transformers torch
```

### Using Different LLM Providers

```yaml
# For actual LLM responses (requires API key)
llm:
  provider: openai
  model: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}  # From environment

# OR use Anthropic
llm:
  provider: anthropic
  model: claude-3-haiku-20240307
  api_key: ${ANTHROPIC_API_KEY}
```

### Embedding Models

The system uses Sentence Transformers for embeddings:
```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  device: cpu  # or "cuda" for GPU
```