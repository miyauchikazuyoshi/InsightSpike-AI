---
status: active
category: llm
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# LLM Provider Selection Improvements

## Current Issues
1. Default to TinyLlama (1.1GB) is too heavy for quick starts
2. Provider selection requires config file editing or complex initialization
3. No clear guidance in main README

## Proposed Solutions

### 1. Update __init__.py to expose quick_start helpers

```python
# In src/insightspike/__init__.py
from .quick_start import create_agent, quick_demo
```

### 2. Add to README.md

```markdown
## Quick Start

```python
from insightspike import create_agent

# Simplest usage (mock provider)
agent = create_agent()
result = agent.process_question("Why is the sky blue?")
print(result.response)

# With OpenAI (set OPENAI_API_KEY first)
agent = create_agent(provider="openai")

# With small local model (CPU-friendly)
agent = create_agent(provider="local", model="google/flan-t5-small")

# Run interactive demo
from insightspike import quick_demo
quick_demo()
```

### 3. Smart defaults based on environment

In L4LLMInterface initialization:

```python
def _auto_select_provider(self) -> str:
    """Auto-select best provider based on environment"""
    
    # Check for API keys
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    
    # Check hardware
    try:
        import torch
        if torch.cuda.is_available():
            return "local"  # Can handle larger models
    except ImportError:
        pass
    
    # CPU-only: use mock or small model
    logger.info("No GPU or API keys found, using mock provider")
    return "mock"
```

### 4. Provider-specific installation extras

In pyproject.toml:

```toml
[tool.poetry.extras]
openai = ["openai>=1.0.0"]
anthropic = ["anthropic>=0.5.0"]
local = ["transformers>=4.30.0", "torch>=2.0.0"]
cpu = ["transformers>=4.30.0", "torch>=2.0.0"]
all = ["openai>=1.0.0", "anthropic>=0.5.0", "transformers>=4.30.0", "torch>=2.0.0"]
```

Usage:
```bash
pip install insightspike-ai[openai]  # Just OpenAI support
pip install insightspike-ai[cpu]     # Optimized for CPU with small models
pip install insightspike-ai[all]     # Everything
```

## Benefits
- Zero-config startup for new users
- Intelligent defaults based on available resources
- Clear upgrade path as users need more features
- Reduced friction for CPU-only users