# Quick Start API Reference

The `insightspike.quick_start` module provides convenience functions for quickly getting started with InsightSpike-AI without complex configuration.

## Functions

### `create_agent(provider: str = "mock", **kwargs) -> MainAgent`

Creates a ready-to-use InsightSpike agent with minimal configuration.

#### Parameters

- **provider** (str, optional): LLM provider to use. Default: "mock"
  - `"mock"`: Testing provider with predetermined responses
  - `"openai"`: OpenAI API (requires `OPENAI_API_KEY` environment variable)
  - `"anthropic"`: Anthropic API (requires `ANTHROPIC_API_KEY` environment variable)
  - `"local"`: ⚠️ **Currently not implemented** - will default to mock provider
  - `"clean"`: Clean provider for testing

- **\*\*kwargs**: Additional configuration options
  - Use `section__field=value` 形式で任意の Pydantic 設定を安全に上書きできます  
    例: `llm__temperature=0.2`, `processing__max_cycles=3`
  - `model` は `llm__model` のショートカットです
  - `preset="cloud"` のようにプリセットを `kwargs` から明示指定可能です
  - 未知のキーや不正なネストを渡すと `ValueError` が送出されます

#### Returns

- **MainAgent**: Initialized agent ready for use

#### Smart Defaults

- Automatically detects CPU/GPU and adjusts model selection for local providers
- Uses experiment preset configuration for optimal performance
- Handles initialization automatically

#### Examples

```python
from insightspike.public import create_agent

# Simple usage with mock provider
agent = create_agent()
res = agent.process_question("What is geDIG?")
print(res.get("response", getattr(res, "response", "")))

# With OpenAI
agent = create_agent(provider="openai")  # Requires OPENAI_API_KEY env var

# Override nested fields
agent = create_agent(
    provider="mock",
    llm__temperature=0.2,
    processing__max_cycles=3,
)

# With custom model (local provider optional; auto CPU fallback)
# agent = create_agent(provider="local", model="google/flan-t5-small")
```

### `quick_demo() -> None`

Runs a quick demonstration of InsightSpike capabilities.

#### Description

This function:
1. Creates an agent with mock provider
2. Adds sample knowledge items
3. Asks demonstration questions
4. Shows how the agent responds and detects insights

#### Example

```python
from insightspike import quick_demo

quick_demo()
```

Output:
```
=== InsightSpike Quick Demo ===

Creating agent...
Adding knowledge...
  ✓ The Earth orbits around the Sun.
  ✓ Water boils at 100 degrees Celsius at sea level.
  ✓ Photosynthesis converts light energy into chemical energy.

Asking questions...

Q: Why does water boil?
A: Water boils when its vapor pressure equals atmospheric pressure...
  💡 Insight detected!

Q: How do plants get energy?
A: Plants get energy through photosynthesis...

Q: What moves around what in our solar system?
A: The Earth orbits around the Sun...

=== Demo Complete ===
```

## Local App Wrapper (InsightAppWrapper)

Use the Streamlit-friendly wrapper for a minimal chat/ingest workflow:

```python
from insightspike.public import InsightAppWrapper

app = InsightAppWrapper(provider="mock")
app.learn("geDIG is a graph distance signal.")
answer = app.ask("What is geDIG?")
```

## Import Paths

These functions are exported at the package level for convenience:

```python
# Recommended import (Public API)
from insightspike.public import create_agent, quick_demo

# Also available from module directly (internal)
from insightspike.quick_start import create_agent, quick_demo
```

## Notes

- The `create_agent()` function returns a standard `MainAgent` instance
- Examples MUST import from `insightspike.public` at top‑level (CI enforced)
- Set `INSIGHTSPIKE_LITE_MODE=1` to avoid heavy imports in constrained environments
- geDIG computations must go through `algorithms.gedig.selector.compute_gedig` (STRICT guard available)
