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
  - `model` (str): Model name to use with the provider
  - Other provider-specific options

#### Returns

- **MainAgent**: Initialized agent ready for use

#### Smart Defaults

- Automatically detects CPU/GPU and adjusts model selection for local providers
- Uses experiment preset configuration for optimal performance
- Handles initialization automatically

#### Examples

```python
from insightspike import create_agent

# Simple usage with mock provider
agent = create_agent()
result = agent.process_question("What is the meaning of life?")
print(result.response)

# With OpenAI
agent = create_agent(provider="openai")  # Requires OPENAI_API_KEY env var

# With custom model (when local provider is implemented)
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

## Import Paths

These functions are exported at the package level for convenience:

```python
# Recommended import
from insightspike import create_agent, quick_demo

# Also available from the module directly
from insightspike.quick_start import create_agent, quick_demo
```

## Notes

- The `create_agent()` function returns a standard `MainAgent` instance
- For production use with large datasets, consider using `DataStoreMainAgent` directly
- Local provider support is planned but not yet implemented
- The mock provider is useful for testing but doesn't provide real AI responses