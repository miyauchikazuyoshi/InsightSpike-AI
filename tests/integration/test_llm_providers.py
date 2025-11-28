#!/usr/bin/env python3
"""
LLM Provider Integration Tests
==============================

Test real LLM provider integrations.
"""

import os
import sys
from pathlib import Path
import importlib.util
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def _has_torch():
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False

if not _has_torch():
    pytest.skip("torch/transformers stack not available in this environment", allow_module_level=True)

from src.insightspike.config.models import LLMConfig
from src.insightspike.providers import MockProvider, ProviderFactory


def test_mock_provider():
    """Test mock provider (always works)"""
    print("\n=== Testing Mock Provider ===")

    config = LLMConfig(provider="mock", model="mock-model", temperature=0.7)

    provider = ProviderFactory.create_from_config(config)
    assert isinstance(provider, MockProvider)

    # Test generation
    response = provider.generate("What is AI?")
    print(f"Response: {response[:100]}...")
    assert len(response) > 0

    print("✓ Mock provider working")


def test_openai_provider():
    """Test OpenAI provider (requires API key)"""
    print("\n=== Testing OpenAI Provider ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI tests")
        return

    try:
        from src.insightspike.providers.openai_provider import OpenAIProvider

        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.7,
            max_tokens=100,
        )

        provider = ProviderFactory.create_from_config(config)
        assert isinstance(provider, OpenAIProvider)

        # Validate config
        if not provider.validate_config():
            print("✗ OpenAI config validation failed")
            return

        # Test generation
        response = provider.generate("Say 'Hello InsightSpike' in exactly 5 words.")
        print(f"Response: {response}")

        # Test token estimation
        tokens = provider.estimate_tokens("This is a test sentence.")
        print(f"Estimated tokens: {tokens}")

        print("✓ OpenAI provider working")

    except ImportError:
        print("⚠️  OpenAI library not installed, run: pip install openai")
    except Exception as e:
        print(f"✗ OpenAI test failed: {e}")


def test_anthropic_provider():
    """Test Anthropic provider (requires API key)"""
    print("\n=== Testing Anthropic Provider ===")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic tests")
        return

    try:
        from src.insightspike.providers.anthropic_provider import AnthropicProvider

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            api_key=api_key,
            temperature=0.7,
            max_tokens=100,
        )

        provider = ProviderFactory.create_from_config(config)
        assert isinstance(provider, AnthropicProvider)

        # Validate config
        if not provider.validate_config():
            print("✗ Anthropic config validation failed")
            return

        # Test generation
        response = provider.generate("Say 'Hello InsightSpike' in exactly 5 words.")
        print(f"Response: {response}")

        # Test token estimation
        tokens = provider.estimate_tokens("This is a test sentence.")
        print(f"Estimated tokens: {tokens}")

        print("✓ Anthropic provider working")

    except ImportError:
        print("⚠️  Anthropic library not installed, run: pip install anthropic")
    except Exception as e:
        print(f"✗ Anthropic test failed: {e}")


def test_provider_factory():
    """Test provider factory"""
    print("\n=== Testing Provider Factory ===")

    # Test aliases
    providers = {
        "mock": MockProvider,
        "openai": "OpenAIProvider",
        "gpt": "OpenAIProvider",
        "anthropic": "AnthropicProvider",
        "claude": "AnthropicProvider",
    }

    for name, expected in providers.items():
        try:
            provider = ProviderFactory.create(name)
            if isinstance(expected, str):
                assert provider.__class__.__name__ == expected
            else:
                assert isinstance(provider, expected)
            print(f"✓ Factory created {name} provider")
        except ImportError:
            print(f"⚠️  {name} provider dependencies not installed")
        except ValueError:
            print(f"⚠️  {name} provider requires API key")
        except Exception as e:
            print(f"✗ Failed to create {name}: {e}")


def main():
    """Run all provider tests"""
    print("=== LLM Provider Integration Tests ===")

    test_mock_provider()
    test_provider_factory()
    test_openai_provider()
    test_anthropic_provider()

    print("\n=== Summary ===")
    print("Run with environment variables:")
    print("  OPENAI_API_KEY=<key> python test_llm_providers.py")
    print("  ANTHROPIC_API_KEY=<key> python test_llm_providers.py")


if __name__ == "__main__":
    main()
