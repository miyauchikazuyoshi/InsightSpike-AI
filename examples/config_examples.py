#!/usr/bin/env python3
"""
Configuration Examples for InsightSpike
======================================

Shows how to use the new Pydantic-based configuration system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.config.models import InsightSpikeConfig, LLMConfig, MemoryConfig
from insightspike.config.presets import ConfigPresets
from insightspike.config.loader import ConfigLoader
from insightspike.core.base.datastore import DataStore


def example_basic_usage():
    """Basic Pydantic configuration usage"""
    print("=== Basic Pydantic Configuration Usage ===\n")
    
    # Create default configuration
    config = InsightSpikeConfig()
    print(f"Default environment: {config.environment}")
    print(f"LLM provider: {config.llm.provider}")
    print(f"LLM model: {config.llm.model}")
    print(f"Embedding model: {config.embedding.model_name}")
    print()


def example_presets():
    """Using configuration presets"""
    print("=== Configuration Presets ===\n")
    
    # Development preset
    dev_config = ConfigPresets.development()
    print("Development preset:")
    print(f"  Environment: {dev_config.environment}")
    print(f"  LLM provider: {dev_config.llm.provider}")
    print(f"  Debug logging: {dev_config.logging.level}")
    print()
    
    # Production preset
    prod_config = ConfigPresets.production()
    print("Production preset:")
    print(f"  Environment: {prod_config.environment}")
    print(f"  LLM provider: {prod_config.llm.provider}")
    print(f"  Max tokens: {prod_config.llm.max_tokens}")
    print()
    
    # Research preset
    research_config = ConfigPresets.research()
    print("Research preset:")
    print(f"  Environment: {research_config.environment}")
    print(f"  Enable monitoring: {research_config.monitoring.enabled}")
    print(f"  Performance tracking: {research_config.monitoring.performance_tracking}")
    print()


def example_custom_config():
    """Creating custom configuration"""
    print("=== Custom Configuration ===\n")
    
    # Create custom configuration
    config = InsightSpikeConfig(
        environment="custom",
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=1024,
            api_key="sk-..."  # Would use env var in practice
        ),
        memory=MemoryConfig(
            max_episodes=2000,
            max_retrieved_docs=15,
            similarity_threshold=0.8
        )
    )
    
    print(f"Custom configuration:")
    print(f"  Environment: {config.environment}")
    print(f"  LLM model: {config.llm.model}")
    print(f"  Max episodes: {config.memory.max_episodes}")
    print(f"  Similarity threshold: {config.memory.similarity_threshold}")
    print()


def example_config_loader():
    """Using ConfigLoader for file-based configuration"""
    print("=== ConfigLoader Usage ===\n")
    
    # Create a config loader
    loader = ConfigLoader()
    
    # Example: Create a temporary config file
    config_data = {
        "environment": "experiment",
        "llm": {
            "provider": "local",
            "model": "distilgpt2",
            "temperature": 0.7
        },
        "graph": {
            "spike_ged_threshold": -0.4,
            "spike_ig_threshold": 0.25
        }
    }
    
    import json
    config_path = Path("experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Load from file
    config = loader.load_from_file(config_path)
    print(f"Loaded configuration from {config_path}:")
    print(f"  Environment: {config.environment}")
    print(f"  LLM model: {config.llm.model}")
    print(f"  Spike thresholds: GED={config.graph.spike_ged_threshold}, IG={config.graph.spike_ig_threshold}")
    
    # Clean up
    config_path.unlink()
    print()


def example_env_overrides():
    """Environment variable overrides"""
    print("=== Environment Variable Overrides ===\n")
    
    # Set environment variables
    os.environ["INSIGHTSPIKE_ENVIRONMENT"] = "testing"
    os.environ["INSIGHTSPIKE_LLM__PROVIDER"] = "anthropic"
    os.environ["INSIGHTSPIKE_LLM__MODEL"] = "claude-2"
    os.environ["INSIGHTSPIKE_LLM__TEMPERATURE"] = "0.8"
    
    # Load config with env overrides
    loader = ConfigLoader()
    config = loader.load_config()
    
    print("Configuration with env overrides:")
    print(f"  Environment: {config.environment}")
    print(f"  LLM provider: {config.llm.provider}")
    print(f"  LLM model: {config.llm.model}")
    print(f"  Temperature: {config.llm.temperature}")
    
    # Clean up
    for key in list(os.environ.keys()):
        if key.startswith("INSIGHTSPIKE_"):
            del os.environ[key]
    print()


def example_validation():
    """Configuration validation examples"""
    print("=== Configuration Validation ===\n")
    
    # Valid configuration
    try:
        config = InsightSpikeConfig(
            llm=LLMConfig(
                provider="openai",
                temperature=0.7,  # Valid: 0.0-2.0
                max_tokens=500    # Valid: > 0
            )
        )
        print("✓ Valid configuration created successfully")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Invalid configuration examples
    print("\nValidation errors:")
    try:
        config = InsightSpikeConfig(
            llm=LLMConfig(
                provider="invalid_provider",  # Will fail validation
                temperature=3.0  # Out of range
            )
        )
    except Exception as e:
        print(f"✗ Provider validation: {e}")
    
    print()


def example_with_agent():
    """Using Pydantic configuration with MainAgent"""
    print("=== Using Pydantic Configuration with Agent ===\n")
    
    # Create configuration
    config = ConfigPresets.development()
    
    # Create a simple in-memory datastore
    datastore = DataStore()
    
    # Create agent with Pydantic config directly
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent = MainAgent(config=config, datastore=datastore)
    agent.initialize()
    
    print("Agent initialized with Pydantic configuration:")
    print(f"  Environment: {config.environment}")
    print(f"  LLM provider: {config.llm.provider}")
    print(f"  Embedding dimension: {config.embedding.dimension}")
    
    # Process a simple question
    result = agent.process_question("What is InsightSpike?", max_cycles=1)
    print(f"\nProcessed question, success: {result.success}")
    print()


def example_config_migration():
    """Example of migrating from legacy to Pydantic config"""
    print("=== Config Migration Example ===\n")
    
    # Old way (would use ConfigConverter internally)
    print("Old way (with ConfigConverter):")
    print("  config = get_config()  # Returns legacy format")
    print("  agent = MainAgent(config)")
    print()
    
    # New way (direct Pydantic)
    print("New way (direct Pydantic):")
    print("  config = InsightSpikeConfig()  # Or ConfigPresets.development()")
    print("  agent = MainAgent(config)")
    print()
    
    # Show the difference
    pydantic_config = ConfigPresets.development()
    print("Pydantic config access:")
    print(f"  config.llm.provider = {pydantic_config.llm.provider}")
    print(f"  config.embedding.dimension = {pydantic_config.embedding.dimension}")
    print()


def example_serialization():
    """Configuration serialization examples"""
    print("=== Configuration Serialization ===\n")
    
    # Create a config
    config = ConfigPresets.research()
    
    # Serialize to dict
    config_dict = config.dict()
    print("Serialized to dict:")
    print(f"  Keys: {list(config_dict.keys())}")
    
    # Serialize to JSON
    config_json = config.json(indent=2)
    print(f"\nSerialized to JSON (first 200 chars):")
    print(config_json[:200] + "...")
    
    # Load from dict
    new_config = InsightSpikeConfig(**config_dict)
    print(f"\nLoaded from dict: environment={new_config.environment}")
    print()


def main():
    """Run all examples"""
    examples = [
        example_basic_usage,
        example_presets,
        example_custom_config,
        example_config_loader,
        example_env_overrides,
        example_validation,
        example_with_agent,
        example_config_migration,
        example_serialization
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
