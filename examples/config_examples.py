#!/usr/bin/env python3
"""
Configuration Examples for InsightSpike
======================================

Shows how to use the simplified configuration system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.config import (
    SimpleConfig, ConfigPresets, ConfigManager,
    get_config, create_config_file
)
from insightspike.core.agents.main_agent import MainAgent


def example_basic_usage():
    """Basic configuration usage"""
    print("=== Basic Configuration Usage ===\n")
    
    # 1. Use default configuration
    config = SimpleConfig()
    print(f"Default mode: {config.mode}")
    print(f"Safe mode: {config.safe_mode}")
    print(f"LLM model: {config.llm_model}")
    print()


def example_presets():
    """Using configuration presets"""
    print("=== Configuration Presets ===\n")
    
    # Development preset - fast, safe, debug-friendly
    dev_config = ConfigPresets.development()
    print("Development preset:")
    print(f"  Safe mode: {dev_config.safe_mode}")
    print(f"  Debug: {dev_config.debug}")
    print(f"  Spike sensitivity: {dev_config.spike_sensitivity}")
    print()
    
    # Experiment preset - real LLM, moderate performance
    exp_config = ConfigPresets.experiment()
    print("Experiment preset:")
    print(f"  Safe mode: {exp_config.safe_mode}")
    print(f"  Max tokens: {exp_config.max_tokens}")
    print(f"  Spike sensitivity: {exp_config.spike_sensitivity}")
    print()
    
    # Production preset - optimized for performance
    prod_config = ConfigPresets.production()
    print("Production preset:")
    print(f"  Mode: {prod_config.mode}")
    print(f"  Batch size: {prod_config.batch_size}")
    print(f"  Advanced metrics: {prod_config.use_advanced_metrics}")
    print()


def example_custom_config():
    """Creating custom configuration"""
    print("=== Custom Configuration ===\n")
    
    # Create custom configuration
    config = SimpleConfig(
        mode="cpu",
        safe_mode=False,
        max_tokens=512,
        temperature=0.5,
        spike_sensitivity=1.5,  # More sensitive spike detection
        batch_size=48,
        debug=True
    )
    
    print(f"Custom configuration:")
    print(f"  Mode: {config.mode}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Spike GED threshold: {config.spike_ged_threshold}")  # 0.5 * 1.5 = 0.75
    print()


def example_config_manager():
    """Using ConfigManager for dynamic configuration"""
    print("=== ConfigManager Usage ===\n")
    
    # Create manager with experiment preset
    manager = ConfigManager(ConfigPresets.experiment())
    
    # Get values
    print(f"Current mode: {manager.get('mode')}")
    print(f"Safe mode: {manager.get('safe_mode')}")
    
    # Update values
    manager.set('debug', True)
    manager.update(
        max_tokens=1024,
        temperature=0.7
    )
    
    print(f"\nAfter updates:")
    print(f"  Debug: {manager.config.debug}")
    print(f"  Max tokens: {manager.config.max_tokens}")
    print(f"  Temperature: {manager.config.temperature}")
    print()


def example_save_load():
    """Saving and loading configuration"""
    print("=== Save/Load Configuration ===\n")
    
    # Create configuration
    config = ConfigPresets.experiment()
    
    # Save to file
    config_path = Path("experiment_config.json")
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load from file
    loaded_config = SimpleConfig.load(config_path)
    print(f"\nLoaded configuration:")
    print(f"  Mode: {loaded_config.mode}")
    print(f"  Safe mode: {loaded_config.safe_mode}")
    print(f"  Spike sensitivity: {loaded_config.spike_sensitivity}")
    
    # Clean up
    config_path.unlink()
    print()


def example_env_overrides():
    """Environment variable overrides"""
    print("=== Environment Variable Overrides ===\n")
    
    import os
    
    # Set environment variables
    os.environ["INSIGHTSPIKE_DEBUG"] = "true"
    os.environ["INSIGHTSPIKE_MAX_TOKENS"] = "1024"
    os.environ["INSIGHTSPIKE_SPIKE_SENSITIVITY"] = "2.0"
    
    # Create manager - env vars will override defaults
    manager = ConfigManager()
    
    print("Configuration with env overrides:")
    print(f"  Debug: {manager.config.debug}")
    print(f"  Max tokens: {manager.config.max_tokens}")
    print(f"  Spike sensitivity: {manager.config.spike_sensitivity}")
    
    # Clean up
    for key in ["INSIGHTSPIKE_DEBUG", "INSIGHTSPIKE_MAX_TOKENS", "INSIGHTSPIKE_SPIKE_SENSITIVITY"]:
        del os.environ[key]
    print()


def example_with_agent():
    """Using configuration with MainAgent"""
    print("=== Using Configuration with Agent ===\n")
    
    # Use ConfigManager for easy management
    manager = ConfigManager(ConfigPresets.development())
    
    # Convert to legacy format for MainAgent
    legacy_config = manager.to_legacy_config()
    
    # Create agent
    agent = MainAgent(config=legacy_config)
    agent.initialize()
    
    print("Agent initialized with configuration:")
    print(f"  LLM provider: {legacy_config.llm.provider}")
    print(f"  Safe mode: {legacy_config.llm.safe_mode}")
    print(f"  Spike thresholds: GED={legacy_config.spike.spike_ged}, IG={legacy_config.spike.spike_ig}")
    
    # Add some episodes
    episodes = [
        "システムAは独立して動作する。",
        "システムBも独立して動作する。",
        "AとBを統合すると新しい性質が生まれる。"
    ]
    
    for episode in episodes:
        result = agent.add_episode_with_graph_update(text=episode)
        if result.get("graph_analysis", {}).get("spike_detected"):
            print(f"\n🎯 Spike detected: {episode}")
    print()


def example_quick_start():
    """Quick start example"""
    print("=== Quick Start ===\n")
    
    # Method 1: Use preset directly
    config = get_config("experiment")
    print(f"1. Got experiment config: safe_mode={config.safe_mode}")
    
    # Method 2: Create config file for editing
    create_config_file("my_config.json", "development")
    print(f"\n2. Created config file: my_config.json")
    
    # Method 3: One-liner with environment override
    import os
    os.environ["INSIGHTSPIKE_SAFE_MODE"] = "false"
    manager = ConfigManager(get_config("development"))
    print(f"\n3. Development config with override: safe_mode={manager.config.safe_mode}")
    
    # Clean up
    Path("my_config.json").unlink()
    del os.environ["INSIGHTSPIKE_SAFE_MODE"]


def main():
    """Run all examples"""
    examples = [
        example_basic_usage,
        example_presets,
        example_custom_config,
        example_config_manager,
        example_save_load,
        example_env_overrides,
        example_with_agent,
        example_quick_start
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()