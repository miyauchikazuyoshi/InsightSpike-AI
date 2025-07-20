#!/usr/bin/env python3
"""
Test script to demonstrate initialization speed improvements.

This script compares initialization times with and without the LLMProviderRegistry cache.
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insightspike.implementations.layers.layer4_llm_interface import (
    LLMProviderRegistry, 
    get_llm_provider,
    LLMConfig,
    LLMProviderType
)
from src.insightspike.implementations.agents.main_agent import MainAgent
from src.insightspike.config.presets import ConfigPresets


def test_without_cache():
    """Test initialization without cache (old behavior)"""
    print("\n=== Testing WITHOUT cache (old behavior) ===")
    
    # Clear any existing cache
    LLMProviderRegistry.clear_cache()
    
    # Time the first initialization
    start_time = time.time()
    
    # Create LLM provider without cache
    config = ConfigPresets.development()
    llm = get_llm_provider(config, use_cache=False)
    
    first_init_time = time.time() - start_time
    print(f"First initialization: {first_init_time:.2f} seconds")
    
    # Time the second initialization (no cache)
    start_time = time.time()
    llm2 = get_llm_provider(config, use_cache=False)
    second_init_time = time.time() - start_time
    print(f"Second initialization: {second_init_time:.2f} seconds")
    
    # Time creating agent
    start_time = time.time()
    agent = MainAgent(config)
    agent.initialize()
    agent_init_time = time.time() - start_time
    print(f"Agent initialization: {agent_init_time:.2f} seconds")
    
    return first_init_time, second_init_time, agent_init_time


def test_with_cache():
    """Test initialization with cache (new behavior)"""
    print("\n=== Testing WITH cache (new behavior) ===")
    
    # Clear cache to start fresh
    LLMProviderRegistry.clear_cache()
    
    # Time the first initialization (will populate cache)
    start_time = time.time()
    config = ConfigPresets.development()
    llm = get_llm_provider(config, use_cache=True)
    first_init_time = time.time() - start_time
    print(f"First initialization (populating cache): {first_init_time:.2f} seconds")
    
    # Time the second initialization (from cache)
    start_time = time.time()
    llm2 = get_llm_provider(config, use_cache=True)
    second_init_time = time.time() - start_time
    print(f"Second initialization (from cache): {second_init_time:.2f} seconds")
    
    # Time creating agent (should reuse cached LLM)
    start_time = time.time()
    agent = MainAgent(config)
    agent.initialize()
    agent_init_time = time.time() - start_time
    print(f"Agent initialization (with cached LLM): {agent_init_time:.2f} seconds")
    
    # Show cached providers
    cached = LLMProviderRegistry.get_cached_providers()
    print(f"\nCached providers: {cached}")
    
    return first_init_time, second_init_time, agent_init_time


def test_pre_warming():
    """Test pre-warming effect"""
    print("\n=== Testing pre-warming simulation ===")
    
    # Clear cache
    LLMProviderRegistry.clear_cache()
    
    # Simulate pre-warming (what happens in __main__.py)
    print("Pre-warming models...")
    start_time = time.time()
    
    # Pre-warm clean provider
    clean_config = LLMConfig(provider=LLMProviderType.CLEAN)
    LLMProviderRegistry.get_instance(clean_config)
    
    # Pre-warm local model (if using distilgpt2)
    local_config = LLMConfig(
        provider=LLMProviderType.LOCAL,
        model_name="distilgpt2",
        device="cpu"
    )
    try:
        LLMProviderRegistry.get_instance(local_config)
    except Exception as e:
        print(f"Could not pre-warm local model: {e}")
    
    pre_warm_time = time.time() - start_time
    print(f"Pre-warming completed in: {pre_warm_time:.2f} seconds")
    
    # Now test actual usage (should be instant)
    start_time = time.time()
    config = ConfigPresets.development()
    agent = MainAgent(config)
    agent.initialize()
    usage_time = time.time() - start_time
    print(f"Agent creation after pre-warming: {usage_time:.2f} seconds")
    
    return pre_warm_time, usage_time


def main():
    """Run all tests and show summary"""
    print("InsightSpike-AI Initialization Speed Test")
    print("=" * 50)
    
    # Test without cache
    no_cache_times = test_without_cache()
    
    # Test with cache
    cache_times = test_with_cache()
    
    # Test pre-warming
    pre_warm_times = test_pre_warming()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print("\nWithout cache:")
    print(f"  - Total time for 2 inits + agent: {sum(no_cache_times):.2f} seconds")
    
    print("\nWith cache:")
    print(f"  - Total time for 2 inits + agent: {sum(cache_times):.2f} seconds")
    print(f"  - Speed improvement: {sum(no_cache_times) / sum(cache_times):.1f}x faster")
    
    print("\nWith pre-warming:")
    print(f"  - Pre-warming time: {pre_warm_times[0]:.2f} seconds (one-time cost)")
    print(f"  - Agent creation time: {pre_warm_times[1]:.2f} seconds")
    print(f"  - Subsequent operations will be ~{no_cache_times[2] / pre_warm_times[1]:.1f}x faster")
    
    print("\nRecommendation:")
    print("The pre-warming approach in __main__.py will make experiments start much faster")
    print("after the initial application startup.")


if __name__ == "__main__":
    main()