#!/usr/bin/env python3
"""
Test How Insights are Included in LLM Prompts
==============================================

Tests the prompt building with insights to ensure they are properly formatted.
"""

from insightspike.config import load_config


def test_prompt_with_insights():
    """Test how prompts are built when insights are included"""
    
    print("=== Testing Prompt Building with Insights ===\n")
    
    # Create agent with mock LLM to see prompts
    config = load_config(preset="development")  # Uses mock provider
    config.processing.enable_insight_registration = True
    config.processing.enable_insight_search = True
    
    # Create LLM provider to access prompt building
    from insightspike.implementations.layers.layer4_llm_interface import get_llm_provider  # internal import inside function
    llm_provider = get_llm_provider(config)
    
    # Create test context with regular docs and insights
    test_context = {
        "retrieved_documents": [
            {
                "text": "Water freezes at 0 degrees Celsius.",
                "similarity": 0.85,
                "c_value": 0.7,
                "is_insight": False,
            },
            {
                "text": "[INSIGHT] Temperature determines the state of water",
                "similarity": 0.9,
                "c_value": 0.8,
                "is_insight": True,
                "insight_id": "test_insight_1"
            },
            {
                "text": "Ice is solid water.",
                "similarity": 0.75,
                "c_value": 0.6,
                "is_insight": False,
            },
            {
                "text": "[INSIGHT] Water transitions between states at specific temperatures",
                "similarity": 0.88,
                "c_value": 0.85,
                "is_insight": True,
                "insight_id": "test_insight_2"
            },
        ],
        "graph_analysis": {
            "spike_detected": True,
            "reasoning_quality": 0.8
        },
        "reasoning_quality": 0.8
    }
    
    question = "What happens to water at different temperatures?"
    
    # Test standard prompt style
    print("1. Standard Prompt Style:")
    print("-" * 50)
    prompt = llm_provider._build_prompt(test_context, question)
    print(prompt)
    print("\n")
    
    # Test detailed prompt style
    print("2. Detailed Prompt Style:")
    print("-" * 50)
    config.llm.prompt_style = "detailed"
    config.llm.include_metadata = True
    llm_provider.config = config.llm
    prompt_detailed = llm_provider._build_prompt(test_context, question)
    print(prompt_detailed)
    print("\n")
    
    # Test simple prompt (for lightweight models)
    print("3. Simple Prompt Style:")
    print("-" * 50)
    prompt_simple = llm_provider._build_simple_prompt(test_context, question)
    print(prompt_simple)
    print("\n")
    
    # Analyze prompt structure
    print("=== Prompt Analysis ===")
    print(f"Standard prompt includes 'Key Insights': {'Key Insights:' in prompt}")
    print(f"Detailed prompt includes 'Previously Discovered Insights': {'Previously Discovered Insights:' in prompt_detailed}")
    print(f"[INSIGHT] prefix removed: {'[INSIGHT]' not in prompt}")
    print(f"Insights separated from regular docs: {prompt.count('Key Insights:') == 1}")


def test_prompt_without_insights():
    """Test prompt building when insights are disabled"""
    
    print("\n=== Testing Prompt Building WITHOUT Insights ===\n")
    
    config = load_config(preset="minimal")  # Insights disabled
    from insightspike.implementations.layers.layer4_llm_interface import get_llm_provider  # internal import inside function
    llm_provider = get_llm_provider(config)
    
    # Context will have no insights since search is disabled
    test_context = {
        "retrieved_documents": [
            {
                "text": "Water freezes at 0 degrees Celsius.",
                "similarity": 0.85,
                "c_value": 0.7,
                "is_insight": False,
            },
            {
                "text": "Ice is solid water.",
                "similarity": 0.75,
                "c_value": 0.6,
                "is_insight": False,
            },
        ],
        "graph_analysis": {
            "spike_detected": False,
            "reasoning_quality": 0.6
        },
        "reasoning_quality": 0.6
    }
    
    question = "What happens to water at different temperatures?"
    
    prompt = llm_provider._build_prompt(test_context, question)
    print("Prompt without insights:")
    print("-" * 50)
    print(prompt)
    print("\n")
    
    print("Analysis:")
    print(f"Contains 'Key Insights': {'Key Insights:' in prompt}")
    print(f"Contains insight text: {any('[INSIGHT]' in doc.get('text', '') for doc in test_context['retrieved_documents'])}")


if __name__ == "__main__":
    test_prompt_with_insights()
    test_prompt_without_insights()
    
    print("\n✓ Prompt building tests completed!")
