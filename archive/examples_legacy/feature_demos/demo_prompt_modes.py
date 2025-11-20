#!/usr/bin/env python3
"""
Test Different Prompt Modes and Their Behavior
==============================================

Demonstrates how different prompt modes handle documents and insights.
"""

from insightspike.config import load_config


def test_all_prompt_modes():
    """Test how each prompt mode handles the same context"""
    
    # Create test context with many documents and insights
    test_context = {
        "retrieved_documents": [
            # Regular documents
            {"text": "Document 1: Basic information about water.", "similarity": 0.9, "is_insight": False},
            {"text": "Document 2: Water freezes at 0°C.", "similarity": 0.85, "is_insight": False},
            {"text": "Document 3: Ice is solid water.", "similarity": 0.8, "is_insight": False},
            {"text": "Document 4: Steam is water vapor.", "similarity": 0.75, "is_insight": False},
            {"text": "Document 5: Water boils at 100°C.", "similarity": 0.7, "is_insight": False},
            {"text": "Document 6: Water has three states.", "similarity": 0.65, "is_insight": False},
            {"text": "Document 7: Temperature affects water state.", "similarity": 0.6, "is_insight": False},
            {"text": "Document 8: Pressure also affects boiling point.", "similarity": 0.55, "is_insight": False},
            # Insights
            {"text": "[INSIGHT] Temperature is the key factor in water state changes", "similarity": 0.95, "c_value": 0.9, "is_insight": True},
            {"text": "[INSIGHT] Water molecules remain H2O in all states", "similarity": 0.92, "c_value": 0.85, "is_insight": True},
            {"text": "[INSIGHT] Phase transitions occur at specific temperatures", "similarity": 0.90, "c_value": 0.8, "is_insight": True},
        ],
        "graph_analysis": {"spike_detected": True, "reasoning_quality": 0.8},
        "reasoning_quality": 0.8
    }
    
    question = "How does temperature affect water?"
    
    print("=== Prompt Mode Comparison ===\n")
    print(f"Total documents: {len([d for d in test_context['retrieved_documents'] if not d.get('is_insight', False)])}")
    print(f"Total insights: {len([d for d in test_context['retrieved_documents'] if d.get('is_insight', False)])}")
    print(f"Total items: {len(test_context['retrieved_documents'])}\n")
    
    # Test 1: Standard mode
    print("1. STANDARD MODE:")
    print("-" * 60)
    config = load_config(preset="development")
    config.llm.prompt_style = "standard"
    config.llm.max_context_docs = 5  # Limit for prompt building
    from insightspike.implementations.layers.layer4_llm_interface import get_llm_provider  # internal import inside function
    llm = get_llm_provider(config)
    prompt_standard = llm._build_prompt(test_context, question)
    print(prompt_standard[:500] + "..." if len(prompt_standard) > 500 else prompt_standard)
    print(f"\nPrompt length: {len(prompt_standard)} chars")
    print(f"Estimated tokens: {len(prompt_standard) // 4}")
    
    # Test 2: Detailed mode
    print("\n\n2. DETAILED MODE:")
    print("-" * 60)
    config.llm.prompt_style = "detailed"
    config.llm.include_metadata = True
    llm.config = config.llm
    prompt_detailed = llm._build_prompt(test_context, question)
    print(prompt_detailed[:500] + "..." if len(prompt_detailed) > 500 else prompt_detailed)
    print(f"\nPrompt length: {len(prompt_detailed)} chars")
    print(f"Estimated tokens: {len(prompt_detailed) // 4}")
    
    # Test 3: Simple/Minimal mode
    print("\n\n3. SIMPLE MODE:")
    print("-" * 60)
    config.llm.use_simple_prompt = True
    llm.config = config.llm
    prompt_simple = llm._build_simple_prompt(test_context, question)
    print(prompt_simple)
    print(f"\nPrompt length: {len(prompt_simple)} chars")
    print(f"Estimated tokens: {len(prompt_simple) // 4}")
    
    # Analysis
    print("\n\n=== Analysis ===")
    print(f"Standard mode uses first {config.llm.max_context_docs} documents")
    print(f"Simple mode uses first 2 documents + 1 insight")
    print(f"Detailed mode includes relevance scores and quality metrics")
    
    # Show what gets cut off
    print("\n=== Document Truncation in Layer4 ===")
    print("Note: Layer4's max_context_docs setting limits documents AFTER retrieval")
    print("This happens in _build_prompt(), not in memory search")
    print(f"If max_context_docs=5 and we have 11 items, only first 5 are used!")


def test_mode_recommendations():
    """Recommend appropriate modes for different scenarios"""
    
    print("\n\n=== Mode Recommendations ===\n")
    
    print("1. For GPT-4/Claude (large models):")
    print("   - Use 'detailed' mode with metadata")
    print("   - Can handle 10-15 documents + insights")
    print("   - Set max_context_docs = 10")
    
    print("\n2. For GPT-3.5/smaller models:")
    print("   - Use 'standard' mode")
    print("   - Limit to 5-7 documents + insights")
    print("   - Set max_context_docs = 7")
    
    print("\n3. For DistilGPT2/TinyLlama:")
    print("   - Use 'simple' mode")
    print("   - Severely limited context")
    print("   - Handled by _build_simple_prompt()")
    
    print("\n4. Dynamic adjustment should consider:")
    print("   - Model capacity")
    print("   - Prompt mode")
    print("   - Total token budget")


if __name__ == "__main__":
    test_all_prompt_modes()
    test_mode_recommendations()
