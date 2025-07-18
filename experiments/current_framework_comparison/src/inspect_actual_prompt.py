#!/usr/bin/env python3
"""
Inspect what actual prompts are being generated with insight information
"""

import json
from pathlib import Path
import sys
import os

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from insightspike.config import InsightSpikeConfig
from insightspike.core.layers.layer4_prompt_builder import L4PromptBuilder


def test_prompt_builder_output():
    """Test what prompts are actually built"""
    print("🔍 Inspecting Actual Prompt Content")
    
    config = InsightSpikeConfig()
    builder = L4PromptBuilder(config)
    
    # Test 1: Minimal context (no insight)
    print("\n1️⃣ Minimal context (no documents, no insight):")
    context1 = {
        "retrieved_documents": [],
        "graph_analysis": {},
        "reasoning_quality": 0.2
    }
    prompt1 = builder.build_prompt(context1, "What is energy?")
    print(f"Length: {len(prompt1)} chars")
    print("Content preview:")
    print("-" * 50)
    print(prompt1[:500] + "...")
    
    # Test 2: With documents but no insight spike
    print("\n\n2️⃣ With documents, no insight spike:")
    context2 = {
        "retrieved_documents": [
            {
                "text": "Energy is the capacity to do work.",
                "c_value": 0.8,
                "similarity": 0.9
            },
            {
                "text": "Energy can change forms but cannot be created or destroyed.",
                "c_value": 0.7,
                "similarity": 0.85
            }
        ],
        "graph_analysis": {
            "metrics": {
                "delta_ged": -0.05,  # Small change
                "delta_ig": 0.1      # Small change
            },
            "spike_detected": False,
            "reasoning_quality": 0.5
        },
        "reasoning_quality": 0.5
    }
    prompt2 = builder.build_prompt(context2, "What is energy?")
    print(f"Length: {len(prompt2)} chars")
    print("\nGraph metrics section:")
    print("-" * 50)
    # Extract graph metrics section
    if "Current Reasoning State" in prompt2:
        start = prompt2.find("Current Reasoning State")
        end = prompt2.find("## User Question", start)
        print(prompt2[start:end].strip())
    
    # Test 3: With insight spike detected
    print("\n\n3️⃣ With INSIGHT SPIKE detected:")
    context3 = {
        "retrieved_documents": [
            {
                "text": "Energy is the capacity to do work.",
                "c_value": 0.8,
                "similarity": 0.9
            },
            {
                "text": "Information and entropy have a deep mathematical relationship.",
                "c_value": 0.85,
                "similarity": 0.92
            },
            {
                "text": "Maxwell's demon shows the relationship between information and energy.",
                "c_value": 0.9,
                "similarity": 0.95
            }
        ],
        "graph_analysis": {
            "metrics": {
                "delta_ged": -0.25,  # Large negative change (simplification)
                "delta_ig": 0.35     # Large positive change (information gain)
            },
            "conflicts": {
                "total": 0.05
            },
            "spike_detected": True,
            "reasoning_quality": 0.85
        },
        "reasoning_quality": 0.85
    }
    prompt3 = builder.build_prompt(context3, "How does information relate to energy?")
    print(f"Length: {len(prompt3)} chars")
    
    # Show the insight-related parts
    print("\nInsight-related content:")
    print("-" * 50)
    if "Current Reasoning State" in prompt3:
        start = prompt3.find("Current Reasoning State")
        end = prompt3.find("## User Question", start)
        print(prompt3[start:end].strip())
    
    print("\nDocuments provided:")
    print("-" * 50)
    if "Retrieved Context Documents" in prompt3:
        start = prompt3.find("Retrieved Context Documents")
        end = prompt3.find("## Current Reasoning State", start)
        docs_section = prompt3[start:end].strip()
        # Show first 800 chars
        print(docs_section[:800] + "...")
    
    # Test 4: Check simple prompt for comparison
    print("\n\n4️⃣ Simple prompt (for comparison):")
    simple_docs = [doc["text"] for doc in context3["retrieved_documents"]]
    simple_prompt = builder.build_simple_prompt(simple_docs, "How does information relate to energy?")
    print(f"Length: {len(simple_prompt)} chars")
    print("Content:")
    print("-" * 50)
    print(simple_prompt)
    
    print("\n\n💡 Analysis:")
    print("- Full prompt includes graph metrics (ΔGED, ΔIG)")
    print("- Spike detection is highlighted with 🧠 emoji")
    print("- Documents are ranked by confidence")
    print("- But all this complexity might confuse small models")


if __name__ == "__main__":
    test_prompt_builder_output()