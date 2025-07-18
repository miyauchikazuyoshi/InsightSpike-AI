#!/usr/bin/env python3
"""
Run experiment with fixed LLM prompt formatting
"""

import json
from pathlib import Path
import sys
import os
from datetime import datetime
import time

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from insightspike.config import InsightSpikeConfig
from insightspike.core.agents.main_agent import MainAgent


def monkey_patch_llm_provider(agent):
    """Patch the LLM provider instance to use simpler prompts"""
    if hasattr(agent, 'l4_llm') and hasattr(agent.l4_llm, '_format_prompt'):
        original_format = agent.l4_llm._format_prompt
        
        def simple_format(prompt):
            # For DistilGPT2, don't add special tokens
            return prompt
        
        agent.l4_llm._format_prompt = simple_format
        print("✅ Patched LLM format_prompt method")
        return True
    return False


def test_single_question():
    """Test a single question with the fixed LLM"""
    print("🧪 Testing Single Question with Fixed LLM")
    
    # Create config
    config = InsightSpikeConfig()
    config.core.model_name = "distilgpt2"
    config.core.max_tokens = 100  # Give it more tokens
    config.core.temperature = 0.7
    
    # Initialize agent
    print("\n📦 Initializing agent...")
    agent = MainAgent(config=config)
    
    if not agent.initialize():
        print("❌ Failed to initialize agent")
        return None
    
    # Apply patch
    monkey_patch_llm_provider(agent)
    
    # Add knowledge base
    print("\n📝 Adding knowledge base...")
    knowledge_base_path = Path(__file__).parent.parent / "data" / "episodes.json"
    
    if knowledge_base_path.exists():
        with open(knowledge_base_path, 'r') as f:
            episodes = json.load(f)
        
        for ep in episodes[:20]:  # Add first 20 episodes
            agent.l2_memory.store_episode(
                text=ep['content'],
                c_value=ep['metadata'].get('c_value', 0.5)
            )
        print(f"✅ Added {min(20, len(episodes))} episodes")
    else:
        # Fallback episodes
        episodes = [
            "Energy is the capacity to do work.",
            "Energy can change forms but cannot be created or destroyed.",
            "Entropy is a measure of energy degradation.",
            "Information and entropy have a deep mathematical relationship.",
            "Information is defined as the reduction of uncertainty."
        ]
        for ep in episodes:
            agent.l2_memory.store_episode(text=ep, c_value=0.5)
        print(f"✅ Added {len(episodes)} fallback episodes")
    
    # Test question
    question = "What is energy?"
    print(f"\n❓ Processing: {question}")
    
    start_time = time.time()
    result = agent.process_question(
        question,
        max_cycles=1,
        verbose=False
    )
    processing_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"Processing time: {processing_time:.2f}s")
    
    if isinstance(result, dict):
        response = result.get('response', 'No response')
        print(f"\n📝 Response: {response}")
        print(f"\n🧠 Spike detected: {result.get('spike_detected', False)}")
        print(f"📈 Reasoning quality: {result.get('reasoning_quality', 0.0):.3f}")
        
        # Check response quality
        if response and len(response) > 20:
            if not response.startswith("You are") and not response.startswith("<|"):
                print("\n✅ Response looks reasonable!")
                return result
            else:
                print("\n⚠️  Response contains prompt artifacts")
        else:
            print("\n⚠️  Response is too short")
    
    return result


def run_full_experiment():
    """Run the full experiment with all questions"""
    print("🚀 Running Full Experiment with Fixed LLM")
    
    questions = [
        "What is energy?",
        "How does entropy relate to energy degradation?",
        "What is the relationship between information and entropy?",
        "How does observation affect quantum systems?",
        "Can the universe be viewed as a quantum computer?",
        "What is the role of consciousness in observation?"
    ]
    
    # Create config
    config = InsightSpikeConfig()
    config.core.model_name = "distilgpt2"
    config.core.max_tokens = 150
    config.core.temperature = 0.7
    
    # Initialize agent
    print("\n📦 Initializing agent...")
    agent = MainAgent(config=config)
    
    if not agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    # Apply patch
    monkey_patch_llm_provider(agent)
    
    # Load knowledge base
    print("\n📝 Loading knowledge base...")
    knowledge_base_path = Path(__file__).parent.parent / "data" / "episodes.json"
    
    if knowledge_base_path.exists():
        with open(knowledge_base_path, 'r') as f:
            episodes = json.load(f)
        
        for ep in episodes:
            agent.l2_memory.store_episode(
                text=ep['content'],
                c_value=ep['metadata'].get('c_value', 0.5)
            )
        print(f"✅ Loaded {len(episodes)} episodes")
    
    # Process questions
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = agent.process_question(
            question,
            max_cycles=3,
            verbose=False
        )
        processing_time = time.time() - start_time
        
        if isinstance(result, dict):
            result['question'] = question
            result['processing_time'] = processing_time
            results.append(result)
            
            print(f"\n📝 Response: {result.get('response', 'No response')[:200]}...")
            print(f"🧠 Spike: {result.get('spike_detected', False)}")
            print(f"📈 Quality: {result.get('reasoning_quality', 0.0):.3f}")
            print(f"⏱️  Time: {processing_time:.2f}s")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"insightspike_fixed_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'distilgpt2',
            'method': 'InsightSpike (Fixed)',
            'timestamp': timestamp,
            'questions': questions,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Summary
    print("\n📊 Summary:")
    print(f"Total questions: {len(questions)}")
    print(f"Spike detections: {sum(1 for r in results if r.get('spike_detected', False))}")
    print(f"Avg reasoning quality: {sum(r.get('reasoning_quality', 0) for r in results) / len(results):.3f}")
    print(f"Avg processing time: {sum(r.get('processing_time', 0) for r in results) / len(results):.2f}s")


if __name__ == "__main__":
    # First test single question
    print("1️⃣ Testing single question...\n")
    test_result = test_single_question()
    
    if test_result and isinstance(test_result, dict):
        response = test_result.get('response', '')
        if response and len(response) > 20:
            print("\n2️⃣ Single test successful! Running full experiment...\n")
            time.sleep(2)
            run_full_experiment()
        else:
            print("\n❌ Single test failed. Response quality too low.")
    else:
        print("\n❌ Single test failed.")