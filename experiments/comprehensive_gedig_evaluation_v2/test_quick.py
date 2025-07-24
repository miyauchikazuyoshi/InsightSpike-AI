#!/usr/bin/env python3
"""Quick test script for v2 experiment."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.insightspike.experiments.utils import ExperimentRunner, run_minimal_test


def test_v2_experiment():
    """Test v2 experiment with different providers."""
    
    print("\n" + "="*60)
    print("V2 Experiment Quick Test")
    print("="*60 + "\n")
    
    # First run minimal test
    print("1. Running minimal test...")
    if not run_minimal_test():
        print("❌ Minimal test failed. Please check the configuration.")
        return
    
    print("\n2. Testing different providers...")
    
    # Test mock provider (fastest)
    print("\n--- Testing Mock Provider ---")
    try:
        results = ExperimentRunner.quick_test(provider='mock', n_questions=3)
        print(f"✓ Mock provider: {len(results)} questions processed")
        for r in results:
            print(f"  - Q: {r['question'][:50]}... | Spike: {r['has_spike']} | Time: {r['processing_time']:.2f}s")
    except Exception as e:
        print(f"✗ Mock provider failed: {e}")
    
    # Test clean provider (no data leakage)
    print("\n--- Testing Clean Provider ---")
    try:
        results = ExperimentRunner.quick_test(provider='clean', n_questions=3)
        print(f"✓ Clean provider: {len(results)} questions processed")
        for r in results:
            print(f"  - Q: {r['question'][:50]}... | Spike: {r['has_spike']} | Time: {r['processing_time']:.2f}s")
    except Exception as e:
        print(f"✗ Clean provider failed: {e}")
    
    # Test local provider with distilgpt2 (faster than TinyLlama)
    print("\n--- Testing Local Provider (distilgpt2) ---")
    print("Note: This will download the model on first run (~82MB)")
    
    from src.insightspike.experiments.utils import create_experiment_config
    config = create_experiment_config('local', 'distilgpt2')
    
    try:
        from src.insightspike.implementations.agents.main_agent import MainAgent
        agent = MainAgent(config)
        
        # Add knowledge
        agent.add_knowledge("The sky appears blue due to Rayleigh scattering of light.")
        
        # Test one question
        import time
        start = time.time()
        result = agent.process_question("Why is the sky blue?")
        elapsed = time.time() - start
        
        if hasattr(result, 'response'):
            print(f"✓ Local provider (distilgpt2): Response in {elapsed:.2f}s")
            print(f"  Response: {result.response[:100]}...")
        else:
            print(f"✓ Local provider (distilgpt2): Response in {elapsed:.2f}s")
            
    except Exception as e:
        print(f"✗ Local provider failed: {e}")
        print("  Tip: Make sure transformers is installed: pip install transformers")
    
    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. To test with TinyLlama (slower, ~2GB download):")
    print("   python test_quick.py --provider local --model tinyllama")
    print("\n2. To run full v2 experiment:")
    print("   python src/run_experiment.py")
    print("\n3. To test with OpenAI:")
    print("   export OPENAI_API_KEY=your_key")
    print("   python test_quick.py --provider openai --model gpt-3.5-turbo")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test for v2 experiment")
    parser.add_argument("--provider", default=None, help="Test specific provider")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions")
    
    args = parser.parse_args()
    
    if args.provider:
        # Test specific provider
        print(f"\nTesting {args.provider} provider with {args.model or 'default model'}...")
        model = args.model
        
        if args.provider == 'local' and not model:
            model = 'distilgpt2'  # Default to faster model
        elif args.provider == 'openai' and not model:
            model = 'gpt-3.5-turbo'
        elif not model:
            model = f'{args.provider}-model'
        
        try:
            results = ExperimentRunner.quick_test(
                provider=args.provider,
                n_questions=args.questions
            )
            print(f"\n✓ Test completed successfully!")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
    else:
        # Run default test suite
        test_v2_experiment()