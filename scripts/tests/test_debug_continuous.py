"""
Debug version - test with detailed logging
"""

import time
import traceback
from insightspike.implementations.agents.main_agent import MainAgent

# Smaller test set for debugging
test_knowledge = [
    "The sun is a star.",
    "Water freezes at 0°C.",
    "AI learns from data.",
    "DNA stores information.",
    "Earth orbits the sun."
]

test_questions = [
    "What is the sun?",
    "When does water freeze?",
    "How does AI work?",
    "What is DNA?",
    "How does Earth move?"
]

def test_with_debug(config_name, config):
    """Test with detailed debug info"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    try:
        print("1. Initializing agent...")
        agent = MainAgent(config)
        print("   ✓ Agent initialized")
        
        # Add knowledge
        print("\n2. Adding knowledge items...")
        for i, knowledge in enumerate(test_knowledge):
            print(f"   [{i+1}/5] Adding: {knowledge}")
            k_start = time.time()
            agent.add_knowledge(knowledge)
            k_time = time.time() - k_start
            print(f"   ✓ Added in {k_time:.3f}s")
        
        # Process questions
        print("\n3. Processing questions...")
        spike_count = 0
        
        for i, question in enumerate(test_questions):
            print(f"   [{i+1}/5] Processing: {question}")
            q_start = time.time()
            
            try:
                result = agent.process_question(question)
                q_time = time.time() - q_start
                
                has_spike = getattr(result, 'has_spike', False)
                if has_spike:
                    spike_count += 1
                
                print(f"   ✓ Processed in {q_time:.3f}s | Spike: {has_spike}")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                traceback.print_exc()
        
        print(f"\n4. Summary:")
        print(f"   - Spikes detected: {spike_count}/5")
        print(f"   - Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        traceback.print_exc()

def main():
    """Test different configurations"""
    
    # Test 1: Baseline
    test_with_debug("Baseline (all disabled)", {
        'llm': {'provider': 'mock'},
        'graph': {
            'enable_message_passing': False,
            'enable_graph_search': False,
            'use_gnn': False
        }
    })
    
    # Test 2: Message passing with minimal settings
    test_with_debug("Message Passing (minimal)", {
        'llm': {'provider': 'mock'},
        'graph': {
            'enable_message_passing': True,
            'message_passing': {
                'alpha': 0.3,
                'iterations': 1  # Reduced iterations
            },
            'enable_graph_search': False,
            'use_gnn': False
        }
    })
    
    # Test 3: Graph search only
    test_with_debug("Graph Search only", {
        'llm': {'provider': 'mock'},
        'graph': {
            'enable_message_passing': False,
            'enable_graph_search': True,
            'use_gnn': False
        }
    })

if __name__ == "__main__":
    main()