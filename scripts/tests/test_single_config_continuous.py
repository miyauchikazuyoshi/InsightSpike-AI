"""
Test single configuration with 20 continuous data points
"""

import time
import traceback
from insightspike.implementations.agents.main_agent import MainAgent

# Test data - 20 items
test_knowledge = [
    "The sun is a star at the center of our solar system.",
    "Water freezes at 0 degrees Celsius.",
    "Machine learning uses algorithms to learn from data.",
    "DNA contains genetic information.",
    "The Earth orbits around the sun.",
    "Neurons transmit electrical signals in the brain.",
    "Photosynthesis converts light energy to chemical energy.",
    "Gravity pulls objects toward each other.",
    "Electrons orbit around the nucleus of atoms.",
    "Evolution occurs through natural selection.",
    "Quantum mechanics describes behavior at atomic scales.",
    "The speed of light is approximately 300,000 km/s.",
    "Proteins are made of amino acids.",
    "Climate change is caused by greenhouse gases.",
    "Black holes have extremely strong gravitational fields.",
    "Artificial intelligence can recognize patterns.",
    "The human brain has billions of neurons.",
    "Energy cannot be created or destroyed.",
    "Chemical reactions involve electron transfer.",
    "The universe is expanding."
]

test_questions = [
    "What is the sun?",
    "How does water change with temperature?",
    "What is machine learning?",
    "What stores genetic information?",
    "How does Earth move?",
    "How do neurons work?",
    "What is photosynthesis?",
    "What is gravity?",
    "What are atoms made of?",
    "How does evolution work?",
    "What is quantum mechanics?",
    "How fast is light?",
    "What are proteins?",
    "What causes climate change?",
    "What are black holes?",
    "What can AI do?",
    "How complex is the brain?",
    "Can energy be destroyed?",
    "What happens in chemical reactions?",
    "Is the universe changing?"
]

def test_baseline():
    """Test baseline configuration"""
    print("Testing Baseline Configuration (all features disabled)")
    print("="*60)
    
    config = {
        'llm': {'provider': 'mock'},
        'graph': {
            'enable_message_passing': False,
            'enable_graph_search': False,
            'use_gnn': False
        }
    }
    
    try:
        agent = MainAgent(config)
        print("Agent initialized successfully")
        
        # Add knowledge
        print("\nAdding 20 knowledge items...")
        start_time = time.time()
        
        for i, knowledge in enumerate(test_knowledge):
            k_start = time.time()
            agent.add_knowledge(knowledge)
            k_time = time.time() - k_start
            print(f"[{i+1}/20] Added in {k_time:.3f}s: {knowledge[:50]}...")
        
        add_time = time.time() - start_time
        print(f"\nTotal knowledge addition time: {add_time:.2f}s")
        
        # Process questions
        print("\nProcessing 20 questions...")
        q_start_time = time.time()
        spike_count = 0
        
        for i, question in enumerate(test_questions):
            q_start = time.time()
            result = agent.process_question(question)
            q_time = time.time() - q_start
            
            has_spike = getattr(result, 'has_spike', False)
            if has_spike:
                spike_count += 1
            
            print(f"[{i+1}/20] Q: {question[:30]}... | Spike: {has_spike} | Time: {q_time:.3f}s")
        
        q_total_time = time.time() - q_start_time
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Knowledge addition: {add_time:.2f}s")
        print(f"Question processing: {q_total_time:.2f}s")
        print(f"Average per question: {q_total_time/20:.3f}s")
        print(f"Spikes detected: {spike_count}/20")
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_baseline()