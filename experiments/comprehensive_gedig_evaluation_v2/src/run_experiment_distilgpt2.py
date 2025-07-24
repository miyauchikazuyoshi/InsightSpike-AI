#!/usr/bin/env python3
"""
Run experiment with DistilGPT2 (smaller model, faster loading).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from run_experiment import ComprehensiveGeDIGExperiment


def main():
    """Run experiment with DistilGPT2."""
    print("=== Running Experiment with DistilGPT2 ===")
    
    # Create experiment
    experiment = ComprehensiveGeDIGExperiment(seed=42)
    
    # Generate reduced question set
    print("\nGenerating 10 test questions...")
    questions = experiment.question_generator.generate_questions(
        n_easy=3, n_medium=5, n_hard=2
    )
    experiment.questions = questions[:10]
    
    # Initialize agent with DistilGPT2
    print("\nInitializing agent with DistilGPT2...")
    
    # Override initialize_agent to use DistilGPT2
    def init_distilgpt2_agent():
        from insightspike import MainAgent
        
        legacy_config = type('Config', (), {
            'graph': type('GraphConfig', (), {
                'similarity_threshold': 0.7,
                'conflict_threshold': 0.5,
                'ged_threshold': 0.3
            })(),
            'embedding': type('EmbeddingConfig', (), {
                'dimension': 768,
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            })(),
            'llm': type('LLMConfig', (), {
                'provider': 'local',
                'model_name': 'distilgpt2'  # Default LocalProvider model
            })(),
            'memory': type('MemoryConfig', (), {
                'max_episodes': 1000,
                'compression_enabled': False
            })(),
            'insight': type('InsightConfig', (), {
                'detection_threshold': 0.5,
                'min_confidence': 0.3
            })()
        })()
        
        agent = MainAgent(legacy_config)
        
        # Load knowledge base
        knowledge_path = experiment.data_dir / "input" / "knowledge_base.json"
        if knowledge_path.exists():
            import json
            with open(knowledge_path, 'r') as f:
                knowledge_data = json.load(f)
            
            for item in knowledge_data.get('associations', [])[:20]:  # Limit to 20
                agent.add_knowledge(item['text'])
            
            print(f"Loaded {min(20, len(knowledge_data.get('associations', [])))} knowledge items")
        
        return agent
    
    experiment.initialize_agent = init_distilgpt2_agent
    
    # Run experiment
    experiment.run_experiment()
    
    print("\n=== DistilGPT2 Experiment Completed ===")


if __name__ == "__main__":
    main()