#!/usr/bin/env python3
"""
Run experiment with OpenAI API (if API key is available).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from run_experiment import ComprehensiveGeDIGExperiment


def main():
    """Run experiment with OpenAI."""
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("=== OpenAI API key not found ===")
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("=== Running Experiment with OpenAI GPT-3.5-turbo ===")
    
    # Create experiment
    experiment = ComprehensiveGeDIGExperiment(seed=42)
    
    # Generate reduced question set for cost efficiency
    print("\nGenerating 10 test questions...")
    questions = experiment.question_generator.generate_questions(
        n_easy=3, n_medium=5, n_hard=2
    )
    experiment.questions = questions[:10]
    
    # Initialize agent with OpenAI
    print("\nInitializing agent with OpenAI...")
    
    # Override initialize_agent to use OpenAI
    def init_openai_agent():
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
                'provider': 'openai',
                'model_name': 'gpt-3.5-turbo',
                'api_key': api_key,
                'temperature': 0.7,
                'max_tokens': 150
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
    
    experiment.initialize_agent = init_openai_agent
    
    # Run experiment
    experiment.run_experiment()
    
    print("\n=== OpenAI Experiment Completed ===")


if __name__ == "__main__":
    main()