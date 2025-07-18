#!/usr/bin/env python3
"""
Minimal InsightSpike RAT Experiment
Using the core functionality without complications
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import only essential InsightSpike components
from src.insightspike.core.layers.layer2_memory_manager import L2MemoryManager
from src.insightspike.core.layers.layer4_llm_provider import get_llm_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimalInsightSpikeExperiment:
    """
    Minimal experiment using actual InsightSpike components
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Minimal InsightSpike Experiment...")
        
        # Initialize memory manager
        self.memory = L2MemoryManager(dim=768)
        
        # Initialize LLM provider
        # Use None to get default config, then it will use local model
        self.llm = get_llm_provider(None)
        
        # Load data
        self.definitions_file = Path(__file__).parent.parent / "data" / "input" / "english_definitions.json"
        self.load_data()
        
    def load_data(self):
        """Load definitions and create problems"""
        with open(self.definitions_file, 'r') as f:
            self.definitions = json.load(f)['definitions']
        
        # Simple RAT problems
        self.test_problems = [
            {
                "id": 1,
                "question": "What word associates with COTTAGE, SWISS, and CAKE?",
                "words": ["COTTAGE", "SWISS", "CAKE"],
                "answer": "CHEESE"
            },
            {
                "id": 2,
                "question": "What word connects CREAM, SKATE, and WATER?",
                "words": ["CREAM", "SKATE", "WATER"],
                "answer": "ICE"
            },
            {
                "id": 3,
                "question": "What concept links DUCK, FOLD, and DOLLAR?",
                "words": ["DUCK", "FOLD", "DOLLAR"],
                "answer": "BILL"
            }
        ]
        
    def add_definitions_to_memory(self, words: List[str]):
        """Add definitions for given words to memory"""
        episode_count = 0
        
        for word in words:
            if word in self.definitions:
                for definition in self.definitions[word][:3]:  # First 3 definitions
                    try:
                        # Create embedding (using embedder from memory manager)
                        vector = self.memory.embedder.embed(definition)
                        
                        # Add to memory
                        self.memory.add_episode(
                            vector=vector,
                            text=definition,
                            c_value=0.5
                        )
                        episode_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to add episode: {e}")
        
        logger.info(f"Added {episode_count} episodes to memory")
        return episode_count
    
    def solve_problem(self, problem: Dict) -> Dict:
        """Solve a RAT problem using InsightSpike components"""
        logger.info(f"\n📝 Problem {problem['id']}: {problem['question']}")
        
        # Clear memory for fresh start
        self.memory.episodes = []
        
        # Add definitions to memory
        episodes_added = self.add_definitions_to_memory(problem['words'])
        
        # Retrieve relevant episodes
        try:
            query_vector = self.memory.embedder.embed(problem['question'])
            retrieved = self.memory.retrieve(query_vector, top_k=10)
            
            # Build context from retrieved episodes
            context = "\n".join([ep.text for ep in retrieved])
            
            # Create prompt for LLM
            prompt = f"""Given these word definitions:
{context}

Question: {problem['question']}

Think step by step:
1. What do {problem['words'][0]}, {problem['words'][1]}, and {problem['words'][2]} have in common?
2. What single word connects all three?

Answer with just the connecting word in CAPITALS."""
            
            # Get LLM response
            response = self.llm.complete(prompt)
            
            # Extract answer
            predicted_answer = self._extract_answer(response)
            
            # Check correctness
            is_correct = predicted_answer == problem['answer']
            
            if is_correct:
                logger.info(f"✅ Correct: {predicted_answer}")
            else:
                logger.info(f"❌ Wrong: {predicted_answer} (expected: {problem['answer']})")
            
            return {
                'problem': problem,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response,
                'episodes_used': episodes_added,
                'context_length': len(context)
            }
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                'problem': problem,
                'predicted_answer': "ERROR",
                'is_correct': False,
                'error': str(e)
            }
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from LLM response"""
        # Look for capitalized words
        words = response.split()
        for word in words:
            if word.isupper() and len(word) > 2:
                # Clean punctuation
                clean_word = ''.join(c for c in word if c.isalpha())
                if clean_word:
                    return clean_word
        return "UNKNOWN"
    
    def run_experiment(self):
        """Run the experiment"""
        logger.info("\n🧪 Running Minimal InsightSpike Experiment")
        logger.info("=" * 60)
        
        results = []
        correct = 0
        
        for problem in self.test_problems:
            result = self.solve_problem(problem)
            results.append(result)
            if result['is_correct']:
                correct += 1
        
        # Summary
        accuracy = correct / len(self.test_problems) * 100
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {correct}/{len(self.test_problems)} = {accuracy:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "minimal_experiments"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"minimal_insightspike_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'Minimal InsightSpike RAT',
                    'description': 'Using actual InsightSpike memory and LLM components',
                    'timestamp': timestamp
                },
                'summary': {
                    'accuracy': accuracy,
                    'total': len(self.test_problems)
                },
                'results': results
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    experiment = MinimalInsightSpikeExperiment()
    experiment.run_experiment()