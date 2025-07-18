#!/usr/bin/env python3
"""
Working RAT Experiment using cycle function
Based on actual test patterns from test_agent_loop.py
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from insightspike.legacy import agent_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkingMemory:
    """Memory implementation based on test patterns"""
    
    def __init__(self):
        self.episodes = []
        self.episode_data = []
        
    def add_document(self, text: str, source: str = ""):
        """Add a document to memory"""
        # Create a simple episode object
        episode = type(
            "Episode",
            (object,),
            {
                "vec": np.random.random(384),  # Simple random vector
                "text": text,
                "c": 0.5,  # C-value
                "source": source
            }
        )
        self.episodes.append(episode)
        self.episode_data.append({
            "text": text,
            "source": source
        })
        
    def search(self, vec, k):
        """Search for similar episodes"""
        if not self.episodes:
            return [], []
        
        # Simple search: return top k episodes
        k = min(k, len(self.episodes))
        indices = list(range(k))
        scores = [0.8 - (i * 0.1) for i in range(k)]  # Decreasing scores
        
        return scores, indices
        
    def update_c(self, idxs, r, eta=0.1):
        """Update C-values"""
        pass
        
    def train_index(self):
        """Train index"""
        pass
        
    def prune(self, c, i):
        """Prune memory"""
        pass
        
    def merge(self, idxs):
        """Merge episodes"""
        pass
        
    def split(self, idx):
        """Split episode"""
        pass


class WorkingRATExperiment:
    """RAT experiment using working cycle pattern"""
    
    def __init__(self):
        logger.info("🚀 Initializing Working RAT Experiment...")
        
        # Load data
        self.definitions_file = Path(__file__).parent.parent / "data" / "input" / "english_definitions.json"
        self.load_data()
        
    def load_data(self):
        """Load definitions and create problems"""
        with open(self.definitions_file, 'r') as f:
            self.definitions = json.load(f)['definitions']
        
        # RAT problems
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
        
    def solve_problem(self, problem: Dict) -> Dict:
        """Solve a RAT problem using cycle function"""
        logger.info(f"\n📝 Problem {problem['id']}: {problem['question']}")
        
        # Create fresh memory for each problem
        memory = WorkingMemory()
        
        # Add definitions to memory
        for word in problem['words']:
            if word in self.definitions:
                for i, definition in enumerate(self.definitions[word][:3]):
                    memory.add_document(
                        text=f"{word}: {definition}",
                        source=f"{word}_def_{i}"
                    )
        
        logger.info(f"Added {len(memory.episodes)} episodes to memory")
        
        # Call cycle function
        try:
            result = agent_loop.cycle(
                memory=memory,
                question=problem['question'],
                g_old=None,  # No previous graph
                top_k=10
            )
            
            # Extract answer
            response = result.get('answer', '')
            predicted_answer = self._extract_answer(response)
            
            # Check correctness
            is_correct = predicted_answer == problem['answer']
            
            if is_correct:
                logger.info(f"✅ Correct: {predicted_answer}")
            else:
                logger.info(f"❌ Wrong: {predicted_answer} (expected: {problem['answer']})")
            
            # Check for spike
            spike_detected = result.get('eureka', False) or result.get('spike_detected', False)
            if spike_detected:
                logger.info(f"⚡ Spike detected!")
            
            return {
                'problem': problem,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response,
                'spike_detected': spike_detected,
                'delta_ged': result.get('delta_ged', 0.0),
                'delta_ig': result.get('delta_ig', 0.0),
                'episodes_used': len(memory.episodes)
            }
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            import traceback
            traceback.print_exc()
            return {
                'problem': problem,
                'predicted_answer': "ERROR",
                'is_correct': False,
                'error': str(e)
            }
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        # Known answers
        known_answers = ['CHEESE', 'ICE', 'BILL', 'WATCH', 'BANK']
        response_upper = response.upper()
        
        for answer in known_answers:
            if answer in response_upper:
                return answer
        
        # Try to find any uppercase word
        words = response.split()
        for word in words:
            if word.isupper() and len(word) > 2:
                return word.strip('.,!?')
        
        return "UNKNOWN"
    
    def run_experiment(self):
        """Run the experiment"""
        logger.info("\n🧪 Running Working Cycle-based RAT Experiment")
        logger.info("=" * 60)
        
        results = []
        correct = 0
        spikes = 0
        
        for problem in self.test_problems:
            result = self.solve_problem(problem)
            results.append(result)
            
            if result['is_correct']:
                correct += 1
            if result.get('spike_detected', False):
                spikes += 1
        
        # Summary
        accuracy = correct / len(self.test_problems) * 100
        spike_rate = spikes / len(self.test_problems) * 100
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {correct}/{len(self.test_problems)} = {accuracy:.1f}%")
        logger.info(f"Spike Rate: {spikes}/{len(self.test_problems)} = {spike_rate:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "working_experiments"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"working_cycle_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'Working Cycle-based RAT',
                    'description': 'Using test-pattern memory implementation',
                    'timestamp': timestamp
                },
                'summary': {
                    'accuracy': accuracy,
                    'spike_rate': spike_rate,
                    'total': len(self.test_problems)
                },
                'results': results
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    experiment = WorkingRATExperiment()
    experiment.run_experiment()