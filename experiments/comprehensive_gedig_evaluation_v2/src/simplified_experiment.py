#!/usr/bin/env python3
"""
Simplified experiment using the same approach as v1 experiment.
Focus on direct metric calculation without complex graph handling.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike import MainAgent
from insightspike.config import load_config
from direct_metrics import DirectMetricsCalculator
from question_generator import ExpandedQuestionGenerator


class SimplifiedGeDIGExperiment:
    """Simplified experiment focusing on embedding changes."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        # Paths
        self.experiment_dir = Path(__file__).parent.parent
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'questions': [],
            'raw_results': [],
            'summary': {}
        }
    
    def load_knowledge_base(self) -> List[str]:
        """Load knowledge base."""
        kb_path = self.data_dir / "input" / "knowledge_base_expanded.json"
        with open(kb_path, 'r') as f:
            data = json.load(f)
        
        return [item['text'] for item in data['associations']]
    
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load test questions."""
        q_path = self.data_dir / "input" / "test_questions.json"
        with open(q_path, 'r') as f:
            data = json.load(f)
        
        return data['questions']
    
    def calculate_embedding_change(self, 
                                  embeddings_before: np.ndarray, 
                                  embeddings_after: np.ndarray) -> float:
        """Calculate change in embeddings."""
        if len(embeddings_before) == 0 or len(embeddings_after) == 0:
            return 0.0
        
        # Average embeddings
        avg_before = np.mean(embeddings_before, axis=0)
        avg_after = np.mean(embeddings_after, axis=0)
        
        # Cosine distance
        similarity = cosine_similarity([avg_before], [avg_after])[0, 0]
        
        # Return change (1 - similarity)
        return 1.0 - similarity
    
    def process_question_simple(self, question: Dict[str, Any], knowledge: List[str]) -> Dict[str, Any]:
        """Process a single question with simple metrics."""
        try:
            # Get embeddings before
            embeddings_before = self.embedding_model.encode(knowledge)
            
            # Simulate question processing
            q_embedding = self.embedding_model.encode([question['text']])[0]
            
            # Find relevant knowledge
            similarities = cosine_similarity([q_embedding], embeddings_before)[0]
            top_k = 5
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_knowledge = [knowledge[i] for i in top_indices]
            
            # Simulate adding the question as new knowledge
            new_knowledge = knowledge + [question['text']]
            embeddings_after = self.embedding_model.encode(new_knowledge)
            
            # Calculate metrics
            embedding_change = self.calculate_embedding_change(embeddings_before, embeddings_after)
            
            # Simple spike detection based on embedding change
            has_spike = embedding_change > 0.1
            
            return {
                'question_id': question['id'],
                'question_text': question['text'],
                'difficulty': question['difficulty'],
                'category': question['category'],
                'embedding_change': float(embedding_change),
                'has_spike': has_spike,
                'relevant_knowledge': relevant_knowledge[:3],  # Top 3
                'processing_time': 0.1  # Simulated
            }
            
        except Exception as e:
            return {
                'question_id': question['id'],
                'error': str(e)
            }
    
    def run_experiment(self):
        """Run simplified experiment."""
        print("\n=== Running Simplified Experiment ===")
        
        # Load data
        print("Loading knowledge base...")
        knowledge = self.load_knowledge_base()
        print(f"Loaded {len(knowledge)} knowledge items")
        
        print("Loading questions...")
        questions = self.load_questions()
        print(f"Loaded {len(questions)} questions")
        
        # Process questions
        print("\nProcessing questions...")
        for i, question in enumerate(questions):
            print(f"\rProgress: {i+1}/{len(questions)}", end='', flush=True)
            
            result = self.process_question_simple(question, knowledge)
            self.results['raw_results'].append(result)
        
        print("\n\nCalculating summary...")
        self._calculate_summary()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _calculate_summary(self):
        """Calculate summary statistics."""
        valid_results = [r for r in self.results['raw_results'] if 'error' not in r]
        
        if not valid_results:
            self.results['summary'] = {'error': 'No valid results'}
            return
        
        # Overall stats
        spike_count = sum(1 for r in valid_results if r['has_spike'])
        
        # By difficulty
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in valid_results if r['difficulty'] == difficulty]
            if diff_results:
                diff_spike_count = sum(1 for r in diff_results if r['has_spike'])
                difficulty_stats[difficulty] = {
                    'total': len(diff_results),
                    'spike_count': diff_spike_count,
                    'spike_rate': diff_spike_count / len(diff_results),
                    'avg_embedding_change': np.mean([r['embedding_change'] for r in diff_results])
                }
        
        self.results['summary'] = {
            'total_questions': len(questions),
            'valid_results': len(valid_results),
            'overall_spike_rate': spike_count / len(valid_results) if valid_results else 0,
            'difficulty_stats': difficulty_stats
        }
    
    def _save_results(self):
        """Save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_path = self.results_dir / "outputs" / f"simplified_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        # Print summary
        summary = self.results['summary']
        print("\n=== Summary ===")
        print(f"Total questions: {summary.get('total_questions', 0)}")
        print(f"Valid results: {summary.get('valid_results', 0)}")
        print(f"Overall spike rate: {summary.get('overall_spike_rate', 0):.2%}")
        
        if 'difficulty_stats' in summary:
            print("\nBy difficulty:")
            for diff, stats in summary['difficulty_stats'].items():
                print(f"  {diff}: {stats['spike_rate']:.2%} spike rate ({stats['spike_count']}/{stats['total']})")
                print(f"    Avg embedding change: {stats['avg_embedding_change']:.3f}")


def main():
    """Run simplified experiment."""
    experiment = SimplifiedGeDIGExperiment(seed=42)
    experiment.run_experiment()


if __name__ == "__main__":
    main()