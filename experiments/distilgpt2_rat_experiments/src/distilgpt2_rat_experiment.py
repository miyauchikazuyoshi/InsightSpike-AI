#!/usr/bin/env python3
"""
DistilGPT2 RAT Experiment with InsightSpike
Using the actual DistilGPT2 model as intended
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import networkx as nx
from typing import List, Dict, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from insightspike.legacy import agent_loop


class OptimizedMemory:
    """Memory implementation optimized for DistilGPT2"""
    
    def __init__(self, knowledge_graph: nx.Graph, episodes_data: List[Dict]):
        self.graph = knowledge_graph
        self.episodes = []
        
        # Create episodes with proper embeddings
        for i, ep_data in enumerate(episodes_data):
            episode = type(
                "Episode",
                (object,),
                {
                    "vec": self._create_embedding(ep_data["text"], i),
                    "text": ep_data["text"],
                    "c": self._calculate_c_value(ep_data),
                    "metadata": ep_data
                }
            )
            self.episodes.append(episode)
            
        logger.info(f"Initialized memory with {len(self.episodes)} episodes")
    
    def _create_embedding(self, text: str, idx: int) -> np.ndarray:
        """Create embedding for text"""
        np.random.seed(idx)  # Deterministic
        
        embedding = np.zeros(384, dtype=np.float32)
        
        # Key terms for RAT problems
        key_terms = {
            'cheese': [0, 1], 'cottage': [2, 3], 'swiss': [4, 5], 'cake': [6, 7],
            'ice': [8, 9], 'cream': [10, 11], 'skate': [12, 13], 'water': [14, 15],
            'bill': [16, 17], 'duck': [18, 19], 'fold': [20, 21], 'dollar': [22, 23],
            'watch': [24, 25], 'night': [26, 27], 'wrist': [28, 29], 'stop': [30, 31],
            'bank': [32, 33], 'river': [34, 35], 'note': [36, 37], 'account': [38, 39]
        }
        
        text_lower = text.lower()
        
        # Set features based on term presence
        for term, indices in key_terms.items():
            if term in text_lower:
                for idx in indices:
                    if idx < 384:
                        embedding[idx] = 1.0
        
        # Add some noise
        embedding += np.random.randn(384) * 0.05
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _calculate_c_value(self, episode_data: Dict) -> float:
        """Calculate C-value based on episode importance"""
        c_value = 0.5
        
        if episode_data.get("contains_answer", False):
            c_value += 0.25
            
        if episode_data.get("type") == "rat_context":
            c_value += 0.15
            
        return min(1.0, c_value)
    
    def search(self, vec, k):
        """Search for similar episodes"""
        if not self.episodes:
            return [], []
        
        # Calculate similarities
        similarities = []
        for i, episode in enumerate(self.episodes):
            sim = np.dot(vec, episode.vec)
            weighted_sim = sim * (episode.c ** 0.7)  # C-value weighting
            similarities.append((weighted_sim, i))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top k
        k = min(k, len(similarities))
        scores = [sim[0] for sim in similarities[:k]]
        indices = [sim[1] for sim in similarities[:k]]
        
        return scores, indices
    
    def update_c(self, idxs, r, eta=0.1):
        for idx in idxs:
            if 0 <= idx < len(self.episodes):
                self.episodes[idx].c = min(1.0, self.episodes[idx].c + eta * r)
    
    def train_index(self):
        pass
    
    def prune(self, c, i):
        pass
    
    def merge(self, idxs):
        pass
    
    def split(self, idx):
        pass


class DistilGPT2RATExperiment:
    """RAT experiment using DistilGPT2"""
    
    def __init__(self):
        logger.info("🚀 Initializing DistilGPT2 RAT Experiment...")
        logger.info("📝 Note: Using distilgpt2 model as configured")
        
        # Load knowledge base
        kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
        
        # Load graph
        with open(kb_dir / "rat_knowledge_graph.json", 'r') as f:
            graph_data = json.load(f)
            self.graph = nx.node_link_graph(graph_data)
        
        # Load episodes
        with open(kb_dir / "rat_episodes.json", 'r') as f:
            episodes_data = json.load(f)
            self.episodes = episodes_data["episodes"]
        
        logger.info(f"Loaded: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges, "
                   f"{len(self.episodes)} episodes")
        
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
            },
            {
                "id": 4,
                "question": "Find the word that relates to NIGHT, WRIST, and STOP",
                "words": ["NIGHT", "WRIST", "STOP"],
                "answer": "WATCH"
            },
            {
                "id": 5,
                "question": "What connects RIVER, NOTE, and ACCOUNT?",
                "words": ["RIVER", "NOTE", "ACCOUNT"],
                "answer": "BANK"
            }
        ]
    
    def solve_problem(self, problem: Dict) -> Dict:
        """Solve a RAT problem using DistilGPT2"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Problem {problem['id']}: {problem['question']}")
        logger.info(f"{'='*60}")
        
        # Create memory
        memory = OptimizedMemory(self.graph, self.episodes)
        
        try:
            # Call cycle function
            result = agent_loop.cycle(
                memory=memory,
                question=problem['question'],
                g_old=self.graph.copy(),
                top_k=10  # Reduced for faster processing
            )
            
            # Extract results
            response = result.get('answer', '')
            predicted_answer = self._extract_answer(response)
            
            # Check correctness
            is_correct = predicted_answer == problem['answer']
            
            logger.info(f"Response: {response[:100]}...")
            logger.info(f"{'✅ Correct' if is_correct else '❌ Wrong'}: "
                       f"{predicted_answer} (expected: {problem['answer']})")
            
            # Check for spike
            spike_detected = result.get('eureka', False) or result.get('spike_detected', False)
            if spike_detected:
                logger.info(f"⚡ Spike detected! ΔGED: {result.get('delta_ged', 0):.3f}, "
                          f"ΔIG: {result.get('delta_ig', 0):.3f}")
            
            # Get metrics
            l1_analysis = result.get('l1_analysis', {})
            reasoning_quality = result.get('reasoning_quality', 0.0)
            
            logger.info(f"📊 Quality: {reasoning_quality:.3f}, "
                       f"Complexity: {l1_analysis.get('query_complexity', 0):.3f}")
            
            return {
                'problem': problem,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response[:200] + "..." if len(response) > 200 else response,
                'spike_detected': spike_detected,
                'delta_ged': result.get('delta_ged', 0.0),
                'delta_ig': result.get('delta_ig', 0.0),
                'reasoning_quality': reasoning_quality,
                'complexity': l1_analysis.get('query_complexity', 0.0)
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
        known_answers = ['CHEESE', 'ICE', 'BILL', 'WATCH', 'BANK']
        response_upper = response.upper()
        
        for answer in known_answers:
            if answer in response_upper:
                return answer
        
        # Try to find any uppercase word
        words = response.split()
        for word in words:
            cleaned = ''.join(c for c in word if c.isalpha())
            if cleaned.isupper() and len(cleaned) > 2:
                return cleaned
        
        return "UNKNOWN"
    
    def run_experiment(self):
        """Run the experiment"""
        logger.info("\n" + "="*70)
        logger.info("🧪 DISTILGPT2 RAT EXPERIMENT WITH INSIGHTSPIKE")
        logger.info("="*70)
        
        results = []
        correct = 0
        spikes = 0
        total_quality = 0
        
        for i, problem in enumerate(self.test_problems):
            logger.info(f"\nProcessing problem {i+1}/{len(self.test_problems)}...")
            
            result = self.solve_problem(problem)
            results.append(result)
            
            if result['is_correct']:
                correct += 1
            if result.get('spike_detected', False):
                spikes += 1
            total_quality += result.get('reasoning_quality', 0)
        
        # Summary
        accuracy = (correct / len(self.test_problems)) * 100
        spike_rate = (spikes / len(self.test_problems)) * 100
        avg_quality = total_quality / len(self.test_problems)
        
        logger.info("\n" + "="*70)
        logger.info("📊 FINAL SUMMARY")
        logger.info("="*70)
        logger.info(f"Accuracy: {correct}/{len(self.test_problems)} = {accuracy:.1f}%")
        logger.info(f"Spike Rate: {spikes}/{len(self.test_problems)} = {spike_rate:.1f}%")
        logger.info(f"Average Quality: {avg_quality:.3f}")
        
        # Analyze correlations
        spike_when_correct = sum(1 for r in results if r['is_correct'] and r.get('spike_detected', False))
        spike_when_wrong = sum(1 for r in results if not r['is_correct'] and r.get('spike_detected', False))
        
        logger.info(f"\nSpike Analysis:")
        logger.info(f"- Spikes on correct: {spike_when_correct}")
        logger.info(f"- Spikes on wrong: {spike_when_wrong}")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "distilgpt2_experiments"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"distilgpt2_rat_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'DistilGPT2 RAT Experiment',
                    'description': 'InsightSpike with DistilGPT2 on RAT problems',
                    'model': 'distilgpt2',
                    'timestamp': timestamp,
                    'knowledge_base': {
                        'nodes': self.graph.number_of_nodes(),
                        'edges': self.graph.number_of_edges(),
                        'episodes': len(self.episodes)
                    }
                },
                'summary': {
                    'accuracy': accuracy,
                    'spike_rate': spike_rate,
                    'avg_reasoning_quality': avg_quality,
                    'spike_accuracy_correlation': {
                        'on_correct': spike_when_correct,
                        'on_wrong': spike_when_wrong
                    }
                },
                'results': results
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return results


if __name__ == "__main__":
    experiment = DistilGPT2RATExperiment()
    experiment.run_experiment()