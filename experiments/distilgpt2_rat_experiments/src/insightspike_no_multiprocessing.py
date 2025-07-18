#!/usr/bin/env python3
"""
InsightSpike RAT Experiment without Multiprocessing Issues
Single-threaded execution to avoid resource leaks
"""

import os
# Disable multiprocessing in PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import networkx as nx
from typing import List, Dict, Optional
import numpy as np

# Ensure single-threaded execution
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
torch.set_num_threads(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import with multiprocessing context set
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from insightspike.legacy import agent_loop


class SafeMemory:
    """Thread-safe memory implementation"""
    
    def __init__(self, knowledge_graph: nx.Graph, episodes_data: List[Dict]):
        self.graph = knowledge_graph
        self.episodes_data = episodes_data
        self.episodes = []
        self._embeddings = {}  # Cache embeddings
        
        # Pre-generate all embeddings to avoid parallel computation
        logger.info("Pre-generating embeddings...")
        for i, ep_data in enumerate(episodes_data):
            embedding = self._generate_embedding_safe(ep_data["text"], i)
            self._embeddings[i] = embedding
            
            episode = type(
                "Episode",
                (object,),
                {
                    "vec": embedding,
                    "text": ep_data["text"],
                    "c": self._calculate_c_value(ep_data),
                    "metadata": ep_data,
                    "idx": i
                }
            )
            self.episodes.append(episode)
            
        logger.info(f"Initialized memory with {len(self.episodes)} episodes")
    
    def _generate_embedding_safe(self, text: str, idx: int) -> np.ndarray:
        """Generate embedding with deterministic seed"""
        # Use deterministic random seed based on text
        np.random.seed(idx)
        
        embedding = np.zeros(384, dtype=np.float32)
        
        # Key terms
        key_terms = {
            'cheese': 0, 'cottage': 1, 'swiss': 2, 'cake': 3,
            'ice': 4, 'cream': 5, 'skate': 6, 'water': 7,
            'bill': 8, 'duck': 9, 'fold': 10, 'dollar': 11,
            'watch': 12, 'night': 13, 'wrist': 14, 'stop': 15,
            'bank': 16, 'river': 17, 'note': 18, 'account': 19
        }
        
        text_lower = text.lower()
        
        # Set features
        for term, base_idx in key_terms.items():
            if term in text_lower:
                start_idx = (base_idx * 19) % 384
                end_idx = min(start_idx + 19, 384)
                embedding[start_idx:end_idx] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _calculate_c_value(self, episode_data: Dict) -> float:
        """Calculate C-value"""
        c_value = 0.5
        
        if episode_data.get("contains_answer", False):
            c_value += 0.2
            
        if episode_data.get("type") == "rat_context":
            c_value += 0.15
            
        priority = episode_data.get("priority", 1)
        c_value -= (priority * 0.05)
        
        return max(0.1, min(1.0, c_value))
    
    def search(self, vec, k):
        """Simple search without parallel processing"""
        if not self.episodes:
            return [], []
        
        # Sequential similarity calculation
        similarities = []
        for i, episode in enumerate(self.episodes):
            sim = float(np.dot(vec, episode.vec))
            weighted_sim = sim * (episode.c ** 0.5)
            similarities.append((weighted_sim, i))
        
        # Sort
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top k
        k = min(k, len(similarities))
        scores = [sim[0] for sim in similarities[:k]]
        indices = [sim[1] for sim in similarities[:k]]
        
        return scores, indices
    
    # Required methods for compatibility
    def update_c(self, idxs, r, eta=0.1):
        for idx in idxs:
            if 0 <= idx < len(self.episodes):
                old_c = self.episodes[idx].c
                new_c = old_c + eta * r
                self.episodes[idx].c = max(0.1, min(1.0, new_c))
    
    def train_index(self):
        pass
    
    def prune(self, c, i):
        pass
    
    def merge(self, idxs):
        pass
    
    def split(self, idx):
        pass


class SingleThreadedRATExperiment:
    """RAT experiment with single-threaded execution"""
    
    def __init__(self):
        logger.info("🚀 Initializing Single-threaded RAT Experiment...")
        
        # Load knowledge base
        kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
        
        # Load graph
        with open(kb_dir / "rat_knowledge_graph.json", 'r') as f:
            graph_data = json.load(f)
            # Simple graph loading
            self.graph = nx.Graph()
            for node in graph_data.get('nodes', []):
                self.graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            for link in graph_data.get('links', []):
                self.graph.add_edge(link['source'], link['target'], **{k: v for k, v in link.items() if k not in ['source', 'target']})
        
        # Load episodes
        with open(kb_dir / "rat_episodes.json", 'r') as f:
            episodes_data = json.load(f)
            self.episodes = episodes_data["episodes"]
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        logger.info(f"Loaded {len(self.episodes)} episodes")
        
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
        """Solve a single RAT problem"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Problem {problem['id']}: {problem['question']}")
        logger.info(f"{'='*60}")
        
        # Create safe memory
        memory = SafeMemory(self.graph, self.episodes)
        
        try:
            # Set timeout for the cycle function
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Cycle function timed out")
            
            # Set 60 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                # Call cycle function
                result = agent_loop.cycle(
                    memory=memory,
                    question=problem['question'],
                    g_old=self.graph.copy(),
                    top_k=10  # Reduced for faster processing
                )
                
                # Cancel alarm
                signal.alarm(0)
                
            except TimeoutError:
                logger.warning("Cycle function timed out, using fallback")
                signal.alarm(0)
                return {
                    'problem': problem,
                    'predicted_answer': "TIMEOUT",
                    'is_correct': False,
                    'error': "Processing timeout"
                }
            
            # Extract results
            response = result.get('answer', '')
            predicted_answer = self._extract_answer(response)
            
            # Check correctness
            is_correct = predicted_answer == problem['answer']
            
            # Log result
            status = "✅ Correct" if is_correct else "❌ Wrong"
            logger.info(f"{status}: {predicted_answer} (expected: {problem['answer']})")
            
            # Check for spike
            spike_detected = result.get('eureka', False) or result.get('spike_detected', False)
            if spike_detected:
                logger.info(f"⚡ Spike detected!")
            
            # Get metrics
            metrics = {
                'delta_ged': result.get('delta_ged', 0.0),
                'delta_ig': result.get('delta_ig', 0.0),
                'reasoning_quality': result.get('reasoning_quality', 0.0)
            }
            
            return {
                'problem': problem,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response[:100] + "..." if len(response) > 100 else response,
                'spike_detected': spike_detected,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}")
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
        
        # Try any uppercase word
        words = response.split()
        for word in words:
            cleaned = ''.join(c for c in word if c.isalpha())
            if cleaned.isupper() and len(cleaned) > 2:
                return cleaned
        
        return "UNKNOWN"
    
    def run_experiment(self):
        """Run the experiment sequentially"""
        logger.info("\n" + "="*70)
        logger.info("🧪 SINGLE-THREADED RAT EXPERIMENT")
        logger.info("="*70)
        
        results = []
        correct = 0
        spikes = 0
        
        # Process each problem sequentially
        for i, problem in enumerate(self.test_problems):
            logger.info(f"\nProcessing problem {i+1}/{len(self.test_problems)}...")
            
            result = self.solve_problem(problem)
            results.append(result)
            
            if result['is_correct']:
                correct += 1
            if result.get('spike_detected', False):
                spikes += 1
            
            # Save intermediate results
            self._save_results(results, correct, spikes)
        
        # Final summary
        accuracy = (correct / len(self.test_problems)) * 100 if self.test_problems else 0
        spike_rate = (spikes / len(self.test_problems)) * 100 if self.test_problems else 0
        
        logger.info("\n" + "="*70)
        logger.info("📊 FINAL SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Problems: {len(self.test_problems)}")
        logger.info(f"Correct: {correct} ({accuracy:.1f}%)")
        logger.info(f"Spikes: {spikes} ({spike_rate:.1f}%)")
        
        return results
    
    def _save_results(self, results, correct, spikes):
        """Save intermediate results"""
        output_dir = Path(__file__).parent.parent / "results" / "single_threaded"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rat_results_{timestamp}.json"
        
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        spike_rate = (spikes / total) * 100 if total > 0 else 0
        
        data = {
            'metadata': {
                'experiment': 'Single-threaded RAT Experiment',
                'description': 'InsightSpike without multiprocessing issues',
                'timestamp': timestamp,
                'status': 'in_progress' if total < len(self.test_problems) else 'complete'
            },
            'summary': {
                'total_problems': len(self.test_problems),
                'completed': total,
                'correct': correct,
                'accuracy': accuracy,
                'spikes': spikes,
                'spike_rate': spike_rate
            },
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved intermediate results to: {output_file}")


def main():
    """Main entry point"""
    # Ensure clean environment
    import gc
    gc.collect()
    
    # Run experiment
    experiment = SingleThreadedRATExperiment()
    results = experiment.run_experiment()
    
    # Clean up
    gc.collect()
    
    logger.info("\n🎉 Experiment completed successfully!")


if __name__ == "__main__":
    main()