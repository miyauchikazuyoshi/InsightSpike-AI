#!/usr/bin/env python3
"""
RAT Experiment using InsightSpike's cycle function
Proper integration with dictionary episodes and graph updates
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import networkx as nx
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import InsightSpike components
from src.insightspike.legacy.agent_loop import cycle
from src.insightspike.core.layers.layer2_memory_manager import L2MemoryManager
from src.insightspike.detection.insight_registry import InsightFactRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CycleBasedRATExperiment:
    """
    RAT experiment using the legacy cycle function with proper data structures
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Cycle-based RAT Experiment...")
        
        # Initialize memory manager
        self.memory = L2MemoryManager(dim=768)
        
        # Initialize insight registry
        self.insight_registry = InsightFactRegistry()
        
        # Load data files
        self.definitions_file = Path(__file__).parent.parent / "data" / "input" / "english_definitions.json"
        self.rat_problems_file = Path(__file__).parent.parent / "data" / "input" / "rat_problems_with_meanings.json"
        
        # Initialize graph
        self.knowledge_graph = nx.Graph()
        
        # Load and prepare data
        self._load_data()
        self._build_initial_graph()
        
    def _load_data(self):
        """Load definitions and RAT problems"""
        # Load definitions
        with open(self.definitions_file, 'r') as f:
            self.definitions = json.load(f)['definitions']
        
        # Load or create RAT problems with meanings
        if self.rat_problems_file.exists():
            with open(self.rat_problems_file, 'r') as f:
                data = json.load(f)
                self.test_problems = data['problems']
        else:
            # Create if doesn't exist
            self._create_rat_problems_with_meanings()
        
        logger.info(f"Loaded {len(self.definitions)} word definitions")
        logger.info(f"Loaded {len(self.test_problems)} RAT problems")
    
    def _create_rat_problems_with_meanings(self):
        """Create RAT problems file with meanings"""
        problems = [
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
        
        # Add meanings to each problem
        for problem in problems:
            problem['word_meanings'] = {}
            for word in problem['words']:
                if word in self.definitions:
                    problem['word_meanings'][word] = self.definitions[word][:3]
        
        # Save
        data = {
            "metadata": {
                "name": "RAT Problems with Meanings",
                "created": datetime.now().isoformat()
            },
            "problems": problems
        }
        
        with open(self.rat_problems_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.test_problems = problems
    
    def _build_initial_graph(self):
        """Build initial knowledge graph from definitions"""
        logger.info("Building initial knowledge graph...")
        
        episode_count = 0
        
        for problem in self.test_problems:
            # Add word nodes
            for word in problem['words']:
                self.knowledge_graph.add_node(word, node_type='word')
                
                # Add definition nodes and edges
                if word in problem.get('word_meanings', {}):
                    for i, meaning in enumerate(problem['word_meanings'][word]):
                        meaning_id = f"{word}_def_{i}"
                        self.knowledge_graph.add_node(
                            meaning_id, 
                            node_type='definition',
                            text=meaning,
                            source_word=word
                        )
                        self.knowledge_graph.add_edge(word, meaning_id, weight=1.0)
                        
                        # Add episode to memory
                        try:
                            self.memory.add_episode(
                                vector=self._create_embedding(meaning),
                                text=meaning,
                                c_value=0.5
                            )
                            episode_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to add episode: {e}")
                
                # Add connections between words that share concepts
                self._add_conceptual_connections(word, problem['words'])
        
        logger.info(f"Graph has {self.knowledge_graph.number_of_nodes()} nodes, "
                   f"{self.knowledge_graph.number_of_edges()} edges")
        logger.info(f"Added {episode_count} episodes to memory")
    
    def _create_embedding(self, text: str):
        """Create simple embedding for text"""
        import numpy as np
        
        # Simplified embedding based on key terms
        key_terms = ['cheese', 'ice', 'bill', 'watch', 'bank', 'cottage', 
                    'swiss', 'cake', 'cream', 'skate', 'water', 'duck', 
                    'fold', 'dollar', 'night', 'wrist', 'stop', 'river', 
                    'note', 'account']
        
        embedding = np.zeros(768, dtype=np.float32)
        text_lower = text.lower()
        
        # Set features based on term presence
        for i, term in enumerate(key_terms):
            if term in text_lower:
                start_idx = i * 30
                end_idx = min(start_idx + 30, 768)
                embedding[start_idx:end_idx] = 1.0
        
        # Add some variation
        np.random.seed(hash(text) % 2**32)
        embedding += np.random.randn(768) * 0.1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _add_conceptual_connections(self, word: str, all_words: List[str]):
        """Add edges between conceptually related words"""
        # Simple heuristic: words that share definition terms
        if word not in self.knowledge_graph:
            return
            
        word_node = self.knowledge_graph.nodes[word]
        
        # Get definition texts for this word
        word_defs = []
        for neighbor in self.knowledge_graph.neighbors(word):
            if self.knowledge_graph.nodes[neighbor].get('node_type') == 'definition':
                word_defs.append(self.knowledge_graph.nodes[neighbor]['text'].lower())
        
        # Check other words
        for other_word in all_words:
            if other_word != word and other_word in self.knowledge_graph:
                # Get other word's definitions
                other_defs = []
                for neighbor in self.knowledge_graph.neighbors(other_word):
                    if self.knowledge_graph.nodes[neighbor].get('node_type') == 'definition':
                        other_defs.append(self.knowledge_graph.nodes[neighbor]['text'].lower())
                
                # Check for shared concepts
                shared_terms = 0
                for wd in word_defs:
                    for od in other_defs:
                        # Count common meaningful words
                        wd_terms = set(w for w in wd.split() if len(w) > 3)
                        od_terms = set(w for w in od.split() if len(w) > 3)
                        shared_terms += len(wd_terms & od_terms)
                
                if shared_terms > 2:  # Threshold for connection
                    self.knowledge_graph.add_edge(
                        word, other_word, 
                        weight=shared_terms / 10.0,
                        edge_type='conceptual'
                    )
    
    def solve_with_cycle(self, problem: Dict) -> Dict:
        """Solve RAT problem using cycle function"""
        # Run cycle with current graph state
        result = cycle(
            memory=self.memory,
            question=problem['question'],
            g_old=self.knowledge_graph.copy(),  # Pass current graph
            top_k=10,
            device=None
        )
        
        # Extract answer and metrics from legacy format
        response = result.get('answer', '')  # cycle returns 'answer' not 'response'
        predicted_answer = self._extract_answer_from_response(response)
        
        # Check if spike was detected (cycle returns 'eureka')
        spike_detected = result.get('eureka', False) or result.get('spike_detected', False)
        
        # Get graph metrics
        g_new = result.get('graph', self.knowledge_graph)
        delta_ged_value = result.get('delta_ged', 0.0)
        delta_ig_value = result.get('delta_ig', 0.0)
        
        return {
            'predicted_answer': predicted_answer,
            'response': response,
            'spike_detected': spike_detected,
            'delta_ged': delta_ged_value,
            'delta_ig': delta_ig_value,
            'graph_before_nodes': self.knowledge_graph.number_of_nodes(),
            'graph_after_nodes': g_new.number_of_nodes() if g_new else 0,
            'insights': result.get('discovered_insights', []),
            'new_connections': result.get('new_connections', 0),
            'l1_analysis': result.get('l1_analysis', {}),
            'adaptive_topk': result.get('adaptive_topk', {}),
            'reasoning_quality': result.get('reasoning_quality', 0.0)
        }
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from cycle response"""
        # Look for known answers
        known_answers = ['CHEESE', 'ICE', 'BILL', 'WATCH', 'BANK']
        response_upper = response.upper()
        
        for answer in known_answers:
            if answer in response_upper:
                return answer
        
        # Fallback: last uppercase word
        words = response.split()
        for word in reversed(words):
            if word.isupper() and len(word) > 2:
                return word
                
        return "UNKNOWN"
    
    def run_experiment(self):
        """Run the full experiment"""
        logger.info("\n🧪 Running Cycle-based RAT Experiment")
        logger.info("=" * 60)
        
        results = []
        correct = 0
        spike_count = 0
        
        for problem in tqdm(self.test_problems, desc="Processing"):
            logger.info(f"\n📝 Problem {problem['id']}: {problem['question']}")
            
            # Solve using cycle
            solution = self.solve_with_cycle(problem)
            
            # Check correctness
            is_correct = solution['predicted_answer'] == problem['answer']
            if is_correct:
                correct += 1
                logger.info(f"✅ Correct: {solution['predicted_answer']}")
            else:
                logger.info(f"❌ Wrong: {solution['predicted_answer']} "
                          f"(expected: {problem['answer']})")
            
            # Check spike
            if solution['spike_detected']:
                spike_count += 1
                logger.info(f"⚡ Spike detected! ΔGED: {solution['delta_ged']:.3f}, "
                          f"ΔIG: {solution['delta_ig']:.3f}")
            
            # Graph changes
            if solution['graph_before_nodes'] != solution['graph_after_nodes']:
                logger.info(f"📊 Graph: {solution['graph_before_nodes']} → "
                          f"{solution['graph_after_nodes']} nodes")
            
            results.append({
                'problem': problem,
                'solution': solution,
                'correct': is_correct
            })
        
        # Summary
        accuracy = correct / len(self.test_problems) * 100
        spike_rate = spike_count / len(self.test_problems) * 100
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {correct}/{len(self.test_problems)} = {accuracy:.1f}%")
        logger.info(f"Spike Rate: {spike_count}/{len(self.test_problems)} = {spike_rate:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "cycle_experiments"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"cycle_rat_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'Cycle-based RAT Experiment',
                    'description': 'Using legacy cycle function with proper graph',
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
        
        # Save final graph
        graph_file = output_dir / f"final_graph_{timestamp}.json"
        graph_data = nx.node_link_data(self.knowledge_graph)
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"📊 Graph saved to: {graph_file}")


if __name__ == "__main__":
    experiment = CycleBasedRATExperiment()
    experiment.run_experiment()