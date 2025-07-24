#!/usr/bin/env python3
"""
Run v2 experiment with TinyLlama directly (bypassing provider system).
Based on successful implementation from v1 experiment.
"""

import os
import sys
import json
import time
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from insightspike import MainAgent
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets

from question_generator import ExpandedQuestionGenerator

# Use full InsightSpike implementations
from insightspike.algorithms.graph_edit_distance import GraphEditDistance
from insightspike.algorithms.pyg_adapter import PyGAdapter
from insightspike.algorithms.information_gain import InformationGain


class TinyLlamaExperiment:
    """Run v2 experiment with TinyLlama."""
    
    def __init__(self, seed: int = 42):
        """Initialize experiment with TinyLlama."""
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set up paths
        self.experiment_dir = Path(__file__).parent.parent
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        
        # Load configuration
        self.config = load_config(preset="experiment")
        
        # Initialize components with full implementations
        self.ged_calculator = GraphEditDistance(
            optimization_level='standard',
            timeout_seconds=5.0
        )
        self.ig_calculator = InformationGain(method='clustering')
        self.question_generator = ExpandedQuestionGenerator(seed=seed)
        
        # Load TinyLlama
        print("Loading TinyLlama model...")
        self.model, self.tokenizer = self._load_tinyllama()
        
        # Results storage
        self.results = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'seed': seed,
                'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'config': self.config.dict()
            },
            'questions': [],
            'raw_results': [],
            'summary': {}
        }
    
    def _load_tinyllama(self):
        """Load TinyLlama model and tokenizer."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.eval()  # Set to evaluation mode
        print("TinyLlama loaded successfully!")
        
        return model, tokenizer
    
    def _generate_with_tinyllama(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response with TinyLlama."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in full_response:
            return full_response.split("<|assistant|>")[-1].strip()
        else:
            return full_response[len(prompt):].strip()
    
    def create_tinyllama_agent(self) -> MainAgent:
        """Create MainAgent with custom TinyLlama wrapper."""
        print("\n=== Initializing Agent with TinyLlama ===")
        
        # Create a custom provider that uses our TinyLlama
        class TinyLlamaProvider:
            def __init__(self, generate_fn):
                self.generate_fn = generate_fn
                self.provider = 'tinyllama'
                self.model_name = 'TinyLlama-1.1B-Chat-v1.0'
                self.initialized = True
            
            def initialize(self) -> bool:
                """Required by InsightSpike's provider interface."""
                return True
            
            def generate(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
                # Build prompt
                retrieved_docs = context.get('retrieved_documents', [])
                
                prompt_parts = ["<|system|>\nYou are a helpful AI assistant capable of reasoning and making connections between concepts.\n</s>"]
                
                if retrieved_docs:
                    prompt_parts.append("<|user|>\nRelevant information:")
                    for doc in retrieved_docs[:3]:  # Top 3 docs
                        prompt_parts.append(f"- {doc['text']}")
                    prompt_parts.append(f"\nQuestion: {question}\n</s>")
                else:
                    prompt_parts.append(f"<|user|>\n{question}\n</s>")
                
                prompt_parts.append("<|assistant|>")
                prompt = "\n".join(prompt_parts)
                
                # Generate response
                response = self.generate_fn(prompt)
                
                return {
                    'response': response,
                    'success': True,
                    'provider': 'tinyllama'
                }
            
            def generate_response(self, prompt: str, max_length: int = 150, **kwargs) -> str:
                """Alternative method name that might be called."""
                # Ensure prompt is string
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                return self.generate_fn(prompt, max_new_tokens=max_length)
        
        # Create legacy config with custom provider
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
                'provider': 'mock',  # We'll replace this after init
                'model_name': 'mock'
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
        
        # Initialize agent
        agent = MainAgent(legacy_config)
        
        # Replace the LLM provider with our TinyLlama wrapper
        agent.l4_llm = TinyLlamaProvider(self._generate_with_tinyllama)
        
        # Load knowledge base
        knowledge_path = self.data_dir / "input" / "knowledge_base.json"
        if knowledge_path.exists():
            with open(knowledge_path, 'r') as f:
                knowledge_data = json.load(f)
            
            # Add knowledge to agent
            for item in knowledge_data.get('associations', [])[:30]:  # Limit for speed
                try:
                    agent.add_knowledge(item['text'])
                except:
                    pass  # Ignore storage errors
            
            print(f"Loaded {len(knowledge_data.get('associations', [])[:30])} knowledge items")
        
        return agent
    
    def run_experiment(self, n_questions: int = 10):
        """Run experiment with limited questions for testing."""
        print("\n=== Running TinyLlama Experiment ===")
        
        # Generate questions
        print(f"Generating {n_questions} test questions...")
        all_questions = self.question_generator.generate_questions(
            n_easy=3, n_medium=5, n_hard=2
        )
        questions = all_questions[:n_questions]
        
        # Initialize agent
        agent = self.create_tinyllama_agent()
        
        # Run questions
        print(f"\nProcessing {len(questions)} questions...")
        for i, question in enumerate(questions):
            print(f"\rProgress: {i+1}/{len(questions)}", end='', flush=True)
            
            try:
                # Get graph state before
                pyg_graph_before = None
                if agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
                    pyg_graph_before = agent.l3_graph.previous_graph
                
                nx_graph_before = PyGAdapter.pyg_to_networkx(pyg_graph_before) if pyg_graph_before else nx.Graph()
                
                # Process question
                start_time = time.time()
                result = agent.process_question(question.text)
                processing_time = time.time() - start_time
                
                # Get graph state after
                pyg_graph_after = None
                if agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
                    pyg_graph_after = agent.l3_graph.previous_graph
                
                nx_graph_after = PyGAdapter.pyg_to_networkx(pyg_graph_after) if pyg_graph_after else nx.Graph()
                
                # Calculate metrics
                delta_ged = self.ged_calculator.compute_delta_ged(nx_graph_before, nx_graph_after)
                
                embeddings_before = self._extract_embeddings(nx_graph_before)
                embeddings_after = self._extract_embeddings(nx_graph_after)
                delta_ig = self.ig_calculator.compute_delta_ig(embeddings_before, embeddings_after)
                
                # Store result
                self.results['raw_results'].append({
                    'question_id': question.id,
                    'question_text': question.text,
                    'difficulty': question.difficulty,
                    'category': question.category,
                    'response': result.response if hasattr(result, 'response') else str(result),
                    'has_spike_detected': result.spike_detected if hasattr(result, 'spike_detected') else False,
                    'metrics': {
                        'delta_ged': delta_ged,
                        'delta_ig': delta_ig,
                    },
                    'processing_time': processing_time,
                    'graph_stats': {
                        'nodes_before': nx_graph_before.number_of_nodes(),
                        'edges_before': nx_graph_before.number_of_edges(),
                        'nodes_after': nx_graph_after.number_of_nodes(),
                        'edges_after': nx_graph_after.number_of_edges()
                    }
                })
                
            except Exception as e:
                print(f"\nError processing question {question.id}: {e}")
                self.results['raw_results'].append({
                    'question_id': question.id,
                    'error': str(e)
                })
        
        print("\n\nExperiment completed!")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _extract_embeddings(self, graph: nx.Graph) -> np.ndarray:
        """Extract embeddings from NetworkX graph nodes."""
        embeddings = []
        for node, data in graph.nodes(data=True):
            if 'feature' in data:
                embeddings.append(data['feature'])
            elif 'embedding' in data:
                embeddings.append(data['embedding'])
        
        if embeddings:
            return np.array(embeddings)
        else:
            return np.array([]).reshape(0, 768)
    
    def _save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create TinyLlama results directory
        tinyllama_results = self.results_dir / "tinyllama"
        tinyllama_results.mkdir(exist_ok=True)
        
        # Save results
        results_path = tinyllama_results / f"results_tinyllama_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_path}")
        
        # Print summary
        valid_results = [r for r in self.results['raw_results'] if 'error' not in r]
        if valid_results:
            print(f"\nQuick Summary:")
            print(f"Valid results: {len(valid_results)}/{len(self.results['raw_results'])}")
            
            # Average metrics
            avg_delta_ged = np.mean([r['metrics']['delta_ged'] for r in valid_results])
            avg_delta_ig = np.mean([r['metrics']['delta_ig'] for r in valid_results])
            
            print(f"Avg ΔGED: {avg_delta_ged:.3f}")
            print(f"Avg ΔIG: {avg_delta_ig:.3f}")
            
            # Graph growth
            for r in valid_results[:3]:  # Show first 3
                print(f"\nQuestion: {r['question_text'][:50]}...")
                print(f"  Response: {r['response'][:100]}...")
                print(f"  Graph: {r['graph_stats']['nodes_before']} → {r['graph_stats']['nodes_after']} nodes")
                print(f"  ΔGED: {r['metrics']['delta_ged']:.3f}, ΔIG: {r['metrics']['delta_ig']:.3f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run v2 experiment with TinyLlama")
    parser.add_argument('--n-questions', type=int, default=10, help='Number of questions to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = TinyLlamaExperiment(seed=args.seed)
    experiment.run_experiment(n_questions=args.n_questions)


if __name__ == "__main__":
    main()