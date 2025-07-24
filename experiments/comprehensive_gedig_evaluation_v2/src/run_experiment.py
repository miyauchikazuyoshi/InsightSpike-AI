#!/usr/bin/env python3
"""
Main experiment runner for comprehensive geDIG evaluation v2.
Implements direct ΔGED/ΔIG calculation with expanded test set.
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

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike import MainAgent
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets

from question_generator import ExpandedQuestionGenerator

# Use full InsightSpike implementations
from insightspike.algorithms.graph_edit_distance import GraphEditDistance
from insightspike.algorithms.pyg_adapter import PyGAdapter
from insightspike.algorithms.information_gain import InformationGain


class ComprehensiveGeDIGExperiment:
    """Run comprehensive evaluation with direct metrics."""
    
    def __init__(self, config_path: str = None, seed: int = 42):
        """
        Initialize experiment.
        
        Args:
            config_path: Path to config file
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Set up paths
        self.experiment_dir = Path(__file__).parent.parent
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        
        # Create directories
        for dir_path in [self.data_dir / "input", 
                         self.data_dir / "processed",
                         self.results_dir / "metrics",
                         self.results_dir / "outputs",
                         self.results_dir / "visualizations"]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config(preset="experiment")
        
        # Initialize components with full implementations
        self.ged_calculator = GraphEditDistance(
            optimization_level='standard',
            timeout_seconds=5.0
        )
        self.ig_calculator = InformationGain(method='clustering')
        self.question_generator = ExpandedQuestionGenerator(seed=seed)
        
        # Results storage
        self.results = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'seed': seed,
                'config': self.config.dict()
            },
            'questions': [],
            'raw_results': [],
            'summary': {}
        }
    
    def prepare_data(self):
        """Prepare experimental data following CLAUDE.md guidelines."""
        print("=== Preparing Data ===")
        
        # Copy knowledge base from project root if needed
        project_data_dir = project_root / "data"
        knowledge_source = project_data_dir / "synthetic" / "mathematical_associations.json"
        
        if knowledge_source.exists():
            import shutil
            shutil.copy2(knowledge_source, self.data_dir / "input" / "knowledge_base.json")
            print(f"Copied knowledge base from {knowledge_source}")
        
        # Generate expanded test set
        print("Generating 100 test questions...")
        questions = self.question_generator.generate_questions(
            n_easy=25, n_medium=50, n_hard=25
        )
        
        # Save questions
        questions_path = self.data_dir / "input" / "test_questions.json"
        self.question_generator.save_questions(questions, str(questions_path))
        print(f"Saved {len(questions)} questions to {questions_path}")
        
        self.questions = questions
        return questions
    
    def initialize_agent(self) -> MainAgent:
        """Initialize InsightSpike agent with knowledge."""
        print("\n=== Initializing Agent ===")
        
        # Check command line arguments for provider
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--provider', default='local', 
                          choices=['local', 'openai', 'mock', 'clean'],
                          help='LLM provider to use')
        parser.add_argument('--model', default=None,
                          help='Model name (default: provider-specific)')
        args, _ = parser.parse_known_args()
        
        # Set provider and model
        llm_provider = args.provider
        if args.model:
            llm_model = args.model
        elif llm_provider == 'local':
            llm_model = 'distilgpt2'  # Use smaller model by default
            print(f"Using {llm_model} via LocalProvider (use --model tinyllama for TinyLlama)")
        elif llm_provider == 'openai':
            llm_model = 'gpt-3.5-turbo'
        else:
            llm_model = f'{llm_provider}-model'
        
        print(f"Using {llm_model} with {llm_provider} provider")
        
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
                'provider': llm_provider,
                'model_name': llm_model,
                'temperature': 0.7,
                'max_tokens': 512,
                'device': 'cpu'
            })(),
            'memory': type('MemoryConfig', (), {
                'max_episodes': 1000,
                'compression_enabled': False,
                'max_retrieved_docs': 10
            })(),
            'insight': type('InsightConfig', (), {
                'detection_threshold': 0.5,
                'min_confidence': 0.3
            })()
        })()
        
        # Initialize agent with legacy config
        try:
            agent = MainAgent(legacy_config)
        except Exception as e:
            print(f"Error initializing agent: {e}")
            print("Falling back to mock provider...")
            legacy_config.llm.provider = 'mock'
            legacy_config.llm.model_name = 'mock-model'
            agent = MainAgent(legacy_config)
        
        # Load knowledge base - try expanded first, then fall back
        knowledge_path = self.data_dir / "input" / "knowledge_base_expanded.json"
        if not knowledge_path.exists():
            knowledge_path = self.data_dir / "input" / "knowledge_base.json"
        
        if knowledge_path.exists():
            with open(knowledge_path, 'r') as f:
                knowledge_data = json.load(f)
            
            # Add knowledge to agent with error handling
            associations = knowledge_data.get('associations', [])
            successful_adds = 0
            for i, item in enumerate(associations):
                try:
                    result = agent.add_knowledge(item['text'])
                    if result.get('success', False):
                        successful_adds += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Added {i + 1}/{len(associations)} knowledge items...")
                except Exception as e:
                    print(f"  Warning: Failed to add knowledge item {i}: {e}")
            
            print(f"Successfully loaded {successful_adds}/{len(associations)} knowledge items")
        
        return agent
    
    def run_single_question(self, 
                           agent: MainAgent, 
                           question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single question and calculate direct metrics using full implementations.
        
        Returns:
            Result dictionary with metrics and response
        """
        # Get PyG graph state before from Layer3
        pyg_graph_before = None
        if agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
            pyg_graph_before = agent.l3_graph.previous_graph
        
        # Convert to NetworkX using official adapter
        nx_graph_before = PyGAdapter.pyg_to_networkx(pyg_graph_before) if pyg_graph_before else nx.Graph()
        
        # Process question
        start_time = time.time()
        result = agent.process_question(question['text'])
        processing_time = time.time() - start_time
        
        # Get PyG graph state after from Layer3
        pyg_graph_after = None
        if agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
            pyg_graph_after = agent.l3_graph.previous_graph
        
        # Convert to NetworkX using official adapter
        nx_graph_after = PyGAdapter.pyg_to_networkx(pyg_graph_after) if pyg_graph_after else nx.Graph()
        
        # Calculate metrics using full implementations
        # ΔGED calculation (instantaneous formula)
        delta_ged = self.ged_calculator.compute_delta_ged(nx_graph_before, nx_graph_after)
        
        # ΔIG calculation
        embeddings_before = self._extract_embeddings(nx_graph_before)
        embeddings_after = self._extract_embeddings(nx_graph_after)
        delta_ig = self.ig_calculator.compute_delta_ig(embeddings_before, embeddings_after)
        
        # Insight detection based on thresholds
        has_insight_metrics = (
            delta_ged < self.config.graph.spike_ged_threshold and
            delta_ig > self.config.graph.spike_ig_threshold
        )
        
        metrics = {
            'delta_ged': delta_ged,
            'delta_ig': delta_ig,
            'ged_threshold': self.config.graph.spike_ged_threshold,
            'ig_threshold': self.config.graph.spike_ig_threshold,
            'has_insight': has_insight_metrics,
            'ged_stats': self.ged_calculator.get_statistics()
        }
        
        # Extract response from CycleResult
        if hasattr(result, 'spike_detected'):
            has_spike = result.spike_detected
            response = result.response
        else:
            # Fallback
            has_spike = False
            response = str(result)
        
        return {
            'question_id': question['id'],
            'question_text': question['text'],
            'difficulty': question['difficulty'],
            'category': question['category'],
            'response': response,
            'has_spike_detected': has_spike,
            'has_insight_metrics': has_insight_metrics,
            'metrics': metrics,
            'processing_time': processing_time,
            'graph_stats': {
                'nodes_before': nx_graph_before.number_of_nodes(),
                'edges_before': nx_graph_before.number_of_edges(),
                'nodes_after': nx_graph_after.number_of_nodes(),
                'edges_after': nx_graph_after.number_of_edges()
            }
        }
    
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
    
    def run_experiment(self):
        """Run complete experiment."""
        print("\n=== Running Experiment ===")
        
        # Prepare data
        if not hasattr(self, 'questions'):
            self.prepare_data()
        
        # Initialize agent
        agent = self.initialize_agent()
        
        # Run questions
        print(f"\nProcessing {len(self.questions)} questions...")
        for i, question in enumerate(self.questions):
            if i % 10 == 0:  # Progress update every 10 questions
                print(f"\nProgress: {i+1}/{len(self.questions)} - Processing '{question.text[:50]}...'")
            
            try:
                result = self.run_single_question(agent, question.__dict__)
                self.results['raw_results'].append(result)
            except Exception as e:
                warnings.warn(f"Error processing question {question.id}: {e}")
                self.results['raw_results'].append({
                    'question_id': question.id,
                    'error': str(e)
                })
        
        print("\n\nExperiment completed!")
        
        # Calculate summary statistics
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
        
        # Overall accuracy
        spike_correct = sum(1 for r in valid_results 
                           if r['has_spike_detected'] == r['has_insight_metrics'])
        
        # By difficulty
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in valid_results if r['difficulty'] == difficulty]
            if diff_results:
                diff_correct = sum(1 for r in diff_results 
                                  if r['has_spike_detected'] == r['has_insight_metrics'])
                difficulty_stats[difficulty] = {
                    'total': len(diff_results),
                    'correct': diff_correct,
                    'accuracy': diff_correct / len(diff_results),
                    'avg_delta_ged': np.mean([r['metrics']['delta_ged'] for r in diff_results]),
                    'avg_delta_ig': np.mean([r['metrics']['delta_ig'] for r in diff_results])
                }
        
        # GED and IG statistics
        ged_stats = self.ged_calculator.get_statistics()
        ig_stats = self.ig_calculator.get_statistics()
        
        # Aggregate metrics
        all_delta_ged = [r['metrics']['delta_ged'] for r in valid_results]
        all_delta_ig = [r['metrics']['delta_ig'] for r in valid_results]
        
        metrics_summary = {
            'ged_calculator': ged_stats,
            'ig_calculator': ig_stats,
            'delta_ged': {
                'mean': np.mean(all_delta_ged),
                'std': np.std(all_delta_ged),
                'min': np.min(all_delta_ged),
                'max': np.max(all_delta_ged)
            },
            'delta_ig': {
                'mean': np.mean(all_delta_ig),
                'std': np.std(all_delta_ig),
                'min': np.min(all_delta_ig),
                'max': np.max(all_delta_ig)
            }
        }
        
        self.results['summary'] = {
            'total_questions': len(self.questions),
            'valid_results': len(valid_results),
            'overall_accuracy': spike_correct / len(valid_results) if valid_results else 0,
            'difficulty_stats': difficulty_stats,
            'metrics_summary': metrics_summary,
            'avg_processing_time': np.mean([r['processing_time'] for r in valid_results])
        }
    
    def _save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert PosixPath to string for JSON serialization
        serializable_results = json.loads(
            json.dumps(self.results, default=str)
        )
        
        # Save detailed results
        results_path = self.results_dir / "outputs" / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary with numpy type conversion
        summary_path = self.results_dir / "metrics" / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types
            summary_serializable = json.loads(
                json.dumps(self.results['summary'], default=str)
            )
            json.dump(summary_serializable, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - {results_path}")
        print(f"  - {summary_path}")
        
        # Print summary
        print("\n=== Experiment Summary ===")
        summary = self.results.get('summary', {})
        if summary:
            print(f"Total questions: {summary.get('total_questions', 'N/A')}")
            print(f"Valid results: {summary.get('valid_results', 'N/A')}")
            if 'overall_accuracy' in summary:
                print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
        else:
            print("Summary not available due to errors")
        
        if summary and 'difficulty_stats' in summary:
            print("\nBy difficulty:")
            for difficulty, stats in summary['difficulty_stats'].items():
                print(f"  {difficulty}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
                print(f"    Avg ΔGED: {stats['avg_delta_ged']:.3f}")
                print(f"    Avg ΔIG: {stats['avg_delta_ig']:.3f}")


def main():
    """Main entry point."""
    import argparse
    
    # Enable detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Run comprehensive geDIG evaluation v2")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    print(f"Starting experiment with seed={args.seed}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    
    try:
        # Run experiment
        experiment = ComprehensiveGeDIGExperiment(
            config_path=args.config,
            seed=args.seed
        )
        
        experiment.run_experiment()
    except Exception as e:
        print(f"ERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()