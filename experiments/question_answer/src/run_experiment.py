#!/usr/bin/env python3
"""
Question-Answer Experiment Runner with Branching Detection

This script runs the experiment using adaptive_loop with custom branching prompts.
"""

import argparse
import json
import logging
import os
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.config import load_config
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.implementations.layers.layer1_error_monitor import ErrorMonitor
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager  
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager
from insightspike.adaptive.core.exploration_loop import ExplorationLoop
from insightspike.adaptive.strategies.expanding import ExpandingStrategy
from insightspike.adaptive.core.adaptive_processor import AdaptiveProcessor
from insightspike.adaptive.calculators.adaptive_topk import AdaptiveTopKCalculator
from insightspike.processing.embedder import EmbeddingManager

# Branching detection is now built into the main code's association_extended prompt style


class QuestionAnswerExperiment:
    """Main experiment class for question-answer evaluation with branching."""
    
    def __init__(self, config_path: str):
        """Initialize experiment with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.setup_logging()
        self.setup_directories()
        
        # Initialize embedder
        self.embedder = EmbeddingManager()
        
        # Results storage
        self.results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'config': self.config
            },
            'knowledge_loading': [],
            'question_results': []
        }
        
    def setup_logging(self):
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['data']['output_dir'],
            self.config['data']['snapshot_dir'],
            os.path.join(self.config['data']['output_dir'], 'responses'),
            os.path.join(self.config['data']['output_dir'], 'insights'),
            '../results/metrics',
            '../results/visualizations'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base from JSON file."""
        with open(self.config['data']['knowledge_base_path'], 'r') as f:
            data = json.load(f)
        return data['knowledge_entries']
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load question sets from JSON file."""
        with open(self.config['data']['questions_path'], 'r') as f:
            data = json.load(f)
        return data['questions']
        
    def initialize_components(self) -> Tuple[Any, AdaptiveProcessor]:
        """Initialize InsightSpike components."""
        # Initialize DataStore
        datastore = DataStoreFactory.create(
            self.config['insightspike']['datastore']['type'],
            base_path=self.config['insightspike']['datastore']['base_path']
        )
        
        # Initialize layers
        l1_error = ErrorMonitor()
        
        # Use CachedMemoryManager for better performance
        l2_memory = CachedMemoryManager(
            datastore=datastore,
            cache_size=100,
            embedder=self.embedder
        )
        
        # Initialize LLM config
        llm_config = {
            'provider': self.config['model']['provider'],
            'model': self.config['model']['model'],
            'temperature': self.config['model']['temperature'],
            'max_tokens': self.config['model']['max_tokens'],
            'prompt_style': 'association_extended',  # Always use association_extended for branching
            'branching_threshold': self.config.get('branching', {}).get('threshold', 0.8),
            'branching_min_branches': self.config.get('branching', {}).get('min_branches', 2),
            'branching_max_gap': self.config.get('branching', {}).get('max_gap', 0.15)
        }
        
        # Add API key if available
        if self.config['model']['provider'] == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                llm_config['api_key'] = api_key
            else:
                self.logger.warning("No ANTHROPIC_API_KEY found in environment")
                
        # Initialize L4 LLM
        l4_llm = L4LLMInterface(config={'llm': llm_config})
        if not l4_llm.initialize():
            raise Exception("Failed to initialize LLM")
            
        # Initialize L3 Graph Reasoner
        graph_config = self.config['insightspike']['graph']
        l3_graph = L3GraphReasoner(config={'graph': graph_config})
        l3_graph.initialize()
        
        # Create exploration loop
        exploration_loop = ExplorationLoop(
            l1_monitor=l1_error,
            l2_memory=l2_memory,
            l3_graph=l3_graph
        )
        
        # Create adaptive processor
        strategy = ExpandingStrategy()
        topk_calculator = AdaptiveTopKCalculator()
        
        adaptive_processor = AdaptiveProcessor(
            exploration_loop=exploration_loop,
            strategy=strategy,
            topk_calculator=topk_calculator,
            l4_llm=l4_llm,
            datastore=datastore,
            max_attempts=5
        )
        
        return datastore, adaptive_processor
        
    def add_knowledge_with_tracking(
        self, 
        datastore: Any,
        memory_manager: CachedMemoryManager,
        knowledge_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add knowledge and track if insight is generated."""
        start_time = time.time()
        
        # Create episode
        episode = {
            'id': f"episode_{knowledge_entry['id']}",
            'text': knowledge_entry['content'],
            'vec': self.embedder.get_embedding(knowledge_entry['content']),
            'c_value': 0.5,
            'metadata': {
                'tags': knowledge_entry['tags'],
                'difficulty': knowledge_entry['difficulty'],
                'related_concepts': knowledge_entry['related_concepts']
            }
        }
        
        # Add to memory (this might trigger insight detection)
        result = memory_manager.add_episode(
            text=episode['text'],
            c_value=episode['c_value'],
            metadata=episode['metadata']
        )
        
        # Check if insight was generated
        # Note: During knowledge loading, insights are not detected
        # Insights only occur during question processing when graph changes
        has_insight = False
        insight_content = None
        
        return {
            'knowledge_id': knowledge_entry['id'],
            'content': knowledge_entry['content'],
            'has_insight': has_insight,
            'insight_content': insight_content,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
    def process_question_with_branching(
        self,
        adaptive_processor: AdaptiveProcessor,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a question with branching detection and custom prompts."""
        start_time = time.time()
        
        # Process the question
        result = adaptive_processor.process(question['question'], verbose=True)
        
        # Extract all needed information
        processing_time = time.time() - start_time
        
        # Get branching info from result if available
        branching_info = result.get('branching_info', {})
        llm_prompt = result.get('prompt', None)
        
        # Get response vector
        response_vector = None
        if result.get('response'):
            response_vector = self.embedder.get_embedding(result['response'])
            
        # Calculate similarity to insight episode if spike detected
        insight_similarity = 0.0
        if result.get('has_spike') and result.get('spike_episode_id'):
            # TODO: Get actual insight episode vector and calculate similarity
            insight_similarity = 0.0  # Set to 0 until proper implementation
        
        # Extract GED and IG values from result
        ged_value = result.get('ged_value', None)
        ig_value = result.get('ig_value', None)
        
        # Extract multihop results if available
        multihop_results = None
        if 'adaptive_metadata' in result and 'final_metrics' in result['adaptive_metadata']:
            # Check if graph_analysis contains multihop results
            graph_analysis = result.get('graph_analysis', {})
            if 'multihop_results' in graph_analysis:
                multihop_results = graph_analysis['multihop_results']
        
        return {
            'question_id': question['id'],
            'question': question['question'],
            'difficulty': question['difficulty'],
            'expected_tags': question['expected_tags'],
            'has_insight': result.get('has_spike', False),
            'has_branching': branching_info.get('has_branching', False),
            'insight_episode_vector': None,  # TODO: Extract from result
            'branching_episode_vector': None,  # TODO: Extract if branching
            'llm_prompt': llm_prompt,
            'llm_response': result.get('response', ''),
            'llm_response_vector': response_vector.tolist() if response_vector is not None else None,
            'insight_similarity': insight_similarity,
            'ged_value': ged_value,
            'ig_value': ig_value,
            'multihop_results': multihop_results,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'total_attempts': result.get('total_attempts', 1),
                'reasoning_quality': result.get('reasoning_quality', 0.0),
                'retrieved_doc_count': result.get('retrieved_doc_count', 0),
                'branching_metadata': branching_info,
                'cycles': result.get('cycles', 1)
            }
        }
        
    def save_results(self):
        """Save experiment results in JSON and CSV formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full JSON results
        json_path = f"../results/metrics/experiment_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save question results as CSV
        csv_path = f"../results/metrics/question_results_{timestamp}.csv"
        if self.results['question_results']:
            fieldnames = [
                'question_id', 'question', 'difficulty', 'has_insight', 
                'has_branching', 'ged_1hop', 'ig_1hop', 'ged_2hop', 'ig_2hop',
                'processing_time', 'insight_similarity', 'high_relevance_count', 'top_relevance_gap'
            ]
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results['question_results']:
                    # Extract hop-specific values
                    ged_1hop = result.get('ged_value', '')  # Default single value
                    ig_1hop = result.get('ig_value', '')
                    ged_2hop = ''
                    ig_2hop = ''
                    
                    # Check for multihop results
                    if result.get('multihop_results') and 'hop_details' in result['multihop_results']:
                        hop_details = result['multihop_results']['hop_details']
                        if 1 in hop_details:
                            ged_1hop = hop_details[1].get('ged', '')
                            ig_1hop = hop_details[1].get('ig', '')
                        if 2 in hop_details:
                            ged_2hop = hop_details[2].get('ged', '')
                            ig_2hop = hop_details[2].get('ig', '')
                    
                    row = {
                        'question_id': result['question_id'],
                        'question': result['question'][:100],  # Truncate for CSV
                        'difficulty': result['difficulty'],
                        'has_insight': result['has_insight'],
                        'has_branching': result['has_branching'],
                        'ged_1hop': ged_1hop,
                        'ig_1hop': ig_1hop,
                        'ged_2hop': ged_2hop,
                        'ig_2hop': ig_2hop,
                        'processing_time': result['processing_time'],
                        'insight_similarity': result['insight_similarity'],
                        'high_relevance_count': result['metadata']['branching_metadata'].get('high_relevance_count', 0),
                        'top_relevance_gap': result['metadata']['branching_metadata'].get('top_relevance_gap', 1.0)
                    }
                    writer.writerow(row)
                    
        self.logger.info(f"Results saved to {json_path} and {csv_path}")
        
    def finalize_experiment(self, datastore: Any):
        """Finalize experiment and create snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(
            self.config['data']['snapshot_dir'],
            timestamp
        )
        
        # Create snapshot directory
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy datastore if filesystem-based
        if hasattr(datastore, 'base_path'):
            import shutil
            shutil.copytree(
                datastore.base_path,
                os.path.join(snapshot_dir, 'datastore')
            )
            
        # Save experiment metadata
        metadata = {
            'timestamp': timestamp,
            'config': self.config,
            'summary': {
                'total_knowledge': len(self.results['knowledge_loading']),
                'total_questions': len(self.results['question_results']),
                'insight_rate': sum(1 for r in self.results['question_results'] if r['has_insight']) / len(self.results['question_results']) if self.results['question_results'] else 0,
                'branching_rate': sum(1 for r in self.results['question_results'] if r['has_branching']) / len(self.results['question_results']) if self.results['question_results'] else 0
            }
        }
        
        with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Experiment snapshot created at {snapshot_dir}")
        
    def run(self, test_mode: bool = False):
        """Run the complete experiment."""
        self.logger.info("Starting Question-Answer Experiment with Branching Detection")
        
        # Initialize components
        datastore, adaptive_processor = self.initialize_components()
        memory_manager = adaptive_processor.exploration_loop.l2_memory
        
        # Phase 1: Load knowledge base
        self.logger.info("Phase 1: Loading knowledge base...")
        knowledge_base = self.load_knowledge_base()
        
        # Limit for testing
        if test_mode:
            knowledge_base = knowledge_base[:10]
            
        for i, entry in enumerate(knowledge_base):
            if i % 50 == 0:
                self.logger.info(f"Loading knowledge {i+1}/{len(knowledge_base)}")
                
            result = self.add_knowledge_with_tracking(datastore, memory_manager, entry)
            self.results['knowledge_loading'].append(result)
            
            if result['has_insight']:
                self.logger.info(f"Insight generated for {entry['id']}")
                
        # Phase 2: Process questions
        self.logger.info("Phase 2: Processing questions...")
        questions = self.load_questions()
        
        # Limit for testing
        if test_mode:
            questions = questions[:5]
            
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}: {question['id']}")
            
            result = self.process_question_with_branching(adaptive_processor, question)
            self.results['question_results'].append(result)
            
            # Log key findings
            if result['has_insight']:
                self.logger.info(f"  → Insight detected!")
            if result['has_branching']:
                self.logger.info(f"  → Branching detected (high_relevance: {result['metadata']['branching_metadata'].get('high_relevance_count', 0)})")
                
        # Save results
        self.results['metadata']['end_time'] = datetime.now().isoformat()
        self.save_results()
        
        # Finalize
        self.finalize_experiment(datastore)
        
        # Print summary
        self.logger.info("\n=== Experiment Summary ===")
        self.logger.info(f"Total knowledge loaded: {len(self.results['knowledge_loading'])}")
        self.logger.info(f"Total questions processed: {len(self.results['question_results'])}")
        
        if self.results['question_results']:
            insight_count = sum(1 for r in self.results['question_results'] if r['has_insight'])
            branching_count = sum(1 for r in self.results['question_results'] if r['has_branching'])
            
            self.logger.info(f"Insight rate: {insight_count}/{len(self.results['question_results'])} ({insight_count/len(self.results['question_results'])*100:.1f}%)")
            self.logger.info(f"Branching rate: {branching_count}/{len(self.results['question_results'])} ({branching_count/len(self.results['question_results'])*100:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Question-Answer Experiment')
    parser.add_argument(
        '--config',
        default='experiment_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with limited data'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = QuestionAnswerExperiment(args.config)
    experiment.run(test_mode=args.test)
    

if __name__ == '__main__':
    main()