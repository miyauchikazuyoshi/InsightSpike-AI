#!/usr/bin/env python3
"""
Question-Answer Experiment Runner using MainAgent (not adaptive_loop)
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

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.processing.embedder import EmbeddingManager


class QuestionAnswerExperimentMainAgent:
    """Experiment using MainAgent.process_question instead of adaptive_loop."""
    
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
                'config': self.config,
                'approach': 'MainAgent.process_question'
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
        
    def initialize_agent(self) -> MainAgent:
        """Initialize MainAgent with configuration."""
        # Convert experiment config to MainAgent config format
        agent_config = {
            "embedder": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "llm": {
                "provider": self.config['model']['provider'],
                "model": self.config['model']['model'],
                "temperature": self.config['model']['temperature'],
                "max_tokens": self.config['model']['max_tokens'],
                "prompt_style": "association_extended"
            },
            "vector_storage": {
                "index_type": "flat",
                "dimension": 384
            },
            "graph": {
                "edge_formation": {
                    "strategy": "radius",
                    "radius": 0.3
                },
                "algorithms": {
                    "metrics": {
                        "ged": "pyg",
                        "ig": "pyg"
                    }
                },
                "spike_detection": {
                    "ged_threshold": self.config['insightspike']['spike_detection']['ged_threshold'],
                    "ig_threshold": self.config['insightspike']['spike_detection']['ig_threshold']
                }
            }
        }
        
        return MainAgent(config=agent_config)
        
    def add_knowledge_with_tracking(
        self, 
        agent: MainAgent,
        knowledge_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add knowledge and track if insight is generated."""
        start_time = time.time()
        
        # Add knowledge
        agent.add_knowledge(knowledge_entry['content'])
        
        # Note: MainAgent.add_knowledge doesn't return spike info
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
        
    def process_question(
        self,
        agent: MainAgent,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a question using MainAgent."""
        start_time = time.time()
        
        # Process the question
        result = agent.process_question(question['question'])
        
        # Extract all needed information
        processing_time = time.time() - start_time
        
        # Get response vector
        response_vector = None
        if hasattr(result, 'response') and result.response:
            response_vector = self.embedder.get_embedding(result.response)
            
        # Extract GED/IG values from graph_analysis
        ged_value = None
        ig_value = None
        if hasattr(result, 'graph_analysis') and result.graph_analysis:
            metrics = result.graph_analysis.get('metrics', {})
            ged_value = metrics.get('delta_ged')
            ig_value = metrics.get('delta_ig')
        
        return {
            'question_id': question['id'],
            'question': question['question'],
            'difficulty': question['difficulty'],
            'expected_tags': question['expected_tags'],
            'has_insight': result.spike_detected if hasattr(result, 'spike_detected') else False,
            'has_branching': False,  # MainAgent doesn't support branching detection
            'insight_episode_vector': None,
            'branching_episode_vector': None,
            'llm_prompt': None,  # Not available in MainAgent
            'llm_response': result.response if hasattr(result, 'response') else '',
            'llm_response_vector': response_vector.tolist() if response_vector is not None else None,
            'insight_similarity': 0.0,
            'ged_value': ged_value,
            'ig_value': ig_value,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'total_attempts': 1,
                'reasoning_quality': result.reasoning_quality if hasattr(result, 'reasoning_quality') else 0.0,
                'retrieved_doc_count': len(result.retrieved_documents) if hasattr(result, 'retrieved_documents') else 0,
                'branching_metadata': {},
                'cycles': result.cycle_number if hasattr(result, 'cycle_number') else 1
            }
        }
        
    def save_results(self):
        """Save experiment results in JSON and CSV formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full JSON results
        json_path = f"../results/metrics/mainagent_experiment_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save question results as CSV
        csv_path = f"../results/metrics/mainagent_question_results_{timestamp}.csv"
        if self.results['question_results']:
            fieldnames = [
                'question_id', 'question', 'difficulty', 'has_insight', 
                'has_branching', 'ged_value', 'ig_value', 'processing_time', 
                'insight_similarity', 'high_relevance_count', 'top_relevance_gap'
            ]
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results['question_results']:
                    row = {
                        'question_id': result['question_id'],
                        'question': result['question'][:100],  # Truncate for CSV
                        'difficulty': result['difficulty'],
                        'has_insight': result['has_insight'],
                        'has_branching': result['has_branching'],
                        'ged_value': result.get('ged_value', ''),
                        'ig_value': result.get('ig_value', ''),
                        'processing_time': result['processing_time'],
                        'insight_similarity': result['insight_similarity'],
                        'high_relevance_count': 0,
                        'top_relevance_gap': 1.0
                    }
                    writer.writerow(row)
                    
        self.logger.info(f"Results saved to {json_path} and {csv_path}")
        
    def finalize_experiment(self):
        """Finalize experiment and create snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(
            self.config['data']['snapshot_dir'],
            f"mainagent_{timestamp}"
        )
        
        # Create snapshot directory
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            
        # Save experiment metadata
        metadata = {
            'timestamp': timestamp,
            'approach': 'MainAgent',
            'config': self.config,
            'summary': {
                'total_knowledge': len(self.results['knowledge_loading']),
                'total_questions': len(self.results['question_results']),
                'insight_rate': sum(1 for r in self.results['question_results'] if r['has_insight']) / len(self.results['question_results']) if self.results['question_results'] else 0,
                'branching_rate': 0  # Not supported
            }
        }
        
        with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Experiment snapshot created at {snapshot_dir}")
        
    def run(self, test_mode: bool = False):
        """Run the complete experiment."""
        self.logger.info("Starting Question-Answer Experiment with MainAgent")
        
        # Initialize agent
        agent = self.initialize_agent()
        
        # Phase 1: Load knowledge base
        self.logger.info("Phase 1: Loading knowledge base...")
        knowledge_base = self.load_knowledge_base()
        
        # Limit for testing
        if test_mode:
            knowledge_base = knowledge_base[:10]
            
        for i, entry in enumerate(knowledge_base):
            if i % 50 == 0:
                self.logger.info(f"Loading knowledge {i+1}/{len(knowledge_base)}")
                
            result = self.add_knowledge_with_tracking(agent, entry)
            self.results['knowledge_loading'].append(result)
                
        # Phase 2: Process questions
        self.logger.info("Phase 2: Processing questions...")
        questions = self.load_questions()
        
        # Limit for testing
        if test_mode:
            questions = questions[:5]
            
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}: {question['id']}")
            
            result = self.process_question(agent, question)
            self.results['question_results'].append(result)
            
            # Log key findings
            if result['has_insight']:
                self.logger.info(f"  → Insight detected!")
            if result['ged_value'] is not None:
                self.logger.info(f"  → GED: {result['ged_value']:.2f}, IG: {result['ig_value']:.4f}")
                
        # Save results
        self.results['metadata']['end_time'] = datetime.now().isoformat()
        self.save_results()
        
        # Finalize
        self.finalize_experiment()
        
        # Print summary
        self.logger.info("\n=== Experiment Summary ===")
        self.logger.info(f"Total knowledge loaded: {len(self.results['knowledge_loading'])}")
        self.logger.info(f"Total questions processed: {len(self.results['question_results'])}")
        
        if self.results['question_results']:
            insight_count = sum(1 for r in self.results['question_results'] if r['has_insight'])
            
            self.logger.info(f"Insight rate: {insight_count}/{len(self.results['question_results'])} ({insight_count/len(self.results['question_results'])*100:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Question-Answer Experiment with MainAgent')
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
    experiment = QuestionAnswerExperimentMainAgent(args.config)
    experiment.run(test_mode=args.test)
    

if __name__ == '__main__':
    main()