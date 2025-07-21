"""
English Insight Experiment with DataStore Architecture
======================================================

Re-implementation of the English insight experiment using the new
DataStore-centric architecture for better scalability.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from src.insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from src.insightspike.config.models import (
    InsightSpikeConfig, MemoryConfig, LLMConfig, GraphConfig
)


class EnglishInsightExperiment:
    """Run English insight detection experiment with DataStore"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.data_dir = experiment_dir / "data"
        self.results_dir = experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize DataStore
        db_path = str(self.data_dir / "experiment.db")
        self.datastore = SQLiteDataStore(db_path, vector_dim=384)
        
        # Create configuration
        self.config = self._create_config()
        
        # Initialize agent
        self.agent = DataStoreMainAgent(
            datastore=self.datastore,
            config=self.config
        )
        
    def _create_config(self) -> Dict[str, Any]:
        """Create experiment configuration"""
        # Return a simple dict config
        return {
            'memory': {
                'max_episodes': 10000,
                'batch_size': 32,
                'working_memory_size': 100,
                'search_k': 20,
                'similarity_threshold': 0.7,
                'datastore_namespace': 'english_insights'
            },
            'llm': {
                'provider': 'mock',
                'model': 'mock-model'
            }
        }
    
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base from backup"""
        kb_path = self.data_dir / "input" / "english_knowledge_base.json"
        
        if not kb_path.exists():
            # Try backup location
            backup_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/ISbackups/0718/english_insight_experiment/data/input/english_knowledge_base.json")
            if backup_path.exists():
                print(f"Loading knowledge base from backup: {backup_path}")
                with open(backup_path, 'r') as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, dict):
                        # If it's a dict with 'items' or similar key
                        if 'episodes' in data:
                            return data['episodes']
                        elif 'items' in data:
                            return data['items']
                        elif 'knowledge_base' in data:
                            return data['knowledge_base']
                        else:
                            # Skip metadata and convert remaining items
                            items = []
                            for k, v in data.items():
                                if k != 'metadata':
                                    if isinstance(v, dict):
                                        items.append({'id': k, **v})
                                    else:
                                        items.append({'id': k, 'text': v})
                            return items if items else self._create_sample_knowledge_base()
                    elif isinstance(data, list):
                        return data
                    else:
                        print(f"Warning: Unexpected data format: {type(data)}")
                        return self._create_sample_knowledge_base()
            else:
                print("Warning: No knowledge base found, using sample data")
                return self._create_sample_knowledge_base()
        
        with open(kb_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'episodes' in data:
                return data['episodes']
            return data
    
    def _create_sample_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create sample knowledge base for testing"""
        return [
            {
                "id": "kb_1",
                "text": "Water is essential for life",
                "category": "science",
                "metadata": {"source": "basic_facts"}
            },
            {
                "id": "kb_2", 
                "text": "The Earth orbits around the Sun",
                "category": "science",
                "metadata": {"source": "astronomy"}
            },
            {
                "id": "kb_3",
                "text": "Python is a programming language",
                "category": "technology",
                "metadata": {"source": "computing"}
            }
        ]
    
    def populate_knowledge_base(self, knowledge_base: List[Dict[str, Any]]):
        """Populate DataStore with knowledge base"""
        print(f"\nKnowledge base type: {type(knowledge_base)}")
        if isinstance(knowledge_base, dict):
            print(f"Knowledge base keys: {list(knowledge_base.keys())}")
            # Convert to list if needed
            if 'episodes' in knowledge_base:
                knowledge_base = knowledge_base['episodes']
            else:
                print("Error: Expected 'episodes' key in knowledge base")
                return
                
        print(f"Populating knowledge base with {len(knowledge_base)} items...")
        
        for i, item in enumerate(knowledge_base):
            # Check item type
            if not isinstance(item, dict):
                print(f"  Warning: Item {i} is not a dict: {type(item)}")
                continue
                
            # Store each knowledge item
            result = self.agent.process(
                text=item.get('text', ''),
                metadata={
                    'kb_id': item.get('id'),
                    'category': item.get('category'),
                    'phase': item.get('phase'),
                    'source': 'knowledge_base',
                    **item.get('metadata', {})
                }
            )
            
            if result.get('error'):
                print(f"  Error storing {item.get('id')}: {result['error']}")
            elif i < 3:  # Show first few
                print(f"  Stored {item.get('id')}: {item.get('text', '')[:50]}...")
        
        print("Knowledge base populated")
    
    def run_test_queries(self) -> List[Dict[str, Any]]:
        """Run test queries to detect insights"""
        test_queries = [
            {
                "id": "q1",
                "text": "H2O is the chemical formula for water",
                "expected": "insight",
                "reason": "Connects chemistry to existing knowledge"
            },
            {
                "id": "q2",
                "text": "Water is essential for life",
                "expected": "no_insight",
                "reason": "Already in knowledge base"
            },
            {
                "id": "q3",
                "text": "Machine learning models can be trained using Python",
                "expected": "insight",
                "reason": "Connects ML to programming language"
            },
            {
                "id": "q4",
                "text": "Pluto was reclassified as a dwarf planet",
                "expected": "insight",
                "reason": "New astronomical information"
            },
            {
                "id": "q5",
                "text": "The Sun is a star",
                "expected": "maybe",
                "reason": "Related to existing knowledge but adds detail"
            }
        ]
        
        results = []
        print(f"\nRunning {len(test_queries)} test queries...")
        
        for query in test_queries:
            print(f"\nProcessing: {query['text'][:50]}...")
            
            # Process query
            result = self.agent.process(
                text=query['text'],
                metadata={'query_id': query['id']}
            )
            
            # Add query info to result
            result['query'] = query
            result['expected'] = query['expected']
            
            # Print summary
            if result.get('has_spike'):
                print(f"  ✓ Insight detected (confidence: {result.get('spike_info', {}).get('confidence', 0):.2f})")
            else:
                print(f"  - No insight detected")
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results"""
        analysis = {
            'total_queries': len(results),
            'insights_detected': sum(1 for r in results if r.get('has_spike')),
            'processing_times': [r.get('processing_time', 0) for r in results],
            'accuracy': 0.0
        }
        
        # Calculate accuracy
        correct = 0
        for r in results:
            expected = r.get('expected', 'unknown')
            detected = 'insight' if r.get('has_spike') else 'no_insight'
            
            if expected == 'maybe':
                correct += 0.5  # Partial credit
            elif expected == detected:
                correct += 1
        
        analysis['accuracy'] = correct / len(results) if results else 0
        analysis['avg_processing_time'] = np.mean(analysis['processing_times'])
        
        return analysis
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"datastore_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'english_insight_datastore',
                'timestamp': timestamp,
                'results': results,
                'analysis': analysis,
                'config': self.config.dict()
            }, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for r in results:
            summary_data.append({
                'query_id': r['query']['id'],
                'text': r['query']['text'],
                'expected': r['expected'],
                'detected': 'insight' if r.get('has_spike') else 'no_insight',
                'confidence': r.get('spike_info', {}).get('confidence', 0),
                'processing_time': r.get('processing_time', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"datastore_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")
    
    def run(self):
        """Run the complete experiment"""
        print("=== English Insight Experiment (DataStore Architecture) ===")
        
        # Load and populate knowledge base
        knowledge_base = self.load_knowledge_base()
        self.populate_knowledge_base(knowledge_base)
        
        # Get initial stats
        stats = self.agent.get_stats()
        print(f"\nInitial stats: {json.dumps(stats, indent=2)}")
        
        # Run test queries
        results = self.run_test_queries()
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Print analysis
        print("\n=== Analysis ===")
        print(f"Total queries: {analysis['total_queries']}")
        print(f"Insights detected: {analysis['insights_detected']}")
        print(f"Accuracy: {analysis['accuracy']:.2%}")
        print(f"Avg processing time: {analysis['avg_processing_time']:.3f}s")
        
        # Save results
        self.save_results(results, analysis)
        
        # Final stats
        final_stats = self.agent.get_stats()
        print(f"\nFinal stats: {json.dumps(final_stats, indent=2)}")
        
        print("\n=== Experiment Complete ===")


def main():
    """Main entry point"""
    # Use v2 experiment directory
    experiment_dir = Path(__file__).parent.parent
    
    # Create experiment
    experiment = EnglishInsightExperiment(experiment_dir)
    
    # Run experiment
    experiment.run()


if __name__ == "__main__":
    main()