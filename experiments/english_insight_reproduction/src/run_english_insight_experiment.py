#!/usr/bin/env python3
"""
English Insight Reproduction Experiment
======================================

Reproducing the successful English insight experiment with current implementation.
Using DistilGPT2 for local execution.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Capture stderr during imports to suppress warnings
from io import StringIO
old_stderr = sys.stderr
sys.stderr = StringIO()

try:
    from src.insightspike.config import load_config
    from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
    from src.insightspike.implementations.agents.main_agent import MainAgent
finally:
    sys.stderr = old_stderr

# Set random seeds for reproducibility
random.seed(42)
import numpy as np
np.random.seed(42)

# Set environment variable to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EnglishInsightExperiment:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.data_dir = experiment_dir / "data"
        self.results_dir = experiment_dir / "results"
        
        # Load configuration
        self.config = load_config(config_path=str(experiment_dir / "config_experiment.yaml"))
        
        # Initialize SQLite datastore
        self.db_path = self.data_dir / "processed" / "insightspike.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.datastore = SQLiteDataStore(str(self.db_path))
        
        # Initialize agent
        print("Initializing MainAgent with DistilGPT2...")
        try:
            self.agent = MainAgent(self.config)
            print("✓ MainAgent initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing MainAgent: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load the English knowledge base"""
        kb_path = self.data_dir / "input" / "english_knowledge_base.json"
        with open(kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['episodes']
    
    def load_test_questions(self) -> List[Dict[str, Any]]:
        """Load test questions"""
        q_path = self.data_dir / "input" / "test_questions.json"
        with open(q_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']
    
    def inject_knowledge(self, episodes: List[Dict[str, Any]]):
        """Inject knowledge episodes into the system"""
        print(f"\n=== Injecting {len(episodes)} knowledge episodes ===")
        
        for i, episode in enumerate(episodes):
            # Add knowledge to the system
            self.agent.add_knowledge(
                episode['text'],
                metadata={
                    'episode_id': episode['id'],
                    'phase': episode['phase'],
                    'source': 'english_knowledge_base'
                }
            )
            
            if (i + 1) % 10 == 0:
                print(f"  Injected {i + 1}/{len(episodes)} episodes...")
        
        print(f"  ✓ All {len(episodes)} episodes injected")
        
        # Get initial graph stats
        stats = self.get_graph_stats()
        print(f"\nInitial graph structure:")
        print(f"  - Nodes: {stats['nodes']}")
        print(f"  - Edges: {stats['edges']}")
        print(f"  - Density: {stats['density']:.4f}")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        graph = self.agent.layer3_graph_reasoner.query_graph.graph
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'density': density
        }
    
    def test_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test the system with questions"""
        print(f"\n=== Testing with {len(questions)} questions ===")
        
        results = []
        
        for q in questions:
            print(f"\nQuestion {q['id']}: {q['question']}")
            
            # Get graph stats before
            stats_before = self.get_graph_stats()
            
            # Process question
            start_time = time.time()
            result = self.agent.process_question(q['question'])
            processing_time = time.time() - start_time
            
            # Get graph stats after
            stats_after = self.get_graph_stats()
            
            # Calculate graph complexity change
            complexity_change = (
                (stats_after['nodes'] + stats_after['edges']) / 
                (stats_before['nodes'] + stats_before['edges'])
            ) - 1.0
            
            # Prepare result
            test_result = {
                'question_id': q['id'],
                'question': q['question'],
                'has_spike': result.get('has_spike', False),
                'spike_confidence': result.get('spike_info', {}).get('confidence', 0),
                'response': result.get('response', ''),
                'processing_time': processing_time,
                'graph_stats_before': stats_before,
                'graph_stats_after': stats_after,
                'complexity_change': complexity_change,
                'expected_spike': q['expected_spike']
            }
            
            results.append(test_result)
            
            # Display result
            if result.get('has_spike'):
                print(f"  ✨ SPIKE DETECTED! (confidence: {test_result['spike_confidence']:.3f})")
            else:
                print(f"  📝 No spike detected")
            
            print(f"  Response: {result.get('response', 'No response')[:150]}...")
            print(f"  Graph complexity change: {complexity_change:.1%}")
            print(f"  Processing time: {processing_time:.2f}s")
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results"""
        total_questions = len(results)
        correct_spike_detections = sum(
            1 for r in results 
            if r['has_spike'] == r['expected_spike']
        )
        
        spike_detected = sum(1 for r in results if r['has_spike'])
        avg_confidence = sum(r['spike_confidence'] for r in results if r['has_spike']) / max(spike_detected, 1)
        
        avg_complexity_change = sum(r['complexity_change'] for r in results) / total_questions
        avg_processing_time = sum(r['processing_time'] for r in results) / total_questions
        
        analysis = {
            'total_questions': total_questions,
            'spike_detection_accuracy': correct_spike_detections / total_questions,
            'spikes_detected': spike_detected,
            'average_spike_confidence': avg_confidence,
            'average_complexity_change': avg_complexity_change,
            'average_processing_time': avg_processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / "outputs" / f"detailed_results_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': 'english_insight_reproduction',
                'model': 'distilgpt2',
                'results': results,
                'analysis': analysis
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save summary report
        report_file = self.results_dir / "outputs" / f"summary_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(results, analysis))
        
        print(f"Report saved to: {report_file}")
    
    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate experiment report"""
        report = f"""# English Insight Reproduction Experiment Report

## Experiment Overview
- **Date**: {analysis['timestamp']}
- **Model**: DistilGPT2 (local)
- **Knowledge Episodes**: 50
- **Test Questions**: {analysis['total_questions']}

## Quantitative Results

| Metric | Value |
|--------|-------|
| Spike Detection Accuracy | {analysis['spike_detection_accuracy']:.1%} |
| Spikes Detected | {analysis['spikes_detected']}/{analysis['total_questions']} |
| Average Spike Confidence | {analysis['average_spike_confidence']:.3f} |
| Average Graph Complexity Change | {analysis['average_complexity_change']:.1%} |
| Average Processing Time | {analysis['average_processing_time']:.2f}s |

## Detailed Results by Question

"""
        for r in results:
            spike_status = "✓ SPIKE" if r['has_spike'] else "✗ No spike"
            expected = "✓" if r['has_spike'] == r['expected_spike'] else "✗"
            
            report += f"""### Question {r['question_id']}: {r['question']}
- **Result**: {spike_status} (Expected: {'spike' if r['expected_spike'] else 'no spike'}) {expected}
- **Confidence**: {r['spike_confidence']:.3f}
- **Graph Change**: {r['complexity_change']:.1%} (Nodes: {r['graph_stats_before']['nodes']}→{r['graph_stats_after']['nodes']}, Edges: {r['graph_stats_before']['edges']}→{r['graph_stats_after']['edges']})
- **Response Preview**: {r['response'][:200]}...

"""
        
        report += """## Comparison with Original Experiment

The original English insight experiment achieved:
- **Spike Detection Rate**: 83.3% (5/6 questions)
- **Average Graph Complexity Increase**: 127.4%

Current implementation:
- **Spike Detection Accuracy**: {:.1%}
- **Average Graph Complexity Change**: {:.1%}

""".format(analysis['spike_detection_accuracy'], analysis['average_complexity_change'])
        
        return report
    
    def run(self):
        """Run the complete experiment"""
        print("=" * 60)
        print("English Insight Reproduction Experiment")
        print("Using DistilGPT2 (Local Model)")
        print("=" * 60)
        
        # Load data
        episodes = self.load_knowledge_base()
        questions = self.load_test_questions()
        
        # Inject knowledge
        self.inject_knowledge(episodes)
        
        # Test with questions
        results = self.test_questions(questions)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Display summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Spike Detection Accuracy: {analysis['spike_detection_accuracy']:.1%}")
        print(f"Spikes Detected: {analysis['spikes_detected']}/{analysis['total_questions']}")
        print(f"Average Spike Confidence: {analysis['average_spike_confidence']:.3f}")
        print(f"Average Graph Complexity Change: {analysis['average_complexity_change']:.1%}")
        print(f"Average Processing Time: {analysis['average_processing_time']:.2f}s")
        
        # Save results
        self.save_results(results, analysis)
        
        # Create data snapshot
        self.create_data_snapshot()
        
        print("\n✅ Experiment completed successfully!")
    
    def create_data_snapshot(self):
        """Create a snapshot of experiment data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.experiment_dir / "data_snapshots" / f"snapshot_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy database
        import shutil
        shutil.copy2(self.db_path, snapshot_dir / "insightspike.db")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'experiment': 'english_insight_reproduction',
            'model': 'distilgpt2',
            'config': self.config.dict()
        }
        
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nData snapshot created: {snapshot_dir}")


if __name__ == "__main__":
    experiment_dir = Path(__file__).parent.parent
    experiment = EnglishInsightExperiment(experiment_dir)
    experiment.run()