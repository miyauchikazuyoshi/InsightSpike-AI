#!/usr/bin/env python3
"""
Full InsightSpike-AI RAG Experiment
===================================

Uses the actual InsightSpike-AI implementation for a comprehensive RAG experiment.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
from insightspike.core.agents.main_agent import MainAgent

import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightSpikeRAGExperiment:
    """Full experiment using InsightSpike-AI for RAG tasks"""
    
    def __init__(self, backup_dir="experiment_backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize InsightSpike components
        self.agent = MainAgent()
        self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.STANDARD)
        
        # Initialize agent
        if not self.agent.initialize():
            logger.warning("Failed to fully initialize MainAgent, continuing anyway")
        
        self.insights = []
        self.processing_times = []
        
    def backup_data(self):
        """Backup current data state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"data_backup_{timestamp}"
        
        # Backup data directory if it exists
        data_dir = Path("data")
        if data_dir.exists():
            shutil.copytree(data_dir, backup_path / "data")
            logger.info(f"Backed up data to {backup_path}")
        
        # Backup any agent state
        if hasattr(self.agent, 'save_state'):
            try:
                self.agent.save_state()
                logger.info("Agent state saved")
            except Exception as e:
                logger.warning(f"Could not save agent state: {e}")
    
    def create_test_documents(self, n_docs=100):
        """Create test documents for the experiment"""
        topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision",
            "reinforcement learning", "data science", "algorithms"
        ]
        
        documents = []
        for i in range(n_docs):
            topic = topics[i % len(topics)]
            doc_variations = [
                f"{topic} is a field of artificial intelligence that enables systems to learn from data.",
                f"The principles of {topic} involve mathematical models and computational algorithms.",
                f"Applications of {topic} include pattern recognition, prediction, and automation.",
                f"{topic} has revolutionized how we approach complex computational problems.",
                f"Recent advances in {topic} have led to breakthrough discoveries in various domains."
            ]
            doc = doc_variations[i % len(doc_variations)]
            documents.append(doc)
        
        return documents
    
    def process_documents(self, documents):
        """Process documents through InsightSpike-AI"""
        logger.info(f"Processing {len(documents)} documents...")
        
        results = []
        for i, doc in enumerate(documents):
            start_time = time.time()
            
            # Process through agent
            result = self.agent.process_question(
                f"Learn and remember this information: {doc}",
                max_cycles=2
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Check for insights
            if 'quality_score' in result:
                quality = result['quality_score']
                is_spike = quality > 0.8  # Threshold for spike
                
                if is_spike:
                    self.insights.append({
                        'document': doc,
                        'quality': quality,
                        'processing_time': processing_time,
                        'index': i
                    })
                    logger.info(f"Insight detected at document {i}: quality={quality:.3f}")
            
            results.append(result)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(documents)}")
        
        return results
    
    def test_retrieval(self, queries, documents):
        """Test retrieval performance"""
        logger.info(f"Testing retrieval with {len(queries)} queries...")
        
        retrieval_results = []
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            # Ask agent
            result = self.agent.process_question(query, max_cycles=3)
            
            retrieval_time = time.time() - start_time
            
            # Evaluate retrieval (simplified - check if relevant topic is mentioned)
            topic = query.split()[-1].rstrip('?.')
            relevant_found = topic.lower() in result.get('response', '').lower()
            
            retrieval_results.append({
                'query': query,
                'response': result.get('response', ''),
                'quality': result.get('quality_score', 0),
                'relevant': relevant_found,
                'time': retrieval_time
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Retrieval progress: {i+1}/{len(queries)}")
        
        return retrieval_results
    
    def analyze_results(self, processing_results, retrieval_results):
        """Analyze experiment results"""
        analysis = {
            'processing': {
                'total_documents': len(processing_results),
                'total_insights': len(self.insights),
                'insight_ratio': len(self.insights) / max(1, len(processing_results)),
                'avg_processing_time': np.mean(self.processing_times),
                'total_processing_time': sum(self.processing_times)
            },
            'retrieval': {
                'total_queries': len(retrieval_results),
                'successful_retrievals': sum(1 for r in retrieval_results if r['relevant']),
                'retrieval_accuracy': sum(1 for r in retrieval_results if r['relevant']) / max(1, len(retrieval_results)),
                'avg_retrieval_time': np.mean([r['time'] for r in retrieval_results]),
                'avg_quality_score': np.mean([r['quality'] for r in retrieval_results])
            },
            'insights': {
                'count': len(self.insights),
                'avg_quality': np.mean([i['quality'] for i in self.insights]) if self.insights else 0,
                'indices': [i['index'] for i in self.insights]
            }
        }
        
        return analysis
    
    def visualize_results(self, analysis):
        """Create visualizations"""
        output_dir = Path("results_full_insightspike")
        output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Processing times
        ax = axes[0, 0]
        ax.plot(self.processing_times, alpha=0.7)
        if self.insights:
            insight_indices = [i['index'] for i in self.insights]
            insight_times = [self.processing_times[i] for i in insight_indices]
            ax.scatter(insight_indices, insight_times, color='red', s=50, label='Insights')
        ax.set_xlabel('Document Index')
        ax.set_ylabel('Processing Time (s)')
        ax.set_title('Document Processing Times')
        ax.legend()
        
        # 2. Quality scores distribution
        ax = axes[0, 1]
        if self.insights:
            qualities = [i['quality'] for i in self.insights]
            ax.hist(qualities, bins=20, alpha=0.7, color='green')
            ax.axvline(np.mean(qualities), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(qualities):.3f}')
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Insight Quality Distribution')
            ax.legend()
        
        # 3. Summary metrics
        ax = axes[1, 0]
        metrics = [
            f"Documents: {analysis['processing']['total_documents']}",
            f"Insights: {analysis['processing']['total_insights']}",
            f"Insight Ratio: {analysis['processing']['insight_ratio']:.2%}",
            f"Retrieval Accuracy: {analysis['retrieval']['retrieval_accuracy']:.2%}",
            f"Avg Quality: {analysis['retrieval']['avg_quality_score']:.3f}"
        ]
        ax.text(0.1, 0.9, '\n'.join(metrics), transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Summary Metrics')
        ax.axis('off')
        
        # 4. Performance comparison
        ax = axes[1, 1]
        categories = ['Processing\nSpeed', 'Retrieval\nAccuracy', 'Insight\nDetection']
        values = [
            1.0 / (analysis['processing']['avg_processing_time'] + 0.001),  # Speed (inverse of time)
            analysis['retrieval']['retrieval_accuracy'],
            analysis['processing']['insight_ratio']
        ]
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('Performance Score')
        ax.set_title('InsightSpike-AI Performance')
        ax.set_ylim(0, 1.2)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'insightspike_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save analysis results
        with open(output_dir / 'experiment_analysis.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj
            
            json.dump(convert_to_serializable(analysis), f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        
        return output_dir

def main():
    """Run the full experiment"""
    print("="*60)
    print("Full InsightSpike-AI RAG Experiment")
    print("="*60)
    
    # Create experiment instance
    experiment = InsightSpikeRAGExperiment()
    
    # Backup current data
    print("\nBacking up current data...")
    experiment.backup_data()
    
    try:
        # Create test data
        print("\nCreating test documents...")
        documents = experiment.create_test_documents(100)
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What are the applications of computer vision?",
            "How does reinforcement learning differ from supervised learning?",
            "What is natural language processing used for?",
            "Describe data science methodologies",
            "What algorithms are used in machine learning?"
        ] * 5  # Repeat for more queries
        
        # Process documents
        print("\nProcessing documents through InsightSpike-AI...")
        processing_results = experiment.process_documents(documents)
        
        # Test retrieval
        print("\nTesting retrieval performance...")
        retrieval_results = experiment.test_retrieval(queries[:20], documents)
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = experiment.analyze_results(processing_results, retrieval_results)
        
        # Visualize results
        print("\nCreating visualizations...")
        output_dir = experiment.visualize_results(analysis)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Documents processed: {analysis['processing']['total_documents']}")
        print(f"Insights detected: {analysis['processing']['total_insights']} ({analysis['processing']['insight_ratio']:.1%})")
        print(f"Average processing time: {analysis['processing']['avg_processing_time']:.3f}s")
        print(f"Retrieval accuracy: {analysis['retrieval']['retrieval_accuracy']:.1%}")
        print(f"Average quality score: {analysis['retrieval']['avg_quality_score']:.3f}")
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise
    
    finally:
        # Restore clean data
        print("\n" + "="*60)
        print("Restoring clean data state...")
        print("="*60)
        
        # Run the restore script
        restore_script = Path("scripts/utilities/restore_clean_data.py")
        if restore_script.exists():
            os.system(f"python {restore_script}")
            print("Data restored to clean state!")
        else:
            print("Warning: Could not find restore script")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()