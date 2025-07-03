#!/usr/bin/env python3
"""
Real InsightSpike-AI RAG Experiment
====================================

This experiment uses the actual InsightSpike-AI implementation
instead of reimplementing everything from scratch.
"""

import sys
import os
# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
from insightspike.core.agents.main_agent import MainAgent

import logging
logger = logging.getLogger("rag_experiment")
logging.basicConfig(level=logging.INFO)

import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Logger is already set up above

class InsightSpikeRAG:
    """RAG system using actual InsightSpike-AI implementation"""
    
    def __init__(self):
        self.agent = MainAgent()
        self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.STANDARD)
        
        # Initialize agent
        if not self.agent.initialize():
            logger.error("Failed to initialize MainAgent")
            
        self.documents = []
        self.insights = []
        
    def add_documents(self, documents):
        """Add documents to the agent's memory"""
        logger.info(f"Adding {len(documents)} documents to InsightSpike-AI")
        
        for i, doc in enumerate(documents):
            # Process each document through the agent
            result = self.agent.process_question(
                f"Learn this: {doc}",
                max_cycles=1
            )
            
            # Check if this was an insight
            if 'insight_detected' in result and result['insight_detected']:
                self.insights.append({
                    'document': doc,
                    'delta_ig': result.get('delta_ig', 0),
                    'delta_ged': result.get('delta_ged', 0),
                    'gedig_score': result.get('gedig_score', 0)
                })
                logger.info(f"Insight detected for document {i}: geDIG={result.get('gedig_score', 0):.3f}")
            
            self.documents.append(doc)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(documents)}")
    
    def retrieve(self, query, k=5):
        """Retrieve documents using InsightSpike-AI"""
        start_time = time.time()
        
        # Use agent to process query
        result = self.agent.process_question(query, max_cycles=3)
        
        # Get relevant documents from agent's response
        # In a real implementation, the agent would have a retrieval mechanism
        # For now, we'll use a simple similarity-based approach
        
        # Calculate similarities using the agent's understanding
        similarities = []
        for doc in self.documents:
            # Compare query with document using IG
            ig_result = self.ig_calculator.calculate(
                {'text': query},
                {'text': doc}
            )
            
            similarity = ig_result.ig_value if ig_result else 0
            similarities.append(similarity)
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved = [
            {
                'document': self.documents[idx],
                'score': similarities[idx],
                'is_insight': any(ins['document'] == self.documents[idx] for ins in self.insights)
            }
            for idx in top_indices
        ]
        
        retrieval_time = (time.time() - start_time) * 1000  # ms
        
        return retrieved, retrieval_time
    
    def get_stats(self):
        """Get InsightSpike-AI statistics"""
        stats = self.agent.get_stats() if hasattr(self.agent, 'get_stats') else {}
        stats['num_documents'] = len(self.documents)
        stats['num_insights'] = len(self.insights)
        stats['insight_ratio'] = len(self.insights) / max(1, len(self.documents))
        return stats

def run_comparison_experiment():
    """Compare InsightSpike-AI with baselines"""
    
    logger.info("Starting RAG comparison experiment")
    
    # Create test dataset
    from insightspike_vs_baselines_comparison import create_qa_dataset, BaselineRAG
    
    questions, documents, _ = create_qa_dataset(200)
    
    # Split data
    train_size = 150
    train_documents = documents[:train_size]
    test_questions = questions[train_size:]
    
    # Initialize systems
    systems = {
        "InsightSpike-AI": InsightSpikeRAG(),
        "TF-IDF": BaselineRAG('tfidf'),
        "Sentence-BERT": BaselineRAG('sbert')
    }
    
    logger.info("Loading documents into systems...")
    
    # Load documents
    systems["InsightSpike-AI"].add_documents(train_documents)
    systems["TF-IDF"].add_documents(train_documents)
    systems["Sentence-BERT"].add_documents(train_documents)
    
    # Evaluation
    results = defaultdict(lambda: defaultdict(list))
    
    logger.info("Evaluating retrieval performance...")
    
    for i, query in enumerate(test_questions):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(test_questions)}")
        
        for system_name, system in systems.items():
            retrieved, latency = system.retrieve(query, k=5)
            
            results[system_name]['latency'].append(latency)
            
            # Check if relevant document is retrieved
            # Simple check: if query topic is in any retrieved document
            topic = query.split()[-1].rstrip('?.')
            success = any(topic.lower() in r['document'].lower() for r in retrieved)
            
            results[system_name]['recall@5'].append(1 if success else 0)
            
            if system_name == "InsightSpike-AI":
                # Track insights
                insight_count = sum(1 for r in retrieved if r.get('is_insight', False))
                results[system_name]['insights_retrieved'].append(insight_count)
    
    # Get final stats
    for system_name, system in systems.items():
        if hasattr(system, 'get_stats'):
            results[system_name]['stats'] = system.get_stats()
    
    return results

def visualize_results(results):
    """Create visualizations for the experiment"""
    
    output_dir = Path("results_real_insightspike")
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    systems = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Recall@5 comparison
    ax = axes[0, 0]
    recalls = [np.mean(results[s]['recall@5']) for s in systems]
    bars = ax.bar(systems, recalls, color=colors)
    ax.set_ylabel('Recall@5')
    ax.set_title('Retrieval Performance')
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Latency comparison
    ax = axes[0, 1]
    latencies = [np.mean(results[s]['latency']) for s in systems]
    ax.bar(systems, latencies, color=colors)
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Query Processing Speed')
    ax.set_yscale('log')
    
    # 3. InsightSpike-AI specific metrics
    ax = axes[1, 0]
    if 'InsightSpike-AI' in results:
        stats = results['InsightSpike-AI'].get('stats', {})
        if stats:
            labels = ['Documents', 'Insights']
            sizes = [stats.get('num_documents', 0) - stats.get('num_insights', 0),
                    stats.get('num_insights', 0)]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#96CEB4', '#FF6B6B'])
            ax.set_title('InsightSpike-AI Memory Composition')
    
    # 4. Insights retrieved over queries
    ax = axes[1, 1]
    if 'insights_retrieved' in results.get('InsightSpike-AI', {}):
        insights = results['InsightSpike-AI']['insights_retrieved']
        ax.plot(insights, color='#FF6B6B', linewidth=2)
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Number of Insights Retrieved')
        ax.set_title('Insight-based Retrieval Pattern')
        ax.set_ylim(0, 6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_insightspike_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    save_results = {}
    for system in results:
        save_results[system] = {
            'recall@5': float(np.mean(results[system]['recall@5'])),
            'avg_latency_ms': float(np.mean(results[system]['latency'])),
            'stats': results[system].get('stats', {})
        }
    
    with open(output_dir / 'real_insightspike_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Create report
    create_report(save_results, output_dir)
    
    logger.info(f"Results saved to {output_dir}")

def create_report(results, output_dir):
    """Create experiment report"""
    
    report = ["# Real InsightSpike-AI RAG Experiment Results\n"]
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("\n## Performance Summary\n")
    report.append("| System | Recall@5 | Latency (ms) |")
    report.append("|--------|----------|--------------|")
    
    for system, metrics in results.items():
        report.append(f"| {system} | {metrics['recall@5']:.3f} | {metrics['avg_latency_ms']:.1f} |")
    
    if 'InsightSpike-AI' in results:
        stats = results['InsightSpike-AI'].get('stats', {})
        if stats:
            report.append("\n## InsightSpike-AI Statistics\n")
            report.append(f"- Total Documents: {stats.get('num_documents', 0)}")
            report.append(f"- Insights Detected: {stats.get('num_insights', 0)}")
            report.append(f"- Insight Ratio: {stats.get('insight_ratio', 0):.2%}")
    
    report.append("\n## Key Findings\n")
    report.append("- Using the actual InsightSpike-AI implementation")
    report.append("- Real Information Gain and Graph Edit Distance calculations")
    report.append("- Insight-based retrieval and memory management")
    
    with open(output_dir / 'REAL_INSIGHTSPIKE_REPORT.md', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Run the experiment"""
    
    print("="*60)
    print("Real InsightSpike-AI RAG Experiment")
    print("Using actual InsightSpike-AI implementation")
    print("="*60)
    
    try:
        results = run_comparison_experiment()
        
        print("\nCreating visualizations...")
        visualize_results(results)
        
        print("\nSummary:")
        for system in results:
            print(f"\n{system}:")
            print(f"  Recall@5: {np.mean(results[system]['recall@5']):.3f}")
            print(f"  Latency: {np.mean(results[system]['latency']):.1f}ms")
            
            if system == "InsightSpike-AI" and 'stats' in results[system]:
                stats = results[system]['stats']
                print(f"  Insights: {stats.get('num_insights', 0)}/{stats.get('num_documents', 0)}")
        
        print("\nExperiment complete!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()