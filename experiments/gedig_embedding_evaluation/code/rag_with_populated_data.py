#!/usr/bin/env python3
"""
RAG Experiment with Populated Data Structure
===========================================

Tests RAG performance using the existing populated data structure
(64 episodes in episodes.json and updated graph_pyg.pt).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGWithPopulatedDataExperiment:
    """Test RAG performance with existing populated data structure"""
    
    def __init__(self):
        # Initialize agent and memory
        self.agent = MainAgent()
        self.memory = L2MemoryManager(dim=384)
        
        # Initialize agent
        if not self.agent.initialize():
            logger.warning("Failed to fully initialize MainAgent, continuing anyway")
        
        # Load existing data
        if self.memory.load():
            logger.info(f"Loaded existing memory: {len(self.memory.episodes)} episodes")
        else:
            logger.error("Failed to load existing memory!")
            
        self.results = []
        
    def check_data_state(self):
        """Check current data state"""
        data_dir = Path("data")
        files_to_check = ["episodes.json", "index.faiss", "graph_pyg.pt"]
        
        logger.info("\nCurrent data state:")
        for filename in files_to_check:
            filepath = data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                logger.info(f"  {filename}: {size:,} bytes, modified {mtime}")
                
                if filename == "episodes.json":
                    try:
                        with open(filepath, 'r') as f:
                            episodes = json.load(f)
                        logger.info(f"    Episodes count: {len(episodes)}")
                        # Sample content check
                        if episodes:
                            sample = episodes[0]
                            logger.info(f"    Sample episode text: {sample.get('text', '')[:100]}...")
                    except Exception as e:
                        logger.error(f"    Error reading episodes: {e}")
                        
                elif filename == "graph_pyg.pt":
                    try:
                        if self.agent.l3_graph:
                            graph = self.agent.l3_graph.load_graph()
                            if graph:
                                logger.info(f"    Graph nodes: {graph.num_nodes}")
                                if hasattr(graph, 'edge_index'):
                                    logger.info(f"    Graph edges: {graph.edge_index.size(1)}")
                    except Exception as e:
                        logger.error(f"    Error reading graph: {e}")
            else:
                logger.info(f"  {filename}: NOT FOUND")
    
    def create_diverse_queries(self, n_queries=50):
        """Create diverse queries to test different aspects of RAG"""
        topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision",
            "reinforcement learning", "data science", "algorithms"
        ]
        
        query_templates = [
            "What is {}?",
            "Explain the principles of {}",
            "How does {} work?",
            "What are the applications of {}?",
            "Describe the mathematical models in {}",
            "Tell me about recent advances in {}",
            "How is {} used in practice?",
            "What problems does {} solve?"
        ]
        
        queries = []
        for i in range(n_queries):
            topic = topics[i % len(topics)]
            template = query_templates[i % len(query_templates)]
            query = template.format(topic)
            queries.append(query)
            
        return queries
    
    def test_rag_performance(self, queries):
        """Test RAG retrieval and response quality"""
        logger.info(f"\nTesting RAG with {len(queries)} queries...")
        
        results = []
        baseline_results = []  # For comparison without RAG
        
        for i, query in enumerate(queries):
            # Test with RAG (using populated memory)
            start_time = time.time()
            rag_result = self.agent.process_question(query, max_cycles=3)
            rag_time = time.time() - start_time
            
            # Extract results
            rag_response = rag_result.get('response', '')
            retrieved_docs = rag_result.get('documents', [])
            quality_score = rag_result.get('reasoning_quality', 0)
            
            # Check relevance
            topic_keywords = self._extract_topic_keywords(query)
            relevant_docs_found = sum(1 for doc in retrieved_docs 
                                    if any(kw in doc.get('text', '').lower() 
                                          for kw in topic_keywords))
            
            response_relevant = any(kw in rag_response.lower() for kw in topic_keywords)
            
            results.append({
                'query': query,
                'response': rag_response,
                'retrieved_count': len(retrieved_docs),
                'relevant_docs': relevant_docs_found,
                'response_relevant': response_relevant,
                'quality_score': quality_score,
                'time': rag_time,
                'doc_similarities': [doc.get('similarity', 0) for doc in retrieved_docs]
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(queries)} queries processed")
                avg_retrieval = np.mean([r['retrieved_count'] for r in results])
                avg_relevant = np.mean([r['relevant_docs'] for r in results])
                logger.info(f"  Avg docs retrieved: {avg_retrieval:.1f}, Avg relevant: {avg_relevant:.1f}")
        
        return results
    
    def _extract_topic_keywords(self, query):
        """Extract topic keywords from query"""
        topic_map = {
            "machine learning": ["machine", "learning", "ml", "model", "data"],
            "deep learning": ["deep", "learning", "neural", "layer", "network"],
            "neural networks": ["neural", "network", "neuron", "layer", "activation"],
            "natural language processing": ["natural", "language", "nlp", "text", "processing"],
            "computer vision": ["computer", "vision", "image", "visual", "recognition"],
            "reinforcement learning": ["reinforcement", "learning", "reward", "agent", "policy"],
            "data science": ["data", "science", "analysis", "statistics", "insight"],
            "algorithms": ["algorithm", "computational", "complexity", "efficiency", "method"]
        }
        
        query_lower = query.lower()
        for topic, keywords in topic_map.items():
            if any(kw in query_lower for kw in topic.split()):
                return keywords
        
        # Fallback: extract key words from query
        return [w for w in query_lower.split() if len(w) > 3 and w not in 
                ['what', 'explain', 'describe', 'tell', 'about', 'does', 'work']]
    
    def analyze_memory_efficiency(self):
        """Analyze memory storage efficiency"""
        memory_stats = self.memory.get_memory_stats()
        
        # Calculate compression metrics
        episodes = self.memory.episodes
        total_text_size = sum(len(ep.text.encode('utf-8')) for ep in episodes)
        total_embedding_size = len(episodes) * 384 * 4  # float32
        
        # Get actual file sizes
        data_dir = Path("data")
        episodes_size = (data_dir / "episodes.json").stat().st_size if (data_dir / "episodes.json").exists() else 0
        index_size = (data_dir / "index.faiss").stat().st_size if (data_dir / "index.faiss").exists() else 0
        graph_size = (data_dir / "graph_pyg.pt").stat().st_size if (data_dir / "graph_pyg.pt").exists() else 0
        
        efficiency = {
            'episode_count': len(episodes),
            'total_text_size': total_text_size,
            'total_embedding_size': total_embedding_size,
            'episodes_file_size': episodes_size,
            'index_file_size': index_size,
            'graph_file_size': graph_size,
            'total_storage': episodes_size + index_size + graph_size,
            'text_compression_ratio': total_text_size / (episodes_size + 1),  # Avoid div by 0
            'storage_per_episode': (episodes_size + index_size + graph_size) / max(1, len(episodes)),
            'memory_stats': memory_stats
        }
        
        return efficiency
    
    def compare_with_baseline(self, queries):
        """Compare with baseline (no memory) performance"""
        logger.info("\nTesting baseline (no memory) for comparison...")
        
        # Create a fresh agent with no memory
        baseline_agent = MainAgent()
        baseline_agent.initialize()
        
        baseline_results = []
        for i, query in enumerate(queries[:10]):  # Test subset for baseline
            start_time = time.time()
            result = baseline_agent.process_question(query, max_cycles=2)
            baseline_time = time.time() - start_time
            
            baseline_results.append({
                'query': query,
                'response': result.get('response', ''),
                'quality': result.get('reasoning_quality', 0),
                'time': baseline_time
            })
        
        return baseline_results
    
    def visualize_results(self, rag_results, memory_efficiency, baseline_results=None):
        """Create visualizations of experiment results"""
        output_dir = Path("results_rag_populated_data")
        output_dir.mkdir(exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("RAG Performance with Populated Data Structure", fontsize=16)
        
        # 1. Document retrieval statistics
        ax = axes[0, 0]
        retrieved_counts = [r['retrieved_count'] for r in rag_results]
        relevant_counts = [r['relevant_docs'] for r in rag_results]
        
        ax.hist([retrieved_counts, relevant_counts], bins=10, label=['Total Retrieved', 'Relevant Retrieved'], alpha=0.7)
        ax.set_xlabel('Number of Documents')
        ax.set_ylabel('Frequency')
        ax.set_title('Document Retrieval Distribution')
        ax.legend()
        
        # 2. Response quality distribution
        ax = axes[0, 1]
        quality_scores = [r['quality_score'] for r in rag_results]
        ax.hist(quality_scores, bins=20, alpha=0.7, color='green')
        ax.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(quality_scores):.3f}')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Quality Distribution')
        ax.legend()
        
        # 3. Retrieval accuracy over time
        ax = axes[1, 0]
        response_accuracy = [1 if r['response_relevant'] else 0 for r in rag_results]
        window_size = 10
        moving_avg = np.convolve(response_accuracy, np.ones(window_size)/window_size, mode='valid')
        
        ax.plot(range(len(moving_avg)), moving_avg)
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Accuracy (Moving Average)')
        ax.set_title(f'Response Relevance Over Time (window={window_size})')
        ax.set_ylim(0, 1)
        
        # 4. Memory efficiency metrics
        ax = axes[1, 1]
        metrics = ['Episodes', 'Storage/Episode', 'Compression']
        values = [
            memory_efficiency['episode_count'],
            memory_efficiency['storage_per_episode'] / 1000,  # KB
            memory_efficiency['text_compression_ratio']
        ]
        
        bars = ax.bar(metrics, values)
        ax.set_ylabel('Value')
        ax.set_title('Memory Efficiency Metrics')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rag_populated_analysis.png', dpi=150)
        plt.close()
        
        # Save detailed results
        analysis = {
            'summary': {
                'total_queries': len(rag_results),
                'avg_retrieved_docs': np.mean([r['retrieved_count'] for r in rag_results]),
                'avg_relevant_docs': np.mean([r['relevant_docs'] for r in rag_results]),
                'response_relevance_rate': np.mean(response_accuracy),
                'avg_quality_score': np.mean(quality_scores),
                'avg_response_time': np.mean([r['time'] for r in rag_results])
            },
            'memory_efficiency': memory_efficiency,
            'baseline_comparison': None
        }
        
        if baseline_results:
            baseline_quality = np.mean([r['quality'] for r in baseline_results])
            rag_subset_quality = np.mean([r['quality_score'] for r in rag_results[:len(baseline_results)]])
            
            analysis['baseline_comparison'] = {
                'baseline_avg_quality': baseline_quality,
                'rag_avg_quality': rag_subset_quality,
                'quality_improvement': (rag_subset_quality - baseline_quality) / baseline_quality * 100,
                'baseline_avg_time': np.mean([r['time'] for r in baseline_results]),
                'rag_avg_time': np.mean([r['time'] for r in rag_results[:len(baseline_results)]])
            }
        
        with open(output_dir / 'rag_populated_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=float)
        
        logger.info(f"\nResults saved to {output_dir}")
        
        return analysis

def main():
    """Run RAG experiment with populated data"""
    print("="*60)
    print("RAG Experiment with Populated Data Structure")
    print("="*60)
    
    # Create experiment instance
    experiment = RAGWithPopulatedDataExperiment()
    
    # Check current data state
    print("\nChecking data state...")
    experiment.check_data_state()
    
    # Create test queries
    print("\nCreating test queries...")
    queries = experiment.create_diverse_queries(50)
    
    # Test RAG performance
    print("\nTesting RAG performance...")
    rag_results = experiment.test_rag_performance(queries)
    
    # Analyze memory efficiency
    print("\nAnalyzing memory efficiency...")
    memory_efficiency = experiment.analyze_memory_efficiency()
    
    # Compare with baseline (optional)
    print("\nComparing with baseline...")
    baseline_results = experiment.compare_with_baseline(queries)
    
    # Visualize and save results
    print("\nVisualizing results...")
    analysis = experiment.visualize_results(rag_results, memory_efficiency, baseline_results)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total queries tested: {analysis['summary']['total_queries']}")
    print(f"Average documents retrieved: {analysis['summary']['avg_retrieved_docs']:.1f}")
    print(f"Average relevant documents: {analysis['summary']['avg_relevant_docs']:.1f}")
    print(f"Response relevance rate: {analysis['summary']['response_relevance_rate']:.1%}")
    print(f"Average quality score: {analysis['summary']['avg_quality_score']:.3f}")
    print(f"Average response time: {analysis['summary']['avg_response_time']:.3f}s")
    
    print("\nMemory Efficiency:")
    print(f"  Episodes stored: {memory_efficiency['episode_count']}")
    print(f"  Storage per episode: {memory_efficiency['storage_per_episode']:.0f} bytes")
    print(f"  Text compression ratio: {memory_efficiency['text_compression_ratio']:.1f}x")
    print(f"  Total storage: {memory_efficiency['total_storage']:,} bytes")
    
    if analysis['baseline_comparison']:
        print("\nBaseline Comparison:")
        comp = analysis['baseline_comparison']
        print(f"  Baseline quality: {comp['baseline_avg_quality']:.3f}")
        print(f"  RAG quality: {comp['rag_avg_quality']:.3f}")
        print(f"  Quality improvement: {comp['quality_improvement']:.1f}%")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()