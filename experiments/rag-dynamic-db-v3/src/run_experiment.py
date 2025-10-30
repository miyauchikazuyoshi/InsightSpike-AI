#!/usr/bin/env python3
"""Run comprehensive geDIG-RAG experiment with visualization."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Core imports
from core.config import ExperimentConfig
from core.knowledge_graph import KnowledgeGraph
from core.gedig_evaluator import GeDIGEvaluator, GraphUpdate, UpdateType

# Set style for nice visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MockEmbedder:
    """Mock embedder with concept-based similarity."""
    def __init__(self):
        self.embedding_dim = 384
        # Define concept embeddings for realistic similarity
        np.random.seed(42)
        self.concepts = {
            'python': np.random.normal(0, 1, self.embedding_dim),
            'programming': np.random.normal(0, 1, self.embedding_dim),
            'machine': np.random.normal(0, 1, self.embedding_dim),
            'learning': np.random.normal(0, 1, self.embedding_dim),
            'deep': np.random.normal(0, 1, self.embedding_dim),
            'neural': np.random.normal(0, 1, self.embedding_dim),
            'network': np.random.normal(0, 1, self.embedding_dim),
            'data': np.random.normal(0, 1, self.embedding_dim),
            'science': np.random.normal(0, 1, self.embedding_dim),
            'algorithm': np.random.normal(0, 1, self.embedding_dim),
            'artificial': np.random.normal(0, 1, self.embedding_dim),
            'intelligence': np.random.normal(0, 1, self.embedding_dim),
            'natural': np.random.normal(0, 1, self.embedding_dim),
            'language': np.random.normal(0, 1, self.embedding_dim),
            'processing': np.random.normal(0, 1, self.embedding_dim),
            'computer': np.random.normal(0, 1, self.embedding_dim),
            'vision': np.random.normal(0, 1, self.embedding_dim),
            'blockchain': np.random.normal(0, 1, self.embedding_dim),
            'quantum': np.random.normal(0, 1, self.embedding_dim),
            'ethics': np.random.normal(0, 1, self.embedding_dim),
        }
        
        # Normalize concept embeddings
        for concept in self.concepts:
            self.concepts[concept] /= np.linalg.norm(self.concepts[concept])
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            embedding = np.zeros(self.embedding_dim)
            matched_concepts = 0
            
            # Mix embeddings based on concepts found
            for concept, concept_emb in self.concepts.items():
                if concept in text_lower:
                    embedding += concept_emb
                    matched_concepts += 1
            
            if matched_concepts == 0:
                # Random embedding for unknown concepts
                np.random.seed(hash(text) % 10000)
                embedding = np.random.normal(0, 1, self.embedding_dim)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings)


class SimpleRAGSystem:
    """Simplified RAG system for experiments."""
    
    def __init__(self, method_name: str, config: ExperimentConfig):
        self.method_name = method_name
        self.config = config
        self.knowledge_graph = KnowledgeGraph(embedding_dim=384)
        self.embedder = MockEmbedder()
        self.gedig_evaluator = GeDIGEvaluator(
            k_coefficient=config.gedig.k_coefficient,
            radius=config.gedig.radius
        ) if method_name == "gedig" else None
        
        # Statistics
        self.queries_processed = 0
        self.updates_applied = 0
        self.similarity_history = []
        self.update_history = []
        self.gedig_scores = []
    
    def add_initial_knowledge(self, documents: List[str]) -> int:
        """Add initial knowledge base."""
        added = 0
        for doc in documents:
            embedding = self.embedder.encode(doc)[0]
            node_id = self.knowledge_graph.add_node(
                text=doc,
                embedding=embedding,
                node_type="initial",
                confidence=0.9
            )
            if node_id:
                added += 1
        return added
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        self.queries_processed += 1
        
        # 1. Retrieve relevant knowledge
        query_embedding = self.embedder.encode(query)[0]
        similar_nodes = self.knowledge_graph.find_similar_nodes(
            query_embedding, k=5, min_similarity=0.0
        )
        
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        self.similarity_history.append(max_similarity)
        
        # 2. Generate response (mock)
        if similar_nodes and max_similarity > 0.3:
            context = self.knowledge_graph.nodes[similar_nodes[0][0]].text
            response = f"Based on: {context[:50]}..."
        else:
            response = f"General response for: {query}"
        
        # 3. Decide on knowledge update
        should_update = self._should_update(query, response, max_similarity)
        
        if should_update:
            self._apply_update(query, response, query_embedding, similar_nodes)
            self.updates_applied += 1
            self.update_history.append(1)
        else:
            self.update_history.append(0)
        
        return {
            'query': query,
            'response': response,
            'max_similarity': max_similarity,
            'updated': should_update,
            'graph_size': len(self.knowledge_graph.nodes)
        }
    
    def _should_update(self, query: str, response: str, max_similarity: float) -> bool:
        """Decide whether to update knowledge based on method."""
        if self.method_name == "static":
            return False
        
        elif self.method_name == "frequency":
            # Update for low similarity or first few queries
            return max_similarity < 0.4 or self.queries_processed <= 3
        
        elif self.method_name == "cosine":
            # Update if similarity below threshold
            return max_similarity < self.config.cosine_similarity_threshold
        
        elif self.method_name == "gedig":
            # Use geDIG evaluation
            if max_similarity > 0.95:  # Skip if too similar
                return False
            
            # Create and evaluate update
            update = self._create_update_candidate(query, response)
            result = self.gedig_evaluator.evaluate_update(
                self.knowledge_graph.graph, update
            )
            
            self.gedig_scores.append(result.delta_gedig)
            return result.delta_gedig > 0.0  # Positive geDIG score
        
        return False
    
    def _create_update_candidate(self, query: str, response: str) -> GraphUpdate:
        """Create an update candidate for geDIG evaluation."""
        new_text = f"Q: {query} A: {response}"
        new_embedding = self.embedder.encode(new_text)[0]
        new_node_id = f"node_{self.queries_processed}"
        
        return GraphUpdate(
            update_type=UpdateType.ADD,
            target_nodes=[],
            new_node_data={
                'id': new_node_id,
                'text': new_text,
                'embedding': new_embedding,
                'node_type': 'qa_pair'
            },
            new_edges=[]
        )
    
    def _apply_update(self, query: str, response: str, 
                     query_embedding: np.ndarray, similar_nodes: List[Tuple[str, float]]):
        """Apply knowledge update."""
        new_text = f"Q: {query} A: {response}"
        new_embedding = self.embedder.encode(new_text)[0]
        
        node_id = self.knowledge_graph.add_node(
            text=new_text,
            embedding=new_embedding,
            node_type="qa_pair",
            confidence=0.7
        )
        
        # Add edges to similar nodes
        for similar_id, similarity in similar_nodes[:2]:
            if similarity > 0.2:
                self.knowledge_graph.add_edge(
                    node_id, similar_id,
                    relation="semantic",
                    weight=similarity,
                    semantic_similarity=similarity
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'method': self.method_name,
            'queries_processed': self.queries_processed,
            'updates_applied': self.updates_applied,
            'update_rate': self.updates_applied / max(1, self.queries_processed),
            'graph_nodes': len(self.knowledge_graph.nodes),
            'graph_edges': len(self.knowledge_graph.edges),
            'avg_similarity': np.mean(self.similarity_history) if self.similarity_history else 0,
            'similarity_history': self.similarity_history,
            'update_history': self.update_history,
            'gedig_scores': self.gedig_scores if self.method_name == "gedig" else []
        }


def create_test_data() -> Tuple[List[str], List[str]]:
    """Create test documents and queries."""
    # Initial knowledge base
    initial_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
        "Natural language processing (NLP) enables computers to understand human language.",
        "Computer vision is a field of AI that trains computers to interpret visual information.",
        "Data science combines statistics, mathematics, and programming to extract insights from data.",
        "Artificial intelligence aims to create machines that can think and learn like humans.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Algorithms are step-by-step procedures for solving problems or completing tasks.",
        "Programming languages are formal languages used to communicate instructions to computers."
    ]
    
    # Test queries (mix of related and unrelated)
    test_queries = [
        # Related to existing knowledge
        "What is Python programming?",
        "How does machine learning work?",
        "Explain deep learning",
        "What is NLP?",
        "Tell me about computer vision",
        
        # Partially related
        "What is reinforcement learning?",
        "How do transformers work?",
        "What is supervised learning?",
        "Explain convolutional neural networks",
        "What is gradient descent?",
        
        # New topics
        "What is blockchain technology?",
        "Explain quantum computing",
        "What is edge computing?",
        "Tell me about cybersecurity",
        "What is cloud computing?",
        
        # Revisiting topics
        "More about Python",
        "Advanced machine learning techniques",
        "Deep learning applications",
        "NLP recent advances",
        "Future of AI"
    ]
    
    return initial_docs, test_queries


def run_experiment() -> Dict[str, Any]:
    """Run complete experiment comparing all RAG methods."""
    print("🚀 Starting geDIG-RAG Experiment")
    print("=" * 60)
    
    # Setup
    config = ExperimentConfig()
    initial_docs, test_queries = create_test_data()
    
    # Methods to compare
    methods = ["static", "frequency", "cosine", "gedig"]
    results = {}
    
    for method in methods:
        print(f"\n📊 Testing {method.upper()} RAG...")
        print("-" * 40)
        
        # Initialize system
        system = SimpleRAGSystem(method, config)
        
        # Add initial knowledge
        n_added = system.add_initial_knowledge(initial_docs)
        print(f"  Initial knowledge: {n_added} documents")
        
        # Process queries
        query_results = []
        for i, query in enumerate(test_queries):
            result = system.process_query(query)
            query_results.append(result)
            
            if (i + 1) % 5 == 0:
                stats = system.get_statistics()
                print(f"  After {i+1} queries: {stats['graph_nodes']} nodes, "
                      f"{stats['updates_applied']} updates")
        
        # Final statistics
        final_stats = system.get_statistics()
        final_stats['query_results'] = query_results
        results[method] = final_stats
        
        print(f"  Final: {final_stats['graph_nodes']} nodes, "
              f"{final_stats['updates_applied']} updates "
              f"(rate: {final_stats['update_rate']:.2%})")
    
    return results


def visualize_results(results: Dict[str, Any]):
    """Create comprehensive visualizations."""
    print("\n📈 Generating Visualizations...")
    
    # Create output directory
    output_dir = Path("../results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Knowledge Graph Growth Over Time
    ax1 = plt.subplot(2, 3, 1)
    for method in results:
        stats = results[method]
        graph_sizes = [r['graph_size'] for r in stats['query_results']]
        ax1.plot(graph_sizes, label=method.upper(), marker='o', markersize=4)
    
    ax1.set_xlabel('Query Number')
    ax1.set_ylabel('Knowledge Graph Size (nodes)')
    ax1.set_title('Knowledge Graph Growth Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Update Rate Comparison
    ax2 = plt.subplot(2, 3, 2)
    methods = list(results.keys())
    update_rates = [results[m]['update_rate'] for m in methods]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax2.bar(methods, update_rates, color=colors)
    
    # Add value labels on bars
    for bar, rate in zip(bars, update_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom')
    
    ax2.set_ylabel('Update Rate')
    ax2.set_title('Knowledge Update Rate by Method')
    ax2.set_ylim(0, max(update_rates) * 1.2)
    
    # 3. Similarity Distribution
    ax3 = plt.subplot(2, 3, 3)
    for method in results:
        if results[method]['similarity_history']:
            ax3.hist(results[method]['similarity_history'], 
                    alpha=0.5, label=method.upper(), bins=15)
    
    ax3.set_xlabel('Maximum Similarity Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Query-Knowledge Similarities')
    ax3.legend()
    
    # 4. Cumulative Updates
    ax4 = plt.subplot(2, 3, 4)
    for method in results:
        cumulative_updates = np.cumsum(results[method]['update_history'])
        ax4.plot(cumulative_updates, label=method.upper(), linewidth=2)
    
    ax4.set_xlabel('Query Number')
    ax4.set_ylabel('Cumulative Updates')
    ax4.set_title('Cumulative Knowledge Updates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. geDIG Scores (if available)
    ax5 = plt.subplot(2, 3, 5)
    if 'gedig' in results and results['gedig']['gedig_scores']:
        scores = results['gedig']['gedig_scores']
        x_pos = range(1, len(scores) + 1)
        colors_gedig = ['green' if s > 0 else 'red' for s in scores]
        ax5.bar(x_pos, scores, color=colors_gedig, alpha=0.6)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_xlabel('Update Evaluation Number')
        ax5.set_ylabel('geDIG Score')
        ax5.set_title('geDIG Evaluation Scores (Green=Accept, Red=Reject)')
    else:
        ax5.text(0.5, 0.5, 'geDIG Scores\n(Only for geDIG method)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('geDIG Evaluation Scores')
    
    # 6. Final Statistics Comparison
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['graph_nodes', 'graph_edges', 'updates_applied']
    metric_labels = ['Final Nodes', 'Final Edges', 'Total Updates']
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in methods]
        ax6.bar(x + i*width, values, width, label=label)
    
    ax6.set_xlabel('Method')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels([m.upper() for m in methods])
    ax6.set_ylabel('Count')
    ax6.set_title('Final System Statistics')
    ax6.legend()
    
    plt.suptitle('geDIG-RAG Experiment Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"experiment_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    # Also save individual method comparison
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        stats = results[method]
        
        # Update pattern over time
        update_pattern = stats['update_history']
        query_numbers = range(1, len(update_pattern) + 1)
        
        # Create bar chart with colors
        colors_update = ['green' if u == 1 else 'lightgray' for u in update_pattern]
        ax.bar(query_numbers, update_pattern, color=colors_update, alpha=0.7)
        
        ax.set_xlabel('Query Number')
        ax.set_ylabel('Update Applied (1=Yes, 0=No)')
        ax.set_title(f'{method.upper()} RAG - Update Pattern')
        ax.set_ylim(-0.1, 1.1)
        
        # Add statistics text
        update_rate = stats['update_rate']
        total_updates = stats['updates_applied']
        ax.text(0.02, 0.95, f"Update Rate: {update_rate:.1%}\nTotal Updates: {total_updates}", 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Update Patterns by Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = output_dir / f"update_patterns_{timestamp}.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"  Saved update patterns to: {output_path2}")
    
    plt.show()
    
    return output_dir


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save experimental results to JSON."""
    # Remove non-serializable data
    clean_results = {}
    for method, stats in results.items():
        clean_stats = {k: v for k, v in stats.items() 
                      if k not in ['query_results']}
        clean_stats['final_nodes'] = stats['graph_nodes']
        clean_stats['final_edges'] = stats['graph_edges']
        clean_results[method] = clean_stats
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"experiment_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"  Saved results to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print comprehensive summary."""
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Create comparison table
    print("\n📈 Performance Comparison:")
    print("-" * 60)
    print(f"{'Method':<12} {'Updates':<10} {'Update Rate':<12} {'Final Nodes':<12} {'Final Edges':<12}")
    print("-" * 60)
    
    for method in results:
        stats = results[method]
        print(f"{method.upper():<12} {stats['updates_applied']:<10} "
              f"{stats['update_rate']:<12.1%} {stats['graph_nodes']:<12} "
              f"{stats['graph_edges']:<12}")
    
    print("-" * 60)
    
    # Key findings
    print("\n🔍 Key Findings:")
    
    # Find best performer by different metrics
    max_updates = max(results.values(), key=lambda x: x['updates_applied'])
    min_updates = min(results.values(), key=lambda x: x['updates_applied'])
    max_nodes = max(results.values(), key=lambda x: x['graph_nodes'])
    
    print(f"  • Most adaptive: {[k for k, v in results.items() if v == max_updates][0].upper()} "
          f"({max_updates['updates_applied']} updates)")
    print(f"  • Most conservative: {[k for k, v in results.items() if v == min_updates][0].upper()} "
          f"({min_updates['updates_applied']} updates)")
    print(f"  • Largest knowledge graph: {[k for k, v in results.items() if v == max_nodes][0].upper()} "
          f"({max_nodes['graph_nodes']} nodes)")
    
    if 'gedig' in results and results['gedig']['gedig_scores']:
        gedig_scores = results['gedig']['gedig_scores']
        positive_scores = [s for s in gedig_scores if s > 0]
        print(f"  • geDIG acceptance rate: {len(positive_scores)}/{len(gedig_scores)} "
              f"({len(positive_scores)/len(gedig_scores)*100:.1f}%)")
        if gedig_scores:
            print(f"  • geDIG score range: [{min(gedig_scores):.3f}, {max(gedig_scores):.3f}]")


def main():
    """Main experiment execution."""
    try:
        # Run experiment
        results = run_experiment()
        
        # Visualize results
        output_dir = visualize_results(results)
        
        # Save results
        save_results(results, output_dir)
        
        # Print summary
        print_summary(results)
        
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)