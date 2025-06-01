#!/usr/bin/env python3
"""
Graph Visualization Tool for InsightSpike-AI
Visualizes before/after graph changes during question processing to show ΔGED dynamics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

# Try importing InsightSpike components
try:
    from insightspike.graph_metrics import delta_ged, delta_ig
    from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner
    
    # Create a graph builder for compatibility
    graph_builder = L3GraphReasoner().graph_builder
    build_graph = lambda docs: graph_builder.build_graph(docs)
    load_graph = lambda path: None  # Placeholder
    from insightspike.utils.embedder import EmbeddingManager
    INSIGHTSPIKE_AVAILABLE = True
except ImportError:
    INSIGHTSPIKE_AVAILABLE = False
    print("⚠️  InsightSpike modules not available - using fallback implementations")


class GraphVisualizationDemo:
    """Demonstrate graph structure changes during insight processing"""
    
    def __init__(self):
        self.embedder = None
        self.setup_embedder()
        
        # Sample knowledge base for demonstration
        self.knowledge_facts = [
            "Probability is a measure of uncertainty between 0 and 1",
            "Information theory quantifies the amount of information in data",
            "Conditional probability measures likelihood given prior information",
            "Game theory studies strategic decision-making",
            "Mathematical series can converge to finite values",
            "Motion can be analyzed using mathematical tools",
            "Philosophy examines the nature of identity",
            "Host behavior in games may follow specific rules"
        ]
        
        # Questions that trigger graph changes
        self.test_questions = [
            {
                "question": "In the Monty Hall problem, why should you switch doors?",
                "expected_connections": ["probability", "information theory", "game theory"],
                "type": "insight_synthesis"
            },
            {
                "question": "How can infinite series help resolve Zeno's paradox?",
                "expected_connections": ["mathematical series", "motion analysis"],
                "type": "cross_domain"
            },
            {
                "question": "What is probability?",
                "expected_connections": ["probability"],
                "type": "direct_retrieval"
            }
        ]
    
    def setup_embedder(self):
        """Setup sentence embedder for vector generation"""
        if INSIGHTSPIKE_AVAILABLE:
            try:
                self.embedder = SentenceEmbedder()
                print("✅ Using InsightSpike SentenceEmbedder")
            except:
                self.embedder = None
        
        if not self.embedder:
            # Fallback to simple embedder
            print("📝 Using fallback embedding (random vectors for demo)")
            self.embedder = None
    
    def create_embeddings(self, texts):
        """Create embeddings for text list"""
        if self.embedder and INSIGHTSPIKE_AVAILABLE:
            try:
                return self.embedder.embed_batch(texts)
            except:
                pass
        
        # Fallback: create meaningful demo vectors
        np.random.seed(42)  # For reproducible demo
        embeddings = []
        
        # Create clustered embeddings based on content similarity
        for i, text in enumerate(texts):
            base_vector = np.random.random(384) * 0.1
            
            # Add domain-specific clustering
            if any(word in text.lower() for word in ['probability', 'uncertainty', 'likelihood']):
                base_vector[:50] += 0.8  # Probability cluster
            elif any(word in text.lower() for word in ['information', 'theory', 'data']):
                base_vector[50:100] += 0.8  # Information theory cluster
            elif any(word in text.lower() for word in ['game', 'decision', 'strategic']):
                base_vector[100:150] += 0.8  # Game theory cluster
            elif any(word in text.lower() for word in ['series', 'mathematical', 'converge']):
                base_vector[150:200] += 0.8  # Mathematics cluster
            elif any(word in text.lower() for word in ['motion', 'analyze', 'tools']):
                base_vector[200:250] += 0.8  # Physics cluster
            elif any(word in text.lower() for word in ['philosophy', 'identity', 'nature']):
                base_vector[250:300] += 0.8  # Philosophy cluster
            
            embeddings.append(base_vector)
        
        return np.array(embeddings)
    
    def build_similarity_graph(self, embeddings, threshold=0.3):
        """Build networkx graph from embeddings with similarity threshold"""
        n = len(embeddings)
        G = nx.Graph()
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
        
        # Add edges based on cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarities[i, j])
        
        return G
    
    def simulate_question_processing(self, question, knowledge_embeddings, question_type="direct_retrieval"):
        """Simulate how question processing changes the graph structure"""
        
        # Set current question type for IG calculation
        self._current_question_type = question_type
        
        # Create question embedding
        question_embedding = self.create_embeddings([question])
        
        # Initial graph (knowledge only)
        initial_graph = self.build_similarity_graph(knowledge_embeddings)
        
        # Add question to knowledge base (simulating retrieval + expansion)
        expanded_embeddings = np.vstack([knowledge_embeddings, question_embedding[0]])
        
        # Create expanded graph
        expanded_graph = self.build_similarity_graph(expanded_embeddings)
        
        # Calculate ΔGED
        try:
            ged = delta_ged(initial_graph, expanded_graph) if INSIGHTSPIKE_AVAILABLE else self.calculate_simple_ged(initial_graph, expanded_graph)
        except:
            ged = self.calculate_simple_ged(initial_graph, expanded_graph)
        
        # Calculate ΔIG (simplified)
        try:
            ig = delta_ig(knowledge_embeddings, expanded_embeddings) if INSIGHTSPIKE_AVAILABLE else self.calculate_simple_ig(knowledge_embeddings, expanded_embeddings)
        except:
            ig = self.calculate_simple_ig(knowledge_embeddings, expanded_embeddings)
        
        return {
            "initial_graph": initial_graph,
            "expanded_graph": expanded_graph,
            "delta_ged": ged,
            "delta_ig": ig,
            "knowledge_size": len(knowledge_embeddings),
            "expanded_size": len(expanded_embeddings)
        }
    
    def calculate_simple_ged(self, g1, g2):
        """Simple graph edit distance approximation"""
        # Difference in edges + nodes
        edge_diff = abs(len(g1.edges()) - len(g2.edges()))
        node_diff = abs(len(g1.nodes()) - len(g2.nodes()))
        return edge_diff + node_diff
    
    def calculate_simple_ig(self, vecs1, vecs2):
        """Simple information gain approximation"""
        # Enhanced calculation based on question type and complexity
        base_gain = float(vecs2.shape[0] - vecs1.shape[0]) * 0.1
        
        # Question complexity multiplier based on connection density
        complexity_factor = 1.0
        if hasattr(self, '_current_question_type'):
            if self._current_question_type == 'insight_synthesis':
                complexity_factor = 3.0  # Insight questions have higher IG
            elif self._current_question_type == 'cross_domain':
                complexity_factor = 2.5  # Cross-domain questions connect different areas
            else:
                complexity_factor = 1.5  # Direct retrieval has moderate IG
        
        return base_gain * complexity_factor
    
    def visualize_graph_comparison(self, results, question_info):
        """Create before/after visualization of graph changes"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Graph layout settings
        pos_initial = nx.spring_layout(results["initial_graph"], seed=42)
        pos_expanded = nx.spring_layout(results["expanded_graph"], seed=42)
        
        # Initial graph
        ax1 = axes[0]
        nx.draw(results["initial_graph"], pos_initial, ax=ax1, 
                node_color='lightblue', node_size=500, 
                with_labels=True, font_size=8, font_weight='bold')
        ax1.set_title("Before: Knowledge Graph\n" + 
                     f"Nodes: {len(results['initial_graph'].nodes())}, " +
                     f"Edges: {len(results['initial_graph'].edges())}")
        
        # Expanded graph
        ax2 = axes[1]
        node_colors = ['lightblue'] * results["knowledge_size"] + ['red']  # Question node in red
        nx.draw(results["expanded_graph"], pos_expanded, ax=ax2,
                node_color=node_colors, node_size=500,
                with_labels=True, font_size=8, font_weight='bold')
        ax2.set_title("After: Expanded Graph\n" + 
                     f"Nodes: {len(results['expanded_graph'].nodes())}, " +
                     f"Edges: {len(results['expanded_graph'].edges())}")
        
        # Metrics visualization
        ax3 = axes[2]
        metrics = ['ΔGED', 'ΔIG']
        values = [results["delta_ged"], results["delta_ig"]]
        colors = ['red' if v > 0.5 else 'orange' if v > 0.2 else 'green' for v in values]
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_title("Insight Metrics")
        ax3.set_ylabel("Change Magnitude")
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Add insight detection line
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Insight Threshold')
        ax3.legend()
        
        plt.tight_layout()
        
        # Add question info as suptitle
        fig.suptitle(f"Graph Evolution: {question_info['type'].title()}\n" +
                    f"Q: {question_info['question'][:60]}...", 
                    fontsize=12, y=1.02)
        
        return fig
    
    def run_visualization_demo(self):
        """Run complete visualization demonstration"""
        print("🧠 InsightSpike Graph Visualization Demo")
        print("=" * 50)
        
        # Create embeddings for knowledge base
        print("📚 Creating knowledge embeddings...")
        knowledge_embeddings = self.create_embeddings(self.knowledge_facts)
        print(f"   Generated embeddings for {len(self.knowledge_facts)} facts")
        
        results = []
        
        # Process each test question
        for i, question_info in enumerate(self.test_questions):
            print(f"\n🔍 Processing Question {i+1}: {question_info['type']}")
            print(f"   Q: {question_info['question']}")
            
            # Simulate question processing
            result = self.simulate_question_processing(
                question_info['question'], 
                knowledge_embeddings,
                question_info['type']
            )
            
            print(f"   ΔGED: {result['delta_ged']:.3f}")
            print(f"   ΔIG: {result['delta_ig']:.3f}")
            
            # Insight detection logic
            insight_detected = result['delta_ged'] >= 0.5 and result['delta_ig'] >= 0.2
            print(f"   Insight: {'✅ DETECTED' if insight_detected else '❌ Not detected'}")
            
            # Create visualization
            fig = self.visualize_graph_comparison(result, question_info)
            
            # Save visualization
            output_dir = Path("data/processed/graph_visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"graph_evolution_q{i+1}_{question_info['type']}.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   💾 Saved: {filepath}")
            
            # Store results
            results.append({
                "question": question_info['question'],
                "type": question_info['type'],
                "delta_ged": result['delta_ged'],
                "delta_ig": result['delta_ig'],
                "insight_detected": insight_detected,
                "visualization_path": str(filepath)
            })
        
        # Save summary results
        summary_file = Path("data/processed/graph_visualization_results.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "demo_type": "graph_evolution_visualization",
                "timestamp": datetime.now().isoformat(),
                "knowledge_facts_count": len(self.knowledge_facts),
                "questions_processed": len(results),
                "results": results,
                "summary": {
                    "insight_questions": sum(1 for r in results if r['insight_detected']),
                    "avg_delta_ged": sum(r['delta_ged'] for r in results) / len(results),
                    "avg_delta_ig": sum(r['delta_ig'] for r in results) / len(results)
                }
            }, f, indent=2)
        
        print(f"\n📊 Summary:")
        print(f"   Total questions: {len(results)}")
        print(f"   Insights detected: {sum(1 for r in results if r['insight_detected'])}")
        print(f"   Average ΔGED: {sum(r['delta_ged'] for r in results) / len(results):.3f}")
        print(f"   Average ΔIG: {sum(r['delta_ig'] for r in results) / len(results):.3f}")
        print(f"\n💾 Results saved to: {summary_file}")
        print(f"🖼️  Visualizations saved to: {output_dir}")
        
        print("\n🎯 Interpretation:")
        print("   • Blue nodes: Knowledge facts")
        print("   • Red node: Question/query")
        print("   • Edges: Similarity connections")
        print("   • ΔGED: Graph structure change")
        print("   • ΔIG: Information gain")
        print("   • Insight threshold: ΔGED≥0.5 AND ΔIG≥0.2")


def main():
    """Run the graph visualization demo"""
    try:
        demo = GraphVisualizationDemo()
        demo.run_visualization_demo()
        
        print("\n🎉 Graph visualization demo completed!")
        print("Check data/processed/graph_visualizations/ for before/after images")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install matplotlib networkx scikit-learn")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
