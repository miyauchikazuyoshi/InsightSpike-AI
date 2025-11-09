#!/usr/bin/env python3
"""Improved geDIG-RAG experiment with real geDIG calculations and high-quality data."""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict

# Add InsightSpike to path for geDIG import
sys.path.insert(0, '/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/src')

# Import InsightSpike's geDIG
try:
    from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGResult
    from insightspike.algorithms.linkset_adapter import build_linkset_info
    print("✅ InsightSpike geDIG loaded successfully")
    INSIGHTSPIKE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ InsightSpike geDIG not available: {e}")
    INSIGHTSPIKE_AVAILABLE = False

# Core imports
from core.config import ExperimentConfig
from core.knowledge_graph import KnowledgeGraph

# Set style for nice visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


@dataclass
class HighQualityKnowledge:
    """Structured knowledge with rich semantic information."""
    text: str
    concepts: List[str]
    depth: str  # 'surface', 'technical', 'conceptual', 'practical'
    domain: str
    relationships: List[str] = None


class ImprovedEmbedder:
    """Advanced embedder with semantic understanding."""
    
    def __init__(self):
        self.embedding_dim = 384
        np.random.seed(42)
        
        # Define rich concept embeddings for better semantic relationships
        self.concept_vectors = self._initialize_concept_space()
        
    def _initialize_concept_space(self):
        """Initialize a rich concept embedding space."""
        # Core programming concepts
        concepts = {
            # Python specific
            'python': np.random.RandomState(100).normal(0, 1, self.embedding_dim),
            'typing': np.random.RandomState(101).normal(0, 1, self.embedding_dim),
            'garbage_collection': np.random.RandomState(102).normal(0, 1, self.embedding_dim),
            'gil': np.random.RandomState(103).normal(0, 1, self.embedding_dim),
            'decorator': np.random.RandomState(104).normal(0, 1, self.embedding_dim),
            
            # Machine Learning
            'machine_learning': np.random.RandomState(200).normal(0, 1, self.embedding_dim),
            'overfitting': np.random.RandomState(201).normal(0, 1, self.embedding_dim),
            'underfitting': np.random.RandomState(202).normal(0, 1, self.embedding_dim),
            'gradient_descent': np.random.RandomState(203).normal(0, 1, self.embedding_dim),
            'backpropagation': np.random.RandomState(204).normal(0, 1, self.embedding_dim),
            'regularization': np.random.RandomState(205).normal(0, 1, self.embedding_dim),
            
            # Deep Learning
            'deep_learning': np.random.RandomState(300).normal(0, 1, self.embedding_dim),
            'neural_network': np.random.RandomState(301).normal(0, 1, self.embedding_dim),
            'cnn': np.random.RandomState(302).normal(0, 1, self.embedding_dim),
            'rnn': np.random.RandomState(303).normal(0, 1, self.embedding_dim),
            'transformer': np.random.RandomState(304).normal(0, 1, self.embedding_dim),
            'attention': np.random.RandomState(305).normal(0, 1, self.embedding_dim),
            
            # NLP
            'nlp': np.random.RandomState(400).normal(0, 1, self.embedding_dim),
            'tokenization': np.random.RandomState(401).normal(0, 1, self.embedding_dim),
            'embedding': np.random.RandomState(402).normal(0, 1, self.embedding_dim),
            'bert': np.random.RandomState(403).normal(0, 1, self.embedding_dim),
            'gpt': np.random.RandomState(404).normal(0, 1, self.embedding_dim),
            
            # Data concepts
            'data': np.random.RandomState(500).normal(0, 1, self.embedding_dim),
            'preprocessing': np.random.RandomState(501).normal(0, 1, self.embedding_dim),
            'feature_engineering': np.random.RandomState(502).normal(0, 1, self.embedding_dim),
            'normalization': np.random.RandomState(503).normal(0, 1, self.embedding_dim),
            
            # Performance concepts
            'gpu': np.random.RandomState(600).normal(0, 1, self.embedding_dim),
            'optimization': np.random.RandomState(601).normal(0, 1, self.embedding_dim),
            'parallelization': np.random.RandomState(602).normal(0, 1, self.embedding_dim),
            'memory': np.random.RandomState(603).normal(0, 1, self.embedding_dim),
        }
        
        # Normalize all concept vectors
        for concept in concepts:
            concepts[concept] /= np.linalg.norm(concepts[concept])
            
        return concepts
    
    def encode(self, texts, knowledge_objects=None):
        """Encode texts with semantic understanding."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        
        for i, text in enumerate(texts):
            # If we have structured knowledge, use it
            if knowledge_objects and i < len(knowledge_objects):
                embedding = self._encode_structured(knowledge_objects[i])
            else:
                embedding = self._encode_text(text)
            
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def _encode_text(self, text):
        """Encode plain text."""
        text_lower = text.lower()
        embedding = np.zeros(self.embedding_dim)
        
        # Weighted combination of concept embeddings
        concept_weights = defaultdict(float)
        
        for concept, vec in self.concept_vectors.items():
            if concept.replace('_', ' ') in text_lower or concept in text_lower:
                # Weight based on concept importance
                weight = 1.0
                if concept in ['python', 'machine_learning', 'deep_learning', 'nlp']:
                    weight = 1.5  # Core concepts
                concept_weights[concept] = weight
        
        if concept_weights:
            # Weighted average of concept vectors
            total_weight = sum(concept_weights.values())
            for concept, weight in concept_weights.items():
                embedding += self.concept_vectors[concept] * (weight / total_weight)
            
            # Add some noise for variation
            embedding += np.random.RandomState(hash(text) % 10000).normal(0, 0.1, self.embedding_dim)
        else:
            # Random embedding for unknown concepts
            embedding = np.random.RandomState(hash(text) % 10000).normal(0, 1, self.embedding_dim)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding
    
    def _encode_structured(self, knowledge: HighQualityKnowledge):
        """Encode structured knowledge with richer semantics."""
        embedding = np.zeros(self.embedding_dim)
        
        # Combine concept embeddings
        if knowledge.concepts:
            for concept in knowledge.concepts:
                if concept in self.concept_vectors:
                    embedding += self.concept_vectors[concept]
        
        # Add depth-based modulation
        depth_modulation = {
            'surface': 0.5,
            'technical': 1.0,
            'conceptual': 0.8,
            'practical': 0.9
        }
        
        if knowledge.depth in depth_modulation:
            embedding *= depth_modulation[knowledge.depth]
        
        # Add noise for uniqueness
        embedding += np.random.RandomState(hash(knowledge.text) % 10000).normal(0, 0.05, self.embedding_dim)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding


class EnhancedRAGSystem:
    """Enhanced RAG system with real geDIG evaluation."""
    
    def __init__(self, method_name: str, config: ExperimentConfig):
        self.method_name = method_name
        self.config = config
        self.knowledge_graph = KnowledgeGraph(embedding_dim=384)
        self.embedder = ImprovedEmbedder()
        
        # NetworkX graph for geDIG calculation
        self.nx_graph = nx.Graph()
        
        # Initialize geDIG evaluator for proposed method
        if method_name == "gedig" and INSIGHTSPIKE_AVAILABLE:
            self.gedig_core = GeDIGCore(
                node_cost=1.0,
                edge_cost=0.5,
                normalization='sum',
                efficiency_weight=0.3,
                enable_multihop=True,
                max_hops=2,
                decay_factor=0.7,
                spike_threshold=-0.3  # More lenient threshold
            )
        else:
            self.gedig_core = None
            
        # Statistics
        self.queries_processed = 0
        self.updates_applied = 0
        self.similarity_history = []
        self.update_history = []
        self.gedig_scores = []
        self.ig_values = []
        self.ged_values = []
        
    def add_initial_knowledge(self, knowledge_items: List[HighQualityKnowledge]) -> int:
        """Add initial high-quality knowledge base."""
        added = 0
        
        for item in knowledge_items:
            # Encode with structured information
            embedding = self.embedder._encode_structured(item)
            
            node_id = self.knowledge_graph.add_node(
                text=item.text,
                embedding=embedding,
                node_type="initial",
                confidence=0.95,
                metadata={
                    'concepts': item.concepts,
                    'depth': item.depth,
                    'domain': item.domain
                }
            )
            
            if node_id:
                added += 1
                # Add to NetworkX graph for geDIG
                self.nx_graph.add_node(node_id, 
                                      text=item.text,
                                      concepts=item.concepts,
                                      depth=item.depth)
                
        # Create initial edges based on concept overlap
        self._create_concept_edges(knowledge_items)
        
        return added
    
    def _create_concept_edges(self, knowledge_items: List[HighQualityKnowledge]):
        """Create edges based on concept relationships."""
        nodes = list(self.knowledge_graph.nodes.keys())
        
        for i, node1_id in enumerate(nodes):
            for j, node2_id in enumerate(nodes[i+1:], i+1):
                node1 = self.knowledge_graph.nodes[node1_id]
                node2 = self.knowledge_graph.nodes[node2_id]
                
                # Calculate concept overlap
                if 'concepts' in node1.metadata and 'concepts' in node2.metadata:
                    concepts1 = set(node1.metadata['concepts'])
                    concepts2 = set(node2.metadata['concepts'])
                    
                    overlap = len(concepts1 & concepts2)
                    if overlap > 0:
                        weight = overlap / max(len(concepts1), len(concepts2))
                        if weight > 0.2:  # Minimum threshold
                            self.knowledge_graph.add_edge(
                                node1_id, node2_id,
                                relation="conceptual",
                                weight=weight,
                                semantic_similarity=weight
                            )
                            self.nx_graph.add_edge(node1_id, node2_id, weight=weight)
    
    def process_query(self, query: str, query_depth: str = "technical") -> Dict[str, Any]:
        """Process a query through the enhanced RAG pipeline."""
        self.queries_processed += 1
        
        # 1. Retrieve relevant knowledge
        query_embedding = self.embedder.encode(query)[0]
        similar_nodes = self.knowledge_graph.find_similar_nodes(
            query_embedding, k=5, min_similarity=0.0
        )
        
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        self.similarity_history.append(max_similarity)
        
        # 2. Generate response with context synthesis
        response = self._generate_informed_response(query, similar_nodes, query_depth)
        
        # 3. Decide on knowledge update
        should_update, update_metadata = self._should_update_knowledge(
            query, response, max_similarity, similar_nodes
        )
        
        if should_update:
            self._apply_knowledge_update(query, response, query_embedding, similar_nodes, update_metadata)
            self.updates_applied += 1
            self.update_history.append(1)
        else:
            self.update_history.append(0)
        
        return {
            'query': query,
            'response': response,
            'max_similarity': max_similarity,
            'updated': should_update,
            'graph_size': len(self.knowledge_graph.nodes),
            'graph_edges': len(self.knowledge_graph.edges),
            'update_metadata': update_metadata
        }
    
    def _generate_informed_response(self, query: str, similar_nodes: List, depth: str) -> str:
        """Generate a response that adds value beyond existing knowledge."""
        if not similar_nodes or similar_nodes[0][1] < 0.3:
            # No relevant context - need to create new knowledge
            return f"New insight needed for: {query}. This requires exploration beyond current knowledge."
        
        # Synthesize from multiple sources
        contexts = []
        for node_id, similarity in similar_nodes[:3]:
            if similarity > 0.2:
                node = self.knowledge_graph.nodes[node_id]
                contexts.append(node.text)
        
        # Generate response based on query type and depth
        if "how" in query.lower():
            response = f"Mechanism: {contexts[0][:100]}... This works through..."
        elif "why" in query.lower():
            response = f"Reasoning: Based on {contexts[0][:50]}..., because..."
        elif "compare" in query.lower() or "difference" in query.lower():
            response = f"Comparison: Unlike {contexts[0][:50]}..., this approach..."
        else:
            response = f"Synthesis: Combining insights from {len(contexts)} sources..."
        
        return response
    
    def _should_update_knowledge(self, query: str, response: str, 
                                max_similarity: float, similar_nodes: List) -> Tuple[bool, Dict]:
        """Decide whether to update knowledge based on method."""
        metadata = {'method': self.method_name}
        
        if self.method_name == "static":
            return False, metadata
        
        elif self.method_name == "frequency":
            # Update for low similarity or early queries
            should_update = max_similarity < 0.4 or self.queries_processed <= 5
            metadata['reason'] = 'low_similarity' if max_similarity < 0.4 else 'early_query'
            return should_update, metadata
        
        elif self.method_name == "cosine":
            # Update if below threshold
            should_update = max_similarity < self.config.cosine_similarity_threshold
            metadata['similarity'] = max_similarity
            return should_update, metadata
        
        elif self.method_name == "gedig" and self.gedig_core:
            # Use real geDIG evaluation
            return self._evaluate_with_gedig(query, response, similar_nodes)
        
        return False, metadata
    
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: List) -> Tuple[bool, Dict]:
        """Evaluate update using geDIG (InsightSpike or fallback)."""
        # Create hypothetical graph after update
        g_before = self.nx_graph.copy()
        g_after = self.nx_graph.copy()
        
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        g_after.add_node(new_node_id, text=f"Q: {query} A: {response}")
        
        # Add potential edges
        edges_added = 0
        for node_id, similarity in similar_nodes[:3]:
            if similarity > 0.3:
                g_after.add_edge(new_node_id, node_id, weight=similarity)
                edges_added += 1
        
        # Calculate geDIG
        if self.gedig_core:
            # Use InsightSpike's geDIG
            try:
                # Build a simple linkset from similar_nodes as candidate_pool
                pool = [{"index": str(nid), "similarity": float(sim)} for (nid, sim) in similar_nodes]
                decision = {"index": str(pool[0]["index"]) if pool else None, "similarity": float(pool[0]["similarity"]) if pool else 1.0}
                ls = build_linkset_info(s_link=[], candidate_pool=pool, decision=decision, query_vector=None, base_mode="pool")
                result = self.gedig_core.calculate(
                    g_prev=g_before,
                    g_now=g_after,
                    focal_nodes={new_node_id},
                    linkset_info=ls,
                )
                
                # Store metrics
                self.gedig_scores.append(result.gedig_value)
                self.ig_values.append(result.ig_value)
                self.ged_values.append(result.ged_value)
                
                # Decision based on geDIG score
                should_update = result.gedig_value > 0.0  # Positive geDIG means beneficial
                
                metadata = {
                    'gedig_score': result.gedig_value,
                    'ged': result.ged_value,
                    'ig': result.ig_value,
                    'spike_detected': result.spike
                }
                
                return should_update, metadata
                
            except Exception as e:
                print(f"⚠️ geDIG evaluation failed: {e}")
                # Fall through to simple implementation
        
        # Simple geDIG implementation as fallback
        # Calculate Graph Edit Distance (structural change)
        nodes_added = len(g_after.nodes) - len(g_before.nodes)
        edges_before = len(g_before.edges)
        edges_after = len(g_after.edges)
        edges_change = edges_after - edges_before
        
        # Simple GED: weighted sum of node and edge changes
        ged = nodes_added * 0.1 + edges_change * 0.05
        
        # Calculate Information Gain (simplified)
        # Based on connectivity improvement and novelty
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        novelty = 1.0 - max_similarity  # Higher novelty = higher IG
        
        # Connectivity improvement (more edges = better integration)
        connectivity_score = edges_added * 0.2
        
        # Simple IG calculation
        ig = novelty * 0.5 + connectivity_score
        
        # geDIG score with k=0.3 (balancing structure vs information)
        k = 0.3
        gedig_score = ged - k * ig
        
        # Adjust decision threshold based on context
        # More lenient for novel information, stricter for redundant
        if novelty > 0.7:  # Very novel
            threshold = -0.1  # Accept even slightly negative geDIG
        elif novelty > 0.4:  # Moderately novel
            threshold = 0.0
        else:  # Low novelty
            threshold = 0.1  # Require positive geDIG
        
        should_update = gedig_score > threshold
        
        # Store metrics
        self.gedig_scores.append(gedig_score)
        self.ig_values.append(ig)
        self.ged_values.append(ged)
        
        metadata = {
            'gedig_score': gedig_score,
            'ged': ged,
            'ig': ig,
            'novelty': novelty,
            'threshold_used': threshold,
            'implementation': 'fallback'
        }
        
        return should_update, metadata
    
    def _apply_knowledge_update(self, query: str, response: str,
                               query_embedding: np.ndarray, 
                               similar_nodes: List, metadata: Dict):
        """Apply knowledge update with rich information."""
        # Create enriched knowledge
        new_text = f"Q: {query} A: {response}"
        
        # Extract concepts from query and response
        concepts = []
        for concept in self.embedder.concept_vectors.keys():
            if concept.replace('_', ' ') in (query + response).lower():
                concepts.append(concept)
        
        # Determine depth based on response type
        if "mechanism" in response.lower() or "works through" in response.lower():
            depth = "technical"
        elif "reasoning" in response.lower() or "because" in response.lower():
            depth = "conceptual"
        else:
            depth = "practical"
        
        # Create new knowledge embedding
        new_knowledge = HighQualityKnowledge(
            text=new_text,
            concepts=concepts,
            depth=depth,
            domain="ml"
        )
        
        new_embedding = self.embedder._encode_structured(new_knowledge)
        
        # Add to knowledge graph
        node_id = self.knowledge_graph.add_node(
            text=new_text,
            embedding=new_embedding,
            node_type="qa_pair",
            confidence=0.8,
            metadata={
                'concepts': concepts,
                'depth': depth,
                'gedig_metadata': metadata
            }
        )
        
        # Add to NetworkX graph
        self.nx_graph.add_node(node_id, text=new_text, concepts=concepts)
        
        # Create meaningful edges
        for similar_id, similarity in similar_nodes[:3]:
            if similarity > 0.3:
                self.knowledge_graph.add_edge(
                    node_id, similar_id,
                    relation="semantic",
                    weight=similarity,
                    semantic_similarity=similarity
                )
                self.nx_graph.add_edge(node_id, similar_id, weight=similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'method': self.method_name,
            'queries_processed': self.queries_processed,
            'updates_applied': self.updates_applied,
            'update_rate': self.updates_applied / max(1, self.queries_processed),
            'graph_nodes': len(self.knowledge_graph.nodes),
            'graph_edges': len(self.knowledge_graph.edges),
            'avg_similarity': np.mean(self.similarity_history) if self.similarity_history else 0,
            'similarity_history': self.similarity_history,
            'update_history': self.update_history,
        }
        
        if self.method_name == "gedig":
            stats.update({
                'gedig_scores': self.gedig_scores,
                'ig_values': self.ig_values,
                'ged_values': self.ged_values,
                'avg_gedig': np.mean(self.gedig_scores) if self.gedig_scores else 0,
                'positive_gedig_rate': sum(1 for s in self.gedig_scores if s > 0) / max(1, len(self.gedig_scores))
            })
        
        return stats


def create_high_quality_knowledge_base() -> List[HighQualityKnowledge]:
    """Create a rich, structured knowledge base."""
    knowledge_base = [
        # Python Deep Knowledge
        HighQualityKnowledge(
            text="Python's Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously, ensuring thread safety but limiting true parallelism in CPU-bound tasks.",
            concepts=['python', 'gil', 'parallelization', 'memory'],
            depth='technical',
            domain='programming'
        ),
        HighQualityKnowledge(
            text="Python uses reference counting with cycle detection for memory management, automatically deallocating objects when their reference count reaches zero.",
            concepts=['python', 'garbage_collection', 'memory'],
            depth='technical',
            domain='programming'
        ),
        
        # Machine Learning Fundamentals
        HighQualityKnowledge(
            text="Overfitting occurs when a model learns noise in training data, resulting in poor generalization. Regularization techniques like L1/L2 penalties help prevent this.",
            concepts=['machine_learning', 'overfitting', 'regularization'],
            depth='conceptual',
            domain='ml'
        ),
        HighQualityKnowledge(
            text="Gradient descent optimizes model parameters by iteratively moving in the direction of steepest decrease in the loss function, with learning rate controlling step size.",
            concepts=['machine_learning', 'gradient_descent', 'optimization'],
            depth='technical',
            domain='ml'
        ),
        
        # Deep Learning Architecture
        HighQualityKnowledge(
            text="Convolutional Neural Networks (CNNs) use convolution operations to extract hierarchical features from spatial data, with pooling layers reducing dimensionality.",
            concepts=['deep_learning', 'cnn', 'neural_network'],
            depth='technical',
            domain='dl'
        ),
        HighQualityKnowledge(
            text="Transformers revolutionized NLP by using self-attention mechanisms to process sequences in parallel, eliminating the sequential bottleneck of RNNs.",
            concepts=['transformer', 'attention', 'nlp', 'deep_learning'],
            depth='conceptual',
            domain='dl'
        ),
        HighQualityKnowledge(
            text="Backpropagation calculates gradients through automatic differentiation, propagating error signals backward through the network to update weights.",
            concepts=['deep_learning', 'backpropagation', 'neural_network', 'gradient_descent'],
            depth='technical',
            domain='dl'
        ),
        
        # NLP Specifics
        HighQualityKnowledge(
            text="BERT uses bidirectional transformers with masked language modeling pre-training, enabling deep contextualized word representations for downstream tasks.",
            concepts=['bert', 'transformer', 'nlp', 'embedding'],
            depth='technical',
            domain='nlp'
        ),
        HighQualityKnowledge(
            text="Tokenization strategies like BPE and WordPiece balance vocabulary size with representation quality, enabling subword modeling for handling OOV words.",
            concepts=['tokenization', 'nlp', 'preprocessing'],
            depth='practical',
            domain='nlp'
        ),
        
        # Performance & Optimization
        HighQualityKnowledge(
            text="GPU acceleration leverages thousands of CUDA cores for parallel matrix operations, providing 10-100x speedups for deep learning training.",
            concepts=['gpu', 'optimization', 'deep_learning', 'parallelization'],
            depth='practical',
            domain='performance'
        ),
        HighQualityKnowledge(
            text="Mixed precision training uses FP16 for forward/backward passes while maintaining FP32 master weights, reducing memory usage and improving throughput.",
            concepts=['optimization', 'gpu', 'memory', 'deep_learning'],
            depth='practical',
            domain='performance'
        ),
        
        # Data Engineering
        HighQualityKnowledge(
            text="Feature engineering transforms raw data into informative representations, with techniques like one-hot encoding, scaling, and polynomial features improving model performance.",
            concepts=['feature_engineering', 'preprocessing', 'data', 'machine_learning'],
            depth='practical',
            domain='data'
        ),
        HighQualityKnowledge(
            text="Data normalization techniques like StandardScaler and MinMaxScaler ensure features have similar scales, preventing gradient descent from being dominated by large-scale features.",
            concepts=['normalization', 'preprocessing', 'data', 'optimization'],
            depth='practical',
            domain='data'
        )
    ]
    
    return knowledge_base


def create_meaningful_queries() -> List[Tuple[str, str]]:
    """Create meaningful queries that test different aspects."""
    queries = [
        # Direct knowledge queries (should match existing)
        ("How does Python's GIL affect multithreading?", "technical"),
        ("What causes overfitting in machine learning?", "conceptual"),
        
        # Synthesis queries (combine multiple concepts)
        ("How do transformers compare to RNNs for sequence modeling?", "conceptual"),
        ("What's the relationship between gradient descent and backpropagation?", "technical"),
        
        # Extension queries (build on existing knowledge)
        ("How can we overcome GIL limitations in Python?", "practical"),
        ("What are advanced regularization techniques beyond L1/L2?", "technical"),
        
        # Novel queries (require new knowledge)
        ("How does attention mechanism work in transformers?", "technical"),
        ("What is transfer learning and when to use it?", "practical"),
        ("Explain the vanishing gradient problem", "conceptual"),
        ("How do GANs generate realistic images?", "technical"),
        
        # Comparative queries (require reasoning)
        ("Compare CNN and transformer architectures for vision tasks", "conceptual"),
        ("When to use LSTM vs GRU vs Transformer?", "practical"),
        
        # Deep technical queries
        ("How does automatic differentiation work in PyTorch?", "technical"),
        ("Explain the mathematics behind batch normalization", "technical"),
        
        # Application queries
        ("Best practices for deploying ML models in production", "practical"),
        ("How to handle imbalanced datasets?", "practical"),
        
        # Revisiting with depth
        ("Deep dive into Python memory management internals", "technical"),
        ("Advanced optimization techniques for deep learning", "technical"),
        ("State-of-the-art NLP architectures beyond BERT", "conceptual")
    ]
    
    return queries


def run_improved_experiment() -> Dict[str, Any]:
    """Run improved experiment with real geDIG and quality data."""
    print("🚀 Starting Improved geDIG-RAG Experiment")
    print("=" * 60)
    
    # Setup
    config = ExperimentConfig()
    config.cosine_similarity_threshold = 0.6  # More reasonable threshold
    
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Methods to compare
    methods = ["static", "frequency", "cosine", "gedig"]
    if INSIGHTSPIKE_AVAILABLE:
        print("✅ InsightSpike geDIG evaluation enabled")
    else:
        print("⚠️ Using fallback geDIG implementation")
    
    results = {}
    
    for method in methods:
        print(f"\n📊 Testing {method.upper()} RAG...")
        print("-" * 40)
        
        # Initialize system
        system = EnhancedRAGSystem(method, config)
        
        # Add initial knowledge
        n_added = system.add_initial_knowledge(knowledge_base)
        print(f"  Initial knowledge: {n_added} items")
        print(f"  Initial edges: {len(system.knowledge_graph.edges)}")
        
        # Process queries
        query_results = []
        for i, (query, depth) in enumerate(test_queries):
            result = system.process_query(query, depth)
            query_results.append(result)
            
            if (i + 1) % 5 == 0:
                stats = system.get_statistics()
                print(f"  After {i+1} queries: {stats['graph_nodes']} nodes, "
                      f"{stats['graph_edges']} edges, "
                      f"{stats['updates_applied']} updates")
        
        # Final statistics
        final_stats = system.get_statistics()
        final_stats['query_results'] = query_results
        results[method] = final_stats
        
        print(f"  Final: {final_stats['graph_nodes']} nodes, "
              f"{final_stats['graph_edges']} edges, "
              f"{final_stats['updates_applied']} updates "
              f"(rate: {final_stats['update_rate']:.2%})")
        
        if method == "gedig" and 'positive_gedig_rate' in final_stats:
            print(f"  geDIG acceptance rate: {final_stats['positive_gedig_rate']:.2%}")
    
    return results


def visualize_improved_results(results: Dict[str, Any]):
    """Create comprehensive visualizations for improved experiment."""
    print("\n📈 Generating Improved Visualizations...")
    
    # Create output directory
    output_dir = Path("../results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Knowledge Graph Growth with Edges
    ax1 = plt.subplot(2, 3, 1)
    for method in results:
        stats = results[method]
        graph_sizes = [r['graph_size'] for r in stats['query_results']]
        edge_counts = [r['graph_edges'] for r in stats['query_results']]
        
        x = range(len(graph_sizes))
        ax1.plot(x, graph_sizes, label=f"{method.upper()} nodes", marker='o', markersize=3)
        ax1.plot(x, edge_counts, label=f"{method.upper()} edges", linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Query Number')
    ax1.set_ylabel('Count')
    ax1.set_title('Knowledge Graph Evolution (Nodes & Edges)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Update Decision Quality
    ax2 = plt.subplot(2, 3, 2)
    methods = list(results.keys())
    update_rates = [results[m]['update_rate'] for m in methods]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax2.bar(methods, update_rates, color=colors[:len(methods)])
    for bar, rate in zip(bars, update_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1%}', ha='center', va='bottom')
    
    ax2.set_ylabel('Update Rate')
    ax2.set_title('Knowledge Update Rates by Method')
    ax2.set_ylim(0, max(update_rates) * 1.2)
    
    # 3. Similarity Distribution Over Time
    ax3 = plt.subplot(2, 3, 3)
    for method in results:
        similarities = results[method]['similarity_history']
        if similarities:
            # Moving average for smoother visualization
            window = 3
            ma = np.convolve(similarities, np.ones(window)/window, mode='valid')
            ax3.plot(range(len(ma)), ma, label=method.upper(), alpha=0.8)
    
    ax3.set_xlabel('Query Number')
    ax3.set_ylabel('Max Similarity Score')
    ax3.set_title('Query-Knowledge Similarity Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. geDIG Metrics (if available)
    ax4 = plt.subplot(2, 3, 4)
    if 'gedig' in results and results['gedig'].get('gedig_scores'):
        gedig_data = results['gedig']
        x = range(len(gedig_data['gedig_scores']))
        
        ax4.plot(x, gedig_data['gedig_scores'], label='geDIG', color='purple', linewidth=2)
        ax4.plot(x, gedig_data['ged_values'], label='GED', color='blue', alpha=0.7)
        ax4.plot(x, gedig_data['ig_values'], label='IG', color='green', alpha=0.7)
        
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Evaluation Number')
        ax4.set_ylabel('Score')
        ax4.set_title('geDIG Component Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'geDIG Metrics\n(Not Available)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('geDIG Component Analysis')
    
    # 5. Update Patterns Heatmap
    ax5 = plt.subplot(2, 3, 5)
    update_matrix = []
    method_labels = []
    
    for method in results:
        update_pattern = results[method]['update_history']
        # Reshape to 2D for better visualization
        n_rows = 4
        n_cols = len(update_pattern) // n_rows + (1 if len(update_pattern) % n_rows else 0)
        padded = update_pattern + [0] * (n_rows * n_cols - len(update_pattern))
        reshaped = np.array(padded).reshape(n_rows, n_cols)
        update_matrix.append(reshaped)
        method_labels.append(method.upper())
    
    # Stack matrices
    combined = np.vstack(update_matrix)
    
    im = ax5.imshow(combined, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_ylabel('Method')
    ax5.set_xlabel('Query Batch')
    ax5.set_title('Update Decision Patterns (Green=Update, Red=Skip)')
    
    # Set y-tick labels
    y_positions = []
    for i, label in enumerate(method_labels):
        y_positions.append(i * 4 + 1.5)
    ax5.set_yticks(y_positions)
    ax5.set_yticklabels(method_labels)
    
    # 6. Final Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    
    metrics_data = []
    for method in results:
        metrics_data.append([
            results[method]['graph_nodes'],
            results[method]['graph_edges'],
            results[method]['updates_applied'],
            results[method]['update_rate'] * 100
        ])
    
    metrics_array = np.array(metrics_data).T
    metric_names = ['Nodes', 'Edges', 'Updates', 'Rate(%)']
    
    x = np.arange(len(methods))
    width = 0.2
    
    for i, (metric_values, metric_name) in enumerate(zip(metrics_array, metric_names)):
        offset = (i - 1.5) * width
        ax6.bar(x + offset, metric_values, width, label=metric_name)
    
    ax6.set_xlabel('Method')
    ax6.set_ylabel('Value')
    ax6.set_title('Final Performance Metrics')
    ax6.set_xticks(x)
    ax6.set_xticklabels([m.upper() for m in methods])
    ax6.legend()
    
    plt.suptitle('Improved geDIG-RAG Experiment Results (High-Quality Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"improved_experiment_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved improved visualization to: {output_path}")
    
    plt.show()
    
    return output_dir


def save_improved_results(results: Dict[str, Any], output_dir: Path):
    """Save improved experimental results."""
    # Prepare clean results for JSON
    clean_results = {}
    for method, stats in results.items():
        clean_stats = {k: v for k, v in stats.items() 
                      if k not in ['query_results']}
        
        # Add summary statistics
        clean_stats['summary'] = {
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges'],
            'total_updates': stats['updates_applied'],
            'update_rate': stats['update_rate'],
            'avg_similarity': stats['avg_similarity']
        }
        
        if method == 'gedig' and 'avg_gedig' in stats:
            clean_stats['summary']['avg_gedig'] = stats['avg_gedig']
            clean_stats['summary']['positive_rate'] = stats.get('positive_gedig_rate', 0)
        
        clean_results[method] = clean_stats
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"improved_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"  Saved results to: {output_path}")


def print_improved_summary(results: Dict[str, Any]):
    """Print comprehensive summary of improved experiment."""
    print("\n" + "=" * 60)
    print("📊 IMPROVED EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Performance comparison table
    print("\n📈 Performance Comparison:")
    print("-" * 60)
    print(f"{'Method':<12} {'Updates':<10} {'Rate':<12} {'Nodes':<10} {'Edges':<10} {'Avg Sim':<10}")
    print("-" * 60)
    
    for method in results:
        stats = results[method]
        print(f"{method.upper():<12} {stats['updates_applied']:<10} "
              f"{stats['update_rate']:<12.1%} {stats['graph_nodes']:<10} "
              f"{stats['graph_edges']:<10} {stats['avg_similarity']:<10.3f}")
    
    print("-" * 60)
    
    # Key findings
    print("\n🔍 Key Findings:")
    
    # Best performers
    max_updates = max(results.values(), key=lambda x: x['updates_applied'])
    max_edges = max(results.values(), key=lambda x: x['graph_edges'])
    
    print(f"  • Most adaptive: {[k for k, v in results.items() if v == max_updates][0].upper()} "
          f"({max_updates['updates_applied']} updates)")
    print(f"  • Most connected: {[k for k, v in results.items() if v == max_edges][0].upper()} "
          f"({max_edges['graph_edges']} edges)")
    
    # geDIG specific analysis
    if 'gedig' in results and 'gedig_scores' in results['gedig']:
        gedig_stats = results['gedig']
        if gedig_stats['gedig_scores']:
            positive_scores = [s for s in gedig_stats['gedig_scores'] if s > 0]
            negative_scores = [s for s in gedig_stats['gedig_scores'] if s <= 0]
            
            print(f"\n  📊 geDIG Analysis:")
            print(f"    • Evaluations: {len(gedig_stats['gedig_scores'])}")
            print(f"    • Acceptance rate: {len(positive_scores)}/{len(gedig_stats['gedig_scores'])} "
                  f"({gedig_stats.get('positive_gedig_rate', 0):.1%})")
            print(f"    • Avg geDIG: {gedig_stats.get('avg_gedig', 0):.3f}")
            print(f"    • Score range: [{min(gedig_stats['gedig_scores']):.3f}, "
                  f"{max(gedig_stats['gedig_scores']):.3f}]")
    
    print("\n✅ Improved experiment with high-quality data completed!")


def main():
    """Main execution for improved experiment."""
    try:
        # Run improved experiment
        results = run_improved_experiment()
        
        # Visualize results
        output_dir = visualize_improved_results(results)
        
        # Save results
        save_improved_results(results, output_dir)
        
        # Print summary
        print_improved_summary(results)
        
        print("\n🎉 Improved Experiment Completed Successfully!")
        print("📁 Results saved in:", output_dir)
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
