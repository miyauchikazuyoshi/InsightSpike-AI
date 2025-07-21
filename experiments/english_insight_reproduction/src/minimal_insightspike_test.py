#!/usr/bin/env python3
"""
Minimal InsightSpike Test
========================

A simplified test to demonstrate InsightSpike capabilities without complex initialization.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import random
import math

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleGEDIG:
    """Simplified GED-IG implementation for demonstration"""
    
    def __init__(self):
        self.knowledge_graph = {}
        self.embeddings = {}
    
    def add_knowledge(self, text: str) -> dict:
        """Add knowledge and compute simple embedding"""
        # Simple hash-based embedding
        embedding = [hash(word) % 100 / 100.0 for word in text.split()[:10]]
        embedding.extend([0.5] * (10 - len(embedding)))
        
        node_id = len(self.knowledge_graph)
        self.knowledge_graph[node_id] = {
            'text': text,
            'embedding': embedding,
            'connections': []
        }
        self.embeddings[node_id] = embedding
        
        # Connect to similar nodes
        for other_id, other_embedding in self.embeddings.items():
            if other_id != node_id:
                similarity = self._cosine_similarity(embedding, other_embedding)
                if similarity > 0.3:
                    self.knowledge_graph[node_id]['connections'].append(other_id)
                    self.knowledge_graph[other_id]['connections'].append(node_id)
        
        return {'node_id': node_id, 'connections': len(self.knowledge_graph[node_id]['connections'])}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2 + 1e-8)
    
    def detect_spike(self, question: str) -> Dict[str, Any]:
        """Detect insight spike based on graph structure change"""
        # Get question embedding
        q_embedding = [hash(word) % 100 / 100.0 for word in question.split()[:10]]
        q_embedding.extend([0.5] * (10 - len(q_embedding)))
        
        # Find relevant nodes
        relevant_nodes = []
        for node_id, node in self.knowledge_graph.items():
            similarity = self._cosine_similarity(q_embedding, node['embedding'])
            if similarity > 0.2:
                relevant_nodes.append((node_id, similarity))
        
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        relevant_nodes = relevant_nodes[:5]
        
        # Calculate graph metrics
        if len(relevant_nodes) >= 3:
            # Count cross-connections between relevant nodes
            cross_connections = 0
            for node_id, _ in relevant_nodes:
                for other_id, _ in relevant_nodes:
                    if other_id in self.knowledge_graph[node_id]['connections']:
                        cross_connections += 1
            
            # Simple spike detection: high cross-connectivity indicates insight
            spike_score = cross_connections / (len(relevant_nodes) * 2)
            has_spike = spike_score > 0.5
            
            # Generate response based on connections
            if has_spike:
                connected_texts = []
                for node_id, _ in relevant_nodes[:3]:
                    connected_texts.append(self.knowledge_graph[node_id]['text'])
                
                response = f"Based on the integration of multiple concepts: {', '.join(t[:30] + '...' for t in connected_texts)}, "
                response += "there appears to be a deeper relationship that emerges from their interaction. "
                response += "This represents a potential insight into how these concepts are fundamentally connected."
            else:
                node_id = relevant_nodes[0][0] if relevant_nodes else 0
                response = f"The concept relates to: {self.knowledge_graph[node_id]['text']}"
        else:
            has_spike = False
            spike_score = 0.0
            response = "Insufficient knowledge to generate insights."
        
        return {
            'has_spike': has_spike,
            'spike_confidence': spike_score,
            'relevant_nodes': len(relevant_nodes),
            'response': response
        }


def run_minimal_experiment():
    """Run minimal InsightSpike demonstration"""
    print("=" * 60)
    print("Minimal InsightSpike Demonstration")
    print("Using Simplified GED-IG Algorithm")
    print("=" * 60)
    
    # Initialize simple GED-IG
    gedig = SimpleGEDIG()
    
    # Knowledge base
    knowledge_items = [
        "Energy is the capacity to do work.",
        "Information is defined as the reduction of uncertainty.",
        "Information and entropy have a deep mathematical relationship.",
        "The second law of thermodynamics and Shannon's information theory share the same mathematical structure.",
        "Energy, information, and entropy form the fundamental trinity of the universe.",
        "Life is a dissipative structure that locally decreases entropy.",
        "Can the hard problem of consciousness be solved from an information integration perspective?",
        "Is evolution a process for the universe to recognize itself?",
        "Energy, information, and consciousness are different aspects of the same reality.",
        "All physical laws reduce to laws of information conservation and transformation."
    ]
    
    # Add knowledge
    print("\nBuilding knowledge graph...")
    for i, item in enumerate(knowledge_items):
        result = gedig.add_knowledge(item)
        print(f"  [{i+1:2d}/10] Added: {item[:50]}... (connections: {result['connections']})")
    
    # Test questions
    test_questions = [
        "How are energy and information fundamentally related?",
        "Can consciousness be understood through information theory?",
        "How does life organize information against entropy?"
    ]
    
    results = []
    print("\nTesting insight detection...")
    
    for i, question in enumerate(test_questions):
        print(f"\nQuestion {i+1}: {question}")
        
        start_time = time.time()
        result = gedig.detect_spike(question)
        processing_time = time.time() - start_time
        
        if result['has_spike']:
            print(f"  ✨ SPIKE DETECTED! (confidence: {result['spike_confidence']:.3f})")
        else:
            print(f"  📝 No spike detected")
        
        print(f"  Response: {result['response'][:150]}...")
        print(f"  Processing time: {processing_time:.3f}s")
        
        results.append({
            'question': question,
            'has_spike': result['has_spike'],
            'spike_confidence': result['spike_confidence'],
            'response': result['response'],
            'processing_time': processing_time
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    spikes_detected = sum(1 for r in results if r['has_spike'])
    avg_confidence = sum(r['spike_confidence'] for r in results if r['has_spike']) / max(spikes_detected, 1)
    
    print(f"Spikes Detected: {spikes_detected}/3")
    print(f"Average Spike Confidence: {avg_confidence:.3f}")
    print(f"Total Processing Time: {sum(r['processing_time'] for r in results):.3f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent.parent / "results" / "outputs" / f"minimal_test_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'minimal_insightspike_demo',
            'algorithm': 'simplified_gedig',
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'spikes_detected': spikes_detected,
                'avg_confidence': avg_confidence
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\n✅ Demonstration complete!")


if __name__ == "__main__":
    run_minimal_experiment()