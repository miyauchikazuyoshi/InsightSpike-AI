#!/usr/bin/env python3
"""
Hybrid InsightSpike Experiment
==============================

Combines simplified GED-IG with actual LLM for better responses.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import math

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import actual embedding model
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class HybridGEDIG:
    """Hybrid GED-IG with real embeddings and LLM generation"""
    
    def __init__(self, use_llm: bool = True):
        self.knowledge_graph = {}
        self.embeddings = {}
        self.use_llm = use_llm
        
        # Initialize embedding model
        print("Loading embedding model...", flush=True)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM if requested
        if self.use_llm:
            print("Loading DistilGPT2...", flush=True)
            self.llm = pipeline(
                "text-generation",
                model="distilgpt2",
                device=-1,  # CPU
                max_new_tokens=100
            )
        else:
            self.llm = None
        
        print("Models loaded!", flush=True)
    
    def add_knowledge(self, text: str) -> dict:
        """Add knowledge with real embeddings"""
        # Get real embedding
        embedding = self.embed_model.encode(text).tolist()
        
        node_id = len(self.knowledge_graph)
        self.knowledge_graph[node_id] = {
            'text': text,
            'embedding': embedding,
            'connections': []
        }
        self.embeddings[node_id] = embedding
        
        # Connect to similar nodes based on semantic similarity
        for other_id, other_embedding in self.embeddings.items():
            if other_id != node_id:
                similarity = self._cosine_similarity(embedding, other_embedding)
                if similarity > 0.5:  # Higher threshold for real embeddings
                    self.knowledge_graph[node_id]['connections'].append((other_id, similarity))
                    self.knowledge_graph[other_id]['connections'].append((node_id, similarity))
        
        return {'node_id': node_id, 'connections': len(self.knowledge_graph[node_id]['connections'])}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2 + 1e-8)
    
    def detect_spike(self, question: str) -> Dict[str, Any]:
        """Detect insight spike and generate response"""
        # Get question embedding
        q_embedding = self.embed_model.encode(question).tolist()
        
        # Find semantically relevant nodes
        relevant_nodes = []
        for node_id, node in self.knowledge_graph.items():
            similarity = self._cosine_similarity(q_embedding, node['embedding'])
            if similarity > 0.3:
                relevant_nodes.append((node_id, similarity))
        
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        relevant_nodes = relevant_nodes[:5]
        
        # Calculate insight spike based on graph structure
        if len(relevant_nodes) >= 3:
            # Analyze connectivity pattern
            total_weight = 0
            connection_pairs = 0
            
            for i, (node_id1, sim1) in enumerate(relevant_nodes):
                for j, (node_id2, sim2) in enumerate(relevant_nodes[i+1:], i+1):
                    # Check if nodes are connected
                    for conn_id, conn_weight in self.knowledge_graph[node_id1]['connections']:
                        if conn_id == node_id2:
                            total_weight += conn_weight
                            connection_pairs += 1
            
            # Calculate spike score
            max_possible_pairs = len(relevant_nodes) * (len(relevant_nodes) - 1) / 2
            connectivity_ratio = connection_pairs / max_possible_pairs if max_possible_pairs > 0 else 0
            avg_weight = total_weight / connection_pairs if connection_pairs > 0 else 0
            
            spike_score = connectivity_ratio * avg_weight
            has_spike = spike_score > 0.3
            
            # Generate response
            if self.use_llm and has_spike:
                # Create context from connected knowledge
                context_parts = []
                for node_id, _ in relevant_nodes[:3]:
                    context_parts.append(self.knowledge_graph[node_id]['text'])
                
                context = " ".join(context_parts)
                prompt = f"Based on these concepts: {context}\n\nThe question '{question}' reveals that"
                
                # Generate with LLM
                try:
                    result = self.llm(prompt, max_new_tokens=100, temperature=0.7, do_sample=True)
                    generated = result[0]['generated_text']
                    # Extract only the generated part
                    if "reveals that" in generated:
                        response = generated.split("reveals that")[1].strip()
                    else:
                        response = generated[len(prompt):].strip()
                except Exception as e:
                    print(f"LLM generation error: {e}")
                    response = self._fallback_response(relevant_nodes, has_spike)
            else:
                response = self._fallback_response(relevant_nodes, has_spike)
        else:
            has_spike = False
            spike_score = 0.0
            response = "Insufficient knowledge connections to generate insights."
        
        return {
            'has_spike': has_spike,
            'spike_confidence': spike_score,
            'relevant_nodes': len(relevant_nodes),
            'response': response
        }
    
    def _fallback_response(self, relevant_nodes: List[tuple], has_spike: bool) -> str:
        """Generate response without LLM"""
        if has_spike and relevant_nodes:
            texts = [self.knowledge_graph[nid]['text'] for nid, _ in relevant_nodes[:3]]
            response = f"The integration of concepts ({', '.join(t[:30] + '...' for t in texts)}) "
            response += "reveals emergent relationships that suggest a deeper underlying pattern."
        elif relevant_nodes:
            node_id = relevant_nodes[0][0]
            response = f"This relates to: {self.knowledge_graph[node_id]['text']}"
        else:
            response = "No relevant knowledge found."
        return response


def run_hybrid_experiment(use_llm: bool = True):
    """Run hybrid InsightSpike experiment"""
    print("=" * 60)
    print("Hybrid InsightSpike Experiment")
    print(f"Mode: {'With DistilGPT2' if use_llm else 'Embeddings Only'}")
    print("=" * 60)
    
    # Initialize hybrid system
    print("\nInitializing...")
    gedig = HybridGEDIG(use_llm=use_llm)
    
    # Knowledge base
    knowledge_items = [
        "Energy is the capacity to do work.",
        "Information is defined as the reduction of uncertainty.",
        "Information and entropy have a deep mathematical relationship.",
        "The second law of thermodynamics and Shannon's information theory share the same mathematical structure.",
        "Energy, information, and entropy form the fundamental trinity of the universe.",
        "Life is a dissipative structure that locally decreases entropy.",
        "Consciousness might be quantified by Integrated Information Theory.",
        "Evolution is a process of increasing information processing capability.",
        "Energy, information, and consciousness are different aspects of the same reality.",
        "All physical laws reduce to laws of information conservation and transformation."
    ]
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    for i, item in enumerate(knowledge_items):
        result = gedig.add_knowledge(item)
        print(f"  [{i+1:2d}/10] Added: {item[:50]}... (connections: {result['connections']})")
    
    # Test questions
    test_questions = [
        {
            'question': "How are energy and information fundamentally related?",
            'expected_spike': True
        },
        {
            'question': "Can consciousness be understood through information theory?",
            'expected_spike': True
        },
        {
            'question': "How does life organize information against entropy?",
            'expected_spike': True
        }
    ]
    
    results = []
    print("\nTesting insight detection...")
    
    for i, q in enumerate(test_questions):
        question = q['question']
        print(f"\nQuestion {i+1}: {question}")
        
        start_time = time.time()
        result = gedig.detect_spike(question)
        processing_time = time.time() - start_time
        
        if result['has_spike']:
            print(f"  ✨ SPIKE DETECTED! (confidence: {result['spike_confidence']:.3f})")
        else:
            print(f"  📝 No spike detected")
        
        print(f"  Response: {result['response'][:200]}...")
        print(f"  Processing time: {processing_time:.3f}s")
        
        test_result = {
            'question': question,
            'has_spike': result['has_spike'],
            'spike_confidence': result['spike_confidence'],
            'response': result['response'],
            'processing_time': processing_time,
            'expected_spike': q['expected_spike'],
            'correct': result['has_spike'] == q['expected_spike']
        }
        results.append(test_result)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    correct = sum(1 for r in results if r['correct'])
    spikes_detected = sum(1 for r in results if r['has_spike'])
    avg_confidence = sum(r['spike_confidence'] for r in results if r['has_spike']) / max(spikes_detected, 1)
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")
    print(f"Spikes Detected: {spikes_detected}/{len(results)}")
    print(f"Average Spike Confidence: {avg_confidence:.3f}")
    print(f"Average Processing Time: {avg_time:.3f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "with_llm" if use_llm else "embeddings_only"
    results_file = Path(__file__).parent.parent / "results" / "outputs" / f"hybrid_{mode}_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'hybrid_insightspike',
            'mode': mode,
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'accuracy': correct / len(results),
                'spikes_detected': spikes_detected,
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_time
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\n✅ Experiment complete!")
    
    return results


if __name__ == "__main__":
    # Run without LLM first (faster)
    print("Running embeddings-only experiment...")
    results1 = run_hybrid_experiment(use_llm=False)
    
    print("\n" + "=" * 80 + "\n")
    
    # Run with LLM
    print("Running with DistilGPT2...")
    results2 = run_hybrid_experiment(use_llm=True)