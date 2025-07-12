#!/usr/bin/env python3
"""
English LLM Experiment with DistilGPT-2
======================================

Run the actual comparison experiment in English
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

# For embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For lightweight LLM
from transformers import pipeline, set_seed


class EnglishLLM:
    """English LLM using DistilGPT-2"""
    
    def __init__(self):
        print("🤖 Initializing English LLM...")
        self.generator = pipeline(
            'text-generation',
            model='distilgpt2',
            device=-1  # CPU
        )
        set_seed(42)
        print("✅ LLM ready")
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text"""
        # Truncate prompt if too long
        if len(prompt.split()) > 100:
            words = prompt.split()
            # Keep question and some context
            if "Question:" in prompt:
                question_idx = prompt.find("Question:")
                prompt = prompt[:100] + "..." + prompt[question_idx:]
            else:
                prompt = ' '.join(words[-50:])
        
        # Generate
        outputs = self.generator(
            prompt,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256,
            do_sample=True,
            top_p=0.95
        )
        
        # Extract response
        generated = outputs[0]['generated_text']
        response = generated[len(prompt):].strip()
        
        # Clean up response - take first complete sentence
        if '. ' in response:
            response = response.split('. ')[0] + '.'
        elif '\n' in response:
            response = response.split('\n')[0]
            
        return response if response else "Based on the provided knowledge."


class DirectLLM:
    """Direct LLM approach"""
    
    def __init__(self):
        self.llm = EnglishLLM()
        
    def answer(self, query: str) -> str:
        """Answer directly"""
        prompt = f"Question: {query}\nAnswer:"
        return self.llm.generate(prompt)


class StandardRAG:
    """Standard RAG approach"""
    
    def __init__(self, knowledge_base_path: str):
        # Load knowledge base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
            
        # Initialize embedder
        print("  🔍 Initializing embeddings...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Pre-compute embeddings
        self.texts = []
        for episode in self.knowledge_base['episodes']:
            self.texts.append(episode['text'])
            
        print(f"  📊 Computing embeddings for {len(self.texts)} texts...")
        self.embeddings = self.embedder.encode(self.texts, show_progress_bar=False)
        
        self.llm = EnglishLLM()
        
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant knowledge"""
        query_embedding = self.embedder.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.texts[i] for i in top_indices]
        
    def answer(self, query: str) -> Tuple[str, List[str]]:
        """Answer using RAG"""
        # Retrieve relevant texts
        relevant_texts = self.retrieve(query)
        
        # Build context
        context = "\n".join([f"- {text}" for text in relevant_texts])
        
        # Create prompt
        prompt = f"""Based on the following knowledge:
{context}

Question: {query}
Answer:"""
        
        response = self.llm.generate(prompt)
        
        return response, relevant_texts


class InsightSpike:
    """InsightSpike approach"""
    
    def __init__(self, knowledge_base_path: str):
        # Load knowledge base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
            
        # Initialize components
        print("  🚀 Initializing InsightSpike...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = EnglishLLM()
        
        # Build knowledge graph by phases
        self._build_knowledge_graph()
        
    def _build_knowledge_graph(self):
        """Build knowledge graph"""
        self.phase_groups = {}
        self.phase_embeddings = {}
        
        # Phase names
        phase_names = {
            1: "Basic Concepts",
            2: "Relationships", 
            3: "Deep Integration",
            4: "Emergent Insights",
            5: "Integration and Circulation"
        }
        
        for episode in self.knowledge_base['episodes']:
            phase = phase_names.get(episode.get('phase', 1), 'Unknown')
            if phase not in self.phase_groups:
                self.phase_groups[phase] = []
            self.phase_groups[phase].append(episode['text'])
        
        print("  🧠 Computing phase embeddings...")
        for phase, texts in self.phase_groups.items():
            if texts:
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
                self.phase_embeddings[phase] = {
                    'texts': texts,
                    'embeddings': embeddings
                }
    
    def detect_insight_opportunity(self, query: str) -> Tuple[bool, float, List[Tuple[str, str, float]]]:
        """Detect insight opportunity"""
        query_embedding = self.embedder.encode([query])
        
        # Collect relevant knowledge across phases
        relevant_across_phases = []
        unique_phases = set()
        
        for phase, data in self.phase_embeddings.items():
            similarities = cosine_similarity(query_embedding, data['embeddings'])[0]
            
            # Get top item from each phase
            if len(similarities) > 0:
                top_idx = np.argmax(similarities)
                
                if similarities[top_idx] > 0.3:  # Threshold
                    relevant_across_phases.append((
                        phase,
                        data['texts'][top_idx],
                        similarities[top_idx]
                    ))
                    unique_phases.add(phase)
        
        # Spike detection: when knowledge from 3+ different phases converges
        spike_detected = len(unique_phases) >= 3
        confidence = len(unique_phases) / 5.0  # Out of 5 phases
        
        # Sort by score
        relevant_across_phases.sort(key=lambda x: x[2], reverse=True)
        
        return spike_detected, confidence, relevant_across_phases[:5]
        
    def answer(self, query: str) -> Tuple[str, bool, float, List[str]]:
        """Answer using InsightSpike"""
        # Detect insight opportunity
        spike_detected, confidence, relevant_texts = self.detect_insight_opportunity(query)
        
        if spike_detected and confidence >= 0.6:
            # Insight mode: integrate multiple phases
            context_parts = []
            for phase, text, score in relevant_texts:
                context_parts.append(f"[{phase}]: {text}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""Integrate knowledge from different perspectives to generate a deep insight:

{context}

Question: {query}
Integrated insight:"""
            
        else:
            # Normal mode
            context_parts = []
            for phase, text, score in relevant_texts[:2]:
                context_parts.append(f"- {text}")
            
            context = "\n".join(context_parts) if context_parts else "General knowledge"
            
            prompt = f"""Based on:
{context}

Question: {query}
Answer:"""
        
        response = self.llm.generate(prompt)
        
        # Return context list
        context_list = [f"[{phase}] {text}" for phase, text, _ in relevant_texts]
        
        return response, spike_detected, confidence, context_list


def evaluate_response_quality(response: str) -> Dict[str, float]:
    """Evaluate response quality"""
    quality = {
        'length': min(len(response) / 100, 1.0),
        'depth': 0.0,
        'specificity': 0.0,
        'integration': 0.0,
        'insight': 0.0
    }
    
    # Keyword-based evaluation
    depth_keywords = ['because', 'therefore', 'specifically', 'for example', 'this means']
    specificity_keywords = ['energy', 'information', 'entropy', 'quantum', 'consciousness']
    integration_keywords = ['relationship', 'connection', 'integration', 'unified', 'perspective']
    insight_keywords = ['insight', 'discovery', 'emerges', 'fundamental', 'principle']
    
    text = response.lower()
    
    for keyword in depth_keywords:
        if keyword in text:
            quality['depth'] += 0.2
            
    for keyword in specificity_keywords:
        if keyword in text:
            quality['specificity'] += 0.2
            
    for keyword in integration_keywords:
        if keyword in text:
            quality['integration'] += 0.2
            
    for keyword in insight_keywords:
        if keyword in text:
            quality['insight'] += 0.2
    
    # Cap at 1.0
    for key in quality:
        quality[key] = min(quality[key], 1.0)
    
    # Overall score
    quality['overall'] = np.mean(list(quality.values()))
    
    return quality


def run_experiment():
    """Run the experiment"""
    
    # Knowledge base path
    knowledge_base_path = "english_knowledge_base.json"
    
    # English queries
    queries = [
        "What is the relationship between energy and information?",
        "Why does consciousness emerge?",
        "What is the mechanism of creativity at the edge of chaos?",
        "What is entropy?",
        "Can you explain quantum entanglement?",
        "Is there a principle that unifies all phenomena?"
    ]
    
    # Initialize systems
    print("\n🚀 Initializing systems...")
    direct_llm = DirectLLM()
    standard_rag = StandardRAG(knowledge_base_path)
    insight_spike = InsightSpike(knowledge_base_path)
    
    # Store results
    results = []
    
    print("\n📝 Starting experiment...")
    for i, query in enumerate(queries):
        print(f"\n--- Question {i+1}/{len(queries)}: {query} ---")
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        # Direct LLM
        print("  1️⃣ Direct LLM...")
        start_time = time.time()
        direct_response = direct_llm.answer(query)
        direct_time = time.time() - start_time
        
        result["direct_llm"] = {
            "response": direct_response,
            "time": direct_time,
            "quality": evaluate_response_quality(direct_response)
        }
        print(f"    Response: {direct_response[:80]}...")
        
        # Standard RAG
        print("  2️⃣ Standard RAG...")
        start_time = time.time()
        rag_response, rag_context = standard_rag.answer(query)
        rag_time = time.time() - start_time
        
        result["standard_rag"] = {
            "response": rag_response,
            "context": rag_context,
            "time": rag_time,
            "quality": evaluate_response_quality(rag_response)
        }
        print(f"    Response: {rag_response[:80]}...")
        
        # InsightSpike
        print("  3️⃣ InsightSpike...")
        start_time = time.time()
        spike_response, spike_detected, confidence, spike_context = insight_spike.answer(query)
        spike_time = time.time() - start_time
        
        result["insightspike"] = {
            "response": spike_response,
            "spike_detected": spike_detected,
            "confidence": confidence,
            "context": spike_context,
            "time": spike_time,
            "quality": evaluate_response_quality(spike_response)
        }
        print(f"    Response: {spike_response[:80]}...")
        
        results.append(result)
        
        if spike_detected:
            print(f"  🎯 Insight detected! (confidence: {confidence:.2%})")
    
    # Save results
    output_file = "english_experiment_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Experiment complete! Results saved to: {output_file}")
    
    # Generate summary
    print("\n📊 Summary:")
    insights_detected = sum(1 for r in results if r['insightspike']['spike_detected'])
    print(f"  - Total questions: {len(queries)}")
    print(f"  - Insights detected: {insights_detected}")
    print(f"  - Insight detection rate: {insights_detected/len(queries):.1%}")
    
    # Quality comparison
    direct_quality = np.mean([r['direct_llm']['quality']['overall'] for r in results])
    rag_quality = np.mean([r['standard_rag']['quality']['overall'] for r in results])
    spike_quality = np.mean([r['insightspike']['quality']['overall'] for r in results])
    
    print(f"\n  Average quality scores:")
    print(f"    - Direct LLM: {direct_quality:.3f}")
    print(f"    - Standard RAG: {rag_quality:.3f}")
    print(f"    - InsightSpike: {spike_quality:.3f}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    # Generate CSV
    print("\n📊 Generating CSV...")
    import csv
    
    with open('english_qa_results.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query', 'direct_llm_response', 'standard_rag_response', 
            'insightspike_response', 'spike_detected', 'confidence',
            'new_concepts', 'structure_improvement', 'phases_integrated'
        ])
        
        for r in results:
            # Count phases
            phases = set()
            for ctx in r['insightspike']['context']:
                if ']' in ctx:
                    phase = ctx.split(']')[0][1:]
                    phases.add(phase)
            
            # Detect if RAG actually uses context
            rag_uses_context = any(
                word in r['standard_rag']['response'] 
                for ctx in r['standard_rag']['context']
                for word in ctx.split()[:5]
                if len(word) > 4
            )
            
            # Simple metric for new concepts
            direct_words = set(r['direct_llm']['response'].lower().split())
            spike_words = set(r['insightspike']['response'].lower().split())
            new_concepts = len(spike_words - direct_words)
            
            writer.writerow([
                r['query'],
                r['direct_llm']['response'][:100],
                r['standard_rag']['response'][:100],
                r['insightspike']['response'][:100],
                r['insightspike']['spike_detected'],
                f"{r['insightspike']['confidence']:.2%}",
                f"+{new_concepts} words",
                "Enhanced" if r['insightspike']['spike_detected'] else "Normal",
                len(phases)
            ])
    
    print("✅ CSV saved: english_qa_results.csv")