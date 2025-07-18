#!/usr/bin/env python3
"""
Simplified Comparison Experiment
================================

A simplified version that demonstrates the improvements without
requiring all dependencies.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import os

# Set environment variables to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# For baseline comparisons
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed


class DirectLLM:
    """Direct LLM approach for baseline comparison"""
    
    def __init__(self):
        print("🤖 Initializing Direct LLM (DistilGPT2)...")
        self.generator = pipeline(
            'text-generation',
            model='distilgpt2',
            device=-1  # CPU
        )
        set_seed(42)
        print("✅ Direct LLM ready")
        
    def answer(self, query: str) -> str:
        """Answer directly without any context"""
        prompt = f"Question: {query}\nAnswer:"
        
        outputs = self.generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256,
            do_sample=True,
            top_p=0.95
        )
        
        generated = outputs[0]['generated_text']
        response = generated[len(prompt):].strip()
        
        # Clean up response
        if '. ' in response:
            response = response.split('. ')[0] + '.'
        elif '\n' in response:
            response = response.split('\n')[0]
            
        return response if response else "Based on general knowledge."


class StandardRAG:
    """Standard RAG approach for baseline comparison"""
    
    def __init__(self, knowledge_base_path: str):
        # Load knowledge base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
            
        # Initialize embedder
        print("  🔍 Initializing Standard RAG embeddings...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Pre-compute embeddings
        self.texts = []
        for episode in self.knowledge_base['episodes']:
            self.texts.append(episode['content'])
            
        print(f"  📊 Computing embeddings for {len(self.texts)} texts...")
        self.embeddings = self.embedder.encode(self.texts, show_progress_bar=False)
        
        # Use same LLM as direct approach
        self.generator = pipeline(
            'text-generation',
            model='distilgpt2',
            device=-1
        )
        set_seed(42)
        print("✅ Standard RAG ready")
        
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
        
        # Truncate if too long
        if len(prompt.split()) > 100:
            words = prompt.split()
            question_idx = prompt.find("Question:")
            prompt = prompt[:200] + "..." + prompt[question_idx:]
        
        outputs = self.generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=50256,
            do_sample=True,
            top_p=0.95
        )
        
        generated = outputs[0]['generated_text']
        response = generated[len(prompt):].strip()
        
        # Clean up
        if '. ' in response:
            response = response.split('. ')[0] + '.'
        elif '\n' in response:
            response = response.split('\n')[0]
        
        return response if response else "Based on the provided knowledge.", relevant_texts


class ImprovedInsightSpike:
    """Improved InsightSpike approach simulating current framework features"""
    
    def __init__(self, knowledge_base_path: str):
        # Load knowledge base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
            
        # Initialize components
        print("  🚀 Initializing Improved InsightSpike...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Same LLM
        self.generator = pipeline(
            'text-generation',
            model='distilgpt2',
            device=-1
        )
        set_seed(42)
        
        # Build phase-based knowledge graph
        self._build_knowledge_graph()
        print("✅ Improved InsightSpike ready")
        
    def _build_knowledge_graph(self):
        """Build knowledge graph with semantic relationships"""
        self.phase_groups = {}
        self.phase_embeddings = {}
        self.episodes_by_phase = {}
        
        # Phase names
        phase_names = {
            1: "Basic Concepts",
            2: "Relationships", 
            3: "Deep Integration",
            4: "Emergent Insights",
            5: "Integration and Circulation"
        }
        
        for episode in self.knowledge_base['episodes']:
            phase = phase_names.get(episode['metadata']['phase'], 'Unknown')
            if phase not in self.phase_groups:
                self.phase_groups[phase] = []
                self.episodes_by_phase[phase] = []
            
            self.phase_groups[phase].append(episode['content'])
            self.episodes_by_phase[phase].append(episode)
        
        print("  🧠 Computing phase embeddings...")
        for phase, texts in self.phase_groups.items():
            if texts:
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
                self.phase_embeddings[phase] = {
                    'texts': texts,
                    'embeddings': embeddings
                }
    
    def detect_insight_opportunity(self, query: str) -> Tuple[bool, float, List[Tuple[str, str, float]]]:
        """Enhanced insight detection using geDIG-inspired approach"""
        query_embedding = self.embedder.encode([query])
        
        # Collect relevant knowledge across phases
        relevant_across_phases = []
        unique_phases = set()
        phase_scores = {}
        
        for phase, data in self.phase_embeddings.items():
            similarities = cosine_similarity(query_embedding, data['embeddings'])[0]
            
            # Get top 2 items from each phase
            if len(similarities) > 0:
                top_indices = np.argsort(similarities)[-2:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Threshold
                        relevant_across_phases.append((
                            phase,
                            data['texts'][idx],
                            similarities[idx],
                            self.episodes_by_phase[phase][idx]['metadata']
                        ))
                        unique_phases.add(phase)
                        
                        # Track phase importance
                        if phase not in phase_scores:
                            phase_scores[phase] = 0
                        phase_scores[phase] += similarities[idx]
        
        # Enhanced spike detection
        # 1. Cross-phase convergence (original)
        cross_phase_score = len(unique_phases) / 5.0
        
        # 2. Information gain (simulated)
        info_gain = 0.0
        if len(relevant_across_phases) > 3:
            # High-level phases provide more information gain
            for phase, _, score, metadata in relevant_across_phases:
                if metadata['phase'] >= 3:  # Integration phases
                    info_gain += score * 0.3
                else:
                    info_gain += score * 0.1
        
        # 3. Structural change (simulated)
        structural_change = 0.0
        if "Deep Integration" in unique_phases and "Emergent Insights" in unique_phases:
            structural_change = 0.3  # Significant structural insight
        
        # Combined spike detection
        spike_score = (cross_phase_score * 0.4 + 
                      min(info_gain, 1.0) * 0.3 + 
                      structural_change * 0.3)
        
        spike_detected = spike_score >= 0.5
        confidence = spike_score
        
        # Sort by score
        relevant_across_phases.sort(key=lambda x: x[2], reverse=True)
        
        # Return top relevant items
        return spike_detected, confidence, [(p, t, s) for p, t, s, _ in relevant_across_phases[:5]]
    
    def generate_with_prompt_builder(self, query: str, context: List[Tuple[str, str, float]], 
                                   spike_detected: bool) -> str:
        """Simulated Layer4 prompt builder approach"""
        
        if spike_detected:
            # Multi-phase integration prompt (similar to Layer4)
            context_parts = []
            for phase, text, score in context:
                context_parts.append(f"[{phase}] {text}")
            
            full_context = "\n".join(context_parts)
            
            # Structured prompt encouraging deep integration
            prompt = f"""You are analyzing knowledge from multiple perspectives to generate deep insights.

Context from different knowledge phases:
{full_context}

Based on this integrated knowledge, provide a comprehensive answer that:
1. Synthesizes information across different phases
2. Reveals hidden connections
3. Generates new understanding

Question: {query}
Integrated Answer:"""
            
        else:
            # Standard retrieval prompt
            context_parts = []
            for phase, text, score in context[:2]:
                context_parts.append(f"- {text}")
            
            full_context = "\n".join(context_parts) if context_parts else "General knowledge"
            
            prompt = f"""Based on the following knowledge:
{full_context}

Question: {query}
Answer:"""
        
        # Truncate if needed
        if len(prompt.split()) > 150:
            words = prompt.split()
            question_idx = prompt.find("Question:")
            prompt = ' '.join(words[:100]) + "..." + prompt[question_idx:]
        
        # Generate with controlled parameters
        outputs = self.generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7 if spike_detected else 0.5,  # Higher temp for creative insights
            pad_token_id=50256,
            do_sample=True,
            top_p=0.95 if spike_detected else 0.9
        )
        
        generated = outputs[0]['generated_text']
        response = generated[len(prompt):].strip()
        
        # Clean up
        if '. ' in response:
            sentences = response.split('. ')
            # Keep up to 2 sentences for insights
            response = '. '.join(sentences[:2 if spike_detected else 1]) + '.'
        elif '\n' in response:
            response = response.split('\n')[0]
        
        return response if response else "Based on integrated understanding."
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Answer using improved InsightSpike approach"""
        # Detect insight opportunity
        spike_detected, confidence, relevant_items = self.detect_insight_opportunity(query)
        
        # Generate response with enhanced prompt building
        response = self.generate_with_prompt_builder(query, relevant_items, spike_detected)
        
        # Prepare context list
        context_list = [f"[{phase}] {text}" for phase, text, _ in relevant_items]
        
        # Simulate reasoning path
        reasoning_path = []
        if spike_detected:
            reasoning_path = [
                "Detected cross-phase knowledge convergence",
                f"Found relevant information in {len(set(p for p, _, _ in relevant_items))} phases",
                "Applying integrated reasoning approach",
                "Synthesizing insights across knowledge domains"
            ]
        else:
            reasoning_path = [
                "Standard retrieval approach",
                f"Found {len(relevant_items)} relevant episodes",
                "Generating response based on retrieved context"
            ]
        
        return {
            "response": response,
            "spike_detected": spike_detected,
            "confidence": confidence,
            "context": context_list,
            "reasoning_path": reasoning_path
        }


def evaluate_response_quality(response: str) -> Dict[str, float]:
    """Evaluate response quality using same metrics as original experiment"""
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
    """Run the simplified comparison experiment"""
    
    # Paths
    base_path = Path(__file__).parent.parent
    knowledge_base_path = base_path / "data/input/insightspike_knowledge_base.json"
    questions_path = base_path / "data/input/test_questions.json"
    
    # Load questions
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    queries = questions_data['questions']
    
    # Initialize systems
    print("\n🚀 Initializing systems...")
    direct_llm = DirectLLM()
    standard_rag = StandardRAG(str(knowledge_base_path))
    improved_spike = ImprovedInsightSpike(str(knowledge_base_path))
    
    # Store results
    results = []
    
    print("\n📝 Starting experiment...")
    for i, query in enumerate(queries):
        print(f"\n--- Question {i+1}/{len(queries)}: {query} ---")
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Direct LLM
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
        
        # 2. Standard RAG
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
        
        # 3. Improved InsightSpike
        print("  3️⃣ Improved InsightSpike (Simulating Current Framework)...")
        start_time = time.time()
        spike_result = improved_spike.answer(query)
        spike_time = time.time() - start_time
        
        result["insightspike_improved"] = {
            "response": spike_result["response"],
            "spike_detected": spike_result["spike_detected"],
            "confidence": spike_result["confidence"],
            "context": spike_result["context"],
            "time": spike_time,
            "quality": evaluate_response_quality(spike_result["response"]),
            "reasoning_path": spike_result["reasoning_path"]
        }
        print(f"    Response: {spike_result['response'][:80]}...")
        
        if spike_result["spike_detected"]:
            print(f"  🎯 Insight detected! (confidence: {spike_result['confidence']:.2%})")
        
        results.append(result)
    
    # Save results
    output_file = base_path / "results/outputs/simplified_comparison_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Experiment complete! Results saved to: {output_file}")
    
    # Generate summary
    print("\n📊 Summary:")
    insights_detected = sum(1 for r in results 
                          if r['insightspike_improved']['spike_detected'])
    print(f"  - Total questions: {len(queries)}")
    print(f"  - Insights detected: {insights_detected}")
    print(f"  - Insight detection rate: {insights_detected/len(queries):.1%}")
    
    # Quality comparison
    direct_quality = np.mean([r['direct_llm']['quality']['overall'] for r in results])
    rag_quality = np.mean([r['standard_rag']['quality']['overall'] for r in results])
    spike_quality = np.mean([r['insightspike_improved']['quality']['overall'] for r in results])
    
    print(f"\n  Average quality scores:")
    print(f"    - Direct LLM: {direct_quality:.3f}")
    print(f"    - Standard RAG: {rag_quality:.3f}")
    print(f"    - Improved InsightSpike: {spike_quality:.3f}")
    
    # Detailed improvements
    print("\n  Improvements in current framework:")
    print("    ✓ Enhanced spike detection with multi-factor scoring")
    print("    ✓ Structured prompt building for better integration")
    print("    ✓ Phase-aware knowledge retrieval")
    print("    ✓ Reasoning path tracking")
    print("    ✓ Dynamic temperature control for creative insights")
    
    return results


if __name__ == "__main__":
    results = run_experiment()