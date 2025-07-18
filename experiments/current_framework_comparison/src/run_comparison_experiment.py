#!/usr/bin/env python3
"""
Current Framework Comparison Experiment
======================================

Run the same experiment as english_insight_experiment but using
the current InsightSpike framework implementation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import sys
import os

# Set environment variables to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from insightspike.config import InsightSpikeConfig
from insightspike.core.agents.main_agent import MainAgent

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


def run_insightspike_agent(question: str, episodes_path: str) -> Dict[str, Any]:
    """Run InsightSpike agent on a single question"""
    
    # Create config
    config = InsightSpikeConfig()
    # Override to use DistilGPT2 for comparison
    config.core.model_name = "distilgpt2"
    config.core.temperature = 0.7
    config.core.max_tokens = 100
    
    # Memory settings
    config.memory.max_episodes = 10000
    config.memory.retrieval_top_k = 5
    
    # Graph settings
    # config.graph.enable_hierarchical = True  # This property doesn't exist
    # config.graph.max_nodes = 1000  # This property doesn't exist
    # config.graph.similarity_threshold = 0.3  # This property doesn't exist
    config.graph.spike_ged_threshold = 0.5
    config.graph.spike_ig_threshold = 0.2
    
    # Processing settings (these don't exist in Config, skip for now)
    # config.processing.batch_size = 32
    # config.processing.max_cycles = 3
    # config.processing.convergence_threshold = 0.01
    
    # Initialize agent
    agent = MainAgent(config=config)
    
    # Load episodes into memory
    with open(episodes_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    # Initialize agent
    if not agent.initialize():
        raise Exception("Failed to initialize agent")
    
    # Add episodes to agent's memory
    for episode in kb['episodes']:
        c_value = episode['metadata'].get('c_value', 0.5)
        success = agent.l2_memory.store_episode(
            text=episode['content'],
            c_value=c_value,
            metadata=episode['metadata']
        )
        if not success:
            print(f"Warning: Failed to store episode: {episode['content'][:50]}...")
    
    # Process question
    start_time = time.time()
    result = agent.process_question(
        question,
        max_cycles=3,
        verbose=True
    )
    processing_time = time.time() - start_time
    
    # Extract spike detection info
    spike_detected = False
    confidence = 0.0
    context_used = []
    
    if hasattr(result, 'metadata') and result.metadata:
        spike_detected = result.metadata.get('spike_detected', False)
        confidence = result.metadata.get('confidence', 0.0)
        context_used = result.metadata.get('context', [])
    
    # Handle dictionary result format
    if isinstance(result, dict):
        return {
            "response": result.get('response', 'No response generated'),
            "spike_detected": result.get('spike_detected', False),
            "confidence": result.get('reasoning_quality', 0.0),
            "context": [doc.get('text', '') for doc in result.get('documents', [])],
            "time": processing_time,
            "reasoning_path": result.get('cycle_history', [])
        }
    else:
        # Original object format (if it exists)
        return {
            "response": result.answer if hasattr(result, 'answer') else 'No response',
            "spike_detected": spike_detected,
            "confidence": confidence,
            "context": context_used,
            "time": processing_time,
            "reasoning_path": result.reasoning_path if hasattr(result, 'reasoning_path') else []
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
    """Run the comparison experiment"""
    
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
        
        # 3. InsightSpike (Current Framework)
        print("  3️⃣ InsightSpike (Current Framework)...")
        try:
            spike_result = run_insightspike_agent(query, str(knowledge_base_path))
            
            result["insightspike_current"] = {
                "response": spike_result["response"],
                "spike_detected": spike_result["spike_detected"],
                "confidence": spike_result["confidence"],
                "context": spike_result["context"],
                "time": spike_result["time"],
                "quality": evaluate_response_quality(spike_result["response"]),
                "reasoning_path": spike_result["reasoning_path"]
            }
            print(f"    Response: {spike_result['response'][:80]}...")
            
            if spike_result["spike_detected"]:
                print(f"  🎯 Insight detected! (confidence: {spike_result['confidence']:.2%})")
        
        except Exception as e:
            print(f"  ❌ InsightSpike error: {str(e)}")
            result["insightspike_current"] = {
                "response": f"Error: {str(e)}",
                "spike_detected": False,
                "confidence": 0.0,
                "context": [],
                "time": 0,
                "quality": {"overall": 0.0},
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results
    output_file = base_path / "results/outputs/comparison_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Experiment complete! Results saved to: {output_file}")
    
    # Generate summary
    print("\n📊 Summary:")
    insights_detected = sum(1 for r in results 
                          if 'insightspike_current' in r 
                          and r['insightspike_current'].get('spike_detected', False))
    print(f"  - Total questions: {len(queries)}")
    print(f"  - Insights detected: {insights_detected}")
    print(f"  - Insight detection rate: {insights_detected/len(queries):.1%}")
    
    # Quality comparison
    direct_quality = np.mean([r['direct_llm']['quality']['overall'] for r in results])
    rag_quality = np.mean([r['standard_rag']['quality']['overall'] for r in results])
    spike_quality = np.mean([r.get('insightspike_current', {}).get('quality', {}).get('overall', 0) 
                           for r in results])
    
    print(f"\n  Average quality scores:")
    print(f"    - Direct LLM: {direct_quality:.3f}")
    print(f"    - Standard RAG: {rag_quality:.3f}")
    print(f"    - InsightSpike (Current): {spike_quality:.3f}")
    
    # Save CSV for easy comparison
    import csv
    csv_path = base_path / "results/outputs/comparison_results.csv"
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query', 'direct_llm_response', 'standard_rag_response', 
            'insightspike_current_response', 'spike_detected', 'confidence',
            'direct_quality', 'rag_quality', 'spike_quality'
        ])
        
        for r in results:
            spike_data = r.get('insightspike_current', {})
            writer.writerow([
                r['query'],
                r['direct_llm']['response'][:100],
                r['standard_rag']['response'][:100],
                spike_data.get('response', 'Error')[:100],
                spike_data.get('spike_detected', False),
                f"{spike_data.get('confidence', 0):.2%}",
                f"{r['direct_llm']['quality']['overall']:.3f}",
                f"{r['standard_rag']['quality']['overall']:.3f}",
                f"{spike_data.get('quality', {}).get('overall', 0):.3f}"
            ])
    
    print(f"\n📊 CSV saved: {csv_path}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()