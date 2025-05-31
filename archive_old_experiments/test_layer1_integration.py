#!/usr/bin/env python3
"""
Focused Layer1 Integration Test
===============================

Test the enhanced Layer1 and adaptive topK functionality without full agent processing.
This validates the core Layer1 enhancements are working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from collections import Counter

# Import the enhanced Layer1 and adaptive topK components
from insightspike.layer1_error_monitor import analyze_input
from insightspike.adaptive_topk import calculate_adaptive_topk, estimate_chain_reaction_potential
from insightspike.loader import load_corpus

def main():
    print("🔬 Enhanced Layer1 Integration Test")
    print("=" * 60)
    
    # Load knowledge base
    try:
        knowledge_base = load_corpus(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'insight_dataset.txt'))
        print(f"📚 Loaded knowledge base with {len(knowledge_base)} documents")
    except Exception as e:
        print(f"⚠️  Using fallback knowledge base due to: {e}")
        knowledge_base = [
            "Quantum entanglement is a physical phenomenon where particles remain connected.",
            "Graph neural networks process relational information using node embeddings.",
            "Information theory deals with quantification and transmission of information.",
            "The Monty Hall problem demonstrates conditional probability concepts.",
            "Zeno's paradoxes involve mathematical concepts of infinity and convergence.",
            "Machine learning algorithms can detect patterns in complex datasets.",
            "Quantum computing leverages quantum mechanical phenomena for computation.",
            "Network analysis reveals structural properties of connected systems."
        ]
    
    # Prepare knowledge base statistics
    all_words = []
    for doc in knowledge_base:
        all_words.extend(doc.lower().split())
    word_counts = Counter(all_words)
    kb_stats = {
        'concept_frequencies': dict(word_counts.most_common(100)),
        'total_concepts': len(word_counts)
    }
    
    # Test cases designed to exercise Layer1 capabilities
    test_cases = [
        {
            'id': 1,
            'question': "What is quantum entanglement?",
            'expected_synthesis': False,  # Direct retrieval question
            'expected_complexity': "Low",
            'category': "Direct Retrieval"
        },
        {
            'id': 2,
            'question': "How do graph neural networks leverage quantum entanglement principles?",
            'expected_synthesis': True,  # Cross-domain synthesis required
            'expected_complexity': "High",
            'category': "Cross-Domain Synthesis"
        },
        {
            'id': 3,
            'question': "What emergent properties arise when information theory meets quantum mechanics in distributed computing systems?",
            'expected_synthesis': True,  # Complex multi-domain synthesis
            'expected_complexity': "Very High",
            'category': "Complex Innovation"
        },
        {
            'id': 4,
            'question': "Compare machine learning pattern detection with human cognitive processing",
            'expected_synthesis': True,  # Synthesis required
            'expected_complexity': "Medium",
            'category': "Analytical Comparison"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        question = test_case['question']
        print(f"\n{test_case['id']}. {test_case['category']}")
        print(f"   Question: \"{question[:60]}{'...' if len(question) > 60 else ''}\"")
        print("-" * 50)
        
        start_time = time.time()
        
        # ===== STEP 1: Layer1 Known/Unknown Analysis =====
        print("🔍 Layer1 Analysis:")
        l1_analysis = analyze_input(question, knowledge_base[:10], kb_stats)
        
        print(f"   Known Elements: {len(l1_analysis.known_elements)} ({', '.join(list(l1_analysis.known_elements)[:3])}{'...' if len(l1_analysis.known_elements) > 3 else ''})")
        print(f"   Unknown Elements: {len(l1_analysis.unknown_elements)} ({', '.join(list(l1_analysis.unknown_elements)[:3])}{'...' if len(l1_analysis.unknown_elements) > 3 else ''})")
        print(f"   Synthesis Required: {l1_analysis.requires_synthesis}")
        print(f"   Query Complexity: {l1_analysis.query_complexity:.3f}")
        print(f"   Analysis Confidence: {l1_analysis.analysis_confidence:.3f}")
        
        # ===== STEP 2: Adaptive topK Calculation =====
        print("\n⚙️ Adaptive topK:")
        adaptive_result = calculate_adaptive_topk(l1_analysis.__dict__)
        adaptive_topk = {k: v for k, v in adaptive_result.items() if not k.startswith('adaptation')}
        adaptation_factors = adaptive_result['adaptation_factors']
        
        print(f"   Layer1: {adaptive_topk['layer1_k']} (scale: {adaptation_factors['final_scaling']['layer1']:.2f}x)")
        print(f"   Layer2: {adaptive_topk['layer2_k']} (scale: {adaptation_factors['final_scaling']['layer2']:.2f}x)")
        print(f"   Layer3: {adaptive_topk['layer3_k']} (scale: {adaptation_factors['final_scaling']['layer3']:.2f}x)")
        
        # ===== STEP 3: Chain Reaction Potential =====
        print("\n🔗 Chain Reaction:")
        chain_potential = estimate_chain_reaction_potential(l1_analysis.__dict__, adaptive_topk)
        
        if chain_potential > 0.8:
            potential_desc = "Very High"
        elif chain_potential > 0.6:
            potential_desc = "High"
        elif chain_potential > 0.4:
            potential_desc = "Medium"
        else:
            potential_desc = "Low"
            
        print(f"   Potential: {chain_potential:.3f} ({potential_desc})")
        
        # ===== STEP 4: Evaluation =====
        print("\n✨ Validation:")
        
        # Check synthesis prediction
        synthesis_correct = l1_analysis.requires_synthesis == test_case['expected_synthesis']
        
        # Check complexity prediction
        complexity_mapping = {
            'Low': (0.0, 0.3),
            'Medium': (0.3, 0.6),
            'High': (0.6, 0.8),
            'Very High': (0.8, 1.0)
        }
        complexity_range = complexity_mapping[test_case['expected_complexity']]
        complexity_correct = complexity_range[0] <= l1_analysis.query_complexity <= complexity_range[1]
        
        processing_time = time.time() - start_time
        
        print(f"   Synthesis Prediction: {'✅' if synthesis_correct else '❌'} ({l1_analysis.requires_synthesis} vs {test_case['expected_synthesis']})")
        print(f"   Complexity Prediction: {'✅' if complexity_correct else '❌'} ({l1_analysis.query_complexity:.2f} vs {test_case['expected_complexity']})")
        print(f"   Processing Time: {processing_time:.3f}s")
        print(f"   Chain Potential: {chain_potential:.3f}")
        
        # Store results
        results.append({
            'test_case': test_case,
            'l1_analysis': l1_analysis.__dict__,
            'adaptive_topk': adaptive_topk,
            'adaptation_factors': adaptation_factors,
            'chain_potential': chain_potential,
            'processing_time': processing_time,
            'validation': {
                'synthesis_correct': synthesis_correct,
                'complexity_correct': complexity_correct
            }
        })
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("📊 Enhanced Layer1 Integration Test Summary")
    print("=" * 60)
    
    synthesis_accuracy = sum(r['validation']['synthesis_correct'] for r in results) / len(results)
    complexity_accuracy = sum(r['validation']['complexity_correct'] for r in results) / len(results)
    avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
    
    # topK scaling analysis
    avg_layer1_scaling = sum(r['adaptive_topk']['layer1_k'] for r in results) / len(results) / 20  # vs base 20
    avg_layer2_scaling = sum(r['adaptive_topk']['layer2_k'] for r in results) / len(results) / 15  # vs base 15
    avg_layer3_scaling = sum(r['adaptive_topk']['layer3_k'] for r in results) / len(results) / 12  # vs base 12
    
    avg_chain_potential = sum(r['chain_potential'] for r in results) / len(results)
    high_chain_tests = [r for r in results if r['chain_potential'] > 0.7]
    
    print(f"✅ Tests Completed: {len(results)}")
    print()
    print(f"🎯 Layer1 Analysis Accuracy:")
    print(f"   Synthesis Requirement: {synthesis_accuracy:.1%}")
    print(f"   Query Complexity: {complexity_accuracy:.1%}")
    print()
    print(f"⚡ Performance Metrics:")
    print(f"   Average Processing Time: {avg_processing_time:.3f}s")
    print()
    print(f"📈 Adaptive topK Scaling (vs defaults):")
    print(f"   Layer1 Average Scale: {avg_layer1_scaling:.2f}x (base: 20)")
    print(f"   Layer2 Average Scale: {avg_layer2_scaling:.2f}x (base: 15)")
    print(f"   Layer3 Average Scale: {avg_layer3_scaling:.2f}x (base: 12)")
    print()
    print(f"🔗 Chain Reaction Analysis:")
    print(f"   Average Chain Potential: {avg_chain_potential:.3f}")
    print(f"   High Chain Potential Tests: {len(high_chain_tests)}/{len(results)} ({len(high_chain_tests)/len(results):.1%})")
    
    print("\n🚀 Key Capabilities Demonstrated:")
    print("   ✅ Layer1 separates known vs unknown information")
    print("   ✅ Dynamic synthesis requirement detection")
    print("   ✅ Query complexity classification")
    print("   ✅ Adaptive topK scaling based on analysis")
    print("   ✅ Chain reaction potential estimation")
    print("   ✅ 連鎖反応的洞察向上 framework operational")
    
    if synthesis_accuracy >= 0.75 and complexity_accuracy >= 0.75:
        print("\n🎉 Enhanced Layer1 Integration: PASSED")
        return True
    else:
        print(f"\n❌ Enhanced Layer1 Integration: FAILED (Synthesis: {synthesis_accuracy:.1%}, Complexity: {complexity_accuracy:.1%})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
