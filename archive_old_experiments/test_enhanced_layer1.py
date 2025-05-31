#!/usr/bin/env python3
"""
Test Enhanced Layer1 Known/Unknown Information Separation
=========================================================

This script tests the new Layer1 functionality that separates known vs unknown
information from input queries before proceeding to retrieval layers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.layer1_error_monitor import analyze_input, KnownUnknownAnalysis
import json
from collections import Counter

def create_knowledge_base_stats(documents):
    """Create knowledge base statistics from documents"""
    all_words = []
    for doc in documents:
        all_words.extend(doc.lower().split())
    
    word_counts = Counter(all_words)
    return {
        'concept_frequencies': dict(word_counts.most_common(100)),
        'total_concepts': len(word_counts)
    }

def test_layer1_analysis():
    """Test the enhanced Layer1 functionality"""
    
    print('🧠 Testing Enhanced Layer1 Known/Unknown Information Separation')
    print('=' * 80)
    
    # Simulate knowledge base
    knowledge_base = [
        'Quantum mechanics describes behavior at atomic scale with uncertainty principles',
        'Probability theory deals with uncertainty and randomness in mathematical systems',
        'Information theory studies communication data compression and entropy',
        'The Monty Hall problem involves three doors and conditional probability',
        'Conditional probability changes based on new information and evidence',
        'Bayes theorem updates beliefs with evidence using prior and posterior distributions',
        'Philosophy examines fundamental questions about existence reality and knowledge',
        'Zeno paradoxes involve infinite series motion and mathematical convergence',
        'Calculus deals with rates of change limits and infinite processes',
        'The Ship of Theseus examines identity persistence over time and change',
        'Photosynthesis converts light energy to chemical energy in plants',
        'Physics describes natural phenomena through mathematical laws and theories',
        'Mathematics provides tools for understanding patterns and relationships',
        'Logic studies valid reasoning and argument structures',
        'Paradoxes reveal contradictions in seemingly reasonable assumptions'
    ]
    
    # Create knowledge base statistics
    kb_stats = create_knowledge_base_stats(knowledge_base)
    
    print(f'📚 Knowledge Base: {len(knowledge_base)} documents')
    print(f'📊 Vocabulary: {len(kb_stats["concept_frequencies"])} unique concepts')
    print()
    
    # Test questions with varying complexity and synthesis requirements
    test_cases = [
        {
            'question': 'What is quantum mechanics?',
            'type': 'Simple Definition',
            'expected_synthesis': False,
            'expected_complexity': 'Low'
        },
        {
            'question': 'How does probability theory relate to uncertainty?',
            'type': 'Basic Relationship', 
            'expected_synthesis': True,
            'expected_complexity': 'Medium'
        },
        {
            'question': 'How does the Monty Hall problem demonstrate the relationship between probability theory and information theory?',
            'type': 'Cross-Domain Synthesis',
            'expected_synthesis': True, 
            'expected_complexity': 'High'
        },
        {
            'question': 'Compare philosophical implications of Zeno paradox with modern calculus solutions and their impact on understanding infinity',
            'type': 'Complex Multi-Domain Analysis',
            'expected_synthesis': True,
            'expected_complexity': 'Very High'
        },
        {
            'question': 'What color is the sky?',
            'type': 'Out-of-Domain Question',
            'expected_synthesis': False,
            'expected_complexity': 'Low'
        },
        {
            'question': 'If the Ship of Theseus paradox applies to quantum particles undergoing measurement collapse, how does this relate to the observer effect in information theory?',
            'type': 'Hypothetical Synthesis',
            'expected_synthesis': True,
            'expected_complexity': 'Very High'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        
        print(f'\n{i}. {test_case["type"]}')
        print(f'   Question: "{question}"')
        print(f'   Expected Synthesis: {test_case["expected_synthesis"]}')
        print(f'   Expected Complexity: {test_case["expected_complexity"]}')
        print('-' * 70)
        
        # Perform Layer1 analysis
        analysis = analyze_input(question, knowledge_base, kb_stats)
        
        # Display results
        print(f'📊 Analysis Results:')
        print(f'   Known Elements ({len(analysis.known_elements)}): {analysis.known_elements[:3]}{"..." if len(analysis.known_elements) > 3 else ""}')
        print(f'   Unknown Elements ({len(analysis.unknown_elements)}): {analysis.unknown_elements[:3]}{"..." if len(analysis.unknown_elements) > 3 else ""}')
        print(f'   Requires Synthesis: {analysis.requires_synthesis}')
        print(f'   Query Complexity: {analysis.query_complexity:.3f}')
        print(f'   Error Threshold: {analysis.error_threshold:.3f}')
        print(f'   Analysis Confidence: {analysis.analysis_confidence:.3f}')
        
        # Evaluate prediction accuracy
        synthesis_correct = analysis.requires_synthesis == test_case['expected_synthesis']
        
        # Classify complexity
        if analysis.query_complexity < 0.3:
            predicted_complexity = 'Low'
        elif analysis.query_complexity < 0.6:
            predicted_complexity = 'Medium'
        elif analysis.query_complexity < 0.8:
            predicted_complexity = 'High'
        else:
            predicted_complexity = 'Very High'
            
        complexity_correct = predicted_complexity == test_case['expected_complexity']
        
        print(f'\n✨ Evaluation:')
        print(f'   Synthesis Prediction: {"✅" if synthesis_correct else "❌"} ({analysis.requires_synthesis} vs {test_case["expected_synthesis"]})')
        print(f'   Complexity Prediction: {"✅" if complexity_correct else "❌"} ({predicted_complexity} vs {test_case["expected_complexity"]})')
        
        # Store results for summary
        results.append({
            'question_type': test_case['type'],
            'synthesis_correct': synthesis_correct,
            'complexity_correct': complexity_correct,
            'analysis': {
                'known_count': len(analysis.known_elements),
                'unknown_count': len(analysis.unknown_elements),
                'requires_synthesis': analysis.requires_synthesis,
                'query_complexity': analysis.query_complexity,
                'analysis_confidence': analysis.analysis_confidence
            }
        })
    
    # Summary
    print('\n' + '=' * 80)
    print('📈 Layer1 Analysis Summary')
    print('=' * 80)
    
    synthesis_accuracy = sum(r['synthesis_correct'] for r in results) / len(results)
    complexity_accuracy = sum(r['complexity_correct'] for r in results) / len(results)
    avg_confidence = sum(r['analysis']['analysis_confidence'] for r in results) / len(results)
    avg_complexity = sum(r['analysis']['query_complexity'] for r in results) / len(results)
    
    print(f'✅ Synthesis Prediction Accuracy: {synthesis_accuracy:.1%}')
    print(f'✅ Complexity Prediction Accuracy: {complexity_accuracy:.1%}')
    print(f'📊 Average Analysis Confidence: {avg_confidence:.3f}')
    print(f'📊 Average Query Complexity: {avg_complexity:.3f}')
    
    # Detailed breakdown
    print('\n📋 Detailed Results:')
    for i, result in enumerate(results, 1):
        analysis = result['analysis']
        print(f'{i}. {result["question_type"]}: '
              f'Known={analysis["known_count"]}, '
              f'Unknown={analysis["unknown_count"]}, '
              f'Synthesis={"✅" if result["synthesis_correct"] else "❌"}, '
              f'Complexity={"✅" if result["complexity_correct"] else "❌"}')
    
    print('\n🎯 Key Observations:')
    high_synthesis_queries = [r for r in results if r['analysis']['requires_synthesis']]
    print(f'   • {len(high_synthesis_queries)}/{len(results)} queries required synthesis')
    
    high_complexity_queries = [r for r in results if r['analysis']['query_complexity'] > 0.7]
    print(f'   • {len(high_complexity_queries)}/{len(results)} queries had high complexity (>0.7)')
    
    high_confidence_analyses = [r for r in results if r['analysis']['analysis_confidence'] > 0.7]
    print(f'   • {len(high_confidence_analyses)}/{len(results)} analyses had high confidence (>0.7)')
    
    print('\n🚀 Implications for topK Optimization:')
    synthesis_questions = len([r for r in results if r['analysis']['requires_synthesis']])
    print(f'   • {synthesis_questions} questions require synthesis → Higher topK needed')
    print(f'   • Average complexity {avg_complexity:.3f} → Adaptive topK should scale with complexity')
    print(f'   • Analysis confidence {avg_confidence:.3f} → Layer1 provides reliable guidance')
    
    print('\n✅ Enhanced Layer1 Test Complete!')
    return results

if __name__ == '__main__':
    test_layer1_analysis()
