#!/usr/bin/env python3
"""
Test pipeline with various configuration patterns
"""

import os
import sys
import logging
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insightspike.implementations.agents.main_agent import MainAgent

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "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"

def create_base_config():
    """Base configuration with Anthropic"""
    return {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 500
        },
        "memory": {
            "max_episodes": 1000,
            "consolidation_threshold": 50
        }
    }

def test_config_variation(name, config_override, knowledge_base, test_questions):
    """Test a specific configuration variation"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Create config
    config = create_base_config()
    config.update(config_override)
    
    # Print key settings
    graph_config = config.get('graph', {})
    print(f"Message Passing: {graph_config.get('enable_message_passing', False)}")
    if graph_config.get('enable_message_passing'):
        mp_config = graph_config.get('message_passing', {})
        print(f"  - Alpha: {mp_config.get('alpha', 0.3)}")
        print(f"  - Iterations: {mp_config.get('iterations', 3)}")
    print(f"Graph Search: {graph_config.get('enable_graph_search', False)}")
    print(f"GNN Enabled: {graph_config.get('use_gnn', False)}")
    
    try:
        # Create agent
        agent = MainAgent(config)
        print("✓ Agent created")
        
        # Add knowledge
        for k in knowledge_base:
            agent.add_knowledge(k)
        print(f"✓ Added {len(knowledge_base)} knowledge items")
        
        # Test questions
        results = []
        for question in test_questions:
            print(f"\nQ: {question}")
            
            result = agent.process_question(question)
            
            # Extract info
            response = getattr(result, 'response', 'No response')
            spike = getattr(result, 'spike_detected', False)
            graph_analysis = getattr(result, 'graph_analysis', {})
            
            # Check graph info
            graph_info = {}
            if graph_analysis:
                graph = graph_analysis.get('graph')
                if graph:
                    graph_info['nodes'] = getattr(graph, 'num_nodes', 0)
                    if hasattr(graph, 'edge_index'):
                        graph_info['edges'] = graph.edge_index.shape[1] if hasattr(graph.edge_index, 'shape') else 0
                    
                    # Check for new edges from message passing
                    if hasattr(graph, 'edge_info') and graph.edge_info:
                        new_edges = sum(1 for e in graph.edge_info if e.get('type') == 'new')
                        graph_info['new_edges'] = new_edges
            
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            print(f"Spike: {spike}, Graph: {graph_info}")
            
            results.append({
                'question': question,
                'spike': spike,
                'graph_info': graph_info,
                'response_len': len(response)
            })
        
        return {
            'success': True,
            'results': results
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    print("\n=== CONFIG VARIATIONS TEST ===")
    print(f"Timestamp: {datetime.now()}")
    
    # Knowledge base
    knowledge_base = [
        "Apples are fruits that grow on trees.",
        "Apples can be red, green, or yellow.",
        "Apple Inc. is a technology company.",
        "The iPhone is Apple's flagship product.",
        "Photosynthesis converts sunlight to energy.",
        "Plants use chlorophyll for photosynthesis.",
        "Neural networks mimic brain neurons.",
        "Deep learning uses multiple neural layers.",
        "Quantum computing uses qubits.",
        "Qubits can exist in superposition."
    ]
    
    test_questions = [
        "What is an apple?",
        "How do plants and technology relate?",
        "What connects neural networks and quantum computing?"
    ]
    
    # Configuration variations
    config_variations = [
        # 1. Baseline - no special features
        {
            'name': 'Baseline (No MP, No Graph Search)',
            'config': {
                'graph': {
                    'enable_message_passing': False,
                    'enable_graph_search': False,
                    'use_gnn': False,
                    'similarity_threshold': 0.6
                }
            }
        },
        
        # 2. Message Passing only
        {
            'name': 'Message Passing Only',
            'config': {
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.3,
                        'iterations': 2
                    },
                    'edge_reevaluation': {
                        'new_edge_threshold': 0.8
                    },
                    'enable_graph_search': False,
                    'use_gnn': False
                }
            }
        },
        
        # 3. Aggressive Message Passing
        {
            'name': 'Aggressive Message Passing',
            'config': {
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.7,  # High query influence
                        'iterations': 5  # More iterations
                    },
                    'edge_reevaluation': {
                        'new_edge_threshold': 0.6,  # Lower threshold
                        'max_new_edges_per_node': 10
                    },
                    'enable_graph_search': False
                }
            }
        },
        
        # 4. Message Passing + Graph Search
        {
            'name': 'Message Passing + Graph Search',
            'config': {
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.4,
                        'iterations': 3
                    },
                    'enable_graph_search': True,
                    'search_depth': 2,
                    'use_gnn': False
                }
            }
        },
        
        # 5. Everything enabled
        {
            'name': 'All Features Enabled',
            'config': {
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.5,
                        'iterations': 3,
                        'aggregation': 'weighted_mean'
                    },
                    'edge_reevaluation': {
                        'new_edge_threshold': 0.7,
                        'similarity_threshold': 0.6
                    },
                    'enable_graph_search': True,
                    'use_gnn': True,
                    'gnn_hidden_dim': 128
                }
            }
        },
        
        # 6. Low thresholds for spike detection
        {
            'name': 'Sensitive Spike Detection',
            'config': {
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.4,
                        'iterations': 2
                    },
                    'spike_ged_threshold': 0.1,  # Very sensitive
                    'spike_ig_threshold': 0.9,   # Very sensitive
                    'conflict_threshold': 0.3
                }
            }
        }
    ]
    
    # Run all variations
    all_results = {}
    
    for variation in config_variations:
        result = test_config_variation(
            variation['name'],
            variation['config'],
            knowledge_base,
            test_questions
        )
        all_results[variation['name']] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in all_results.values() if r['success'])
    print(f"\nTotal variations tested: {len(config_variations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(config_variations) - successful}")
    
    # Spike detection comparison
    print("\nSpike Detection Results:")
    for name, result in all_results.items():
        if result['success']:
            spikes = sum(1 for r in result['results'] if r['spike'])
            print(f"  {name}: {spikes}/{len(test_questions)} spikes")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"config_variations_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'variations': config_variations,
            'results': all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()