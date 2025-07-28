#!/usr/bin/env python3
"""
Pipeline test with message passing and query integration enabled.
Tests the complete flow from question to response with new features.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config():
    """Create configuration with message passing enabled."""
    return {
        "model": {
            "name": "InsightSpike-MessagePassing-Test",
            "version": "1.0.0"
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "cache_enabled": True
        },
        "processing": {
            "enable_insight_search": True,
            "max_insights_per_query": 5
        },
        "graph": {
            # Enable message passing
            "enable_message_passing": True,
            "message_passing": {
                "alpha": 0.3,
                "iterations": 3,
                "aggregation": "weighted_mean",
                "self_loop_weight": 0.5,
                "decay_factor": 0.8
            },
            # Edge re-evaluation settings
            "edge_reevaluation": {
                "similarity_threshold": 0.7,
                "new_edge_threshold": 0.8,
                "max_new_edges_per_node": 5,
                "edge_decay_factor": 0.9
            },
            # Standard graph settings
            "similarity_threshold": 0.6,
            "spike_ged_threshold": 0.3,
            "spike_ig_threshold": 0.7,
            "conflict_threshold": 0.5,
            "enable_graph_search": True,
            "use_gnn": False
        },
        "algorithms": {
            "use_advanced_ged": True,
            "use_advanced_ig": True,
            "advanced_ged_algorithm": "hungarian",
            "advanced_ig_algorithm": "normalized"
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful AI assistant that provides clear and concise answers."
        },
        "memory": {
            "max_episodes": 5000,
            "consolidation_threshold": 100
        },
        "logging": {
            "level": "INFO"
        }
    }

def run_pipeline_test():
    """Run comprehensive pipeline test."""
    print("\n" + "="*80)
    print("INSIGHTSPIKE PIPELINE TEST - MESSAGE PASSING ENABLED")
    print("="*80 + "\n")
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Create agent with message passing config
    config = create_test_config()
    print("Initializing agent with message passing enabled...")
    
    try:
        # Debug: Print config to verify
        print(f"Graph config: enable_message_passing = {config['graph'].get('enable_message_passing', False)}")
        
        agent = MainAgent(config)
        print("✓ Agent initialized successfully")
        
        # Verify message passing is enabled
        if hasattr(agent.l3_graph, 'message_passing_enabled'):
            print(f"✓ Message passing enabled: {agent.l3_graph.message_passing_enabled}")
        else:
            print("✗ Message passing status unknown")
            
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Add some knowledge to work with
    print("\nAdding initial knowledge...")
    knowledge_base = [
        "Apples are fruits that grow on trees.",
        "Apples can be red, green, or yellow in color.",
        "Apple Inc. is a technology company founded by Steve Jobs.",
        "The iPhone is Apple's most popular product.",
        "Photosynthesis is how plants convert sunlight into energy.",
        "Trees produce oxygen through photosynthesis.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Quantum computing uses quantum bits or qubits.",
        "Superposition allows qubits to be in multiple states simultaneously."
    ]
    
    for knowledge in knowledge_base:
        agent.add_knowledge(knowledge)
    print(f"✓ Added {len(knowledge_base)} knowledge items")
    
    # Test questions
    test_questions = [
        {
            "question": "What is an apple?",
            "context": "Expecting disambiguation between fruit and company"
        },
        {
            "question": "How do apples relate to photosynthesis?",
            "context": "Testing connection discovery through message passing"
        },
        {
            "question": "What's the connection between neural networks and quantum computing?",
            "context": "Testing distant concept linking"
        }
    ]
    
    print("\n" + "-"*80)
    print("RUNNING TESTS")
    print("-"*80)
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\nTest {i}: {test['question']}")
        print(f"Context: {test['context']}")
        print("-"*40)
        
        try:
            # Process question
            result = agent.process_question(test['question'])
            
            # Extract information
            if hasattr(result, 'has_spike'):
                has_spike = result.has_spike
                response = result.response
                graph_analysis = result.graph_analysis
            else:
                has_spike = result.get('has_spike', False)
                response = result.get('response', '')
                graph_analysis = result.get('graph_analysis', {})
            
            # Display results
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            print(f"Spike Detected: {'Yes' if has_spike else 'No'}")
            
            # Check for message passing indicators
            if graph_analysis:
                metrics = graph_analysis.get('metrics', {})
                print(f"Graph Metrics - GED: {metrics.get('delta_ged', 'N/A')}, IG: {metrics.get('delta_ig', 'N/A')}")
                
                # Check if edge re-evaluation occurred
                graph = graph_analysis.get('graph')
                if graph and hasattr(graph, 'edge_info') and graph.edge_info:
                    new_edges = sum(1 for e in graph.edge_info if e.get('type') == 'new')
                    print(f"New edges discovered: {new_edges}")
            
            results.append({
                "test": i,
                "question": test['question'],
                "has_spike": has_spike,
                "response_length": len(response),
                "success": True
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
            logger.error(f"Test {i} failed", exc_info=True)
            results.append({
                "test": i,
                "question": test['question'],
                "error": str(e),
                "success": False
            })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal tests: {len(test_questions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(test_questions) - successful}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"pipeline_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": config,
            "results": results,
            "summary": {
                "total": len(test_questions),
                "successful": successful,
                "failed": len(test_questions) - successful
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Memory stats
    stats = agent.l2_memory.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"- Total episodes: {stats.get('total_episodes', 0)}")
    print(f"- Vector dimension: {stats.get('vector_dimension', 0)}")

if __name__ == "__main__":
    run_pipeline_test()