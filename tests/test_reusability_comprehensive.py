"""
Comprehensive Integration Test for Reusable Components
====================================================

Test the integration of all reusable components and create 
a comprehensive test suite that demonstrates the reusability
improvements.
"""

import logging
from typing import Dict, Any
import numpy as np

from insightspike.core.agents.agent_factory import create_maze_agent, AgentConfigBuilder
from insightspike.core.reasoners.standalone_l3 import create_standalone_reasoner
from insightspike.core.interfaces.maze_implementation import (
    MazeEnvironmentAdapter, MazeInsightDetector
)

logger = logging.getLogger(__name__)


def test_reusability_comprehensive():
    """Comprehensive test of all reusable components"""
    
    print("🧪 Running Comprehensive Reusability Integration Test")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Multiple agent configurations
    print("\n1. Testing Multiple Agent Configurations:")
    print("-" * 50)
    
    configs = [
        ("Conservative", {"learning_rate": 0.05, "exploration_rate": 0.2}),
        ("Balanced", {"learning_rate": 0.15, "exploration_rate": 0.4}),
        ("Aggressive", {"learning_rate": 0.3, "exploration_rate": 0.6})
    ]
    
    agent_results = {}
    for name, config_params in configs:
        config = (AgentConfigBuilder()
                  .learning_rate(config_params["learning_rate"])
                  .exploration_params(config_params["exploration_rate"], 0.995, 0.05)
                  .insight_thresholds(-0.3, 1.0)
                  .build())
        
        agent = create_maze_agent(maze_size=8, agent_config=config)
        
        # Train for 3 episodes
        total_reward = 0
        total_insights = 0
        for _ in range(3):
            result = agent.train_episode()
            total_reward += result['reward']
            total_insights += result['insights']
        
        avg_reward = total_reward / 3
        agent_results[name] = {
            'avg_reward': avg_reward,
            'total_insights': total_insights,
            'config': config_params
        }
        
        print(f"   {name}: Avg Reward={avg_reward:.2f}, Insights={total_insights}")
    
    results['agent_configurations'] = agent_results
    
    # Test 2: Standalone reasoner with different document sets
    print("\n2. Testing Standalone Reasoner with Different Contexts:")
    print("-" * 50)
    
    reasoner = create_standalone_reasoner()
    
    document_sets = [
        ("Technical", [
            "The algorithm optimized path finding efficiently",
            "Neural network convergence improved significantly", 
            "Computational complexity was reduced by 40%"
        ]),
        ("Exploration", [
            "New territories were discovered systematically",
            "Exploration strategy yielded unexpected results",
            "Breakthrough insights emerged from careful observation"
        ]),
        ("Performance", [
            "Performance metrics exceeded baseline expectations",
            "Efficiency gains were substantial and measurable",
            "Strategic improvements delivered optimal outcomes"
        ])
    ]
    
    reasoner_results = {}
    for name, docs in document_sets:
        result = reasoner.analyze_documents(docs)
        reasoner_results[name] = {
            'spike_detected': result['spike_detected'],
            'reasoning_quality': result['reasoning_quality'],
            'reward': result['reward'],
            'metrics': result['metrics']
        }
        
        print(f"   {name}: Spike={result['spike_detected']}, "
              f"Quality={result['reasoning_quality']:.3f}")
    
    results['reasoner_analysis'] = reasoner_results
    
    # Test 3: Environment adaptability
    print("\n3. Testing Environment Adaptability:")
    print("-" * 50)
    
    env_results = {}
    for size in [6, 10, 15]:
        env = MazeEnvironmentAdapter(maze_size=size)
        detector = MazeInsightDetector(maze_size=size)
        
        # Quick simulation
        state = env.reset()
        insights_detected = 0
        steps = 0
        
        for _ in range(20):  # Limited steps for testing
            action = np.random.randint(4)
            next_state, reward, done, info = env.step(action)
            
            context = {
                'distance_to_goal': info.get('distance_to_goal', 0),
                'exploration_ratio': info.get('exploration_ratio', 0)
            }
            
            insight = detector.detect_insight(state, action, reward, next_state, context)
            if insight:
                insights_detected += 1
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        env_results[f"{size}x{size}"] = {
            'insights_detected': insights_detected,
            'steps_taken': steps,
            'completion_rate': 1.0 if done else 0.0
        }
        
        print(f"   {size}x{size} maze: {insights_detected} insights in {steps} steps")
    
    results['environment_adaptability'] = env_results
    
    # Test 4: Memory and learning persistence
    print("\n4. Testing Memory and Learning Persistence:")
    print("-" * 50)
    
    agent = create_maze_agent(maze_size=8)
    
    # Train and track learning progression
    performance_progression = []
    insight_progression = []
    
    for episode in range(5):
        result = agent.train_episode()
        performance_progression.append(result['reward'])
        insight_progression.append(result['insights'])
    
    # Calculate learning metrics
    initial_performance = np.mean(performance_progression[:2])
    final_performance = np.mean(performance_progression[-2:])
    learning_improvement = final_performance - initial_performance
    
    memory_stats = agent.memory.get_memory_stats()
    
    results['learning_persistence'] = {
        'initial_performance': initial_performance,
        'final_performance': final_performance,
        'learning_improvement': learning_improvement,
        'total_insights': sum(insight_progression),
        'memory_utilization': memory_stats['memory_utilization']
    }
    
    print(f"   Learning improvement: {learning_improvement:.2f}")
    print(f"   Total insights: {sum(insight_progression)}")
    print(f"   Memory utilization: {memory_stats['memory_utilization']:.3f}")
    
    # Test 5: Component interoperability
    print("\n5. Testing Component Interoperability:")
    print("-" * 50)
    
    # Create components independently and verify they work together
    env = MazeEnvironmentAdapter(maze_size=6)
    reasoner = create_standalone_reasoner()
    
    # Simulate agent-environment interaction
    state = env.reset()
    interaction_log = []
    
    for step in range(10):
        action = np.random.randint(4)
        next_state, reward, done, info = env.step(action)
        
        # Document the interaction
        interaction_text = f"Step {step}: action={action}, reward={reward:.3f}, done={done}"
        interaction_log.append(interaction_text)
        
        state = next_state
        if done:
            break
    
    # Analyze interactions with reasoner
    reasoner_analysis = reasoner.analyze_documents(interaction_log)
    
    results['component_interoperability'] = {
        'interactions_logged': len(interaction_log),
        'reasoner_spike': reasoner_analysis['spike_detected'],
        'analysis_quality': reasoner_analysis['reasoning_quality']
    }
    
    print(f"   Logged {len(interaction_log)} interactions")
    print(f"   Reasoner detected spike: {reasoner_analysis['spike_detected']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Comprehensive Test Summary:")
    print("=" * 70)
    
    # Agent configuration diversity
    config_diversity = len([c for c in agent_results.values() if c['total_insights'] > 0])
    print(f"✅ Agent Configurations: {len(agent_results)} tested, {config_diversity} with insights")
    
    # Reasoner analysis capability
    spike_detection_rate = len([r for r in reasoner_results.values() if r['spike_detected']]) / len(reasoner_results)
    print(f"✅ Reasoner Analysis: {len(reasoner_results)} contexts, {spike_detection_rate:.1%} spike rate")
    
    # Environment adaptability
    env_sizes = list(env_results.keys())
    print(f"✅ Environment Adaptability: {len(env_sizes)} sizes tested")
    
    # Learning progression
    learning_effective = learning_improvement > 0
    print(f"✅ Learning Persistence: {'Effective' if learning_effective else 'Needs tuning'}")
    
    # Component integration
    integration_successful = results['component_interoperability']['interactions_logged'] > 0
    print(f"✅ Component Interoperability: {'Successful' if integration_successful else 'Failed'}")
    
    print(f"\n🎯 Reusability Score: {calculate_reusability_score(results):.1f}/100")
    
    return results


def calculate_reusability_score(results: Dict[str, Any]) -> float:
    """Calculate overall reusability score"""
    
    score = 0.0
    
    # Agent configuration flexibility (25 points)
    agent_configs = len(results['agent_configurations'])
    score += min(25, agent_configs * 8)  # Up to 25 points for 3+ configs
    
    # Reasoner versatility (25 points)
    reasoner_contexts = len(results['reasoner_analysis'])
    score += min(25, reasoner_contexts * 8)  # Up to 25 points for 3+ contexts
    
    # Environment adaptability (25 points)
    env_sizes = len(results['environment_adaptability'])
    score += min(25, env_sizes * 8)  # Up to 25 points for 3+ sizes
    
    # Learning and memory (15 points)
    learning_improvement = results['learning_persistence']['learning_improvement']
    if learning_improvement > 0:
        score += 15
    elif learning_improvement > -10:
        score += 10
    else:
        score += 5
    
    # Component integration (10 points)
    if results['component_interoperability']['interactions_logged'] > 0:
        score += 10
    
    return score


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run comprehensive test
    results = test_reusability_comprehensive()
    
    print(f"\n✨ Comprehensive reusability test completed!")
    print(f"🔄 Results available for further analysis")


# Export test function
__all__ = ["test_reusability_comprehensive", "calculate_reusability_score"]
