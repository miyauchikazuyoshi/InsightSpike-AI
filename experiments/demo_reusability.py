#!/usr/bin/env python3
"""
InsightSpike Reusability Demo
============================

Demonstration of the reusable InsightSpike components across different scenarios.
"""

import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_standalone_reasoner():
    """Demonstrate standalone L3 Graph Reasoner"""
    print("🧠 Standalone L3 Graph Reasoner Demo")
    print("=" * 50)
    
    from insightspike.core.reasoners.standalone_l3 import create_standalone_reasoner
    
    # Create reasoner with custom config
    config = {
        'spike_threshold': 0.2,
        'ged_threshold': 0.3,
        'ig_threshold': 0.15,
        'embedding_dim': 64
    }
    
    reasoner = create_standalone_reasoner(config)
    
    # Analyze different document sets
    scenarios = [
        {
            'name': 'Research Papers',
            'documents': [
                "Machine learning algorithms for pattern recognition",
                "Deep neural networks in computer vision applications", 
                "Reinforcement learning in robotics and control systems",
                "Natural language processing with transformer architectures"
            ]
        },
        {
            'name': 'Business Strategy',
            'documents': [
                "Market analysis reveals growing demand for AI solutions",
                "Competitive advantage through technology innovation",
                "Strategic partnerships drive business expansion",
                "Digital transformation accelerates organizational growth"
            ]
        },
        {
            'name': 'Problem Solving',
            'documents': [
                "Initial exploration of the problem space",
                "Breakthrough insight: novel approach discovered",
                "Implementation strategy refined and optimized",
                "Final solution validates theoretical predictions"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\nAnalyzing: {scenario['name']}")
        print("-" * 30)
        
        result = reasoner.analyze_documents(scenario['documents'])
        
        print(f"📊 Analysis Results:")
        print(f"   Documents: {result['document_count']}")
        print(f"   Spike Detected: {'🔥' if result['spike_detected'] else '📈'} {result['spike_detected']}")
        print(f"   Reasoning Quality: {result['reasoning_quality']:.3f}")
        print(f"   Reward Signal: {result['reward']:.3f}")
        print(f"   ΔGED: {result['metrics']['delta_ged']:.3f}")
        print(f"   ΔIG: {result['metrics']['delta_ig']:.3f}")
    
    # Get overall summary
    summary = reasoner.get_analysis_summary()
    print(f"\n🎯 Overall Summary:")
    print(f"   Total Analyses: {summary['total_analyses']}")
    print(f"   Spike Rate: {summary['spike_rate']:.1%}")
    print(f"   Average Quality: {summary['avg_quality']:.3f}")
    
    return True


def demo_maze_agent_variants():
    """Demonstrate different maze agent configurations"""
    print("\n🎮 Maze Agent Variants Demo")
    print("=" * 50)
    
    from insightspike.core.agents.agent_factory import (
        create_maze_agent, create_configured_maze_agent, AgentConfigBuilder
    )
    
    # Test different configurations
    configs = [
        {
            'name': 'Cautious Explorer',
            'params': {
                'learning_rate': 0.05,
                'exploration_rate': 0.2,
                'dged_threshold': -0.5,
                'dig_threshold': 2.0
            }
        },
        {
            'name': 'Aggressive Learner', 
            'params': {
                'learning_rate': 0.3,
                'exploration_rate': 0.6,
                'dged_threshold': -0.1,
                'dig_threshold': 0.5
            }
        },
        {
            'name': 'Balanced Agent',
            'params': {
                'learning_rate': 0.15,
                'exploration_rate': 0.4,
                'dged_threshold': -0.3,
                'dig_threshold': 1.0
            }
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 30)
        
        # Create agent with specific configuration
        agent_config = (AgentConfigBuilder()
                       .learning_rate(config['params']['learning_rate'])
                       .exploration_params(config['params']['exploration_rate'], 0.995, 0.05)
                       .insight_thresholds(config['params']['dged_threshold'], 
                                         config['params']['dig_threshold'])
                       .build())
        
        agent = create_maze_agent(maze_size=8, agent_config=agent_config)
        
        # Train for a few episodes
        episode_results = []
        total_insights = 0
        
        for episode in range(5):
            result = agent.train_episode()
            episode_results.append(result)
            total_insights += result['insights']
            
            if result['insights'] > 0:
                print(f"   Episode {episode+1}: 🧠 {result['insights']} insights! Reward: {result['reward']:.1f}")
        
        # Calculate statistics
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_steps = np.mean([r['steps'] for r in episode_results])
        
        config_result = {
            'name': config['name'],
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'total_insights': total_insights,
            'insight_rate': total_insights / 5
        }
        
        results.append(config_result)
        
        print(f"   📊 Performance:")
        print(f"      Average Reward: {avg_reward:.2f}")
        print(f"      Average Steps: {avg_steps:.1f}")
        print(f"      Total Insights: {total_insights}")
        print(f"      Insight Rate: {total_insights/5:.1f} per episode")
    
    # Compare results
    print(f"\n🏆 Configuration Comparison:")
    print("-" * 50)
    for result in sorted(results, key=lambda x: x['avg_reward'], reverse=True):
        print(f"{result['name']:20} | Reward: {result['avg_reward']:6.2f} | "
              f"Steps: {result['avg_steps']:5.1f} | Insights: {result['total_insights']:2d}")
    
    return True


def demo_cross_domain_adaptability():
    """Demonstrate adaptability across different domains"""
    print("\n🌐 Cross-Domain Adaptability Demo")
    print("=" * 50)
    
    # This demonstrates how the generic interfaces enable easy adaptation
    from insightspike.core.interfaces.generic_interfaces import TaskType
    from insightspike.core.agents.agent_factory import InsightSpikeAgentFactory
    
    # Show different task types and their default configurations
    task_types = [TaskType.NAVIGATION, TaskType.OPTIMIZATION, TaskType.GAME_PLAYING]
    
    print("🎯 Default Configurations by Task Type:")
    print("-" * 40)
    
    for task_type in task_types:
        config = InsightSpikeAgentFactory.get_default_config(task_type)
        print(f"\n{task_type.value.upper()}:")
        print(f"   Learning Rate: {config['learning_rate']}")
        print(f"   Exploration Rate: {config['exploration_rate']}")
        print(f"   ΔGED Threshold: {config['dged_threshold']}")
        print(f"   ΔIG Threshold: {config['dig_threshold']}")
    
    print(f"\n💡 Key Benefits of Generic Design:")
    print("   ✅ Environment-agnostic insight detection")
    print("   ✅ Configurable thresholds per domain")
    print("   ✅ Reusable memory and reasoning components")
    print("   ✅ Standardized interfaces for easy integration")
    print("   ✅ Standalone components for modular use")
    
    return True


def demo_component_modularity():
    """Demonstrate component modularity and standalone usage"""
    print("\n🔧 Component Modularity Demo")
    print("=" * 50)
    
    # Show how components can be used independently
    from insightspike.core.interfaces.maze_implementation import MazeInsightDetector
    from insightspike.core.reasoners.standalone_l3 import analyze_documents_simple
    
    print("🧩 Using Components Independently:")
    print("-" * 40)
    
    # 1. Standalone insight detector
    print("\n1. Standalone Insight Detector:")
    detector = MazeInsightDetector(maze_size=10)
    print("   ✅ Created maze insight detector independently")
    print("   ✅ Can be integrated into any maze-like environment")
    
    # 2. Simple document analysis
    print("\n2. Simple Document Analysis:")
    docs = [
        "Problem identification phase", 
        "Solution breakthrough achieved",
        "Implementation and validation"
    ]
    analysis = analyze_documents_simple(docs)
    print(f"   ✅ Analyzed {len(docs)} documents")
    print(f"   ✅ Spike detected: {analysis['spike_detected']}")
    
    # 3. Component composition
    print("\n3. Easy Component Composition:")
    print("   ✅ Mix and match components for different needs")
    print("   ✅ Replace individual components without affecting others")
    print("   ✅ Add new environments with minimal code changes")
    
    return True


def main():
    """Run all demonstrations"""
    print("🚀 InsightSpike Reusability Demonstration")
    print("=" * 60)
    print("Showcasing the modular, reusable architecture improvements")
    print("=" * 60)
    
    demos = [
        ("Standalone Reasoner", demo_standalone_reasoner),
        ("Maze Agent Variants", demo_maze_agent_variants),
        ("Cross-Domain Adaptability", demo_cross_domain_adaptability),
        ("Component Modularity", demo_component_modularity)
    ]
    
    success_count = 0
    
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                success_count += 1
                print(f"\n✅ {demo_name} completed successfully!")
        except Exception as e:
            print(f"\n❌ {demo_name} failed: {e}")
            logger.exception(f"Demo {demo_name} failed")
    
    print("\n" + "=" * 60)
    print(f"🎉 Demonstration Results: {success_count}/{len(demos)} demos successful")
    
    if success_count == len(demos):
        print("\n🌟 All demos completed successfully!")
        print("🎯 InsightSpike components are now fully modular and reusable!")
        print("\n📋 Key Achievements:")
        print("   ✨ Generic interfaces for any environment")
        print("   🔄 Reusable agents across domains")
        print("   🧠 Standalone reasoning components")
        print("   ⚙️  Configurable insight detection")
        print("   🏗️  Modular architecture design")
    
    return success_count == len(demos)


if __name__ == "__main__":
    main()
