"""
Reusability Examples and Test Cases
===================================

Examples showing how to use the reusable InsightSpike components
in different contexts and environments.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from insightspike.core.agents.agent_factory import create_maze_agent, AgentConfigBuilder
from insightspike.core.interfaces.generic_interfaces import (
    EnvironmentInterface,
    EnvironmentState,
    InsightDetectorInterface,
    RewardNormalizer,
    StateEncoder,
    TaskType,
)
from insightspike.core.interfaces.maze_implementation import MazeEnvironmentAdapter
from insightspike.core.reasoners.standalone_l3 import create_standalone_reasoner

logger = logging.getLogger(__name__)


def example_1_basic_maze_agent():
    """Example 1: Basic maze agent with default settings"""
    print("=== Example 1: Basic Maze Agent ===")

    # Create agent with default settings
    agent = create_maze_agent(maze_size=8)

    # Train for a few episodes
    for episode in range(5):
        result = agent.train_episode()
        print(
            f"Episode {episode + 1}: Reward={result['reward']:.2f}, "
            f"Steps={result['steps']}, Insights={result['insights']}"
        )

    # Get performance summary
    summary = agent.get_performance_summary()
    print(
        f"Summary: {summary['total_insights']} insights in {summary['total_episodes']} episodes"
    )
    print()


def example_2_custom_configured_agent():
    """Example 2: Agent with custom configuration"""
    print("=== Example 2: Custom Configured Agent ===")

    # Build custom configuration
    config = (
        AgentConfigBuilder()
        .learning_rate(0.2)
        .exploration_params(0.5, 0.99, 0.02)
        .insight_thresholds(-0.2, 0.8)
        .build()
    )

    # Create agent with custom config
    agent = create_maze_agent(maze_size=10, agent_config=config)

    # Train and observe behavior
    total_insights = 0
    for episode in range(10):
        result = agent.train_episode()
        total_insights += result["insights"]
        if result["insights"] > 0:
            print(f"Episode {episode + 1}: 🧠 {result['insights']} insights detected!")

    print(f"Total insights: {total_insights}")
    print()


def example_3_standalone_graph_reasoner():
    """Example 3: Using L3 Graph Reasoner independently"""
    print("=== Example 3: Standalone Graph Reasoner ===")

    # Create standalone reasoner
    reasoner = create_standalone_reasoner()

    # Example documents
    documents = [
        "The agent explored the maze systematically",
        "A new path was discovered leading to the goal",
        "The exploration strategy proved highly effective",
        "Unexpected obstacles required adaptive behavior",
    ]

    # Analyze documents
    result = reasoner.analyze_documents(documents)

    print(f"Analysis complete:")
    print(f"  Spike detected: {result['spike_detected']}")
    print(f"  Reasoning quality: {result['reasoning_quality']:.3f}")
    print(f"  Reward signal: {result['reward']:.3f}")
    print(
        f"  Metrics: ΔGED={result['metrics']['delta_ged']:.3f}, "
        f"ΔIG={result['metrics']['delta_ig']:.3f}"
    )

    # Analyze more documents to see evolution
    more_docs = [
        "The agent achieved breakthrough performance",
        "Multiple insights led to strategy optimization",
        "Goal-directed behavior emerged naturally",
    ]

    result2 = reasoner.analyze_documents(more_docs)
    print(
        f"Second analysis - Spike: {result2['spike_detected']}, "
        f"Quality: {result2['reasoning_quality']:.3f}"
    )

    # Get summary
    summary = reasoner.get_analysis_summary()
    print(f"Reasoner summary: {summary}")
    print()


def example_4_different_maze_environments():
    """Example 4: Same agent architecture, different maze sizes"""
    print("=== Example 4: Different Maze Environments ===")

    for size in [6, 10, 15]:
        print(f"Testing maze size {size}x{size}:")

        agent = create_maze_agent(maze_size=size)

        # Quick training
        insights = 0
        for episode in range(3):
            result = agent.train_episode()
            insights += result["insights"]

        summary = agent.get_performance_summary()
        print(
            f"  Insights: {insights}, Avg reward: {summary['avg_episode_reward']:.2f}"
        )

    print()


def example_5_insight_pattern_analysis():
    """Example 5: Analyzing insight patterns across episodes"""
    print("=== Example 5: Insight Pattern Analysis ===")

    agent = create_maze_agent(maze_size=12)
    all_insights = []

    # Train for more episodes to gather insights
    for episode in range(15):
        result = agent.train_episode()
        if result["insights"] > 0:
            # Get actual insight objects from agent
            recent_insights = agent.insight_moments[-result["insights"] :]
            all_insights.extend(recent_insights)
            print(f"Episode {episode + 1}: {len(recent_insights)} insights")

    # Analyze patterns using the agent's reasoner
    if all_insights:
        pattern_analysis = agent.reasoner.analyze_insight_pattern(all_insights)
        print(f"Pattern analysis: {pattern_analysis}")
    else:
        print("No insights detected for pattern analysis")

    print()


class SimpleCustomEnvironment(EnvironmentInterface):
    """Simple custom environment for testing reusability"""

    def __init__(self):
        self.state = 0
        self.target = 10
        self.step_count = 0
        self.episode_count = 0

    def get_state(self) -> EnvironmentState:
        return EnvironmentState(
            state_data={"value": self.state, "target": self.target},
            environment_type="simple_number_game",
            task_type=TaskType.OPTIMIZATION,
            step_count=self.step_count,
            episode_count=self.episode_count,
        )

    def get_action_space(self):
        from insightspike.core.interfaces.generic_interfaces import ActionSpace

        return ActionSpace(
            action_type="discrete",
            action_dim=3,
            discrete_actions=["decrease", "stay", "increase"],
        )

    def step(self, action):
        self.step_count += 1

        # Actions: 0=decrease, 1=stay, 2=increase
        if action == 0:
            self.state -= 1
        elif action == 2:
            self.state += 1
        # action == 1 means stay

        # Calculate reward based on distance to target
        distance = abs(self.state - self.target)
        reward = -distance  # Closer is better

        # Episode ends when target is reached or after 20 steps
        done = (self.state == self.target) or (self.step_count >= 20)

        info = {"distance": distance}
        return self.get_state(), reward, done, info

    def reset(self):
        self.state = 0
        self.step_count = 0
        self.episode_count += 1
        return self.get_state()

    def get_task_type(self):
        return TaskType.OPTIMIZATION


def example_6_custom_environment():
    """Example 6: Using generic agent with custom environment"""
    print("=== Example 6: Custom Environment ===")

    # This example shows how you would create a custom environment
    # and use it with the generic framework (implementation would require
    # custom StateEncoder, RewardNormalizer, and InsightDetector)

    custom_env = SimpleCustomEnvironment()
    print(f"Created custom environment: {custom_env.get_task_type()}")

    # Test basic environment functionality
    state = custom_env.reset()
    print(f"Initial state: {state.state_data}")

    state, reward, done, info = custom_env.step(2)  # Increase
    print(f"After step: state={state.state_data}, reward={reward}, done={done}")
    print()


def run_all_examples():
    """Run all examples to demonstrate reusability"""
    print("Running InsightSpike Reusability Examples\n")
    print("=" * 50)

    try:
        example_1_basic_maze_agent()
        example_2_custom_configured_agent()
        example_3_standalone_graph_reasoner()
        example_4_different_maze_environments()
        example_5_insight_pattern_analysis()
        example_6_custom_environment()

        print("✅ All examples completed successfully!")
        print("\nThese examples demonstrate:")
        print("- Basic agent creation and training")
        print("- Custom configuration")
        print("- Standalone reasoner usage")
        print("- Environment adaptability")
        print("- Insight pattern analysis")
        print("- Custom environment support")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        logger.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run examples
    run_all_examples()


# Export main functions
__all__ = [
    "run_all_examples",
    "example_1_basic_maze_agent",
    "example_2_custom_configured_agent",
    "example_3_standalone_graph_reasoner",
    "example_4_different_maze_environments",
    "example_5_insight_pattern_analysis",
    "example_6_custom_environment",
]
