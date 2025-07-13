"""
Test Generic InsightSpike Components
===================================

Test the reusable components to ensure they work correctly.
"""

import logging
import sys
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all new components can be imported"""
    print("Testing imports...")

    try:
        # Test generic interfaces
        from insightspike.core.interfaces.generic_interfaces import (
            EnvironmentInterface,
            EnvironmentState,
            InsightMoment,
            TaskType,
        )

        print("✅ Generic interfaces imported successfully")

        # Test maze implementation
        from insightspike.core.interfaces.maze_implementation import (
            MazeEnvironmentAdapter,
            MazeInsightDetector,
        )

        print("✅ Maze implementation imported successfully")

        # Test generic agent
        from insightspike.core.agents.generic_agent import GenericInsightSpikeAgent

        print("✅ Generic agent imported successfully")

        # Test agent factory
        from insightspike.core.agents.agent_factory import (
            AgentConfigBuilder,
            create_maze_agent,
        )

        print("✅ Agent factory imported successfully")

        # Test standalone reasoner
        from insightspike.core.reasoners.standalone_l3 import (
            analyze_documents_simple,
            create_standalone_reasoner,
        )

        print("✅ Standalone reasoner imported successfully")

        assert True  # Test passed

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        assert False, f"Import test failed: {e}"
        return False


def test_standalone_reasoner():
    """Test standalone L3 Graph Reasoner"""
    print("\nTesting standalone reasoner...")

    try:
        from insightspike.core.reasoners.standalone_l3 import create_standalone_reasoner

        # Create reasoner
        reasoner = create_standalone_reasoner()
        print("✅ Reasoner created successfully")

        # Test document analysis
        documents = [
            "The agent explored the maze systematically",
            "A breakthrough strategy was discovered",
            "Performance improved significantly",
        ]

        result = reasoner.analyze_documents(documents)
        print(f"✅ Document analysis completed")
        print(f"   Spike detected: {result['spike_detected']}")
        print(f"   Reasoning quality: {result['reasoning_quality']:.3f}")

        # Get summary
        summary = reasoner.get_analysis_summary()
        print(f"✅ Analysis summary: {summary['total_analyses']} analyses")

        assert True  # Test passed

    except Exception as e:
        print(f"❌ Standalone reasoner test failed: {e}")
        traceback.print_exc()
        assert False, f"Standalone reasoner test failed: {e}"


def test_maze_agent_creation():
    """Test maze agent creation and basic functionality"""
    print("\nTesting maze agent creation...")

    try:
        from insightspike.core.agents.agent_factory import (
            AgentConfigBuilder,
            create_maze_agent,
        )

        # Test basic agent creation
        agent = create_maze_agent(maze_size=6)
        print("✅ Basic maze agent created successfully")

        # Test custom configuration
        config = (
            AgentConfigBuilder()
            .learning_rate(0.2)
            .exploration_params(0.4, 0.99, 0.05)
            .insight_thresholds(-0.3, 1.0)
            .build()
        )

        custom_agent = create_maze_agent(maze_size=8, agent_config=config)
        print("✅ Custom configured agent created successfully")

        # Test basic training
        result = agent.train_episode()
        print(f"✅ Training episode completed")
        print(f"   Reward: {result['reward']:.2f}")
        print(f"   Steps: {result['steps']}")
        print(f"   Insights: {result['insights']}")

        assert True  # Test passed

    except Exception as e:
        print(f"❌ Maze agent test failed: {e}")
        traceback.print_exc()
        assert False, f"Maze agent test failed: {e}"


def test_environment_adapter():
    """Test maze environment adapter"""
    print("\nTesting environment adapter...")

    try:
        from insightspike.core.interfaces.generic_interfaces import TaskType
        from insightspike.core.interfaces.maze_implementation import (
            MazeEnvironmentAdapter,
        )

        # Create environment
        env = MazeEnvironmentAdapter(maze_size=5)
        print("✅ Environment created successfully")

        # Test basic functionality
        state = env.reset()
        print(f"✅ Environment reset - state type: {state.environment_type}")
        print(f"   Task type: {env.get_task_type()}")

        # Test action execution
        action_space = env.get_action_space()
        print(
            f"✅ Action space: {action_space.action_type}, dim: {action_space.action_dim}"
        )

        # Execute a few actions
        for i in range(3):
            next_state, reward, done, info = env.step(i % 4)
            print(f"   Step {i+1}: reward={reward:.3f}, done={done}")
            if done:
                break

        assert True  # Test passed

    except Exception as e:
        print(f"❌ Environment adapter test failed: {e}")
        traceback.print_exc()
        assert False, f"Environment adapter test failed: {e}"


def test_insight_detection():
    """Test insight detection functionality"""
    print("\nTesting insight detection...")

    try:
        from insightspike.core.interfaces.maze_implementation import (
            MazeEnvironmentAdapter,
            MazeInsightDetector,
        )

        # Create components
        env = MazeEnvironmentAdapter(maze_size=6)
        detector = MazeInsightDetector(maze_size=6)
        print("✅ Insight detector created successfully")

        # Simulate some transitions
        state = env.reset()
        for i in range(5):
            action = i % 4
            next_state, reward, done, info = env.step(action)

            context = {
                "distance_to_goal": info.get("distance_to_goal", 0),
                "exploration_ratio": info.get("exploration_ratio", 0),
            }

            insight = detector.detect_insight(
                state, action, reward, next_state, context
            )

            if insight:
                print(f"🧠 Insight detected: {insight.insight_type}")
                print(f"   Description: {insight.description}")
                break

            state = next_state
            if done:
                break

        print(f"✅ Insight detection test completed")
        assert True  # Test passed

    except Exception as e:
        print(f"❌ Insight detection test failed: {e}")
        traceback.print_exc()
        assert False, f"Insight detection test failed: {e}"


def test_full_integration():
    """Test full integration - agent training with insights"""
    print("\nTesting full integration...")

    try:
        from insightspike.core.agents.agent_factory import create_maze_agent

        # Create agent
        agent = create_maze_agent(maze_size=6)
        print("✅ Integration agent created")

        # Train for several episodes
        total_insights = 0
        for episode in range(5):
            result = agent.train_episode()
            total_insights += result["insights"]

            if result["insights"] > 0:
                print(f"   Episode {episode+1}: 🧠 {result['insights']} insights!")

        # Get performance summary
        summary = agent.get_performance_summary()
        print(f"✅ Integration test completed")
        print(f"   Total insights: {total_insights}")
        print(f"   Average reward: {summary.get('avg_episode_reward', 0):.3f}")

        assert True  # Test passed

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        assert False, f"Integration test failed: {e}"


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running InsightSpike Reusability Tests")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Standalone Reasoner Test", test_standalone_reasoner),
        ("Maze Agent Creation Test", test_maze_agent_creation),
        ("Environment Adapter Test", test_environment_adapter),
        ("Insight Detection Test", test_insight_detection),
        ("Full Integration Test", test_full_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Reusability implementation successful!")
        assert True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        assert False, f"{total - passed} tests failed"


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
