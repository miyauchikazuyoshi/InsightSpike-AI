#!/usr/bin/env python3
"""Test structured action memory navigator - NO CHEATING!"""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.structured_action_navigator import StructuredActionNavigator, ActionResult
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.config.maze_config import MazeNavigatorConfig

# Suppress verbose output
import logging
logging.getLogger().setLevel(logging.WARNING)


def test_structured_vs_blind():
    """Compare structured action memory vs blind navigation."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    print("Testing Structured Action Memory Navigator")
    print("=" * 70)
    print("Concept: Learning from structural similarity between actions")
    print("NO CHEATING: No prior knowledge, learns everything from experience")
    print("=" * 70)
    
    results = {
        'structured': {'steps': [], 'wall_hits': []},
        'blind': {'steps': [], 'wall_hits': []}
    }
    
    # Test on multiple trials
    for trial in range(5):
        print(f"\n--- Trial {trial + 1} ---")
        
        # Test structured navigator
        np.random.seed(42 + trial)
        maze = SimpleMaze(size=(15, 15), maze_type='dfs')
        structured_nav = StructuredActionNavigator(nav_config)
        
        obs = maze.reset()
        for step in range(500):
            action = structured_nav.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            
            if done and maze.agent_pos == maze.goal_pos:
                results['structured']['steps'].append(step + 1)
                results['structured']['wall_hits'].append(structured_nav.wall_hits)
                print(f"Structured: {step + 1} steps, {structured_nav.wall_hits} wall hits")
                break
        else:
            results['structured']['steps'].append(500)
            results['structured']['wall_hits'].append(structured_nav.wall_hits)
            print(f"Structured: Failed (500 steps)")
        
        # Test blind navigator
        np.random.seed(42 + trial)
        maze = SimpleMaze(size=(15, 15), maze_type='dfs')
        blind_nav = BlindExperienceNavigator(nav_config)
        
        obs = maze.reset()
        for step in range(500):
            action = blind_nav.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            
            if done and maze.agent_pos == maze.goal_pos:
                results['blind']['steps'].append(step + 1)
                results['blind']['wall_hits'].append(blind_nav.wall_hits)
                print(f"Blind:      {step + 1} steps, {blind_nav.wall_hits} wall hits")
                break
        else:
            results['blind']['steps'].append(500)
            results['blind']['wall_hits'].append(blind_nav.wall_hits)
            print(f"Blind:      Failed (500 steps)")
        
        # Show global knowledge learned
        if trial == 0:  # Show for first trial
            print("\nGlobal Knowledge Learned by Structured Navigator:")
            metrics = structured_nav.get_metrics()
            for direction, stats in metrics['global_knowledge'].items():
                print(f"  {direction}: {stats['success_rate']:.1%} success rate "
                      f"({stats['total_attempts']} attempts)")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS:")
    
    print(f"\n🧠 Structured Action Memory:")
    print(f"   Steps: {np.mean(results['structured']['steps']):.1f} ± {np.std(results['structured']['steps']):.1f}")
    print(f"   Wall hits: {np.mean(results['structured']['wall_hits']):.1f} ± {np.std(results['structured']['wall_hits']):.1f}")
    
    print(f"\n🦯 Blind Navigator:")
    print(f"   Steps: {np.mean(results['blind']['steps']):.1f} ± {np.std(results['blind']['steps']):.1f}")
    print(f"   Wall hits: {np.mean(results['blind']['wall_hits']):.1f} ± {np.std(results['blind']['wall_hits']):.1f}")
    
    improvement = (np.mean(results['blind']['steps']) - np.mean(results['structured']['steps'])) / np.mean(results['blind']['steps']) * 100
    print(f"\n📊 Performance Improvement: {improvement:.1f}%")
    
    # Create visualization
    visualize_comparison(results)
    
    # Test on single maze to show similarity learning
    test_similarity_learning()


def visualize_comparison(results):
    """Visualize the comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Steps comparison
    x = range(len(results['structured']['steps']))
    ax1.plot(x, results['structured']['steps'], 'b-o', label='Structured', linewidth=2)
    ax1.plot(x, results['blind']['steps'], 'r-s', label='Blind', linewidth=2)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Steps to Goal')
    ax1.set_title('Navigation Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wall hits comparison
    ax2.scatter(results['structured']['steps'], results['structured']['wall_hits'], 
               s=100, alpha=0.6, label='Structured', color='blue')
    ax2.scatter(results['blind']['steps'], results['blind']['wall_hits'], 
               s=100, alpha=0.6, label='Blind', color='red')
    ax2.set_xlabel('Steps to Goal')
    ax2.set_ylabel('Wall Hits')
    ax2.set_title('Efficiency: Steps vs Wall Hits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('structured_vs_blind_comparison.png', dpi=150)
    plt.show()


def test_similarity_learning():
    """Test how similarity learning works."""
    print("\n" + "=" * 70)
    print("SIMILARITY LEARNING DEMONSTRATION")
    print("=" * 70)
    
    # Create simple maze
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    navigator = StructuredActionNavigator(MazeNavigatorConfig())
    
    # Run for a few steps
    obs = maze.reset()
    for _ in range(30):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        if done:
            break
    
    # Show some similar actions
    print("\nExample of Structural Similarity:")
    
    # Find an unknown action
    unknown_action = None
    for node in navigator.action_nodes.values():
        if node.result == ActionResult.UNKNOWN:
            unknown_action = node
            break
    
    if unknown_action:
        print(f"\nUnknown action: {unknown_action.from_pos} → {unknown_action.to_pos}")
        print(f"Direction: {navigator.DIRECTION_NAMES.get(unknown_action.direction, '?')}")
        
        similar = navigator._find_similar_actions(unknown_action)
        if similar:
            print("\nMost similar known actions:")
            for sim_node, similarity in similar[:3]:
                print(f"  {sim_node.from_pos} → {sim_node.to_pos} "
                      f"({navigator.DIRECTION_NAMES.get(sim_node.direction, '?')}): "
                      f"{sim_node.result.value} (similarity: {similarity:.2f})")
    
    print("\nThis demonstrates NO CHEATING:")
    print("- Navigator starts with ZERO knowledge")
    print("- Learns patterns from actual experience")
    print("- Uses structural similarity (same direction, nearby location)")
    print("- Builds global knowledge (which directions work better)")


if __name__ == "__main__":
    test_structured_vs_blind()