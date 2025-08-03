#!/usr/bin/env python3
"""Test action memory navigator with entropy-based decisions."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.action_memory_navigator import ActionMemoryNavigator, ActionResult
from insightspike.config.maze_config import MazeNavigatorConfig

# Suppress verbose output
import logging
logging.getLogger().setLevel(logging.WARNING)


def test_action_memory():
    """Test action memory navigator."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0  # No random exploration
    
    print("Testing Action Memory Navigator")
    print("=" * 60)
    print("Concept: Each movement attempt (A→B) is a memory node")
    print("Entropy: Unknown=1.0, Blocked=0.5, Passed=0.1")
    print("=" * 60)
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    # Create navigator
    navigator = ActionMemoryNavigator(nav_config)
    
    # Run navigation
    obs = maze.reset()
    trajectory = [obs.position]
    
    for step in range(200):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        trajectory.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ Goal reached in {step + 1} steps!")
            print(f"   Wall hits: {navigator.wall_hits}")
            break
    else:
        print(f"\n❌ Failed to reach goal in 200 steps")
    
    # Print metrics
    metrics = navigator.get_metrics()
    print(f"\nAction Memory Statistics:")
    print(f"  Total action nodes: {metrics['total_action_nodes']}")
    print(f"  Passed actions: {metrics['passed_actions']}")
    print(f"  Blocked actions: {metrics['blocked_actions']}")
    print(f"  Unknown actions: {metrics['unknown_actions']}")
    print(f"  Average entropy: {metrics['average_entropy']:.3f}")
    
    # Visualize action memory
    visualize_action_memory(maze, navigator, trajectory)


def visualize_action_memory(maze, navigator, trajectory):
    """Visualize the action memory nodes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Maze with trajectory
    ax1.set_title("Navigation Trajectory", fontsize=14)
    
    # Draw maze
    for i in range(maze.size[0]):
        for j in range(maze.size[1]):
            if maze.grid[i, j] == 1:  # Wall
                ax1.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                       facecolor='black'))
    
    # Draw trajectory
    if len(trajectory) > 1:
        traj_x = [pos[1] + 0.5 for pos in trajectory]
        traj_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in trajectory]
        ax1.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.5)
    
    # Draw start and goal
    sx, sy = maze.start_pos
    ax1.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
           ha='center', va='center', fontsize=16, color='green', weight='bold')
    
    gx, gy = maze.goal_pos
    ax1.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
           ha='center', va='center', fontsize=16, color='red', weight='bold')
    
    ax1.set_xlim(0, maze.size[1])
    ax1.set_ylim(0, maze.size[0])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Right: Action memory visualization
    ax2.set_title("Action Memory Nodes (Entropy Visualization)", fontsize=14)
    
    # Draw all action nodes as arrows
    for action_node in navigator.action_nodes.values():
        from_y, from_x = action_node.from_pos
        to_y, to_x = action_node.to_pos
        
        # Convert to display coordinates
        from_x_disp = from_x + 0.5
        from_y_disp = maze.size[0] - 1 - from_y + 0.5
        to_x_disp = to_x + 0.5
        to_y_disp = maze.size[0] - 1 - to_y + 0.5
        
        # Color based on result/entropy
        if action_node.result == ActionResult.PASSED:
            color = 'green'
            alpha = 0.8
            linewidth = 2
        elif action_node.result == ActionResult.BLOCKED:
            color = 'red'
            alpha = 0.6
            linewidth = 2
        else:  # UNKNOWN
            color = 'gray'
            alpha = 0.3
            linewidth = 1
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (from_x_disp, from_y_disp),
            (to_x_disp, to_y_disp),
            arrowstyle='->', 
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            mutation_scale=15
        )
        ax2.add_patch(arrow)
    
    # Draw maze outline
    for i in range(maze.size[0]):
        for j in range(maze.size[1]):
            if maze.grid[i, j] == 1:  # Wall
                ax2.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                       facecolor='black', alpha=0.2))
    
    # Add legend
    passed_patch = mpatches.Patch(color='green', label=f'Passed (entropy=0.1)')
    blocked_patch = mpatches.Patch(color='red', label=f'Blocked (entropy=0.5)')
    unknown_patch = mpatches.Patch(color='gray', label=f'Unknown (entropy=1.0)')
    ax2.legend(handles=[passed_patch, blocked_patch, unknown_patch], loc='upper right')
    
    ax2.set_xlim(0, maze.size[1])
    ax2.set_ylim(0, maze.size[0])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_memory_visualization.png', dpi=150)
    plt.show()
    
    # Analyze entropy distribution
    analyze_entropy_distribution(navigator)


def analyze_entropy_distribution(navigator):
    """Analyze the entropy distribution of action nodes."""
    entropies = [node.entropy for node in navigator.action_nodes.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of entropies
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts, _, patches = ax.hist(entropies, bins=bins, edgecolor='black')
    
    # Color code the bars
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Entropy', fontsize=12)
    ax.set_ylabel('Number of Action Nodes', fontsize=12)
    ax.set_title('Distribution of Action Node Entropies', fontsize=14)
    
    # Add text annotations
    ax.text(0.1, max(counts)*0.9, f"Passed: {sum(1 for e in entropies if e < 0.2)}", 
            fontsize=10, color='green', weight='bold')
    ax.text(0.5, max(counts)*0.9, f"Blocked: {sum(1 for e in entropies if 0.4 < e < 0.6)}", 
            fontsize=10, color='red', weight='bold')
    ax.text(0.9, max(counts)*0.9, f"Unknown: {sum(1 for e in entropies if e > 0.9)}", 
            fontsize=10, color='darkred', weight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_entropy_distribution.png', dpi=150)
    plt.show()
    
    print(f"\nEntropy Analysis:")
    print(f"  Average entropy: {np.mean(entropies):.3f}")
    print(f"  Min entropy: {min(entropies):.3f}")
    print(f"  Max entropy: {max(entropies):.3f}")
    print(f"  Entropy reduction: {1.0 - np.mean(entropies):.1%}")


if __name__ == "__main__":
    test_action_memory()