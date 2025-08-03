#!/usr/bin/env python3
"""Analyze geDIG values during navigation."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def analyze_gediq_values():
    """Analyze geDIG, GED, and IG values during navigation."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    print("Analyzing geDIG values during navigation...")
    print(f"Maze size: {maze.size}, Goal at: {maze.goal_pos}")
    print("=" * 50)
    
    # Create navigator
    navigator = ExperienceMemoryNavigator(nav_config)
    
    # Run navigation
    obs = maze.reset()
    max_steps = 300
    
    for step in range(max_steps):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\nGoal reached in {step + 1} steps!")
            break
    
    # Extract history
    history = navigator.gediq_history
    
    if not history:
        print("No geDIG history recorded.")
        return
    
    # Convert to arrays for plotting
    steps = [h['step'] for h in history]
    geds = [h['ged'] for h in history]
    igs = [h['ig'] for h in history]
    fs = [h['f'] for h in history]
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot GED values
    ax1 = axes[0]
    ax1.plot(steps, geds, 'b-', linewidth=2, label='GED (Movement Cost)')
    ax1.set_ylabel('GED', fontsize=12)
    ax1.set_title('GED (Generalized Euclidean Distance) - Movement Cost', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot IG values
    ax2 = axes[1]
    ax2.plot(steps, igs, 'g-', linewidth=2, label='IG (Information Gain)')
    ax2.set_ylabel('IG', fontsize=12)
    ax2.set_title('IG (Information Gain) - Discovery Value', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Find goal discovery point
    goal_discovery_step = None
    for i, h in enumerate(history):
        if navigator.goal_position and h['position'] == navigator.goal_position:
            goal_discovery_step = h['step']
            break
    
    if goal_discovery_step:
        ax2.axvline(x=goal_discovery_step, color='red', linestyle='--', alpha=0.7, label='Goal Discovered')
        ax2.legend()
    
    # Plot geDIG objective values
    ax3 = axes[2]
    ax3.plot(steps, fs, 'r-', linewidth=2, label='f = w*GED - k*IG')
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('geDIG Objective', fontsize=12)
    ax3.set_title(f'geDIG Objective Function (w={navigator.w_ged}, k={navigator.k_ig})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add zero line for reference
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gediq_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Average GED: {np.mean(geds):.2f}")
    print(f"  Average IG: {np.mean(igs):.2f}")
    print(f"  Average geDIG: {np.mean(fs):.2f}")
    print(f"  Max IG: {np.max(igs):.2f} at step {steps[np.argmax(igs)]}")
    print(f"  Min geDIG: {np.min(fs):.2f} at step {steps[np.argmin(fs)]}")
    
    # Show some interesting moments
    print("\nInteresting moments:")
    
    # High IG moments
    high_ig_indices = np.argsort(igs)[-5:]
    print("\nTop 5 High Information Gain moments:")
    for idx in reversed(high_ig_indices):
        h = history[idx]
        print(f"  Step {h['step']}: IG={h['ig']:.2f}, Action={['up','right','down','left'][h['action']]}, Position={h['position']}")
    
    # Low geDIG moments (best actions)
    low_f_indices = np.argsort(fs)[:5]
    print("\nTop 5 Best geDIG scores (most attractive actions):")
    for idx in low_f_indices:
        h = history[idx]
        print(f"  Step {h['step']}: f={h['f']:.2f}, IG={h['ig']:.2f}, Action={['up','right','down','left'][h['action']]}, Position={h['position']}")


if __name__ == "__main__":
    analyze_gediq_values()