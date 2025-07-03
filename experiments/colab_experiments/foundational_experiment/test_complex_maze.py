#!/usr/bin/env python3
"""
Quick test of complex maze environment
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the complex maze environment
import sys
sys.path.append(str(Path(__file__).parent))
from intrinsic_motivation_complex_maze import ComplexMazeEnvironment

def test_maze_generation():
    """Test and visualize maze generation"""
    
    print("Testing Complex Maze Generation")
    print("="*50)
    
    # Create maze
    env = ComplexMazeEnvironment(size=12, num_rooms=4, num_keys=2)
    
    # Generate a few mazes
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i in range(4):
        env.reset()
        env.render(axes[i])
        axes[i].set_title(f'Maze Example {i+1}')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_complex_maze")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'maze_examples.png', dpi=150, bbox_inches='tight')
    print(f"Maze examples saved to {output_dir}")
    
    # Test basic navigation
    print("\nTesting Basic Navigation:")
    env.reset()
    
    # Take some random actions
    total_reward = 0
    for step in range(50):
        action = np.random.randint(4)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode ended at step {step}")
            print(f"Success: {info['success']}")
            print(f"Total reward: {total_reward:.2f}")
            break
        
        if step % 10 == 0:
            print(f"Step {step}: Keys={info['keys_collected']}/{env.num_keys}, "
                  f"Subgoals={info['subgoals_visited']}/{len(env.subgoals)}")
    
    print("\nMaze features:")
    print(f"- Size: {env.size}x{env.size}")
    print(f"- Rooms: {env.num_rooms}")
    print(f"- Keys: {env.num_keys}")
    print(f"- Subgoals: {len(env.subgoals)}")
    print(f"- State space: {env.state_space_size} dimensions")
    
    # Visualize state representation
    print("\nState representation shape:", state.shape)
    
    # Show complexity compared to simple grid
    simple_grid_states = 6 * 6  # 6x6 grid positions
    complex_maze_states = env.state_space_size
    
    print(f"\nComplexity comparison:")
    print(f"- Simple 6x6 grid: {simple_grid_states} state dimensions")
    print(f"- Complex 12x12 maze: {complex_maze_states} state dimensions")
    print(f"- Increase: {complex_maze_states/simple_grid_states:.1f}x")

if __name__ == "__main__":
    test_maze_generation()