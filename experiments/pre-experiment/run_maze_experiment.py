"""Run maze navigation experiments with geDIG."""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze, MazeObservation
from insightspike.navigators.simple_gediq_navigator import SimpleGeDIGNavigator
from insightspike.navigators.pure_gediq_navigator import PureGeDIGNavigator
from insightspike.config.maze_config import MazeExperimentConfig, MazeNavigatorConfig
import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineNavigator:
    """Base class for baseline navigators."""
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        raise NotImplementedError
    
    def new_episode(self):
        """Called at start of new episode."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {}


class RandomNavigator(BaselineNavigator):
    """Random action selection."""
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        if observation.possible_moves:
            return np.random.choice(observation.possible_moves)
        return 0


class WallFollowerNavigator(BaselineNavigator):
    """Simple wall following strategy."""
    
    def __init__(self):
        self.last_action = 0
        self.following_right = True
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        if not observation.possible_moves:
            return 0
        
        # Try to follow right wall
        if self.following_right:
            # Priority: right, forward, left, back
            action_priority = [
                (self.last_action + 1) % 4,  # Right turn
                self.last_action,             # Forward
                (self.last_action + 3) % 4,  # Left turn (was -1 which gives negative)
                (self.last_action + 2) % 4   # Back
            ]
        else:
            # Priority: left, forward, right, back
            action_priority = [
                (self.last_action + 3) % 4,  # Left turn
                self.last_action,             # Forward
                (self.last_action + 1) % 4,  # Right turn
                (self.last_action + 2) % 4   # Back
            ]
        
        # Choose first available action from priority
        for action in action_priority:
            if action in observation.possible_moves:
                self.last_action = action
                return action
        
        return observation.possible_moves[0]


def run_episode(maze: SimpleMaze, navigator: Any, 
                max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
    """Run a single episode."""
    obs = maze.reset()
    done = False
    step = 0
    
    trajectory = [maze.agent_pos]
    actions = []
    rewards = []
    
    while not done and step < max_steps:
        # Navigator decides action
        action = navigator.decide_action(obs, maze)
        actions.append(action)
        
        # Take action
        obs, reward, done, info = maze.step(action)
        rewards.append(reward)
        trajectory.append(maze.agent_pos)
        
        if render and step % 10 == 0:
            print(f"\nStep {step}:")
            print(maze.render('ascii'))
            time.sleep(0.1)
        
        step += 1
    
    # Episode metrics
    metrics = {
        'steps': step,
        'success': done and maze.agent_pos == maze.goal_pos,
        'wall_hits': info.get('wall_hits', 0),
        'total_reward': sum(rewards),
        'trajectory_length': len(set(trajectory)),  # Unique positions visited
        'trajectory': trajectory,
        'actions': actions
    }
    
    return metrics


def run_experiment(config: MazeExperimentConfig):
    """Run the full experiment."""
    results = {}
    
    # Set random seed
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Initialize maze
    maze = SimpleMaze(size=config.maze.size)
    
    # Run each algorithm
    for algo_name in config.algorithms:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running algorithm: {algo_name}")
        logger.info(f"{'='*50}")
        
        # Create navigator
        if algo_name == 'random':
            navigator = RandomNavigator()
        elif algo_name == 'wall_follower':
            navigator = WallFollowerNavigator()
        elif algo_name == 'gediq':
            navigator = SimpleGeDIGNavigator(config.navigator)
        elif algo_name == 'pure_gediq':
            navigator = PureGeDIGNavigator(config.navigator)
        else:
            logger.warning(f"Unknown algorithm: {algo_name}, skipping")
            continue
        
        # Metrics storage
        episode_metrics = []
        
        # Run episodes
        for episode in range(config.maze.num_episodes):
            # New episode
            navigator.new_episode()
            
            # Run episode
            render = (episode % config.maze.render_frequency == 0) and config.maze.render_mode != 'none'
            metrics = run_episode(maze, navigator, config.maze.max_steps, render=render)
            
            episode_metrics.append(metrics)
            
            # Log progress
            if episode % 10 == 0:
                recent_success = np.mean([m['success'] for m in episode_metrics[-10:]])
                recent_steps = np.mean([m['steps'] for m in episode_metrics[-10:]])
                logger.info(f"Episode {episode}: Success rate: {recent_success:.2f}, "
                           f"Avg steps: {recent_steps:.1f}")
        
        # Store results
        results[algo_name] = {
            'episodes': episode_metrics,
            'navigator_metrics': navigator.get_metrics() if hasattr(navigator, 'get_metrics') else {},
            'summary': {
                'success_rate': np.mean([m['success'] for m in episode_metrics]),
                'avg_steps': np.mean([m['steps'] for m in episode_metrics]),
                'avg_wall_hits': np.mean([m['wall_hits'] for m in episode_metrics]),
                'final_success_rate': np.mean([m['success'] for m in episode_metrics[-20:]]),
                'final_avg_steps': np.mean([m['steps'] for m in episode_metrics[-20:]])
            }
        }
    
    return results, maze


def plot_results(results: Dict[str, Any], save_path: Optional[Path] = None):
    """Plot experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Success rate over episodes
    ax = axes[0, 0]
    for algo_name, algo_results in results.items():
        episodes = algo_results['episodes']
        success_rates = []
        
        # Moving average
        window = 10
        for i in range(len(episodes)):
            start = max(0, i - window + 1)
            success_rates.append(np.mean([e['success'] for e in episodes[start:i+1]]))
        
        ax.plot(success_rates, label=algo_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (10-episode avg)')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True)
    
    # 2. Steps to goal
    ax = axes[0, 1]
    for algo_name, algo_results in results.items():
        episodes = algo_results['episodes']
        successful_episodes = [e for e in episodes if e['success']]
        
        if successful_episodes:
            steps = []
            window = 5
            for i in range(len(successful_episodes)):
                start = max(0, i - window + 1)
                steps.append(np.mean([e['steps'] for e in successful_episodes[start:i+1]]))
            
            ax.plot(steps, label=algo_name)
    
    ax.set_xlabel('Successful Episode')
    ax.set_ylabel('Steps to Goal (5-episode avg)')
    ax.set_title('Path Efficiency')
    ax.legend()
    ax.grid(True)
    
    # 3. Wall hits
    ax = axes[1, 0]
    for algo_name, algo_results in results.items():
        episodes = algo_results['episodes']
        wall_hits = []
        
        window = 10
        for i in range(len(episodes)):
            start = max(0, i - window + 1)
            wall_hits.append(np.mean([e['wall_hits'] for e in episodes[start:i+1]]))
        
        ax.plot(wall_hits, label=algo_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wall Hits (10-episode avg)')
    ax.set_title('Collision Avoidance Learning')
    ax.legend()
    ax.grid(True)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Final Performance (last 20 episodes):\n\n"
    for algo_name, algo_results in results.items():
        summary = algo_results['summary']
        summary_text += f"{algo_name}:\n"
        summary_text += f"  Success Rate: {summary['final_success_rate']:.2%}\n"
        summary_text += f"  Avg Steps: {summary['final_avg_steps']:.1f}\n"
        
        # geDIG specific metrics
        if 'navigator_metrics' in algo_results and algo_results['navigator_metrics']:
            nav_metrics = algo_results['navigator_metrics']
            if 'total_nodes' in nav_metrics:
                summary_text += f"  Memory Nodes: {nav_metrics['total_nodes']}\n"
        summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def visualize_memory_map(results: Dict[str, Any], maze: SimpleMaze, 
                        save_path: Optional[Path] = None):
    """Visualize the memory map created by geDIG."""
    if 'gediq' not in results:
        logger.warning("No geDIG results to visualize")
        return
    
    gediq_results = results['gediq']
    nav_metrics = gediq_results.get('navigator_metrics', {})
    
    if 'node_positions' not in nav_metrics:
        logger.warning("No memory nodes to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze
    maze_array = maze.grid
    ax.imshow(maze_array, cmap='gray_r', alpha=0.3)
    
    # Draw memory nodes
    node_positions = nav_metrics['node_positions']
    node_types = nav_metrics.get('node_types', ['unknown'] * len(node_positions))
    
    # Color map for node types
    type_colors = {
        'wall': 'red',
        'junction': 'blue',
        'dead_end': 'orange',
        'corridor': 'green',
        'goal': 'gold',
        'unknown': 'gray'
    }
    
    for pos, node_type in zip(node_positions, node_types):
        color = type_colors.get(node_type, 'gray')
        ax.scatter(pos[1], pos[0], c=color, s=100, alpha=0.8, edgecolors='black')
    
    # Draw start and goal
    ax.scatter(maze.start_pos[1], maze.start_pos[0], c='green', s=200, marker='s', label='Start')
    ax.scatter(maze.goal_pos[1], maze.goal_pos[0], c='red', s=200, marker='*', label='Goal')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=node_type.replace('_', ' ').title())
        for node_type, color in type_colors.items()
        if any(t == node_type for t in node_types)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f'geDIG Memory Map ({len(node_positions)} nodes)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved memory map to {save_path}")
    
    plt.show()


def main():
    """Main experiment runner."""
    # Load configuration
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        experiment_config = MazeExperimentConfig(**config_dict)
    else:
        # Use default configuration
        experiment_config = MazeExperimentConfig()
        
        # Save default config for future use
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config.dict(), f, default_flow_style=False)
        logger.info(f"Created default config at {config_path}")
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Run experiment
    logger.info("Starting maze navigation experiment...")
    results, maze = run_experiment(experiment_config)
    
    # Save results
    results_file = results_dir / f"maze_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for algo, algo_results in results.items():
            json_results[algo] = {
                'summary': algo_results['summary'],
                'navigator_metrics': algo_results['navigator_metrics']
            }
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Plot results
    plot_path = results_dir / f"maze_results_{int(time.time())}.png"
    plot_results(results, plot_path)
    
    # Visualize memory map
    memory_map_path = results_dir / f"memory_map_{int(time.time())}.png"
    visualize_memory_map(results, maze, memory_map_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for algo_name, algo_results in results.items():
        summary = algo_results['summary']
        print(f"\n{algo_name.upper()}:")
        print(f"  Overall Success Rate: {summary['success_rate']:.2%}")
        print(f"  Final Success Rate: {summary['final_success_rate']:.2%}")
        print(f"  Average Steps: {summary['avg_steps']:.1f}")
        print(f"  Average Wall Hits: {summary['avg_wall_hits']:.1f}")
        
        if algo_name == 'gediq' and 'navigator_metrics' in algo_results:
            nav_metrics = algo_results['navigator_metrics']
            print(f"  Total Memory Nodes: {nav_metrics.get('total_nodes', 'N/A')}")
            print(f"  Total Energy Spent: {nav_metrics.get('total_energy', 'N/A'):.1f}")


if __name__ == "__main__":
    main()