#!/usr/bin/env python3
"""
Very quick test of complex maze - 20 episodes only
"""

import numpy as np
import torch
from run_complex_maze_experiment import SimplifiedComplexAgent, ComplexMazeEnvironment
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

def quick_test():
    """Run quick 20-episode test"""
    
    print("QUICK COMPLEX MAZE TEST - 20 Episodes")
    print("="*50)
    
    # Test configurations
    configs = [
        {"name": "Full_GED_IG", "use_ged": True, "use_ig": True, "weight": 0.2},
        {"name": "IG_Only", "use_ged": False, "use_ig": True, "weight": 0.2},
        {"name": "Baseline", "use_ged": False, "use_ig": False, "weight": 0.0}
    ]
    
    results = {}
    episodes = 20
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        env = ComplexMazeEnvironment(size=12, num_rooms=4, num_keys=2)
        agent = SimplifiedComplexAgent(
            env.state_space_size,
            env.action_space_size,
            use_ged=config['use_ged'],
            use_ig=config['use_ig'],
            intrinsic_weight=config['weight']
        )
        
        episode_rewards = []
        episode_success = []
        keys_collected = []
        
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:  # Limit steps
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done, info)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train
                if len(agent.memory) > 16 and steps % 8 == 0:
                    agent.replay(batch_size=16)
            
            episode_rewards.append(total_reward)
            episode_success.append(info.get('success', False))
            keys_collected.append(info.get('keys_collected', 0))
            
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}: Reward={total_reward:.2f}, "
                      f"Keys={info.get('keys_collected', 0)}/2, "
                      f"Success={info.get('success', False)}")
        
        results[config['name']] = {
            'rewards': episode_rewards,
            'success': episode_success,
            'keys': keys_collected,
            'avg_reward': np.mean(episode_rewards),
            'success_rate': np.mean(episode_success),
            'avg_keys': np.mean(keys_collected)
        }
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY (20 episodes)")
    print("="*50)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Success rate: {data['success_rate']:.2%}")
        print(f"  Avg reward: {data['avg_reward']:.2f}")
        print(f"  Avg keys: {data['avg_keys']:.2f}/2")
    
    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    names = list(results.keys())
    success_rates = [results[n]['success_rate'] for n in names]
    ax1.bar(names, success_rates)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate Comparison (20 episodes)')
    ax1.set_ylim(0, 1)
    
    # Average keys collected
    avg_keys = [results[n]['avg_keys'] for n in names]
    ax2.bar(names, avg_keys)
    ax2.set_ylabel('Average Keys Collected')
    ax2.set_title('Exploration Progress')
    ax2.set_ylim(0, 2.5)
    ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Max keys')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_complex_maze")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'quick_test_results.png', dpi=150)
    
    # Save data
    save_data = {
        'results': results,
        'episodes': episodes,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'quick_test_data.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Show one maze example
    print("\nGenerating maze visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env.reset()
    env.render(ax)
    plt.savefig(output_dir / 'maze_example.png', dpi=150)
    print("Maze example saved!")

if __name__ == "__main__":
    quick_test()