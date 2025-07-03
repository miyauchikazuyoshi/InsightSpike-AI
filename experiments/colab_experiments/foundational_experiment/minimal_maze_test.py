#!/usr/bin/env python3
"""
Minimal test - just 3 episodes per config
"""

import numpy as np
import torch
import time
from run_complex_maze_experiment import SimplifiedComplexAgent
from intrinsic_motivation_complex_maze import ComplexMazeEnvironment
from pathlib import Path
import json

np.random.seed(42)
torch.manual_seed(42)

output_dir = Path("experiments/foundational_intrinsic_motivation/results_complex_maze")
output_dir.mkdir(parents=True, exist_ok=True)

print("MINIMAL COMPLEX MAZE TEST")
print("="*40)

# Just test 2 configs, 3 episodes each
configs = [
    {"name": "Full_GED_IG", "use_ged": True, "use_ig": True},
    {"name": "Baseline", "use_ged": False, "use_ig": False}
]

results = {}

for config in configs:
    print(f"\n{config['name']}:")
    
    env = ComplexMazeEnvironment(size=12, num_rooms=4, num_keys=2)
    agent = SimplifiedComplexAgent(
        env.state_space_size,
        env.action_space_size,
        use_ged=config['use_ged'],
        use_ig=config['use_ig'],
        intrinsic_weight=0.2
    )
    
    total_rewards = []
    successes = []
    keys = []
    
    for ep in range(3):
        start = time.time()
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Just take 100 steps
        for _ in range(100):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        
        elapsed = time.time() - start
        total_rewards.append(total_reward)
        successes.append(info.get('success', False))
        keys.append(info.get('keys_collected', 0))
        
        print(f"  Ep {ep+1}: Steps={steps}, Reward={total_reward:.2f}, "
              f"Keys={info['keys_collected']}/2, Success={info['success']}, "
              f"Time={elapsed:.1f}s")
    
    results[config['name']] = {
        'avg_reward': np.mean(total_rewards),
        'success_rate': np.mean(successes),
        'avg_keys': np.mean(keys)
    }

print("\n" + "="*40)
print("RESULTS:")
for name, data in results.items():
    print(f"{name}: Reward={data['avg_reward']:.2f}, "
          f"Success={data['success_rate']:.0%}, "
          f"Keys={data['avg_keys']:.1f}")

# Save results
with open(output_dir / 'minimal_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {output_dir}")

# Test intrinsic rewards
if hasattr(agent, 'intrinsic_rewards') and agent.intrinsic_rewards:
    print(f"\nIntrinsic rewards (last agent):")
    print(f"  Mean: {np.mean(agent.intrinsic_rewards):.3f}")
    print(f"  Max: {np.max(agent.intrinsic_rewards):.3f}")
    
if hasattr(agent, 'ig_values') and agent.ig_values:
    print(f"\nIG values: Mean={np.mean(agent.ig_values):.3f}")
    
if hasattr(agent, 'ged_values') and agent.ged_values:
    print(f"GED values: Mean={np.mean(agent.ged_values):.3f}")