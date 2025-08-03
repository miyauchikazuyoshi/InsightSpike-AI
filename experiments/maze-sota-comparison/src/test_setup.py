#!/usr/bin/env python3
"""Test setup for maze comparison."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Test imports
try:
    from environments.standard_maze import make_maze_env
    print("✓ Environment import successful")
    
    from baselines.classical_agents import create_classical_agent
    print("✓ Classical agents import successful")
    
    from agents.gedig_gym_wrapper import create_gedig_agent
    print("✓ geDIG agent import successful")
    
    # Test environment creation
    env = make_maze_env("standard", size=(5, 5))
    print("✓ Environment creation successful")
    
    # Test agent creation
    agents = {
        'random': create_classical_agent('random'),
        'astar': create_classical_agent('astar'),
        'gedig_blind': create_gedig_agent('blind'),
        'gedig_visual': create_gedig_agent('visual'),
    }
    print("✓ All agents created successfully")
    
    # Test single step
    obs, _ = env.reset(seed=42)
    for name, agent in agents.items():
        agent.reset()
        if hasattr(agent, 'compute_path'):
            grid, start, goal = env.get_state_for_classical()
            agent.compute_path(grid, start, goal)
        action = agent.act(obs)
        print(f"✓ {name} agent action: {action}")
    
    print("\n✅ All tests passed! Ready to run comparison.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()