#!/usr/bin/env python3
"""
Correct GED Implementation for Intrinsic Motivation
===================================================

This implementation correctly measures how new information (high IG) 
contributes to structural optimization of knowledge in long-term memory.

Key concepts:
- GED: Measures structural simplification in knowledge representation
- IG: Measures novelty/information gain
- True insight: When new information leads to simpler knowledge structure
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import networkx as nx

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class KnowledgeGraph:
    """Represents agent's knowledge as a graph structure"""
    
    def __init__(self, max_nodes=100):
        self.graph = nx.Graph()
        self.max_nodes = max_nodes
        self.state_to_node = {}  # Map states to graph nodes
        self.node_visits = defaultdict(int)
        self.reference_graph = None  # Optimal/goal structure
        
    def add_state_transition(self, state_from, state_to, reward):
        """Add a state transition to the knowledge graph"""
        # Convert states to hashable keys
        key_from = tuple(state_from.flatten())
        key_to = tuple(state_to.flatten())
        
        # Add nodes if not present
        if key_from not in self.state_to_node:
            node_id = len(self.state_to_node)
            self.state_to_node[key_from] = node_id
            self.graph.add_node(node_id, state=key_from, value=0)
        
        if key_to not in self.state_to_node:
            node_id = len(self.state_to_node)
            self.state_to_node[key_to] = node_id
            self.graph.add_node(node_id, state=key_to, value=0)
        
        # Get node IDs
        node_from = self.state_to_node[key_from]
        node_to = self.state_to_node[key_to]
        
        # Update node visits
        self.node_visits[node_from] += 1
        self.node_visits[node_to] += 1
        
        # Add or update edge with reward as weight
        if self.graph.has_edge(node_from, node_to):
            # Update edge weight with running average
            old_weight = self.graph[node_from][node_to]['weight']
            new_weight = 0.9 * old_weight + 0.1 * reward
            self.graph[node_from][node_to]['weight'] = new_weight
        else:
            self.graph.add_edge(node_from, node_to, weight=reward)
        
        # Prune if too many nodes (keep most visited)
        if len(self.graph) > self.max_nodes:
            self._prune_least_visited()
    
    def _prune_least_visited(self):
        """Remove least visited nodes to maintain size limit"""
        # Sort nodes by visit count
        nodes_by_visits = sorted(self.node_visits.items(), key=lambda x: x[1])
        
        # Remove bottom 10%
        num_to_remove = len(self.graph) // 10
        for node_id, _ in nodes_by_visits[:num_to_remove]:
            if node_id in self.graph:
                self.graph.remove_node(node_id)
                # Clean up mappings
                for state, nid in list(self.state_to_node.items()):
                    if nid == node_id:
                        del self.state_to_node[state]
                del self.node_visits[node_id]
    
    def compute_structural_complexity(self):
        """Compute complexity of current knowledge structure"""
        if len(self.graph) == 0:
            return 0
        
        # Complexity based on:
        # 1. Number of nodes (states discovered)
        # 2. Number of edges (transitions learned)  
        # 3. Graph diameter (longest shortest path)
        # 4. Average clustering coefficient
        
        num_nodes = len(self.graph)
        num_edges = self.graph.number_of_edges()
        
        # Diameter (max shortest path length)
        if nx.is_connected(self.graph):
            diameter = nx.diameter(self.graph)
        else:
            # Use largest component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            diameter = nx.diameter(subgraph) if len(subgraph) > 1 else 0
        
        # Average clustering
        avg_clustering = nx.average_clustering(self.graph)
        
        # Combine metrics (lower is simpler/better)
        complexity = (num_nodes + num_edges) / (1 + diameter) * (1 - avg_clustering)
        
        return complexity
    
    def find_shortest_path_to_goal(self, current_state, goal_state):
        """Find shortest path in knowledge graph to goal"""
        current_key = tuple(current_state.flatten())
        goal_key = tuple(goal_state.flatten())
        
        if current_key not in self.state_to_node or goal_key not in self.state_to_node:
            return None
        
        try:
            path = nx.shortest_path(
                self.graph, 
                self.state_to_node[current_key],
                self.state_to_node[goal_key]
            )
            return len(path) - 1  # Number of steps
        except nx.NetworkXNoPath:
            return None


class CorrectGEDAgent:
    """Agent with correct GED implementation using long-term memory"""
    
    def __init__(self, state_size, action_size, 
                 use_ged=True, use_ig=True, 
                 intrinsic_weight=0.1):
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.intrinsic_weight = intrinsic_weight
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=5000)
        
        # Knowledge graph (long-term memory structure)
        self.knowledge_graph = KnowledgeGraph(max_nodes=200)
        
        # State visit counts for IG
        self.state_visits = defaultdict(int)
        
        # Track structural complexity over time
        self.complexity_history = deque(maxlen=100)
        self.prev_complexity = None
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Goal state for reference
        self.goal_state = None
    
    def set_goal_state(self, goal_state):
        """Set the goal state for structural optimization"""
        self.goal_state = goal_state
    
    def _calculate_ig(self, state):
        """Calculate Information Gain (novelty)"""
        if not self.use_ig:
            return 1.0
        
        state_key = tuple(state.flatten())
        old_visits = self.state_visits[state_key]
        self.state_visits[state_key] += 1
        
        # IG based on entropy change
        total_visits = sum(self.state_visits.values())
        if total_visits == 0:
            return 1.0
        
        # Calculate entropy before and after
        old_probs = np.array(list(self.state_visits.values())) / total_visits
        old_entropy = -np.sum(old_probs * np.log2(old_probs + 1e-10))
        
        # Update with new visit
        self.state_visits[state_key] = old_visits + 1
        new_total = total_visits + 1
        new_probs = np.array(list(self.state_visits.values())) / new_total
        new_entropy = -np.sum(new_probs * np.log2(new_probs + 1e-10))
        
        # Information gain is reduction in entropy
        ig = old_entropy - new_entropy
        
        # Boost for completely new states
        if old_visits == 0:
            ig += 0.5
        
        return max(0, ig)
    
    def _calculate_ged(self, state, action, next_state, reward):
        """Calculate GED based on structural optimization in long-term memory"""
        if not self.use_ged:
            return 0.0
        
        # Add transition to knowledge graph
        self.knowledge_graph.add_state_transition(state, next_state, reward)
        
        # Calculate current structural complexity
        current_complexity = self.knowledge_graph.compute_structural_complexity()
        self.complexity_history.append(current_complexity)
        
        if self.prev_complexity is None:
            self.prev_complexity = current_complexity
            return 0.0
        
        # ΔGED = change in structural complexity
        # Negative means simplification (good)
        delta_ged = current_complexity - self.prev_complexity
        self.prev_complexity = current_complexity
        
        # Additional bonus for finding shorter paths to goal
        if self.goal_state is not None:
            old_path_length = self.knowledge_graph.find_shortest_path_to_goal(state, self.goal_state)
            new_path_length = self.knowledge_graph.find_shortest_path_to_goal(next_state, self.goal_state)
            
            if old_path_length is not None and new_path_length is not None:
                path_improvement = old_path_length - new_path_length
                delta_ged -= path_improvement * 0.1  # Bonus for getting closer
        
        return -delta_ged  # Invert so positive is good
    
    def _calculate_intrinsic_reward(self, state, action, next_state, reward):
        """Calculate intrinsic reward combining IG and GED"""
        if not self.use_ged and not self.use_ig:
            return 0.0
        
        # Calculate components
        ig = self._calculate_ig(next_state)
        ged = self._calculate_ged(state, action, next_state, reward)
        
        # Combine: high reward when new information (high IG) 
        # leads to structural simplification (positive GED)
        intrinsic_reward = ig * ged
        
        # Additional boost for "eureka moments"
        # When both IG and GED are significantly positive
        if ig > 0.2 and ged > 0.5:
            intrinsic_reward *= 2.0  # Eureka spike!
        
        return intrinsic_reward
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with intrinsic reward"""
        intrinsic_reward = self._calculate_intrinsic_reward(state, action, next_state, reward)
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        
        self.memory.append((state, action, total_reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the network on experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_knowledge_stats(self):
        """Get statistics about knowledge structure"""
        kg = self.knowledge_graph
        
        return {
            'num_states': len(kg.graph),
            'num_transitions': kg.graph.number_of_edges(),
            'complexity': kg.compute_structural_complexity(),
            'avg_visits': np.mean(list(kg.node_visits.values())) if kg.node_visits else 0,
            'complexity_trend': np.mean(np.diff(list(self.complexity_history)[-10:])) if len(self.complexity_history) > 1 else 0
        }


# Simple test environment (same as before)
class SimpleGridWorld:
    """Grid world environment"""
    
    def __init__(self, size=6, num_obstacles=3, goal_reward=10.0):
        self.size = size
        self.num_obstacles = num_obstacles
        self.goal_reward = goal_reward
        self.step_penalty = -0.01
        self.collision_penalty = -0.1
        self.timeout_penalty = -1.0
        self.reset()
    
    def _check_path_exists(self):
        """Check if path to goal exists"""
        from collections import deque
        
        visited = set()
        queue = deque([self.start_pos])
        visited.add(self.start_pos)
        
        while queue:
            pos = queue.popleft()
            if pos == self.goal_pos:
                return True
            
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (row + dr, col + dc)
                if (0 <= new_pos[0] < self.size and 
                    0 <= new_pos[1] < self.size and
                    new_pos not in visited and 
                    self.grid[new_pos] != -1):
                    visited.add(new_pos)
                    queue.append(new_pos)
        return False
    
    def reset(self):
        """Reset environment"""
        self.grid = np.zeros((self.size, self.size))
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        
        # Place obstacles ensuring path exists
        for _ in range(10):
            self.grid.fill(0)
            obstacles = set()
            
            while len(obstacles) < self.num_obstacles:
                pos = (np.random.randint(self.size), np.random.randint(self.size))
                if pos != self.start_pos and pos != self.goal_pos:
                    obstacles.add(pos)
            
            for pos in obstacles:
                self.grid[pos] = -1
            
            if self._check_path_exists():
                break
        
        self.current_pos = self.start_pos
        self.step_count = 0
        self.max_steps = self.size * self.size * 2
        
        return self._get_state()
    
    def _get_state(self):
        """Get state representation"""
        state = np.zeros(self.size * self.size)
        state[self.current_pos[0] * self.size + self.current_pos[1]] = 1
        return state
    
    def get_goal_state(self):
        """Get goal state representation"""
        state = np.zeros(self.size * self.size)
        state[self.goal_pos[0] * self.size + self.goal_pos[1]] = 1
        return state
    
    def step(self, action):
        """Execute action"""
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        
        new_pos = (self.current_pos[0] + dr, self.current_pos[1] + dc)
        
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.grid[new_pos] != -1):
            self.current_pos = new_pos
            reward = self.step_penalty
        else:
            reward = self.collision_penalty
        
        self.step_count += 1
        done = False
        info = {'success': False}
        
        if self.current_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
            info['success'] = True
        elif self.step_count >= self.max_steps:
            reward = self.timeout_penalty
            done = True
        
        return self._get_state(), reward, done, info
    
    @property
    def state_space_size(self):
        return self.size * self.size
    
    @property
    def action_space_size(self):
        return 4


def run_correct_ged_experiment(episodes=100, trials=3):
    """Run experiment with correct GED implementation"""
    
    configs = [
        {"name": "Correct_Full", "ged": True, "ig": True, "weight": 0.1},
        {"name": "Correct_IG_Only", "ged": False, "ig": True, "weight": 0.1},
        {"name": "Correct_GED_Only", "ged": True, "ig": False, "weight": 0.1},
        {"name": "Baseline", "ged": False, "ig": False, "weight": 0.0}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        config_results = {
            'success_rates': [],
            'convergence_episodes': [],
            'knowledge_stats': [],
            'complexity_reduction': []
        }
        
        for trial in range(trials):
            # Set seed
            seed = RANDOM_SEED + trial * 10
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            env = SimpleGridWorld(size=6, num_obstacles=3)
            agent = CorrectGEDAgent(
                env.state_space_size,
                env.action_space_size,
                use_ged=config['ged'],
                use_ig=config['ig'],
                intrinsic_weight=config['weight']
            )
            
            # Set goal state for GED calculation
            agent.set_goal_state(env.get_goal_state())
            
            successes = []
            convergence_ep = episodes
            initial_complexity = None
            
            for ep in range(episodes):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    if done:
                        successes.append(1 if info['success'] else 0)
                
                # Train
                if len(agent.memory) > 32:
                    for _ in range(4):
                        agent.replay()
                
                # Track knowledge statistics
                if ep == 10:  # Early complexity
                    initial_complexity = agent.knowledge_graph.compute_structural_complexity()
                
                # Check convergence
                if len(successes) >= 10:
                    recent_rate = np.mean(successes[-10:])
                    if recent_rate >= 0.5 and convergence_ep == episodes:
                        convergence_ep = ep
            
            # Final statistics
            final_stats = agent.get_knowledge_stats()
            final_complexity = final_stats['complexity']
            
            if initial_complexity and initial_complexity > 0:
                complexity_reduction = (initial_complexity - final_complexity) / initial_complexity
            else:
                complexity_reduction = 0
            
            config_results['success_rates'].append(np.mean(successes))
            config_results['convergence_episodes'].append(convergence_ep)
            config_results['knowledge_stats'].append(final_stats)
            config_results['complexity_reduction'].append(complexity_reduction)
            
            print(f"  Trial {trial+1}: Success={np.mean(successes):.3f}, "
                  f"Conv={convergence_ep}, States={final_stats['num_states']}, "
                  f"Complexity reduction={complexity_reduction:.3f}")
        
        results[config['name']] = config_results
    
    return results


def analyze_correct_ged_results(results):
    """Analyze results with focus on structural optimization"""
    
    print("\n" + "="*70)
    print("CORRECT GED IMPLEMENTATION - RESULTS ANALYSIS")
    print("="*70)
    
    # Success rates
    print("\nSuccess Rates:")
    for name, data in results.items():
        mean_success = np.mean(data['success_rates'])
        std_success = np.std(data['success_rates'])
        print(f"  {name}: {mean_success:.3f} ± {std_success:.3f}")
    
    # Structural optimization
    print("\nStructural Optimization (Complexity Reduction):")
    for name, data in results.items():
        mean_reduction = np.mean(data['complexity_reduction'])
        print(f"  {name}: {mean_reduction:.3%} reduction")
    
    # Knowledge statistics
    print("\nKnowledge Graph Statistics:")
    for name, data in results.items():
        avg_states = np.mean([s['num_states'] for s in data['knowledge_stats']])
        avg_transitions = np.mean([s['num_transitions'] for s in data['knowledge_stats']])
        print(f"  {name}: {avg_states:.1f} states, {avg_transitions:.1f} transitions")
    
    # Statistical tests
    print("\nStatistical Significance (vs Baseline):")
    baseline_success = results['Baseline']['success_rates']
    
    for name in ['Correct_Full', 'Correct_IG_Only', 'Correct_GED_Only']:
        if name in results:
            test_success = results[name]['success_rates']
            if len(test_success) == len(baseline_success) > 1:
                t_stat, p_val = stats.ttest_rel(test_success, baseline_success)
                print(f"  {name}: p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
    
    return results


def create_visualization(results, output_dir):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = list(results.keys())
    
    # Success rates
    ax = axes[0, 0]
    means = [np.mean(results[n]['success_rates']) for n in names]
    stds = [np.std(results[n]['success_rates']) for n in names]
    ax.bar(names, means, yerr=stds, capsize=5)
    ax.set_ylabel('Success Rate')
    ax.set_title('Performance Comparison')
    ax.set_ylim(0, 1.1)
    
    # Convergence speed
    ax = axes[0, 1]
    conv_means = [np.mean(results[n]['convergence_episodes']) for n in names]
    ax.bar(names, conv_means)
    ax.set_ylabel('Episodes to Convergence')
    ax.set_title('Learning Speed')
    
    # Complexity reduction
    ax = axes[1, 0]
    reduction_means = [np.mean(results[n]['complexity_reduction']) * 100 for n in names]
    ax.bar(names, reduction_means)
    ax.set_ylabel('Complexity Reduction (%)')
    ax.set_title('Structural Optimization')
    
    # Knowledge graph size
    ax = axes[1, 1]
    states = [np.mean([s['num_states'] for s in results[n]['knowledge_stats']]) for n in names]
    transitions = [np.mean([s['num_transitions'] for s in results[n]['knowledge_stats']]) for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, states, width, label='States')
    ax.bar(x + width/2, transitions, width, label='Transitions')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Knowledge Graph Size')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correct_ged_results.png', dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}")


def main():
    """Run experiment with correct GED implementation"""
    
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_correct_ged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("INTRINSIC MOTIVATION WITH CORRECT GED IMPLEMENTATION")
    print("="*70)
    print("\nGED now measures structural optimization in long-term memory")
    print("High reward when new information (IG) simplifies knowledge structure")
    
    # Run experiment
    results = run_correct_ged_experiment(episodes=100, trials=3)
    
    # Analyze
    results = analyze_correct_ged_results(results)
    
    # Save results
    save_data = {
        'results': {
            name: {
                'success_rate_mean': float(np.mean(data['success_rates'])),
                'success_rate_std': float(np.std(data['success_rates'])),
                'convergence_mean': float(np.mean(data['convergence_episodes'])),
                'complexity_reduction_mean': float(np.mean(data['complexity_reduction'])),
                'avg_knowledge_states': float(np.mean([s['num_states'] for s in data['knowledge_stats']])),
                'avg_knowledge_transitions': float(np.mean([s['num_transitions'] for s in data['knowledge_stats']]))
            }
            for name, data in results.items()
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'episodes': 100,
            'trials': 3,
            'description': 'Correct GED implementation measuring structural optimization'
        }
    }
    
    with open(output_dir / 'correct_ged_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Visualize
    create_visualization(results, output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find best performer
    best = max(results.keys(), key=lambda k: np.mean(results[k]['success_rates']))
    print(f"\nBest performer: {best}")
    
    # Check if structural optimization helps
    full_reduction = np.mean(results['Correct_Full']['complexity_reduction'])
    baseline_reduction = np.mean(results['Baseline']['complexity_reduction'])
    
    if full_reduction > baseline_reduction:
        print(f"\nCorrect Full shows {(full_reduction - baseline_reduction)*100:.1f}% "
              f"more structural optimization than Baseline")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()