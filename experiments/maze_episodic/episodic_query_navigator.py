#!/usr/bin/env python3
"""
Episodic Query-Based Navigator
- Stores movement episodes: (x, y, direction, success, wall/path, visit_count, goal)
- Queries memory to create insight episodes
- Extracts direction from insights
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class EpisodicQueryNavigator:
    """Navigator using query-based episodic memory"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        
        # Episode memory as vectors
        self.episode_vectors = []  # 7D vectors
        self.episode_metadata = []  # Additional info
        
        # Visit tracking
        self.visit_counts = {}
        self.path = [self.position]
        self.wall_hits = 0
        
        # Action mapping
        self.actions = ['up', 'right', 'down', 'left']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def _find_start(self) -> Tuple[int, int]:
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (1, 1)
    
    def _find_goal(self) -> Tuple[int, int]:
        for i in range(self.height-1, -1, -1):
            for j in range(self.width-1, -1, -1):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (self.height-2, self.width-2)
    
    def _update_visit_count(self):
        """Update visit count"""
        pos = self.position
        if pos not in self.visit_counts:
            self.visit_counts[pos] = 0
        self.visit_counts[pos] += 1
    
    def _create_episode_vector(self, x: int, y: int, direction: str, 
                               success: bool, is_path: bool, 
                               visit_count: int, is_goal: bool) -> np.ndarray:
        """Create 7D episode vector"""
        vec = np.zeros(7, dtype=np.float32)
        
        # Position (normalized)
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction (normalized index)
        vec[2] = self.action_to_idx[direction] / 3.0
        
        # Success (binary)
        vec[3] = 1.0 if success else 0.0
        
        # Wall/Path
        vec[4] = 1.0 if is_path else -1.0
        
        # Visit count (log normalized)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # Goal
        vec[6] = 1.0 if is_goal else 0.0
        
        return vec
    
    def _create_query_vector(self) -> np.ndarray:
        """Create query vector for current state"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction: null (0.5 = any direction)
        vec[2] = 0.5
        
        # Want successful moves
        vec[3] = 1.0
        
        # Path/wall: neutral (will match with actual observations)
        vec[4] = 0.0
        
        # Visit count: current state
        vec[5] = np.log1p(self.visit_counts.get((x, y), 0)) / 10.0
        
        # Goal: neutral (0.5)
        vec[6] = 0.5
        
        return vec
    
    def _search_episodes(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search for k most similar episodes"""
        if not self.episode_vectors:
            return []
        
        # Convert to numpy array for similarity computation
        episodes_array = np.array(self.episode_vectors)
        query_2d = query.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_2d, episodes_array)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return indices with scores
        results = [(idx, similarities[idx]) for idx in top_k_indices]
        return results
    
    def _create_insight_episode(self, retrieved_episodes: List[Tuple[int, float]]) -> np.ndarray:
        """Create insight episode from retrieved memories"""
        if not retrieved_episodes:
            # No memory - return neutral vector
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Weighted average of retrieved episodes
        insight = np.zeros(7, dtype=np.float32)
        total_weight = 0
        
        for idx, score in retrieved_episodes:
            if idx < len(self.episode_vectors):
                vec = self.episode_vectors[idx]
                metadata = self.episode_metadata[idx]
                
                # Weight by similarity score and success
                weight = score
                if metadata['success']:
                    weight *= 2.0  # Boost successful episodes
                
                insight += vec * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            insight = insight / total_weight
        
        return insight
    
    def _extract_direction_probabilities(self, insight: np.ndarray) -> np.ndarray:
        """Extract direction probabilities from insight episode"""
        # Direction component is at index 2 (normalized 0-1)
        direction_value = insight[2] * 3.0  # Denormalize to 0-3
        
        # Success component affects confidence
        success_confidence = insight[3]
        
        # Create probability distribution
        probs = np.ones(4) * 0.1  # Base probability
        
        # Add weight to indicated direction
        if 0 <= direction_value <= 3:
            # Soft assignment to nearby directions
            main_dir = int(round(direction_value))
            remainder = direction_value - main_dir
            
            if main_dir < 4:
                probs[main_dir] += 0.6 * success_confidence
                
                # Soft assignment to adjacent directions
                if remainder > 0 and main_dir < 3:
                    probs[main_dir + 1] += 0.2 * remainder * success_confidence
                elif remainder < 0 and main_dir > 0:
                    probs[main_dir - 1] += 0.2 * abs(remainder) * success_confidence
        
        # Normalize to sum to 1
        probs = probs / np.sum(probs)
        
        return probs
    
    def _add_visual_observations(self):
        """Add visual observations as episodes"""
        x, y = self.position
        visit_count = self.visit_counts.get((x, y), 0)
        
        for action in self.actions:
            dx, dy = self.action_deltas[action]
            nx, ny = x + dx, y + dy
            
            # Check if path or wall
            is_path = False
            if 0 <= nx < self.height and 0 <= ny < self.width:
                is_path = (self.maze[nx, ny] == 0)
            
            # Create visual episode (not actual movement)
            vec = self._create_episode_vector(
                x, y, action, 
                success=False,  # Not a movement yet
                is_path=is_path,
                visit_count=visit_count,
                is_goal=(nx, ny) == self.goal
            )
            
            self.episode_vectors.append(vec)
            self.episode_metadata.append({
                'type': 'visual',
                'position': (x, y),
                'action': action,
                'success': False,
                'is_path': is_path
            })
    
    def get_action(self) -> str:
        """Get action using query-based memory search"""
        self._update_visit_count()
        
        # Add visual observations
        self._add_visual_observations()
        
        # Create query
        query = self._create_query_vector()
        
        # Search memory (retrieve k episodes)
        k = min(20, len(self.episode_vectors))
        retrieved = self._search_episodes(query, k=k)
        
        # Create insight episode
        insight = self._create_insight_episode(retrieved)
        
        # Extract direction probabilities
        probs = self._extract_direction_probabilities(insight)
        
        # Sample action
        return np.random.choice(self.actions, p=probs)
    
    def move(self, action: str) -> bool:
        """Execute action and store episode"""
        if action not in self.actions:
            return False
        
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
        # Check if valid move
        success = False
        is_path = False
        
        if 0 <= new_x < self.height and 0 <= new_y < self.width:
            is_path = (self.maze[new_x, new_y] == 0)
            if is_path:
                self.position = (new_x, new_y)
                self.path.append(self.position)
                success = True
        
        if not success:
            self.wall_hits += 1
        
        # Store movement episode
        visit_count = self.visit_counts.get((x, y), 0)
        vec = self._create_episode_vector(
            x, y, action,
            success=success,
            is_path=is_path,
            visit_count=visit_count,
            is_goal=(new_x, new_y) == self.goal if success else False
        )
        
        self.episode_vectors.append(vec)
        self.episode_metadata.append({
            'type': 'movement',
            'position': (x, y),
            'action': action,
            'success': success,
            'is_path': is_path
        })
        
        return success
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate using episodic query memory"""
        start_time = time.time()
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                
                print(f"\n🎉 SUCCESS with episodic query memory!")
                print(f"  Steps: {step}")
                print(f"  Episodes: {len(self.episode_vectors)}")
                print(f"  Wall hits: {self.wall_hits}")
                
                return {
                    'success': True,
                    'steps': step,
                    'episodes': len(self.episode_vectors),
                    'wall_hits': self.wall_hits,
                    'path': self.path,
                    'visit_counts': self.visit_counts,
                    'total_time': total_time
                }
            
            # Get and execute action
            action = self.get_action()
            self.move(action)
            
            # Progress report
            if step % 100 == 0 and step > 0:
                dist = abs(self.position[0]-self.goal[0]) + abs(self.position[1]-self.goal[1])
                hit_rate = self.wall_hits / step * 100
                max_visits = max(self.visit_counts.values()) if self.visit_counts else 0
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"wall_hits={self.wall_hits} ({hit_rate:.1f}%), "
                      f"episodes={len(self.episode_vectors)}, max_visits={max_visits}")
        
        total_time = time.time() - start_time
        
        return {
            'success': False,
            'steps': max_steps,
            'episodes': len(self.episode_vectors),
            'wall_hits': self.wall_hits,
            'path': self.path,
            'visit_counts': self.visit_counts,
            'total_time': total_time
        }


def test_episodic_query():
    """Test episodic query navigator"""
    # Simple maze
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("Episodic Query Navigation Test")
    print("=" * 40)
    print("Process:")
    print("1. Query memory with current state")
    print("2. Create insight episode from retrieved memories")
    print("3. Extract direction from insight")
    print("4. Normalize to action probabilities")
    print("=" * 40)
    print("\nMaze (0=path, 1=wall):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    print(f"Start: (0,0), Goal: (4,4)")
    print("-" * 40)
    
    navigator = EpisodicQueryNavigator(maze)
    result = navigator.navigate(max_steps=1000)
    
    if result['success']:
        print(f"\n✅ Solved in {result['steps']} steps!")
        print(f"Path length: {len(result['path'])}")
        
        # Show memory usage
        movement_episodes = sum(1 for m in navigator.episode_metadata if m['type'] == 'movement')
        visual_episodes = sum(1 for m in navigator.episode_metadata if m['type'] == 'visual')
        print(f"\nMemory usage:")
        print(f"  Movement episodes: {movement_episodes}")
        print(f"  Visual episodes: {visual_episodes}")
        
        # Show most visited
        sorted_visits = sorted(result['visit_counts'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        print(f"\nMost visited positions:")
        for pos, count in sorted_visits:
            print(f"  {pos}: {count} visits")
    else:
        print(f"\n❌ Failed after {result['steps']} steps")
    
    return result


if __name__ == "__main__":
    test_episodic_query()