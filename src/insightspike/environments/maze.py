"""Maze environment for InsightSpike experiments."""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .complex_maze import ComplexMazeGenerator
from .proper_maze_generator import ProperMazeGenerator


class CellType(Enum):
    """Types of cells in the maze."""
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3


@dataclass
class MazeObservation:
    """Observation from the maze environment."""
    position: Tuple[int, int]
    cell_type: CellType
    num_paths: int  # Number of possible moves
    possible_moves: List[int]  # Available actions
    hit_wall: bool = False
    is_junction: bool = False
    is_dead_end: bool = False
    is_goal: bool = False
    
    def to_features(self) -> Dict[str, Any]:
        """Convert observation to feature dictionary for memory."""
        return {
            'type': self.get_location_type(),
            'num_paths': self.num_paths,
            'position': self.position,
            'distinctive_features': self.get_distinctive_features()
        }
    
    def get_location_type(self) -> str:
        """Determine the type of location."""
        if self.is_goal:
            return 'goal'
        elif self.hit_wall:
            return 'wall'
        elif self.is_dead_end:
            return 'dead_end'
        elif self.is_junction:
            return 'junction'
        else:
            return 'corridor'
    
    def get_distinctive_features(self) -> List[str]:
        """Get distinctive features of this location."""
        features = []
        if self.num_paths >= 3:
            features.append(f'{self.num_paths}_way_junction')
        if self.hit_wall:
            features.append('wall_collision')
        return features


class SimpleMaze:
    """Simple 2D maze environment for InsightSpike experiments."""
    
    # Action mappings
    ACTIONS = {
        0: (-1, 0),  # Up
        1: (0, 1),   # Right
        2: (1, 0),   # Down
        3: (0, -1)   # Left
    }
    
    ACTION_NAMES = {
        0: 'up',
        1: 'right',
        2: 'down',
        3: 'left'
    }
    
    def __init__(self, size: Tuple[int, int] = (20, 20), 
                 maze_layout: Optional[np.ndarray] = None,
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 maze_type: str = 'simple'):
        """Initialize maze environment.
        
        Args:
            size: Size of the maze (height, width)
            maze_layout: Optional pre-defined maze layout
            start_pos: Starting position (default: top-left)
            goal_pos: Goal position (default: bottom-right)
            maze_type: Type of maze ('simple', 'complex', 'spiral', 'rooms')
        """
        self.size = size
        self.height, self.width = size
        
        # Initialize maze
        if maze_layout is not None:
            self.grid = maze_layout.copy()
        else:
            if maze_type == 'simple':
                self.grid = self._generate_simple_maze()
            elif maze_type == 'complex':
                self.grid = ComplexMazeGenerator.generate_maze(size, complexity=0.75, density=0.75)
            elif maze_type == 'spiral':
                self.grid = ComplexMazeGenerator.generate_spiral_maze(size)
            elif maze_type == 'rooms':
                self.grid = ComplexMazeGenerator.generate_rooms_maze(size, num_rooms=6)
            elif maze_type == 'dfs':
                self.grid = ProperMazeGenerator.generate_dfs_maze(size)
            elif maze_type == 'kruskal':
                self.grid = ProperMazeGenerator.generate_kruskal_maze(size)
            elif maze_type == 'prim':
                self.grid = ProperMazeGenerator.generate_prim_maze(size)
            elif maze_type == 'dfs_loops':
                base_maze = ProperMazeGenerator.generate_dfs_maze(size)
                self.grid = ProperMazeGenerator.add_loops(base_maze, loop_probability=0.1)
            else:
                self.grid = self._generate_simple_maze()
        
        # Set positions
        self.start_pos = start_pos or (1, 1)
        self.goal_pos = goal_pos or (self.height - 2, self.width - 2)
        self.agent_pos = self.start_pos
        
        # Mark start and goal
        self.grid[self.start_pos] = CellType.START.value
        self.grid[self.goal_pos] = CellType.GOAL.value
        
        # Episode tracking
        self.steps = 0
        self.trajectory = [self.start_pos]
        self.wall_hits = 0
        
    def _generate_simple_maze(self) -> np.ndarray:
        """Generate a simple maze with some walls."""
        # Start with empty maze
        maze = np.zeros((self.height, self.width), dtype=int)
        
        # Add border walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value
        
        # Create a more complex maze with multiple paths
        # Vertical walls
        maze[2:8, 5] = CellType.WALL.value
        maze[12:18, 5] = CellType.WALL.value
        
        maze[2:10, 10] = CellType.WALL.value
        maze[14:18, 10] = CellType.WALL.value
        
        maze[4:16, 15] = CellType.WALL.value
        
        # Horizontal walls
        maze[5, 2:5] = CellType.WALL.value
        maze[5, 6:10] = CellType.WALL.value
        maze[5, 11:15] = CellType.WALL.value
        
        maze[10, 2:10] = CellType.WALL.value
        maze[10, 15:18] = CellType.WALL.value
        
        maze[15, 2:5] = CellType.WALL.value
        maze[15, 6:10] = CellType.WALL.value
        maze[15, 11:15] = CellType.WALL.value
        
        # Create some dead ends and interesting paths
        maze[3, 7:9] = CellType.WALL.value
        maze[7, 12:14] = CellType.WALL.value
        maze[13, 3:5] = CellType.WALL.value
        maze[17, 7:9] = CellType.WALL.value
        
        # Ensure there's at least one clear path
        maze[10, 10] = CellType.EMPTY.value  # Central junction
        maze[5, 15] = CellType.EMPTY.value   # Upper right passage
        maze[15, 5] = CellType.EMPTY.value   # Lower left passage
        
        return maze
    
    def reset(self) -> MazeObservation:
        """Reset the environment."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.trajectory = [self.start_pos]
        self.wall_hits = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[MazeObservation, float, bool, Dict]:
        """Take an action in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: Current observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate new position
        delta = self.ACTIONS[action]
        new_pos = (
            self.agent_pos[0] + delta[0],
            self.agent_pos[1] + delta[1]
        )
        
        # Check if hitting wall
        hit_wall = False
        done = False  # Initialize done
        reward = 0.0  # No extrinsic reward/penalty by default
        if self._is_wall(new_pos):
            hit_wall = True
            self.wall_hits += 1
        else:
            self.agent_pos = new_pos
            self.trajectory.append(new_pos)
            
            # Check if reached goal
            if new_pos == self.goal_pos:
                done = True
            else:
                done = False
        
        self.steps += 1
        
        # Get observation
        obs = self._get_observation(hit_wall=hit_wall)
        
        # Additional info
        info = {
            'steps': self.steps,
            'wall_hits': self.wall_hits,
            'trajectory_length': len(self.trajectory),
            'position': self.agent_pos
        }
        
        # Episode timeout
        if self.steps >= 1000:
            done = True
            info['timeout'] = True
        
        return obs, reward, done, info
    
    def _get_observation(self, hit_wall: bool = False) -> MazeObservation:
        """Get current observation."""
        # Count possible moves
        possible_moves = []
        for action, delta in self.ACTIONS.items():
            next_pos = (
                self.agent_pos[0] + delta[0],
                self.agent_pos[1] + delta[1]
            )
            if not self._is_wall(next_pos):
                possible_moves.append(action)
        
        num_paths = len(possible_moves)
        
        # Determine special properties
        is_junction = num_paths >= 3
        is_dead_end = num_paths == 1
        is_goal = self.agent_pos == self.goal_pos
        
        return MazeObservation(
            position=self.agent_pos,
            cell_type=self._get_cell_type(self.agent_pos),
            num_paths=num_paths,
            possible_moves=possible_moves,
            hit_wall=hit_wall,
            is_junction=is_junction,
            is_dead_end=is_dead_end,
            is_goal=is_goal
        )
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall."""
        row, col = pos
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] == CellType.WALL.value
        return True  # Out of bounds is considered wall
    
    def _get_cell_type(self, pos: Tuple[int, int]) -> CellType:
        """Get cell type at position."""
        if pos == self.goal_pos:
            return CellType.GOAL
        elif pos == self.start_pos:
            return CellType.START
        elif self._is_wall(pos):
            return CellType.WALL
        else:
            return CellType.EMPTY
    
    def get_state_vector(self) -> np.ndarray:
        """Get state as vector for compatibility."""
        # Simple state: normalized position + one-hot cell type
        state = np.zeros(6)  # 2 for position, 4 for cell type
        state[0] = self.agent_pos[0] / self.height
        state[1] = self.agent_pos[1] / self.width
        state[2 + self._get_cell_type(self.agent_pos).value] = 1.0
        return state
    
    def render(self, mode: str = 'ascii') -> Optional[str]:
        """Render the maze."""
        if mode == 'ascii':
            # Create display grid
            display = self.grid.copy()
            
            # Mark agent position
            display[self.agent_pos] = 4  # Special marker for agent
            
            # Convert to string
            symbols = {
                CellType.EMPTY.value: ' ',
                CellType.WALL.value: '#',
                CellType.START.value: 'S',
                CellType.GOAL.value: 'G',
                4: '@'  # Agent
            }
            
            lines = []
            for row in display:
                line = ''.join(symbols.get(cell, '?') for cell in row)
                lines.append(line)
            
            return '\n'.join(lines)
        
        return None
