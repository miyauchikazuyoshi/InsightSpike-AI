"""Proper maze generation algorithms."""

import numpy as np
import random
from typing import Tuple, List, Set


class ProperMazeGenerator:
    """Generate proper mazes using various algorithms."""
    
    @staticmethod
    def generate_dfs_maze(size: Tuple[int, int], seed: int = None) -> np.ndarray:
        """Generate maze using depth-first search (穴掘り法).
        
        This creates a perfect maze with exactly one solution.
        
        Args:
            size: (height, width) - must be odd numbers
            seed: Random seed
            
        Returns:
            2D numpy array where 1 = wall, 0 = path
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        height, width = size
        # Ensure odd dimensions for proper maze
        if height % 2 == 0:
            height += 1
        if width % 2 == 0:
            width += 1
            
        # Initialize maze filled with walls
        maze = np.ones((height, width), dtype=int)
        
        # Directions: up, right, down, left
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        
        def is_valid(y, x):
            return 0 < y < height - 1 and 0 < x < width - 1
        
        def carve_path(y, x):
            """Carve path using DFS."""
            maze[y, x] = 0  # Mark current cell as path
            
            # Randomize directions
            random.shuffle(directions)
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                wall_y, wall_x = y + dy // 2, x + dx // 2
                
                # If the neighbor hasn't been visited yet
                if is_valid(ny, nx) and maze[ny, nx] == 1:
                    maze[wall_y, wall_x] = 0  # Remove wall between cells
                    carve_path(ny, nx)  # Recursively carve from neighbor
        
        # Start carving from (1, 1)
        carve_path(1, 1)
        
        # Ensure start and goal are accessible
        maze[1, 1] = 0
        maze[height-2, width-2] = 0
        
        return maze
    
    @staticmethod
    def generate_kruskal_maze(size: Tuple[int, int], seed: int = None) -> np.ndarray:
        """Generate maze using Kruskal's algorithm.
        
        Creates more branching paths than DFS.
        
        Args:
            size: (height, width) - must be odd numbers
            seed: Random seed
            
        Returns:
            2D numpy array where 1 = wall, 0 = path
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        height, width = size
        # Ensure odd dimensions
        if height % 2 == 0:
            height += 1
        if width % 2 == 0:
            width += 1
            
        # Initialize maze
        maze = np.ones((height, width), dtype=int)
        
        # Create cells and walls lists
        cells = []
        walls = []
        
        # Initialize cells (odd coordinates)
        for y in range(1, height, 2):
            for x in range(1, width, 2):
                maze[y, x] = 0
                cells.append((y, x))
        
        # Create walls between adjacent cells
        for y in range(1, height, 2):
            for x in range(1, width, 2):
                if x + 2 < width:
                    walls.append(((y, x), (y, x + 2), (y, x + 1)))
                if y + 2 < height:
                    walls.append(((y, x), (y + 2, x), (y + 1, x)))
        
        # Shuffle walls
        random.shuffle(walls)
        
        # Union-Find structure
        parent = {cell: cell for cell in cells}
        
        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]
        
        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                parent[root1] = root2
                return True
            return False
        
        # Process walls
        for cell1, cell2, wall in walls:
            if union(cell1, cell2):
                maze[wall[0], wall[1]] = 0
        
        return maze
    
    @staticmethod
    def generate_prim_maze(size: Tuple[int, int], seed: int = None) -> np.ndarray:
        """Generate maze using Prim's algorithm.
        
        Creates mazes with a different texture than DFS.
        
        Args:
            size: (height, width) - must be odd numbers
            seed: Random seed
            
        Returns:
            2D numpy array where 1 = wall, 0 = path
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        height, width = size
        # Ensure odd dimensions
        if height % 2 == 0:
            height += 1
        if width % 2 == 0:
            width += 1
            
        # Initialize maze
        maze = np.ones((height, width), dtype=int)
        
        # Start with a random cell
        current = (1, 1)
        maze[current[0], current[1]] = 0
        
        # Walls list
        walls = []
        
        # Add walls of starting cell
        for dy, dx in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 < ny < height - 1 and 0 < nx < width - 1:
                walls.append(((current[0], current[1]), (ny, nx), 
                            (current[0] + dy // 2, current[1] + dx // 2)))
        
        # Process walls
        while walls:
            # Pick random wall
            idx = random.randint(0, len(walls) - 1)
            cell1, cell2, wall = walls.pop(idx)
            
            # If cell2 hasn't been visited
            if maze[cell2[0], cell2[1]] == 1:
                maze[cell2[0], cell2[1]] = 0
                maze[wall[0], wall[1]] = 0
                
                # Add cell2's walls
                for dy, dx in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                    ny, nx = cell2[0] + dy, cell2[1] + dx
                    if (0 < ny < height - 1 and 0 < nx < width - 1 and 
                        maze[ny, nx] == 1):
                        walls.append((cell2, (ny, nx), 
                                    (cell2[0] + dy // 2, cell2[1] + dx // 2)))
        
        return maze
    
    @staticmethod
    def add_loops(maze: np.ndarray, loop_probability: float = 0.05) -> np.ndarray:
        """Add loops to a perfect maze by removing some walls.
        
        Args:
            maze: Perfect maze
            loop_probability: Probability of removing each wall
            
        Returns:
            Maze with some loops
        """
        height, width = maze.shape
        maze_with_loops = maze.copy()
        
        # Find all walls that can be removed
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze[y, x] == 1:  # If it's a wall
                    # Check if removing it would connect two paths
                    neighbors = []
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0:
                            neighbors.append((ny, nx))
                    
                    # If wall is between exactly 2 paths, consider removing it
                    if len(neighbors) == 2:
                        if random.random() < loop_probability:
                            maze_with_loops[y, x] = 0
        
        return maze_with_loops