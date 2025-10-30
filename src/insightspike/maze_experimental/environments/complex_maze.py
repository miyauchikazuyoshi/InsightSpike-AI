"""Complex maze generator for more challenging navigation."""

import numpy as np
from typing import Tuple, List
import random


class ComplexMazeGenerator:
    """Generate complex mazes using recursive backtracking algorithm."""
    
    @staticmethod
    def generate_maze(size: Tuple[int, int], complexity: float = 0.75, 
                     density: float = 0.75, seed: int = None) -> np.ndarray:
        """Generate a complex maze using recursive division.
        
        Args:
            size: (height, width) of the maze
            complexity: Determines how complex the maze is (0-1)
            density: Determines how dense the walls are (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            2D numpy array where 1 = wall, 0 = empty
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        height, width = size
        
        # Build outer walls
        maze = np.zeros((height, width), dtype=int)
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        # Generate maze using recursive division
        def divide(x1, y1, x2, y2):
            """Recursively divide the space with walls."""
            # Check if space is too small to divide
            if x2 - x1 < 4 or y2 - y1 < 4:
                return
            
            # Choose orientation based on space shape
            if x2 - x1 > y2 - y1:
                horizontal = False
            elif y2 - y1 > x2 - x1:
                horizontal = True
            else:
                horizontal = random.random() < 0.5
            
            if horizontal:
                # Add horizontal wall
                wall_y = random.randint(y1 + 2, y2 - 2)
                if wall_y % 2 == 0:  # Ensure odd position for better maze
                    wall_y += 1 if wall_y < y2 - 2 else -1
                
                # Create wall with a passage
                maze[wall_y, x1+1:x2] = 1
                passage_x = random.randint(x1 + 1, x2 - 1)
                if passage_x % 2 == 0:
                    passage_x += 1 if passage_x < x2 - 1 else -1
                maze[wall_y, passage_x] = 0
                
                # Recursively divide the two spaces
                if random.random() < complexity:
                    divide(x1, y1, x2, wall_y - 1)
                    divide(x1, wall_y + 1, x2, y2)
            else:
                # Add vertical wall
                wall_x = random.randint(x1 + 2, x2 - 2)
                if wall_x % 2 == 0:
                    wall_x += 1 if wall_x < x2 - 2 else -1
                
                # Create wall with a passage
                maze[y1+1:y2, wall_x] = 1
                passage_y = random.randint(y1 + 1, y2 - 1)
                if passage_y % 2 == 0:
                    passage_y += 1 if passage_y < y2 - 1 else -1
                maze[passage_y, wall_x] = 0
                
                # Recursively divide the two spaces
                if random.random() < complexity:
                    divide(x1, y1, wall_x - 1, y2)
                    divide(wall_x + 1, y1, x2, y2)
        
        # Start recursive division
        divide(0, 0, width - 1, height - 1)
        
        # Add some random obstacles based on density
        for _ in range(int(height * width * density * 0.05)):
            x = random.randint(2, width - 3)
            y = random.randint(2, height - 3)
            if maze[y, x] == 0 and not (x == 1 and y == 1) and not (x == width-2 and y == height-2):
                # Add small wall segments
                direction = random.choice(['h', 'v', 'cross', 'corner'])
                if direction == 'h':
                    maze[y, max(1, x-1):min(width-1, x+2)] = 1
                elif direction == 'v':
                    maze[max(1, y-1):min(height-1, y+2), x] = 1
                elif direction == 'cross':
                    maze[y, x] = 1
                    if y > 1: maze[y-1, x] = 1
                    if y < height-2: maze[y+1, x] = 1
                    if x > 1: maze[y, x-1] = 1
                    if x < width-2: maze[y, x+1] = 1
                else:  # corner
                    maze[y:min(height-1, y+2), x:min(width-1, x+2)] = 1
        
        # Ensure start and goal are clear
        maze[1, 1] = 0
        maze[height-2, width-2] = 0
        
        # Create some guaranteed passages near start and goal
        maze[1, 2] = 0
        maze[2, 1] = 0
        maze[height-2, width-3] = 0
        maze[height-3, width-2] = 0
        
        return maze
    
    @staticmethod
    def generate_spiral_maze(size: Tuple[int, int]) -> np.ndarray:
        """Generate a spiral maze pattern."""
        height, width = size
        maze = np.ones((height, width), dtype=int)
        
        # Create spiral pattern
        x, y = 1, 1
        dx, dy = 1, 0
        steps = 0
        max_steps = height * width
        
        while steps < max_steps and 0 < x < width-1 and 0 < y < height-1:
            maze[y, x] = 0
            
            # Check if we need to turn
            next_x = x + dx
            next_y = y + dy
            
            if (next_x <= 0 or next_x >= width-1 or 
                next_y <= 0 or next_y >= height-1 or 
                (dx == 1 and maze[y, next_x+1] == 0) or
                (dx == -1 and maze[y, next_x-1] == 0) or
                (dy == 1 and maze[next_y+1, x] == 0) or
                (dy == -1 and maze[next_y-1, x] == 0)):
                # Turn right
                dx, dy = -dy, dx
            
            x += dx
            y += dy
            steps += 1
            
            # Add some breaks in the spiral
            if steps % 20 == 0 and random.random() < 0.3:
                break_x = random.randint(2, width-3)
                break_y = random.randint(2, height-3)
                maze[break_y, break_x] = 0
        
        # Ensure goal is reachable
        maze[height-2, width-2] = 0
        
        return maze
    
    @staticmethod
    def generate_rooms_maze(size: Tuple[int, int], num_rooms: int = 6) -> np.ndarray:
        """Generate a maze with interconnected rooms."""
        height, width = size
        maze = np.ones((height, width), dtype=int)
        
        # Create rooms
        rooms = []
        for _ in range(num_rooms):
            room_w = random.randint(4, 8)
            room_h = random.randint(4, 8)
            room_x = random.randint(1, width - room_w - 1)
            room_y = random.randint(1, height - room_h - 1)
            
            # Clear room interior
            maze[room_y:room_y+room_h, room_x:room_x+room_w] = 0
            rooms.append((room_x, room_y, room_w, room_h))
            
            # Add doors
            for _ in range(random.randint(1, 3)):
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top' and room_y > 1:
                    door_x = room_x + random.randint(1, room_w-2)
                    maze[room_y-1, door_x] = 0
                elif side == 'bottom' and room_y + room_h < height - 1:
                    door_x = room_x + random.randint(1, room_w-2)
                    maze[room_y+room_h, door_x] = 0
                elif side == 'left' and room_x > 1:
                    door_y = room_y + random.randint(1, room_h-2)
                    maze[door_y, room_x-1] = 0
                elif side == 'right' and room_x + room_w < width - 1:
                    door_y = room_y + random.randint(1, room_h-2)
                    maze[door_y, room_x+room_w] = 0
        
        # Create corridors between rooms
        for i in range(len(rooms) - 1):
            x1, y1, w1, h1 = rooms[i]
            x2, y2, w2, h2 = rooms[i + 1]
            
            # Connect centers
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            
            # Create L-shaped corridor
            if random.random() < 0.5:
                # Horizontal first
                maze[cy1, min(cx1, cx2):max(cx1, cx2)+1] = 0
                maze[min(cy1, cy2):max(cy1, cy2)+1, cx2] = 0
            else:
                # Vertical first
                maze[min(cy1, cy2):max(cy1, cy2)+1, cx1] = 0
                maze[cy2, min(cx1, cx2):max(cx1, cx2)+1] = 0
        
        # Ensure start and goal are clear
        maze[1, 1] = 0
        maze[height-2, width-2] = 0
        
        return maze