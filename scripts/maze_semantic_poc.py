#!/usr/bin/env python3
"""
Maze Semantic Space PoC
=======================
Proof of Concept for "Semantic Space Cultivation" and "Deductive NN Teacher" loop.

Goal:
- Use a simple Maze environment.
- Implement "Structure Learning":
    - Wake: Explore and collect traces.
    - Sleep: Classify traces into Positive (Good Plan) / Negative (Revisit/Collision).
    - Train: A simple NN (MLP) to predict "Good Move" from (State, Action) embeddings.
    - Next Wake: Use the NN to guide exploration.

Dependencies: sklearn, numpy, matplotlib (optional for viz)
"""

import sys
import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Try importing, if fails we might need to mock or adjust path
try:
    from insightspike.maze_experimental.environments.complex_maze import ComplexMazeGenerator
except ImportError:
    # Fallback if src structure is different or complex_maze depends on other things
    print("WARNING: Could not import ComplexMazeGenerator from src. Using local mock.")
    class ComplexMazeGenerator:
        @staticmethod
        def generate_maze(size, seed=None):
            h, w = size
            maze = np.zeros((h, w), dtype=int)
            # Simple boundary walls
            maze[0, :] = 1
            maze[-1, :] = 1
            maze[:, 0] = 1
            maze[:, -1] = 1
            # Some random walls
            np.random.seed(seed)
            for _ in range(int(h*w*0.2)):
                y, x = np.random.randint(1, h-1), np.random.randint(1, w-1)
                maze[y, x] = 1
            maze[1, 1] = 0
            maze[h-2, w-2] = 0
            return maze

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MAZE_SIZE = (7, 7)
N_EPISODES = 20
MEMORY_SIZE = 1000
TRAIN_FREQ = 1  # Train every episode

@dataclass
class Experience:
    state_vec: np.ndarray
    action_vec: np.ndarray
    label: int  # 1 (Positive), 0 (Negative/Neutral)

class SemanticNavigator:
    def __init__(self, maze_shape):
        self.maze_shape = maze_shape
        self.h, self.w = maze_shape
        
        # Structure Predictor (The "Intuition")
        # Input: State(2) + Action(2) = 4 dim
        # Output: Probability of being "Good"
        self.brain = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            learning_rate_init=0.01,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.memory: List[Experience] = []
        
    def embed_state(self, pos: Tuple[int, int]) -> np.ndarray:
        """Embed discrete state (x,y) into continuous semantic space."""
        # Simple normalized coordinates for POC
        return np.array([pos[1] / self.w, pos[0] / self.h]) # x, y normalized

    def embed_action(self, action_delta: Tuple[int, int]) -> np.ndarray:
        """Embed discrete action (dx, dy) into vector."""
        return np.array(action_delta)

    def decide(self, current_pos, valid_moves, epsilon=0.1):
        """Decide next move using Brain + Exploration."""
        if random.random() < epsilon or not self.is_trained:
            return random.choice(valid_moves)
        
        # Rate all valid moves
        s_vec = self.embed_state(current_pos)
        scores = []
        
        for move in valid_moves:
            a_vec = self.embed_action(move)
            # Input: [s_x, s_y, a_dx, a_dy]
            X = np.concatenate([s_vec, a_vec]).reshape(1, -1)
            # X = self.scaler.transform(X) # Skipping scaler for single inference for speed
            prob = self.brain.predict_proba(X)[0][1] # Prob of class 1 (Good)
            scores.append((prob, move))
            
        # Greedy choice based on brain
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def sleep(self):
        """Review memories and train the brain."""
        if len(self.memory) < 10:
            return
            
        # Prepare training data
        X = []
        y = []
        
        # Weight recent memories more? Or just shuffle.
        batch = self.memory[-MEMORY_SIZE:]
        
        for exp in batch:
            input_vec = np.concatenate([exp.state_vec, exp.action_vec])
            X.append(input_vec)
            y.append(exp.label)
            
        X = np.array(X)
        y = np.array(y)
        
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Sleep Data Balance: {dict(zip(unique, counts))}")

        # Train
        # self.scaler.partial_fit(X) if online... for generic MLP we fit all
        self.brain.fit(X, y)
        self.is_trained = True
        logger.info(f"Sleep Complete. Trained on {len(X)} samples. Train Score: {self.brain.score(X, y):.3f}")

def run_poc():
    # 1. Generate Maze
    maze = ComplexMazeGenerator.generate_maze(MAZE_SIZE, seed=42)
    start_pos = (1, 1)
    goal_pos = (MAZE_SIZE[0]-2, MAZE_SIZE[1]-2)
    
    print("Maze generated:")
    # Simple ASCII print
    for y in range(MAZE_SIZE[0]):
        row = ""
        for x in range(MAZE_SIZE[1]):
            if (y, x) == start_pos: row += "S"
            elif (y, x) == goal_pos: row += "G"
            else: row += "#" if maze[y,x] == 1 else "."
        print(row)

    navigator = SemanticNavigator(MAZE_SIZE)
    
    history_steps = []

    for episode in range(N_EPISODES):
        logger.info(f"--- Episode {episode+1} ---")
        
        pos = start_pos
        path = [pos]
        steps = 0
        max_steps = MAZE_SIZE[0] * MAZE_SIZE[1] * 2
        
        episode_memory = [] # (s, a, next_s)
        
        # Wake: Explore
        while pos != goal_pos and steps < max_steps:
            # Get valid moves
            moves = []
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]: # y, x
                ny, nx = pos[0]+dy, pos[1]+dx
                if 0 <= ny < MAZE_SIZE[0] and 0 <= nx < MAZE_SIZE[1]:
                    if maze[ny, nx] == 0:
                        moves.append((dy, dx))
            
            if not moves: break # trapped
            
            # Decision
            # Decaying epsilon
            epsilon = max(0.05, 0.5 - episode * 0.05)
            move = navigator.decide(pos, moves, epsilon=epsilon)
            
            # Execute
            new_pos = (pos[0]+move[0], pos[1]+move[1])
            episode_memory.append((pos, move, new_pos))
            pos = new_pos
            path.append(pos)
            steps += 1
        
        success = (pos == goal_pos)
        logger.info(f"Episode finished. Steps: {steps}. Success: {success}")
        history_steps.append(steps)
        
        # Sleep: Labeling (Structure Learning)
        # 1. Identify "Negative" behaviors: Revisits (loops)
        visited_counts = {}
        for p in path:
            visited_counts[p] = visited_counts.get(p, 0) + 1
            
        current_episode_exps = []
        path_set = set(path)
        
        for s, a, next_s in episode_memory:
            s_vec = navigator.embed_state(s)
            a_vec = navigator.embed_action(a)
            
            label = 0 # Neutral/Unknown
            
            # Rule 1: Revisit is Bad (Negative)
            if visited_counts[next_s] > 1:
                label = 0 # Or -1 if using regression, but we utilize 0/1 class
            
            # Rule 2: Progress toward goal is Good (Positive)
            # If successful, the whole non-looping path is good?
            # Or just local gradient?
            # Let's use simple Euclidean heuristic for teaching
            dist_curr = abs(s[0]-goal_pos[0]) + abs(s[1]-goal_pos[1])
            dist_next = abs(next_s[0]-goal_pos[0]) + abs(next_s[1]-goal_pos[1])
            
            if dist_next < dist_curr:
                label = 1
                
            # If success, boost path
            if success and next_s in path_set and visited_counts[next_s] == 1:
                 label = 1
                 
            current_episode_exps.append(Experience(s_vec, a_vec, label))
            
        navigator.memory.extend(current_episode_exps)
        navigator.sleep()

    print("\nTraining Results:")
    print(f"Steps per episode: {history_steps}")
    # Ideally should decrease

if __name__ == "__main__":
    run_poc()
