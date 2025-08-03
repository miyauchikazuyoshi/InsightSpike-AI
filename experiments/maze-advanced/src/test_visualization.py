#!/usr/bin/env python3
"""
Test maze agent with visualization
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # バックエンドを非インタラクティブに設定
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from maze_agent_wrapper import MazeAgentWrapper

def main():
    """メイン実行関数"""
    # テスト用の迷路（5x5）
    test_maze = {
        "type": "maze",
        "maze": [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        "start": (0, 0),
        "goal": (4, 4)
    }
    
    # MazeAgentWrapperを作成
    agent = MazeAgentWrapper()
    
    # 迷路を解く（ビジュアライゼーション有効）
    print("Starting maze solving with visualization...")
    result = agent.solve_maze(test_maze, visualize=True)
    
    # 結果を表示
    print("\n=== Maze Solving Result ===")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Unique positions visited: {result['unique_positions']}")
    print(f"Efficiency: {result['unique_positions'] / result['steps']:.2%}")
    
    # 最終的な図を保存
    plt.savefig('experiments/maze-agent-integration/results/maze_solution.png')
    print("\nVisualization saved to results/maze_solution.png")
    
    # アニメーションGIFを作成（仮の実装）
    if hasattr(agent.visualizer, 'save_animation'):
        agent.visualizer.save_animation('experiments/maze-agent-integration/results/maze_solving.gif')
        print("Animation saved to results/maze_solving.gif")

if __name__ == "__main__":
    main()