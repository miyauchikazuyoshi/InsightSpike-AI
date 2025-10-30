#!/usr/bin/env python3
"""
より複雑な袋小路でのgeDIG値観察（ゴールを遠くに配置）
"""

import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def create_complex_deadend_maze():
    """複雑な袋小路を持つ迷路（ゴールまでの経路が限定的）"""
    maze = np.ones((17, 17), dtype=int)
    
    # スタート地点から最初の分岐まで
    for y in range(13, 16):
        maze[y, 2] = 0
    
    # 分岐点（2, 13）から3方向に分かれる
    maze[13, 2] = 0
    
    # 袋小路1：左方向（短い）
    for x in range(1, 3):
        maze[13, x] = 0
    
    # 袋小路2：上方向（長い迷路）
    for y in range(8, 14):
        maze[y, 2] = 0
    # 上部で左右に分岐
    for x in range(1, 6):
        maze[8, x] = 0
    # さらに袋小路
    for y in range(5, 9):
        maze[y, 5] = 0
    
    # 正解ルート：右方向
    for x in range(2, 8):
        maze[13, x] = 0
    
    # 右から上に曲がる
    for y in range(9, 14):
        maze[y, 7] = 0
    
    # 上部で右に曲がってゴールへ
    for x in range(7, 15):
        maze[9, x] = 0
    
    # 追加の袋小路（ゴール手前）
    for y in range(5, 10):
        maze[y, 11] = 0
    
    # ゴールへの最終通路
    for y in range(2, 10):
        maze[y, 14] = 0
    
    return maze


class DetailedDeadEndAnalyzer(MazeNavigator):
    """詳細な袋小路分析用ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_visit_history = []  # (step, position, gedig)のリスト
        self.deadend_entries = []  # 袋小路への進入記録
        self.deadend_exits = []   # 袋小路からの脱出記録
        self.gedig_at_positions = {}  # 位置ごとのgeDIG値履歴
    
    def step(self) -> bool:
        """ステップ実行（詳細記録付き）"""
        prev_pos = self.current_pos
        
        # 親クラスのstep実行
        result = super().step()
        
        # 現在のgeDIG値
        current_gedig = self.gedig_history[-1] if self.gedig_history else 0
        
        # 位置履歴に記録
        self.position_visit_history.append({
            'step': self.step_count,
            'position': self.current_pos,
            'gedig': current_gedig,
            'from_position': prev_pos
        })
        
        # 位置別geDIG履歴
        if self.current_pos not in self.gedig_at_positions:
            self.gedig_at_positions[self.current_pos] = []
        self.gedig_at_positions[self.current_pos].append(current_gedig)
        
        # 袋小路の検出（同じ場所に3回以上 & 負のgeDIG）
        pos_visits = [h for h in self.position_visit_history 
                     if h['position'] == self.current_pos]
        
        if len(pos_visits) == 3:  # 3回目の訪問時
            if current_gedig < -0.05:
                self.deadend_entries.append({
                    'step': self.step_count,
                    'position': self.current_pos,
                    'gedig': current_gedig
                })
        
        # 袋小路からの脱出検出（負→正のgeDIG変化）
        if len(self.gedig_history) >= 2:
            prev_gedig = self.gedig_history[-2]
            if prev_gedig < -0.1 and current_gedig > 0.05:
                self.deadend_exits.append({
                    'step': self.step_count,
                    'position': self.current_pos,
                    'gedig_change': current_gedig - prev_gedig
                })
        
        return result


def main():
    print("="*60)
    print("COMPLEX DEAD-END geDIG ANALYSIS")
    print("="*60)
    
    # 迷路作成
    maze = create_complex_deadend_maze()
    start_pos = (2, 15)  # 左下
    goal_pos = (14, 2)   # 右上
    
    print("\n迷路構造（◎:袋小路の端）:")
    h, w = maze.shape
    for y in range(h):
        row = ""
        for x in range(w):
            if (x, y) == start_pos:
                row += "S "
            elif (x, y) == goal_pos:
                row += "G "
            elif maze[y, x] == 1:
                row += "█ "
            else:
                # 袋小路の端を検出
                neighbors = 0
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                        neighbors += 1
                if neighbors == 1:
                    row += "◎ "
                else:
                    row += "· "
        print(row)
    
    print("\n実験開始...")
    print("-"*60)
    
    # 複数の設定でテスト
    configs = [
        {'name': 'Sensitive', 'threshold': -0.1},
        {'name': 'Normal', 'threshold': -0.15},
        {'name': 'Tolerant', 'threshold': -0.2}
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for idx, config in enumerate(configs):
        print(f"\nテスト: {config['name']} (threshold={config['threshold']})")
        
        # 分析用ナビゲーター
        analyzer = DetailedDeadEndAnalyzer(
            maze=maze,
            start_pos=start_pos,
            goal_pos=goal_pos,
            temperature=0.1,
            gedig_threshold=0.3,
            backtrack_threshold=config['threshold'],
            wiring_strategy='simple'
        )
        
        # 実行
        success = analyzer.run(max_steps=500)
        
        # 結果表示
        print(f"  結果: {'成功' if success else '失敗'}")
        print(f"  ステップ数: {analyzer.step_count}")
        print(f"  袋小路進入: {len(analyzer.deadend_entries)}回")
        print(f"  袋小路脱出: {len(analyzer.deadend_exits)}回")
        
        if analyzer.gedig_history:
            negative_ratio = sum(1 for g in analyzer.gedig_history if g < 0) / len(analyzer.gedig_history)
            print(f"  負のgeDIG率: {negative_ratio:.1%}")
        
        # 可視化（各設定で1行）
        
        # 1. 迷路とパス
        ax = axes[idx, 0]
        for y in range(h):
            for x in range(w):
                if maze[y, x] == 1:
                    rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                                   facecolor='black', edgecolor='gray')
                    ax.add_patch(rect)
        
        if analyzer.path:
            path_x = [p[0] for p in analyzer.path]
            path_y = [p[1] for p in analyzer.path]
            ax.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.5)
            
            # 袋小路進入点をマーク
            for entry in analyzer.deadend_entries:
                ax.plot(entry['position'][0], entry['position'][1], 
                       'rx', markersize=8, markeredgewidth=2)
        
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=8)
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=8)
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(-0.5, h-0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f'{config["name"]}: Path (red X = deadend)')
        ax.grid(True, alpha=0.3)
        
        # 2. geDIG時系列
        ax = axes[idx, 1]
        if analyzer.gedig_history:
            steps = range(len(analyzer.gedig_history))
            ax.plot(steps, analyzer.gedig_history, 'b-', linewidth=1)
            
            # 袋小路イベントをマーク
            for entry in analyzer.deadend_entries:
                if entry['step'] < len(analyzer.gedig_history):
                    ax.plot(entry['step'], entry['gedig'], 'ro', markersize=6)
            
            for exit_event in analyzer.deadend_exits:
                if exit_event['step'] < len(analyzer.gedig_history):
                    ax.plot(exit_event['step'], analyzer.gedig_history[exit_event['step']], 
                           'g^', markersize=8)
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(y=config['threshold'], color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel('geDIG Value')
            ax.set_title(f'geDIG Evolution (threshold={config["threshold"]})')
            ax.grid(True, alpha=0.3)
        
        # 3. 訪問頻度マップ
        ax = axes[idx, 2]
        visit_map = np.zeros_like(maze, dtype=float)
        for pos in analyzer.path:
            visit_map[pos[1], pos[0]] += 1
        
        visit_map[maze == 1] = -1
        im = ax.imshow(visit_map, cmap='hot', interpolation='nearest')
        ax.set_title(f'Visit Frequency (max={int(visit_map.max())})')
        plt.colorbar(im, ax=ax, label='Visits')
    
    plt.suptitle('Comparison of Backtrack Thresholds on Complex Dead-ends', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/complex_deadend_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()