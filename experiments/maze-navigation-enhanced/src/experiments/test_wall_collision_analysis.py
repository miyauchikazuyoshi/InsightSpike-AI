#!/usr/bin/env python3
"""
壁衝突の詳細分析
ソフトマックス選択で壁を選ぶ確率と実際の衝突を調査
"""

import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


@dataclass
class WallCollisionEvent:
    """壁衝突イベント"""
    step: int
    position: Tuple[int, int]
    attempted_direction: Tuple[int, int]
    softmax_probs: Dict[Tuple[int, int], float]
    wall_selection_prob: float
    gedig_value: float


class WallCollisionAnalyzer(MazeNavigator):
    """壁衝突分析用ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_collision_events = []
        self.softmax_history = []  # ソフトマックス確率の履歴
        self.wall_selection_attempts = 0  # 壁を選んだ回数
        self.successful_moves = 0  # 成功した移動回数
        
    def select_direction(self, similarities: dict) -> tuple:
        """方向選択（詳細記録付き）"""
        # 温度パラメータでスケーリング
        scaled = {k: v / self.temperature for k, v in similarities.items()}
        
        # ソフトマックス
        max_val = max(scaled.values())
        exp_values = {k: np.exp(v - max_val) for k, v in scaled.items()}
        sum_exp = sum(exp_values.values())
        
        if sum_exp == 0:
            probs = {k: 1.0 / len(exp_values) for k in exp_values}
        else:
            probs = {k: v / sum_exp for k, v in exp_values.items()}
        
        # 記録
        self.softmax_history.append({
            'step': self.step_count,
            'position': self.current_pos,
            'probs': probs.copy(),
            'similarities': similarities.copy()
        })
        
        # 壁方向の確率を計算
        episodes = self.episode_manager.get_episodes_at_position(self.current_pos)
        wall_probs = {}
        passage_probs = {}
        
        for direction, prob in probs.items():
            key = (self.current_pos, direction)
            if key in episodes and episodes[key].is_wall:
                wall_probs[direction] = prob
            else:
                passage_probs[direction] = prob
        
        # 選択
        directions = list(probs.keys())
        probabilities = list(probs.values())
        selected = directions[np.random.choice(len(directions), p=probabilities)]
        
        # 壁を選んだかチェック
        selected_key = (self.current_pos, selected)
        if selected_key in episodes and episodes[selected_key].is_wall:
            self.wall_selection_attempts += 1
            
            # 壁衝突イベント記録
            self.wall_collision_events.append(WallCollisionEvent(
                step=self.step_count,
                position=self.current_pos,
                attempted_direction=selected,
                softmax_probs=probs.copy(),
                wall_selection_prob=probs[selected],
                gedig_value=self.gedig_history[-1] if self.gedig_history else 0
            ))
        
        return selected
    
    def step(self) -> bool:
        """ステップ実行（壁衝突記録付き）"""
        prev_pos = self.current_pos
        result = super().step()
        
        # 移動成功をチェック
        if self.current_pos != prev_pos:
            self.successful_moves += 1
        
        return result


def create_deadend_heavy_maze():
    """袋小路が多い迷路（壁衝突が起きやすい）"""
    maze = np.ones((15, 15), dtype=int)
    
    # メイン通路
    for y in range(1, 14):
        maze[y, 7] = 0
    
    # 多数の袋小路（様々な深さ）
    # 左側の袋小路群
    for y in [3, 5, 7, 9, 11]:
        for x in range(2, 7):
            maze[y, x] = 0
    
    # 右側の袋小路群
    for y in [2, 4, 6, 8, 10, 12]:
        for x in range(8, 13):
            maze[y, x] = 0
    
    # ゴールへの経路
    for x in range(7, 13):
        maze[1, x] = 0
    
    return maze


def analyze_wall_collisions(analyzer):
    """壁衝突の詳細分析"""
    
    stats = {
        'total_steps': analyzer.step_count,
        'wall_selection_attempts': analyzer.wall_selection_attempts,
        'successful_moves': analyzer.successful_moves,
        'wall_collision_rate': analyzer.wall_selection_attempts / analyzer.step_count if analyzer.step_count > 0 else 0,
        'move_success_rate': analyzer.successful_moves / analyzer.step_count if analyzer.step_count > 0 else 0
    }
    
    # 壁選択確率の分析
    if analyzer.wall_collision_events:
        wall_probs = [e.wall_selection_prob for e in analyzer.wall_collision_events]
        stats['avg_wall_selection_prob'] = np.mean(wall_probs)
        stats['max_wall_selection_prob'] = max(wall_probs)
        stats['min_wall_selection_prob'] = min(wall_probs)
    
    # 位置別の壁衝突頻度
    position_collisions = {}
    for event in analyzer.wall_collision_events:
        pos = event.position
        if pos not in position_collisions:
            position_collisions[pos] = 0
        position_collisions[pos] += 1
    
    stats['position_collisions'] = position_collisions
    
    # geDIG値と壁衝突の関係
    if analyzer.wall_collision_events:
        collision_gedigs = [e.gedig_value for e in analyzer.wall_collision_events]
        stats['collision_gedig_mean'] = np.mean(collision_gedigs)
        stats['collision_gedig_std'] = np.std(collision_gedigs)
    
    return stats


def visualize_wall_collision_analysis(analyzer, maze, stats):
    """壁衝突分析の可視化"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. 迷路と壁衝突位置
    ax1 = plt.subplot(3, 4, 1)
    h, w = maze.shape
    
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax1.add_patch(rect)
    
    # パス描画
    if analyzer.path:
        path_x = [p[0] for p in analyzer.path]
        path_y = [p[1] for p in analyzer.path]
        ax1.plot(path_x, path_y, 'b-', linewidth=0.5, alpha=0.3)
    
    # 壁衝突位置をマーク（頻度で色分け）
    if stats['position_collisions']:
        max_collisions = max(stats['position_collisions'].values())
        for pos, count in stats['position_collisions'].items():
            intensity = count / max_collisions
            ax1.plot(pos[0], pos[1], 'ro', markersize=8+count*2, 
                    alpha=0.3+0.7*intensity)
            ax1.text(pos[0]+0.2, pos[1], str(count), fontsize=8)
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title(f'Wall Collisions (Total: {analyzer.wall_selection_attempts})')
    ax1.grid(True, alpha=0.3)
    
    # 2. 壁選択確率の時系列
    ax2 = plt.subplot(3, 4, (2, 4))
    
    if analyzer.wall_collision_events:
        steps = [e.step for e in analyzer.wall_collision_events]
        probs = [e.wall_selection_prob for e in analyzer.wall_collision_events]
        
        ax2.scatter(steps, probs, c=range(len(steps)), cmap='viridis', s=30, alpha=0.7)
        ax2.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='25% threshold')
        ax2.axhline(y=np.mean(probs), color='g', linestyle='-', alpha=0.5, 
                   label=f'Mean: {np.mean(probs):.3f}')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Wall Selection Probability')
        ax2.set_title('Probability of Selecting Wall Direction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. ソフトマックス温度の影響
    ax3 = plt.subplot(3, 4, 5)
    
    # 温度別の確率分布例を表示
    temps = [0.1, 0.3, 0.5, 1.0]
    similarity_diff = np.linspace(-0.5, 0.5, 100)
    
    for temp in temps:
        probs = 1 / (1 + np.exp(-similarity_diff/temp))
        ax3.plot(similarity_diff, probs, label=f'T={temp}')
    
    ax3.set_xlabel('Similarity Difference (wall - passage)')
    ax3.set_ylabel('Wall Selection Probability')
    ax3.set_title('Temperature Effect on Wall Selection')
    ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 壁衝突時のgeDIG値分布
    ax4 = plt.subplot(3, 4, 6)
    
    if analyzer.wall_collision_events:
        gedigs = [e.gedig_value for e in analyzer.wall_collision_events]
        ax4.hist(gedigs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.mean(gedigs), color='darkred', linestyle='--',
                   label=f'Mean: {np.mean(gedigs):.4f}')
        ax4.set_xlabel('geDIG Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('geDIG at Wall Collisions')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 壁選択確率の分布
    ax5 = plt.subplot(3, 4, 7)
    
    if analyzer.wall_collision_events:
        wall_probs = [e.wall_selection_prob for e in analyzer.wall_collision_events]
        ax5.hist(wall_probs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='25%')
        ax5.axvline(x=np.mean(wall_probs), color='green', linestyle='-',
                   label=f'Mean: {np.mean(wall_probs):.3f}')
        ax5.set_xlabel('Wall Selection Probability')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Wall Selection Probabilities')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 時間経過による壁衝突率
    ax6 = plt.subplot(3, 4, 8)
    
    if analyzer.wall_collision_events:
        window = 50  # 50ステップのウィンドウ
        collision_rate = []
        steps_range = []
        
        for i in range(0, analyzer.step_count, window):
            end = min(i + window, analyzer.step_count)
            collisions_in_window = sum(1 for e in analyzer.wall_collision_events 
                                      if i <= e.step < end)
            rate = collisions_in_window / (end - i)
            collision_rate.append(rate)
            steps_range.append(i + window/2)
        
        ax6.plot(steps_range, collision_rate, 'b-', linewidth=2)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Collision Rate')
        ax6.set_title(f'Wall Collision Rate Over Time (window={window})')
        ax6.grid(True, alpha=0.3)
    
    # 7. 詳細な壁衝突イベント（最初の10個）
    ax7 = plt.subplot(3, 4, (9, 10))
    ax7.axis('off')
    
    text = "First 10 Wall Collision Events:\n\n"
    for i, event in enumerate(analyzer.wall_collision_events[:10]):
        text += f"{i+1}. Step {event.step}: pos{event.position}\n"
        text += f"   Wall prob: {event.wall_selection_prob:.3f}\n"
        text += f"   geDIG: {event.gedig_value:.4f}\n"
    
    ax7.text(0.05, 0.95, text, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax7.set_title('Wall Collision Details')
    
    # 8. 統計サマリー
    ax8 = plt.subplot(3, 4, (11, 12))
    ax8.axis('off')
    
    summary = f"""Wall Collision Statistics:

Total Steps: {stats['total_steps']}
Wall Selection Attempts: {stats['wall_selection_attempts']}
Successful Moves: {stats['successful_moves']}

Collision Rate: {stats['wall_collision_rate']:.1%}
Move Success Rate: {stats['move_success_rate']:.1%}

Wall Selection Probability:
  Average: {stats.get('avg_wall_selection_prob', 0):.3f}
  Maximum: {stats.get('max_wall_selection_prob', 0):.3f}
  Minimum: {stats.get('min_wall_selection_prob', 0):.3f}

geDIG at Collisions:
  Mean: {stats.get('collision_gedig_mean', 0):.4f}
  Std: {stats.get('collision_gedig_std', 0):.4f}

Temperature: {analyzer.temperature if hasattr(analyzer, 'temperature') else 'N/A'}"""
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Statistical Summary')
    
    plt.suptitle('Wall Collision Analysis: Softmax Selection Behavior', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    print("="*60)
    print("WALL COLLISION ANALYSIS")
    print("Investigating wall selection with softmax probabilities")
    print("="*60)
    
    # 複数の温度で実験
    temperatures = [0.1, 0.3, 0.5]
    all_results = []
    
    for temp in temperatures:
        print(f"\n実験: Temperature = {temp}")
        print("-"*40)
        
        # 迷路作成
        maze = create_deadend_heavy_maze()
        start_pos = (7, 13)
        goal_pos = (12, 1)
        
        # 分析用ナビゲーター
        analyzer = WallCollisionAnalyzer(
            maze=maze,
            start_pos=start_pos,
            goal_pos=goal_pos,
            temperature=temp,
            gedig_threshold=0.3,
            backtrack_threshold=-0.1,
            wiring_strategy='simple'
        )
        
        # 実行
        success = analyzer.run(max_steps=500)
        
        # 分析
        stats = analyze_wall_collisions(analyzer)
        
        print(f"  結果: {'成功' if success else '失敗'}")
        print(f"  総ステップ数: {stats['total_steps']}")
        print(f"  壁選択回数: {stats['wall_selection_attempts']}")
        print(f"  壁衝突率: {stats['wall_collision_rate']:.1%}")
        print(f"  平均壁選択確率: {stats.get('avg_wall_selection_prob', 0):.3f}")
        
        all_results.append({
            'temperature': temp,
            'analyzer': analyzer,
            'stats': stats,
            'success': success
        })
    
    # 温度による比較
    print("\n" + "="*60)
    print("TEMPERATURE COMPARISON")
    print("="*60)
    
    for result in all_results:
        print(f"\nT={result['temperature']}:")
        print(f"  壁衝突率: {result['stats']['wall_collision_rate']:.1%}")
        print(f"  平均壁選択確率: {result['stats'].get('avg_wall_selection_prob', 0):.3f}")
        print(f"  最大壁選択確率: {result['stats'].get('max_wall_selection_prob', 0):.3f}")
    
    # 最も詳細な結果を可視化（中間温度）
    middle_result = all_results[1]  # T=0.3
    fig = visualize_wall_collision_analysis(
        middle_result['analyzer'], 
        create_deadend_heavy_maze(),
        middle_result['stats']
    )
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/wall_collision_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()