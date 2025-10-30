#!/usr/bin/env python3
"""
行き詰まり（袋小路）でのgeDIG値の変化を観察
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


def create_deadend_maze():
    """明確な袋小路を持つ迷路を作成"""
    maze = np.ones((15, 15), dtype=int)
    
    # メイン通路（縦）
    for y in range(1, 14):
        maze[y, 7] = 0
    
    # 袋小路1（短い、3マス）
    for x in range(5, 8):
        maze[3, x] = 0
    
    # 袋小路2（中程度、5マス）
    for x in range(3, 8):
        maze[7, x] = 0
    
    # 袋小路3（長い、7マス）  
    for x in range(1, 8):
        maze[11, x] = 0
    
    # ゴールへの通路
    for x in range(7, 14):
        maze[1, x] = 0
    
    return maze


class DeadEndAnalyzer(MazeNavigator):
    """行き詰まり分析用の拡張ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadend_events = []  # 袋小路イベント記録
        self.position_gedig_history = {}  # 位置ごとのgeDIG履歴
    
    def step(self) -> bool:
        """ステップ実行（geDIG値を詳細記録）"""
        current_pos = self.current_pos
        
        # 親クラスのstep実行
        result = super().step()
        
        # geDIG値を位置別に記録
        if self.gedig_history:
            current_gedig = self.gedig_history[-1] if self.gedig_history else 0
            
            if current_pos not in self.position_gedig_history:
                self.position_gedig_history[current_pos] = []
            self.position_gedig_history[current_pos].append({
                'step': self.step_count,
                'gedig': current_gedig
            })
            
            # 袋小路検出（3回以上同じ位置を訪問 & 負のgeDIG）
            if len(self.position_gedig_history[current_pos]) >= 3:
                if current_gedig < -0.05:
                    self.deadend_events.append({
                        'step': self.step_count,
                        'position': current_pos,
                        'gedig': current_gedig,
                        'visit_count': len(self.position_gedig_history[current_pos])
                    })
        
        return result


def analyze_deadend_behavior(maze, start_pos, goal_pos):
    """袋小路での挙動を分析"""
    
    # 分析用ナビゲーター
    analyzer = DeadEndAnalyzer(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=-0.15,  # 比較的敏感に設定
        wiring_strategy='simple'
    )
    
    # 実行
    success = analyzer.run(max_steps=500)
    
    return analyzer, success


def visualize_gedig_at_deadends(analyzer, maze):
    """袋小路でのgeDIG値を可視化"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 迷路とパス
    ax1 = plt.subplot(2, 3, 1)
    h, w = maze.shape
    
    # 迷路描画
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
        ax1.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.5)
        
        # 袋小路イベントをマーク
        for event in analyzer.deadend_events[:10]:  # 最初の10個
            ax1.plot(event['position'][0], event['position'][1], 
                    'rx', markersize=10, markeredgewidth=2)
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title('Maze with Dead-End Events (red X)')
    ax1.grid(True, alpha=0.3)
    
    # 2. geDIG値の時系列
    ax2 = plt.subplot(2, 3, 2)
    if analyzer.gedig_history:
        steps = range(len(analyzer.gedig_history))
        ax2.plot(steps, analyzer.gedig_history, 'b-', linewidth=1, alpha=0.7)
        
        # 袋小路イベントをマーク
        for event in analyzer.deadend_events:
            if event['step'] < len(analyzer.gedig_history):
                ax2.plot(event['step'], event['gedig'], 'ro', markersize=8)
                ax2.annotate(f"{event['position']}", 
                           (event['step'], event['gedig']),
                           fontsize=8, ha='center')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=-0.15, color='r', linestyle='--', alpha=0.5, 
                   label='Backtrack Threshold')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('geDIG Value')
        ax2.set_title('geDIG Evolution (Dead-ends marked)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 袋小路別のgeDIG分布
    ax3 = plt.subplot(2, 3, 3)
    
    # 袋小路エリアの定義
    deadend_areas = {
        'Short (3,y)': [(x, 3) for x in range(5, 8)],
        'Medium (7,y)': [(x, 7) for x in range(3, 8)],
        'Long (11,y)': [(x, 11) for x in range(1, 8)]
    }
    
    deadend_gedigs = {name: [] for name in deadend_areas}
    
    for pos, history in analyzer.position_gedig_history.items():
        for name, positions in deadend_areas.items():
            if pos in positions:
                deadend_gedigs[name].extend([h['gedig'] for h in history])
    
    # 箱ひげ図
    data_to_plot = [values for values in deadend_gedigs.values() if values]
    labels = [name for name, values in deadend_gedigs.items() if values]
    
    if data_to_plot:
        bp = ax3.boxplot(data_to_plot, labels=labels)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axhline(y=-0.15, color='r', linestyle='--', alpha=0.5)
        ax3.set_ylabel('geDIG Value')
        ax3.set_title('geDIG Distribution by Dead-end Type')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 訪問回数ヒートマップ
    ax4 = plt.subplot(2, 3, 4)
    visit_map = np.zeros_like(maze, dtype=float)
    for pos in analyzer.path:
        visit_map[pos[1], pos[0]] += 1
    
    visit_map[maze == 1] = -1
    im = ax4.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax4.set_title('Visit Frequency Heatmap')
    plt.colorbar(im, ax=ax4, label='Visit Count')
    
    # 5. geDIG値の頻度分布
    ax5 = plt.subplot(2, 3, 5)
    if analyzer.gedig_history:
        positive_gedigs = [g for g in analyzer.gedig_history if g > 0]
        negative_gedigs = [g for g in analyzer.gedig_history if g < 0]
        
        bins = np.linspace(-0.5, 0.5, 30)
        ax5.hist([positive_gedigs, negative_gedigs], bins=bins, 
                label=['Positive (Exploration)', 'Negative (Dead-end)'],
                color=['green', 'red'], alpha=0.7, stacked=False)
        
        ax5.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax5.axvline(x=-0.15, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('geDIG Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('geDIG Value Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 袋小路イベントの詳細
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    text = "Dead-end Events Summary:\n\n"
    text += f"Total events: {len(analyzer.deadend_events)}\n\n"
    
    for i, event in enumerate(analyzer.deadend_events[:5]):
        text += f"Event {i+1}:\n"
        text += f"  Step: {event['step']}\n"
        text += f"  Position: {event['position']}\n"
        text += f"  geDIG: {event['gedig']:.4f}\n"
        text += f"  Visits: {event['visit_count']}\n\n"
    
    ax6.text(0.1, 0.9, text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Dead-end Event Details')
    
    plt.suptitle('Dead-end Detection Analysis with geDIG', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    print("="*60)
    print("DEAD-END geDIG ANALYSIS")
    print("="*60)
    
    # 迷路作成
    maze = create_deadend_maze()
    start_pos = (7, 13)
    goal_pos = (13, 1)
    
    print("\n迷路構造:")
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
                    row += "◎ "  # 袋小路の端
                else:
                    row += "· "
        print(row)
    
    print("\n分析開始...")
    print("-"*60)
    
    # 分析実行
    analyzer, success = analyze_deadend_behavior(maze, start_pos, goal_pos)
    
    # 結果表示
    print(f"\n結果: {'成功' if success else '失敗'}")
    print(f"総ステップ数: {analyzer.step_count}")
    print(f"ユニーク位置: {len(set(analyzer.path))}")
    print(f"袋小路イベント数: {len(analyzer.deadend_events)}")
    
    if analyzer.gedig_history:
        negative_count = sum(1 for g in analyzer.gedig_history if g < 0)
        print(f"負のgeDIG値の割合: {negative_count}/{len(analyzer.gedig_history)} "
              f"({negative_count*100/len(analyzer.gedig_history):.1f}%)")
        
        min_gedig = min(analyzer.gedig_history)
        max_gedig = max(analyzer.gedig_history)
        avg_gedig = np.mean(analyzer.gedig_history)
        print(f"geDIG範囲: [{min_gedig:.4f}, {max_gedig:.4f}], 平均: {avg_gedig:.4f}")
    
    # 袋小路イベントの分析
    if analyzer.deadend_events:
        print("\n袋小路イベント詳細:")
        for i, event in enumerate(analyzer.deadend_events[:5]):
            print(f"  {i+1}. Step {event['step']}: {event['position']} "
                  f"(geDIG={event['gedig']:.4f}, visits={event['visit_count']})")
    
    # 可視化
    fig = visualize_gedig_at_deadends(analyzer, maze)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/deadend_analysis_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()
    
    # 詳細な統計
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    
    # 位置別の訪問回数とgeDIG
    position_stats = {}
    for pos, history in analyzer.position_gedig_history.items():
        if len(history) > 1:
            gedigs = [h['gedig'] for h in history]
            position_stats[pos] = {
                'visits': len(history),
                'avg_gedig': np.mean(gedigs),
                'min_gedig': min(gedigs)
            }
    
    # 最も頻繁に訪問された位置
    print("\n最頻訪問位置（上位5個）:")
    sorted_positions = sorted(position_stats.items(), 
                            key=lambda x: x[1]['visits'], 
                            reverse=True)
    for pos, stats in sorted_positions[:5]:
        print(f"  {pos}: {stats['visits']}回訪問, "
              f"平均geDIG={stats['avg_gedig']:.4f}, "
              f"最小geDIG={stats['min_gedig']:.4f}")
    
    # バックトラックの効果
    backtrack_events = [e for e in analyzer.event_log if e['type'] == 'backtrack_trigger']
    print(f"\nバックトラックトリガー: {len(backtrack_events)}回")
    
    if backtrack_events:
        print("バックトラックイベント（最初の3個）:")
        for event in backtrack_events[:3]:
            print(f"  Step {event['step']}: {event['message']}")


if __name__ == "__main__":
    main()