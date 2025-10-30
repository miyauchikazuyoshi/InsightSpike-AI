#!/usr/bin/env python3
"""
複雑な迷路でのgeDIG値詳細分析
- 多数の分岐と袋小路
- 長い経路
- 複数の解候補
"""

import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def create_complex_maze():
    """複雑な迷路（確実に接続されている版）"""
    maze = np.ones((21, 21), dtype=int)
    
    # メインの通路網
    # 縦の基幹通路
    for y in range(1, 20):
        maze[y, 10] = 0  # 中央縦通路
    
    # 横の基幹通路
    for x in range(1, 20):
        maze[5, x] = 0   # 上部横通路
        maze[10, x] = 0  # 中央横通路
        maze[15, x] = 0  # 下部横通路
    
    # 左右の縦通路（部分的）
    for y in range(1, 10):
        maze[y, 3] = 0   # 左側上部
    for y in range(11, 20):
        maze[y, 3] = 0   # 左側下部
    
    for y in range(1, 9):
        maze[y, 17] = 0  # 右側上部
    for y in range(12, 20):
        maze[y, 17] = 0  # 右側下部
    
    # 袋小路を追加（接続を壊さないように）
    
    # 短い袋小路
    maze[3, 5] = 0
    maze[3, 6] = 0
    
    maze[7, 18] = 0
    maze[7, 19] = 0
    
    # 中程度の袋小路
    for x in range(5, 8):
        maze[8, x] = 0
    
    for y in range(12, 14):
        maze[y, 5] = 0
    
    # 長い袋小路
    for x in range(12, 16):
        maze[13, x] = 0
    maze[14, 14] = 0
    
    # 追加の接続路（迷路を面白くする）
    maze[7, 7] = 0
    maze[7, 8] = 0
    maze[13, 7] = 0
    maze[13, 8] = 0
    
    return maze


class ComplexMazeAnalyzer(MazeNavigator):
    """複雑な迷路分析用ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_history = []  # 状態履歴
        self.deadend_visits = {}  # 袋小路訪問記録
        self.branch_decisions = []  # 分岐点での決定
        self.gedig_at_states = {
            'exploring_new': [],
            'revisiting': [],
            'deadend': [],
            'branch': [],
            'backtrack': []
        }
    
    def step(self) -> bool:
        """ステップ実行（詳細な状態分類）"""
        prev_pos = self.current_pos
        
        # 現在の状態を分析
        episodes = self.episode_manager.get_episodes_at_position(self.current_pos)
        available = sum(1 for ep in episodes.values() if not ep.is_wall)
        visited = sum(1 for ep in episodes.values() if ep.visit_count > 0)
        
        # 状態を判定
        state = self._determine_state(available, visited, prev_pos)
        
        # 親クラスのstep実行
        result = super().step()
        
        # geDIG値を記録
        current_gedig = self.gedig_history[-1] if self.gedig_history else 0
        
        # 状態別にgeDIG値を記録
        self.gedig_at_states[state].append(current_gedig)
        
        # 履歴に追加
        self.state_history.append({
            'step': self.step_count,
            'position': self.current_pos,
            'state': state,
            'gedig': current_gedig,
            'available': available,
            'visited': visited
        })
        
        # 袋小路の記録
        if available == 1:  # 袋小路の端
            if self.current_pos not in self.deadend_visits:
                self.deadend_visits[self.current_pos] = []
            self.deadend_visits[self.current_pos].append({
                'step': self.step_count,
                'gedig': current_gedig
            })
        
        # 分岐点の記録
        if available >= 3:  # 分岐点
            self.branch_decisions.append({
                'step': self.step_count,
                'position': self.current_pos,
                'gedig': current_gedig,
                'options': available,
                'visited_options': visited
            })
        
        return result
    
    def _determine_state(self, available, visited, prev_pos):
        """現在の状態を判定"""
        # 袋小路
        if available == 1:
            return 'deadend'
        
        # 分岐点
        if available >= 3:
            return 'branch'
        
        # バックトラック（すべて訪問済み）
        if visited == available and available > 0:
            return 'backtrack'
        
        # 再訪問
        if self.current_pos in [h['position'] for h in self.state_history]:
            return 'revisiting'
        
        # 新規探索
        return 'exploring_new'


def analyze_complex_behavior(analyzer):
    """複雑な迷路での挙動分析"""
    
    stats = {}
    
    # 状態別geDIG統計
    for state, gedigs in analyzer.gedig_at_states.items():
        if gedigs:
            stats[state] = {
                'count': len(gedigs),
                'mean': np.mean(gedigs),
                'std': np.std(gedigs),
                'min': min(gedigs),
                'max': max(gedigs),
                'negative_ratio': sum(1 for g in gedigs if g < 0) / len(gedigs)
            }
    
    # 袋小路の統計
    deadend_stats = []
    for pos, visits in analyzer.deadend_visits.items():
        if visits:
            gedigs = [v['gedig'] for v in visits]
            deadend_stats.append({
                'position': pos,
                'visit_count': len(visits),
                'avg_gedig': np.mean(gedigs),
                'first_gedig': gedigs[0] if gedigs else None,
                'last_gedig': gedigs[-1] if gedigs else None
            })
    
    # 分岐点の統計
    branch_stats = {
        'total_branches': len(analyzer.branch_decisions),
        'avg_options': np.mean([b['options'] for b in analyzer.branch_decisions]) if analyzer.branch_decisions else 0,
        'revisited_branches': sum(1 for b in analyzer.branch_decisions if b['visited_options'] > 1)
    }
    
    return {
        'state_stats': stats,
        'deadend_stats': deadend_stats,
        'branch_stats': branch_stats
    }


def visualize_complex_analysis(analyzer, maze, start_pos, goal_pos):
    """複雑な迷路分析の可視化"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 迷路とパス（状態別色分け）
    ax1 = plt.subplot(3, 4, (1, 5))
    h, w = maze.shape
    
    # 迷路描画
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray', linewidth=0.5)
                ax1.add_patch(rect)
    
    # パスを状態別に色分け
    if analyzer.state_history:
        state_colors = {
            'exploring_new': 'blue',
            'revisiting': 'cyan',
            'deadend': 'red',
            'branch': 'yellow',
            'backtrack': 'orange'
        }
        
        for i in range(len(analyzer.state_history) - 1):
            current = analyzer.state_history[i]
            next_hist = analyzer.state_history[i + 1]
            
            color = state_colors.get(current['state'], 'gray')
            ax1.plot([current['position'][0], next_hist['position'][0]],
                    [current['position'][1], next_hist['position'][1]],
                    color=color, linewidth=2, alpha=0.7)
    
    # スタートとゴール
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'ro', markersize=12, label='Goal')
    
    # 袋小路をマーク
    for pos in analyzer.deadend_visits.keys():
        ax1.plot(pos[0], pos[1], 'rx', markersize=8, markeredgewidth=2)
    
    # 分岐点をマーク
    branch_positions = set(b['position'] for b in analyzer.branch_decisions)
    for pos in branch_positions:
        ax1.plot(pos[0], pos[1], 'y*', markersize=10)
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title('Complex Maze Navigation\n(States: Blue=New, Red=DeadEnd, Yellow=Branch)')
    ax1.grid(True, alpha=0.3)
    
    # 凡例作成
    patches = [mpatches.Patch(color=color, label=state) 
              for state, color in state_colors.items()]
    ax1.legend(handles=patches, loc='upper left', fontsize=8)
    
    # 2. geDIG時系列（状態別）
    ax2 = plt.subplot(3, 4, (2, 3))
    
    if analyzer.gedig_history:
        steps = range(len(analyzer.gedig_history))
        
        # 背景に全体のgeDIG
        ax2.plot(steps, analyzer.gedig_history, 'k-', linewidth=0.5, alpha=0.3)
        
        # 状態別にマーカー
        for hist in analyzer.state_history:
            if hist['step'] < len(analyzer.gedig_history):
                color = state_colors.get(hist['state'], 'gray')
                ax2.plot(hist['step'], hist['gedig'], 'o', 
                        color=color, markersize=4, alpha=0.7)
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=-0.15, color='r', linestyle='--', alpha=0.5, label='Backtrack Threshold')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('geDIG Value')
        ax2.set_title('geDIG Evolution with States')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 状態別geDIG分布
    ax3 = plt.subplot(3, 4, 6)
    
    state_data = []
    state_labels = []
    for state in ['exploring_new', 'deadend', 'branch', 'backtrack', 'revisiting']:
        if analyzer.gedig_at_states[state]:
            state_data.append(analyzer.gedig_at_states[state])
            state_labels.append(f"{state}\n(n={len(analyzer.gedig_at_states[state])})")
    
    if state_data:
        bp = ax3.boxplot(state_data, tick_labels=state_labels)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('geDIG Value')
        ax3.set_title('geDIG Distribution by State')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. 訪問頻度ヒートマップ
    ax4 = plt.subplot(3, 4, 7)
    
    visit_map = np.zeros_like(maze, dtype=float)
    for hist in analyzer.state_history:
        x, y = hist['position']
        visit_map[y, x] += 1
    
    visit_map[maze == 1] = -1
    im = ax4.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax4.set_title(f'Visit Frequency (max={int(visit_map.max())})')
    plt.colorbar(im, ax=ax4, label='Visits')
    
    # 5. 袋小路でのgeDIG変化
    ax5 = plt.subplot(3, 4, 8)
    
    if analyzer.deadend_visits:
        for i, (pos, visits) in enumerate(analyzer.deadend_visits.items()):
            if visits:
                steps = [v['step'] for v in visits]
                gedigs = [v['gedig'] for v in visits]
                ax5.plot(steps, gedigs, 'o-', label=f"Dead-end {i+1}", alpha=0.7)
        
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('geDIG Value')
        ax5.set_title('geDIG at Dead-ends Over Time')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # 6. 分岐点でのgeDIG
    ax6 = plt.subplot(3, 4, 9)
    
    if analyzer.branch_decisions:
        branch_gedigs = [b['gedig'] for b in analyzer.branch_decisions]
        branch_options = [b['options'] for b in analyzer.branch_decisions]
        
        scatter = ax6.scatter(branch_options, branch_gedigs, 
                            c=range(len(branch_gedigs)), cmap='viridis',
                            s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax6, label='Time (step)')
        
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Number of Options')
        ax6.set_ylabel('geDIG Value')
        ax6.set_title('geDIG at Branch Points')
        ax6.grid(True, alpha=0.3)
    
    # 7. geDIG値のヒストグラム（正vs負）
    ax7 = plt.subplot(3, 4, 10)
    
    if analyzer.gedig_history:
        positive = [g for g in analyzer.gedig_history if g > 0]
        negative = [g for g in analyzer.gedig_history if g < 0]
        zero = [g for g in analyzer.gedig_history if g == 0]
        
        bins = np.linspace(-0.3, 0.3, 30)
        ax7.hist([positive, negative], bins=bins, 
                label=[f'Positive (n={len(positive)})', f'Negative (n={len(negative)})'],
                color=['green', 'red'], alpha=0.7)
        
        ax7.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax7.set_xlabel('geDIG Value')
        ax7.set_ylabel('Frequency')
        ax7.set_title('geDIG Value Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. 統計サマリー
    ax8 = plt.subplot(3, 4, (11, 12))
    ax8.axis('off')
    
    analysis = analyze_complex_behavior(analyzer)
    
    text = "State Statistics:\n" + "="*30 + "\n"
    for state, stats in analysis['state_stats'].items():
        text += f"\n{state}:\n"
        text += f"  Count: {stats['count']}\n"
        text += f"  Mean: {stats['mean']:.4f}\n"
        text += f"  Negative ratio: {stats['negative_ratio']:.1%}\n"
    
    text += f"\n\nMaze Statistics:\n" + "="*30 + "\n"
    text += f"Dead-ends found: {len(analysis['deadend_stats'])}\n"
    text += f"Branch points: {analysis['branch_stats']['total_branches']}\n"
    text += f"Revisited branches: {analysis['branch_stats']['revisited_branches']}\n"
    text += f"\nTotal steps: {analyzer.step_count}\n"
    text += f"Unique positions: {len(set(h['position'] for h in analyzer.state_history))}"
    
    ax8.text(0.05, 0.95, text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Statistical Summary')
    
    plt.suptitle('Complex Maze geDIG Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    print("="*60)
    print("COMPLEX MAZE geDIG ANALYSIS")
    print("Detailed behavior in a challenging maze")
    print("="*60)
    
    # 迷路作成
    maze = create_complex_maze()
    start_pos = (3, 19)   # 左下
    goal_pos = (17, 1)    # 右上
    
    print(f"\nMaze size: {maze.shape}")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    
    # 通路の統計
    passages = np.sum(maze == 0)
    total = maze.size
    print(f"Passages: {passages}/{total} ({passages*100/total:.1f}%)")
    
    # 迷路の構造分析
    print("\n迷路構造の特徴:")
    print("- 3本の縦通路と3本の横通路")
    print("- 複数の袋小路（短・中・長）")
    print("- ループ構造あり")
    print("- 部分的な通路封鎖")
    
    print("\n実験実行...")
    print("-"*60)
    
    # 分析用ナビゲーター（温度を上げて探索的にする）
    analyzer = ComplexMazeAnalyzer(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        temperature=0.5,  # より探索的に
        gedig_threshold=0.3,
        backtrack_threshold=-0.10,  # より敏感に
        wiring_strategy='simple'
    )
    
    # 実行
    success = analyzer.run(max_steps=1000)
    
    # 結果表示
    print(f"\n結果: {'成功' if success else '失敗'}")
    print(f"総ステップ数: {analyzer.step_count}")
    print(f"ユニーク位置: {len(set(h['position'] for h in analyzer.state_history))}")
    
    # 詳細分析
    analysis = analyze_complex_behavior(analyzer)
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # 状態別統計
    print("\n状態別geDIG統計:")
    for state, stats in analysis['state_stats'].items():
        print(f"\n{state}:")
        print(f"  発生回数: {stats['count']}")
        print(f"  平均geDIG: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  範囲: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  負の値の割合: {stats['negative_ratio']:.1%}")
    
    # 袋小路統計
    if analysis['deadend_stats']:
        print(f"\n袋小路統計:")
        print(f"  発見数: {len(analysis['deadend_stats'])}")
        
        avg_first_gedig = np.mean([d['first_gedig'] for d in analysis['deadend_stats'] if d['first_gedig'] is not None])
        avg_last_gedig = np.mean([d['last_gedig'] for d in analysis['deadend_stats'] if d['last_gedig'] is not None])
        
        print(f"  初回到達時の平均geDIG: {avg_first_gedig:.4f}")
        print(f"  最終訪問時の平均geDIG: {avg_last_gedig:.4f}")
        
        # 最も訪問された袋小路
        most_visited = max(analysis['deadend_stats'], key=lambda x: x['visit_count'])
        print(f"  最頻訪問袋小路: {most_visited['position']} ({most_visited['visit_count']}回)")
    
    # 分岐点統計
    print(f"\n分岐点統計:")
    print(f"  総分岐点数: {analysis['branch_stats']['total_branches']}")
    print(f"  平均選択肢数: {analysis['branch_stats']['avg_options']:.1f}")
    print(f"  再訪問分岐点: {analysis['branch_stats']['revisited_branches']}")
    
    # 重要な発見
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    if 'exploring_new' in analysis['state_stats'] and 'deadend' in analysis['state_stats']:
        exp_stats = analysis['state_stats']['exploring_new']
        dead_stats = analysis['state_stats']['deadend']
        
        print(f"\n1. 新規探索時 vs 袋小路:")
        print(f"   新規探索: 平均geDIG = {exp_stats['mean']:.4f}")
        print(f"   袋小路:   平均geDIG = {dead_stats['mean']:.4f}")
        print(f"   差: {dead_stats['mean'] - exp_stats['mean']:.4f}")
    
    if 'branch' in analysis['state_stats']:
        branch_stats = analysis['state_stats']['branch']
        print(f"\n2. 分岐点でのgeDIG:")
        print(f"   平均: {branch_stats['mean']:.4f}")
        print(f"   負の値の割合: {branch_stats['negative_ratio']:.1%}")
    
    if 'backtrack' in analysis['state_stats']:
        back_stats = analysis['state_stats']['backtrack']
        print(f"\n3. バックトラック時のgeDIG:")
        print(f"   平均: {back_stats['mean']:.4f}")
        print(f"   負の値の割合: {back_stats['negative_ratio']:.1%}")
    
    # geDIG履歴の全体統計
    if analyzer.gedig_history:
        negative_ratio = sum(1 for g in analyzer.gedig_history if g < 0) / len(analyzer.gedig_history)
        print(f"\n4. 全体的なgeDIG傾向:")
        print(f"   負の値の割合: {negative_ratio:.1%}")
        print(f"   平均値: {np.mean(analyzer.gedig_history):.4f}")
        print(f"   標準偏差: {np.std(analyzer.gedig_history):.4f}")
    
    # 可視化
    fig = visualize_complex_analysis(analyzer, maze, start_pos, goal_pos)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/complex_maze_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()