#!/usr/bin/env python3
"""
袋小路の探索フェーズ別geDIG値詳細分析
- Phase 1: 袋小路への進入（探索中）
- Phase 2: 突き当たり到達（壁に衝突）
- Phase 3: 引き返し開始（バックトラック）
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
class DeadEndPhase:
    """袋小路探索のフェーズ"""
    phase_type: str  # 'entering', 'hit_wall', 'backtracking'
    start_step: int
    end_step: int
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    gedig_values: List[float]
    avg_gedig: float
    min_gedig: float
    max_gedig: float


def create_controlled_deadend_maze():
    """制御された袋小路を持つ迷路（分析しやすい構造）"""
    maze = np.ones((13, 13), dtype=int)
    
    # メイン通路
    for y in range(1, 12):
        maze[y, 6] = 0
    
    # 袋小路1: 短い（2マス）
    maze[3, 5] = 0
    maze[3, 4] = 0
    
    # 袋小路2: 中程度（4マス）
    for x in range(2, 6):
        maze[6, x] = 0
    
    # 袋小路3: 長い（6マス）
    for x in range(0, 6):
        maze[9, x] = 0
    
    # ゴールへの通路
    for x in range(6, 12):
        maze[1, x] = 0
    
    return maze


class DeadEndPhaseAnalyzer(MazeNavigator):
    """袋小路フェーズ分析用ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detailed_history = []  # 詳細履歴
        self.wall_hits = []  # 壁衝突記録
        self.direction_changes = []  # 方向転換記録
        self.deadend_phases = []  # フェーズ記録
        self.current_deadend_entry = None  # 現在の袋小路進入情報
        
    def step(self) -> bool:
        """ステップ実行（詳細記録付き）"""
        prev_pos = self.current_pos
        prev_episodes = self.episode_manager.get_episodes_at_position(prev_pos)
        
        # 親クラスのstep実行
        result = super().step()
        
        # 現在のgeDIG値
        current_gedig = self.gedig_history[-1] if self.gedig_history else 0
        
        # 現在のエピソード情報
        current_episodes = self.episode_manager.get_episodes_at_position(self.current_pos)
        
        # 詳細履歴に記録
        self.detailed_history.append({
            'step': self.step_count,
            'position': self.current_pos,
            'prev_position': prev_pos,
            'gedig': current_gedig,
            'available_directions': sum(1 for ep in current_episodes.values() if not ep.is_wall),
            'visited_directions': sum(1 for ep in current_episodes.values() if ep.visit_count > 0)
        })
        
        # 壁衝突の検出
        if self.current_pos == prev_pos:  # 移動失敗
            self.wall_hits.append({
                'step': self.step_count,
                'position': self.current_pos,
                'gedig': current_gedig
            })
            self._check_deadend_hit()
        
        # 方向転換の検出（180度ターン）
        if len(self.path) >= 3:
            p1, p2, p3 = self.path[-3], self.path[-2], self.path[-1]
            if p1 == p3 and p1 != p2:  # 戻った
                self.direction_changes.append({
                    'step': self.step_count,
                    'position': p2,
                    'gedig': current_gedig,
                    'type': 'u_turn'
                })
                self._check_backtrack_start()
        
        # 袋小路への進入検出
        self._check_deadend_entry()
        
        return result
    
    def _check_deadend_entry(self):
        """袋小路への進入を検出"""
        if len(self.detailed_history) < 2:
            return
        
        current = self.detailed_history[-1]
        prev = self.detailed_history[-2]
        
        # メイン通路から横道へ（利用可能方向が減少）
        if prev['available_directions'] >= 3 and current['available_directions'] == 2:
            if not self.current_deadend_entry:
                self.current_deadend_entry = {
                    'start_step': self.step_count,
                    'entry_pos': prev['position'],
                    'gedig_at_entry': current['gedig'],
                    'gedig_history': [current['gedig']]
                }
    
    def _check_deadend_hit(self):
        """袋小路の突き当たり到達を検出"""
        if not self.current_deadend_entry:
            return
        
        current = self.detailed_history[-1]
        
        # 突き当たり（利用可能方向が1つだけ = 来た道のみ）
        if current['available_directions'] == 1:
            phase = DeadEndPhase(
                phase_type='exploration_to_hit',
                start_step=self.current_deadend_entry['start_step'],
                end_step=self.step_count,
                start_pos=self.current_deadend_entry['entry_pos'],
                end_pos=current['position'],
                gedig_values=self.current_deadend_entry['gedig_history'],
                avg_gedig=np.mean(self.current_deadend_entry['gedig_history']),
                min_gedig=min(self.current_deadend_entry['gedig_history']),
                max_gedig=max(self.current_deadend_entry['gedig_history'])
            )
            self.deadend_phases.append(phase)
            
            # 突き当たり後のフェーズ開始
            self.current_deadend_entry = {
                'start_step': self.step_count,
                'entry_pos': current['position'],
                'gedig_at_entry': current['gedig'],
                'gedig_history': [current['gedig']],
                'is_hit': True
            }
    
    def _check_backtrack_start(self):
        """バックトラック開始を検出"""
        if not self.current_deadend_entry:
            return
        
        if self.current_deadend_entry.get('is_hit'):
            # 突き当たり後の引き返し
            phase = DeadEndPhase(
                phase_type='hit_to_backtrack',
                start_step=self.current_deadend_entry['start_step'],
                end_step=self.step_count,
                start_pos=self.current_deadend_entry['entry_pos'],
                end_pos=self.detailed_history[-1]['position'],
                gedig_values=self.current_deadend_entry['gedig_history'],
                avg_gedig=np.mean(self.current_deadend_entry['gedig_history']),
                min_gedig=min(self.current_deadend_entry['gedig_history']),
                max_gedig=max(self.current_deadend_entry['gedig_history'])
            )
            self.deadend_phases.append(phase)
            self.current_deadend_entry = None


def analyze_phases(analyzer):
    """フェーズ別の詳細分析"""
    
    # フェーズタイプ別に分類
    phase_groups = {
        'exploration_to_hit': [],
        'hit_to_backtrack': []
    }
    
    for phase in analyzer.deadend_phases:
        if phase.phase_type in phase_groups:
            phase_groups[phase.phase_type].append(phase)
    
    # 統計計算
    stats = {}
    for phase_type, phases in phase_groups.items():
        if phases:
            all_gedigs = []
            for p in phases:
                all_gedigs.extend(p.gedig_values)
            
            stats[phase_type] = {
                'count': len(phases),
                'avg_gedig': np.mean(all_gedigs),
                'std_gedig': np.std(all_gedigs),
                'min_gedig': min(all_gedigs),
                'max_gedig': max(all_gedigs),
                'avg_duration': np.mean([p.end_step - p.start_step for p in phases])
            }
    
    return stats


def visualize_phase_analysis(analyzer, maze):
    """フェーズ別分析の可視化"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. 迷路とパス
    ax1 = plt.subplot(3, 4, 1)
    h, w = maze.shape
    
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax1.add_patch(rect)
    
    if analyzer.path:
        path_x = [p[0] for p in analyzer.path]
        path_y = [p[1] for p in analyzer.path]
        ax1.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.5)
        
        # 壁衝突点をマーク
        for hit in analyzer.wall_hits:
            ax1.plot(hit['position'][0], hit['position'][1], 
                    'rx', markersize=8, markeredgewidth=2)
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title('Maze Path (red X = wall hits)')
    ax1.grid(True, alpha=0.3)
    
    # 2. geDIG時系列（フェーズ別色分け）
    ax2 = plt.subplot(3, 4, (2, 4))
    
    if analyzer.gedig_history:
        steps = range(len(analyzer.gedig_history))
        ax2.plot(steps, analyzer.gedig_history, 'k-', linewidth=0.5, alpha=0.3)
        
        # フェーズ別に色分け
        colors = {'exploration_to_hit': 'blue', 'hit_to_backtrack': 'red'}
        for phase in analyzer.deadend_phases:
            phase_steps = range(phase.start_step, min(phase.end_step + 1, len(analyzer.gedig_history)))
            phase_gedigs = analyzer.gedig_history[phase.start_step:min(phase.end_step + 1, len(analyzer.gedig_history))]
            if phase_gedigs:
                ax2.plot(phase_steps, phase_gedigs, 
                        color=colors.get(phase.phase_type, 'gray'),
                        linewidth=2, alpha=0.8,
                        label=phase.phase_type)
        
        # 壁衝突をマーク
        for hit in analyzer.wall_hits:
            if hit['step'] < len(analyzer.gedig_history):
                ax2.plot(hit['step'], hit['gedig'], 'ro', markersize=8)
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=-0.15, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('geDIG Value')
        ax2.set_title('geDIG Evolution by Phase')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. フェーズ別geDIG分布（箱ひげ図）
    ax3 = plt.subplot(3, 4, 5)
    
    phase_data = {
        'Exploring': [],
        'Hit Wall': [],
        'Backtrack': []
    }
    
    for phase in analyzer.deadend_phases:
        if phase.phase_type == 'exploration_to_hit':
            phase_data['Exploring'].extend(phase.gedig_values[:len(phase.gedig_values)//2])
            phase_data['Hit Wall'].extend(phase.gedig_values[len(phase.gedig_values)//2:])
        elif phase.phase_type == 'hit_to_backtrack':
            phase_data['Backtrack'].extend(phase.gedig_values)
    
    data_to_plot = [v for v in phase_data.values() if v]
    labels = [k for k, v in phase_data.items() if v]
    
    if data_to_plot:
        bp = ax3.boxplot(data_to_plot, tick_labels=labels)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('geDIG Value')
        ax3.set_title('geDIG Distribution by Phase')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. geDIG変化率
    ax4 = plt.subplot(3, 4, 6)
    
    if len(analyzer.gedig_history) > 1:
        gedig_diff = np.diff(analyzer.gedig_history)
        steps = range(1, len(analyzer.gedig_history))
        ax4.plot(steps, gedig_diff, 'g-', linewidth=1, alpha=0.7)
        
        # 壁衝突時の変化率をマーク
        for hit in analyzer.wall_hits:
            if 0 < hit['step'] < len(gedig_diff):
                ax4.plot(hit['step'], gedig_diff[hit['step']-1], 'ro', markersize=8)
        
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('geDIG Change Rate')
        ax4.set_title('geDIG Derivative')
        ax4.grid(True, alpha=0.3)
    
    # 5. 壁衝突時のgeDIG値分布
    ax5 = plt.subplot(3, 4, 7)
    
    if analyzer.wall_hits:
        wall_hit_gedigs = [hit['gedig'] for hit in analyzer.wall_hits]
        ax5.hist(wall_hit_gedigs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax5.axvline(x=np.mean(wall_hit_gedigs), color='darkred', linestyle='--', 
                   label=f'Mean: {np.mean(wall_hit_gedigs):.3f}')
        ax5.set_xlabel('geDIG Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('geDIG at Wall Hits')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 方向転換時のgeDIG値
    ax6 = plt.subplot(3, 4, 8)
    
    if analyzer.direction_changes:
        turn_gedigs = [turn['gedig'] for turn in analyzer.direction_changes]
        ax6.hist(turn_gedigs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax6.axvline(x=np.mean(turn_gedigs), color='darkorange', linestyle='--',
                   label=f'Mean: {np.mean(turn_gedigs):.3f}')
        ax6.set_xlabel('geDIG Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('geDIG at Direction Changes')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7-8. フェーズ統計サマリー
    ax7 = plt.subplot(3, 4, (9, 10))
    ax7.axis('off')
    
    stats = analyze_phases(analyzer)
    
    text = "Phase Statistics Summary:\n\n"
    for phase_type, stat in stats.items():
        text += f"{phase_type}:\n"
        text += f"  Count: {stat['count']}\n"
        text += f"  Avg geDIG: {stat['avg_gedig']:.4f} ± {stat['std_gedig']:.4f}\n"
        text += f"  Range: [{stat['min_gedig']:.4f}, {stat['max_gedig']:.4f}]\n"
        text += f"  Avg Duration: {stat['avg_duration']:.1f} steps\n\n"
    
    ax7.text(0.1, 0.9, text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax7.set_title('Phase Statistics')
    
    # 9. 詳細イベントログ
    ax8 = plt.subplot(3, 4, (11, 12))
    ax8.axis('off')
    
    text = "Event Log:\n\n"
    text += f"Total Wall Hits: {len(analyzer.wall_hits)}\n"
    text += f"Total Direction Changes: {len(analyzer.direction_changes)}\n"
    text += f"Total Phases Detected: {len(analyzer.deadend_phases)}\n\n"
    
    if analyzer.wall_hits[:3]:
        text += "First 3 Wall Hits:\n"
        for i, hit in enumerate(analyzer.wall_hits[:3]):
            text += f"  {i+1}. Step {hit['step']}: {hit['position']}, geDIG={hit['gedig']:.4f}\n"
    
    ax8.text(0.1, 0.9, text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Event Details')
    
    plt.suptitle('Dead-end Phase Analysis: Exploration vs Hit vs Backtrack', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    print("="*60)
    print("DEAD-END PHASE ANALYSIS")
    print("Comparing: Exploration -> Wall Hit -> Backtrack")
    print("="*60)
    
    # 迷路作成
    maze = create_controlled_deadend_maze()
    start_pos = (6, 11)
    goal_pos = (11, 1)
    
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
                neighbors = sum(1 for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]
                              if 0 <= x+dx < w and 0 <= y+dy < h and maze[y+dy, x+dx] == 0)
                if neighbors == 1:
                    row += "◎ "  # 袋小路の端
                else:
                    row += "· "
        print(row)
    
    print("\n分析開始...")
    print("-"*60)
    
    # フェーズ分析用ナビゲーター
    analyzer = DeadEndPhaseAnalyzer(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=-0.15,
        wiring_strategy='simple'
    )
    
    # 実行
    success = analyzer.run(max_steps=300)
    
    # 結果表示
    print(f"\n結果: {'成功' if success else '失敗'}")
    print(f"総ステップ数: {analyzer.step_count}")
    print(f"壁衝突回数: {len(analyzer.wall_hits)}")
    print(f"方向転換回数: {len(analyzer.direction_changes)}")
    print(f"検出フェーズ数: {len(analyzer.deadend_phases)}")
    
    # フェーズ別統計
    stats = analyze_phases(analyzer)
    
    print("\n" + "="*60)
    print("PHASE-WISE ANALYSIS")
    print("="*60)
    
    for phase_type, stat in stats.items():
        print(f"\n{phase_type}:")
        print(f"  発生回数: {stat['count']}")
        print(f"  平均geDIG: {stat['avg_gedig']:.4f}")
        print(f"  標準偏差: {stat['std_gedig']:.4f}")
        print(f"  最小値: {stat['min_gedig']:.4f}")
        print(f"  最大値: {stat['max_gedig']:.4f}")
        print(f"  平均継続時間: {stat['avg_duration']:.1f} steps")
    
    # 重要な発見
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    if analyzer.wall_hits and analyzer.gedig_history:
        wall_hit_gedigs = [hit['gedig'] for hit in analyzer.wall_hits]
        print(f"\n壁衝突時の平均geDIG: {np.mean(wall_hit_gedigs):.4f}")
        
        # 壁衝突前後の変化
        for hit in analyzer.wall_hits[:3]:
            step = hit['step']
            if 0 < step < len(analyzer.gedig_history) - 1:
                before = analyzer.gedig_history[step-1]
                at_hit = analyzer.gedig_history[step]
                after = analyzer.gedig_history[step+1] if step+1 < len(analyzer.gedig_history) else at_hit
                
                print(f"\n壁衝突 Step {step}:")
                print(f"  衝突前: {before:.4f}")
                print(f"  衝突時: {at_hit:.4f}")
                print(f"  衝突後: {after:.4f}")
                print(f"  変化量: {at_hit - before:.4f} → {after - at_hit:.4f}")
    
    # 可視化
    fig = visualize_phase_analysis(analyzer, maze)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/phase_analysis_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()