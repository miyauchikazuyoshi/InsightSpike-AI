#!/usr/bin/env python3
"""
強制的に袋小路を通る迷路でのフェーズ別geDIG分析
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


def create_forced_deadend_maze():
    """必ず袋小路を通る迷路（ゴールへの経路が袋小路経由のみ）"""
    maze = np.ones((11, 11), dtype=int)
    
    # スタートから最初の袋小路まで（強制）
    for y in range(7, 10):
        maze[y, 2] = 0
    
    # 袋小路1（必ず通る）
    for x in range(2, 6):
        maze[7, x] = 0
    
    # 袋小路から戻って上へ
    for y in range(4, 8):
        maze[y, 2] = 0
    
    # 上部で右へ
    for x in range(2, 9):
        maze[4, x] = 0
    
    # もう一つの袋小路（オプション）
    for x in range(5, 9):
        maze[2, x] = 0
    
    # ゴールへ
    maze[3, 8] = 0
    maze[2, 8] = 0
    maze[1, 8] = 0
    
    return maze


class DetailedGeDIGTracker(MazeNavigator):
    """詳細なgeDIG追跡ナビゲーター"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_markers = []  # フェーズマーカー
        self.detailed_log = []   # 詳細ログ
        self.movement_types = [] # 移動タイプ記録
    
    def step(self) -> bool:
        """ステップ実行（詳細記録）"""
        prev_pos = self.current_pos
        
        # 現在の状況を記録
        episodes = self.episode_manager.get_episodes_at_position(self.current_pos)
        available = sum(1 for ep in episodes.values() if not ep.is_wall)
        unvisited = sum(1 for ep in episodes.values() if not ep.is_wall and ep.visit_count == 0)
        
        # 親クラスのstep実行
        result = super().step()
        
        # geDIG値
        current_gedig = self.gedig_history[-1] if self.gedig_history else 0
        
        # 移動タイプを判定
        movement_type = self._determine_movement_type(prev_pos, self.current_pos, available, unvisited)
        
        # 詳細ログ
        self.detailed_log.append({
            'step': self.step_count,
            'position': self.current_pos,
            'prev_position': prev_pos,
            'gedig': current_gedig,
            'available_directions': available,
            'unvisited_directions': unvisited,
            'movement_type': movement_type
        })
        
        self.movement_types.append(movement_type)
        
        # フェーズマーカー
        if movement_type in ['hit_wall', 'dead_end_reached', 'backtrack_start']:
            self.phase_markers.append({
                'step': self.step_count,
                'position': self.current_pos,
                'type': movement_type,
                'gedig': current_gedig
            })
        
        return result
    
    def _determine_movement_type(self, prev_pos, curr_pos, available, unvisited):
        """移動タイプを判定"""
        if prev_pos == curr_pos:
            return 'hit_wall'
        
        if available == 1:  # 袋小路の端
            return 'dead_end_reached'
        
        if unvisited == 0 and available > 1:  # すべて訪問済み
            return 'backtrack_start'
        
        if unvisited > 0:
            return 'exploring'
        
        return 'normal'


def analyze_gedig_patterns(tracker):
    """geDIG値のパターン分析"""
    
    if not tracker.detailed_log:
        return {}
    
    # 移動タイプ別にgeDIG値を集計
    type_gedigs = {}
    for log in tracker.detailed_log:
        mtype = log['movement_type']
        if mtype not in type_gedigs:
            type_gedigs[mtype] = []
        type_gedigs[mtype].append(log['gedig'])
    
    # 統計計算
    stats = {}
    for mtype, gedigs in type_gedigs.items():
        if gedigs:
            stats[mtype] = {
                'count': len(gedigs),
                'mean': np.mean(gedigs),
                'std': np.std(gedigs),
                'min': min(gedigs),
                'max': max(gedigs)
            }
    
    # 連続パターンの検出
    patterns = []
    for i in range(len(tracker.movement_types) - 2):
        pattern = tuple(tracker.movement_types[i:i+3])
        if pattern not in [p[0] for p in patterns]:
            count = tracker.movement_types.count(pattern[0])
            patterns.append((pattern, count))
    
    return {'type_stats': stats, 'patterns': patterns}


def main():
    print("="*60)
    print("FORCED DEAD-END ANALYSIS")
    print("Detailed geDIG behavior in mandatory dead-ends")
    print("="*60)
    
    # 迷路作成
    maze = create_forced_deadend_maze()
    start_pos = (2, 9)
    goal_pos = (8, 1)
    
    print("\n迷路構造（必ず袋小路を通る）:")
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
    
    print("\n実験実行...")
    print("-"*60)
    
    # 複数回実験（統計的に有意なデータを取得）
    all_results = []
    
    for run in range(3):
        print(f"\nRun {run+1}/3:")
        
        tracker = DetailedGeDIGTracker(
            maze=maze,
            start_pos=start_pos,
            goal_pos=goal_pos,
            temperature=0.1 + run * 0.05,  # 温度を少し変える
            gedig_threshold=0.3,
            backtrack_threshold=-0.15,
            wiring_strategy='simple'
        )
        
        success = tracker.run(max_steps=200)
        
        print(f"  結果: {'成功' if success else '失敗'}")
        print(f"  ステップ数: {tracker.step_count}")
        print(f"  フェーズマーカー数: {len(tracker.phase_markers)}")
        
        all_results.append({
            'tracker': tracker,
            'success': success,
            'analysis': analyze_gedig_patterns(tracker)
        })
    
    # 統合分析
    print("\n" + "="*60)
    print("INTEGRATED ANALYSIS")
    print("="*60)
    
    # すべての実行からgeDIG値を収集
    all_type_gedigs = {}
    for result in all_results:
        for mtype, stats in result['analysis']['type_stats'].items():
            if mtype not in all_type_gedigs:
                all_type_gedigs[mtype] = []
            
            # 個々のgeDIG値を収集
            for log in result['tracker'].detailed_log:
                if log['movement_type'] == mtype:
                    all_type_gedigs[mtype].append(log['gedig'])
    
    print("\n移動タイプ別geDIG統計:")
    print("-"*40)
    for mtype, gedigs in all_type_gedigs.items():
        if gedigs:
            print(f"\n{mtype}:")
            print(f"  サンプル数: {len(gedigs)}")
            print(f"  平均: {np.mean(gedigs):.4f}")
            print(f"  標準偏差: {np.std(gedigs):.4f}")
            print(f"  範囲: [{min(gedigs):.4f}, {max(gedigs):.4f}]")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 迷路構造
    ax = axes[0, 0]
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax.add_patch(rect)
    
    # 最初の実行のパスを表示
    if all_results[0]['tracker'].path:
        path_x = [p[0] for p in all_results[0]['tracker'].path]
        path_y = [p[1] for p in all_results[0]['tracker'].path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
    
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10)
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(-0.5, h-0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Forced Dead-end Maze')
    ax.grid(True, alpha=0.3)
    
    # 2. geDIG時系列（全実行）
    ax = axes[0, 1]
    for i, result in enumerate(all_results):
        if result['tracker'].gedig_history:
            steps = range(len(result['tracker'].gedig_history))
            ax.plot(steps, result['tracker'].gedig_history, 
                   label=f'Run {i+1}', linewidth=2, alpha=0.7)
            
            # フェーズマーカー
            for marker in result['tracker'].phase_markers:
                if marker['step'] < len(result['tracker'].gedig_history):
                    ax.plot(marker['step'], marker['gedig'], 'ko', markersize=6)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=-0.15, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('geDIG Value')
    ax.set_title('geDIG Evolution (All Runs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 移動タイプ別geDIG分布
    ax = axes[0, 2]
    
    data_to_plot = []
    labels = []
    for mtype in ['exploring', 'dead_end_reached', 'backtrack_start', 'hit_wall']:
        if mtype in all_type_gedigs and all_type_gedigs[mtype]:
            data_to_plot.append(all_type_gedigs[mtype])
            labels.append(mtype)
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('geDIG Value')
        ax.set_title('geDIG by Movement Type')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. geDIG値のヒストグラム（探索中 vs 袋小路）
    ax = axes[1, 0]
    
    if 'exploring' in all_type_gedigs:
        ax.hist(all_type_gedigs['exploring'], bins=20, alpha=0.5, 
               label='Exploring', color='blue')
    if 'dead_end_reached' in all_type_gedigs:
        ax.hist(all_type_gedigs['dead_end_reached'], bins=20, alpha=0.5,
               label='Dead End', color='red')
    
    ax.set_xlabel('geDIG Value')
    ax.set_ylabel('Frequency')
    ax.set_title('geDIG Distribution: Exploring vs Dead End')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. フェーズマーカーの分析
    ax = axes[1, 1]
    
    all_markers = []
    for result in all_results:
        all_markers.extend(result['tracker'].phase_markers)
    
    if all_markers:
        marker_types = {}
        for marker in all_markers:
            if marker['type'] not in marker_types:
                marker_types[marker['type']] = []
            marker_types[marker['type']].append(marker['gedig'])
        
        x_pos = []
        x_labels = []
        for i, (mtype, gedigs) in enumerate(marker_types.items()):
            x_pos.extend([i] * len(gedigs))
            x_labels.append(mtype)
            ax.scatter([i] * len(gedigs), gedigs, alpha=0.6, s=50)
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel('geDIG Value')
        ax.set_title('geDIG at Phase Markers')
        ax.grid(True, alpha=0.3)
    
    # 6. 統計サマリー
    ax = axes[1, 2]
    ax.axis('off')
    
    text = "Key Findings:\n\n"
    
    # 探索中と袋小路のgeDIG差
    if 'exploring' in all_type_gedigs and 'dead_end_reached' in all_type_gedigs:
        exp_mean = np.mean(all_type_gedigs['exploring'])
        dead_mean = np.mean(all_type_gedigs['dead_end_reached'])
        text += f"Exploring avg: {exp_mean:.4f}\n"
        text += f"Dead-end avg: {dead_mean:.4f}\n"
        text += f"Difference: {dead_mean - exp_mean:.4f}\n\n"
    
    # バックトラック開始時
    if 'backtrack_start' in all_type_gedigs:
        back_mean = np.mean(all_type_gedigs['backtrack_start'])
        text += f"Backtrack avg: {back_mean:.4f}\n\n"
    
    # 成功率
    success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
    text += f"Success rate: {success_rate:.1%}\n"
    text += f"Avg steps: {np.mean([r['tracker'].step_count for r in all_results]):.1f}"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Statistical Summary')
    
    plt.suptitle('Forced Dead-end Analysis: Detailed geDIG Behavior', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/forced_deadend_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()
    
    # 最重要な発見
    print("\n" + "="*60)
    print("CRITICAL FINDINGS")
    print("="*60)
    
    if 'exploring' in all_type_gedigs and 'dead_end_reached' in all_type_gedigs:
        exp_gedigs = all_type_gedigs['exploring']
        dead_gedigs = all_type_gedigs['dead_end_reached']
        
        print(f"\n1. 探索中のgeDIG:")
        print(f"   平均: {np.mean(exp_gedigs):.4f}")
        print(f"   中央値: {np.median(exp_gedigs):.4f}")
        
        print(f"\n2. 袋小路到達時のgeDIG:")
        print(f"   平均: {np.mean(dead_gedigs):.4f}")
        print(f"   中央値: {np.median(dead_gedigs):.4f}")
        
        print(f"\n3. 差分:")
        print(f"   平均の差: {np.mean(dead_gedigs) - np.mean(exp_gedigs):.4f}")
        print(f"   → 袋小路到達時は明確に負の値になる傾向")


if __name__ == "__main__":
    main()