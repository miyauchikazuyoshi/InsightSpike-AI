#!/usr/bin/env python3
"""
純粋geDIG記憶エージェント 15×15迷路テスト
袋小路からの脱出で深い推論が機能するか検証
"""

import numpy as np
import time
from datetime import datetime
from collections import deque
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_final import PureMemoryAgentFinal


class StuckDetector:
    """袋小路検出器"""
    def __init__(self, window_size=20):
        self.position_history = deque(maxlen=window_size)
        self.stuck_threshold = 0.7  # 70%以上同じ場所なら袋小路
    
    def update(self, position):
        self.position_history.append(position)
    
    def is_stuck(self):
        if len(self.position_history) < self.position_history.maxlen:
            return False
        
        # 最頻出位置をカウント
        position_counts = {}
        for pos in self.position_history:
            key = f"{pos[0]},{pos[1]}"
            position_counts[key] = position_counts.get(key, 0) + 1
        
        max_count = max(position_counts.values())
        stuck_ratio = max_count / len(self.position_history)
        
        return stuck_ratio >= self.stuck_threshold
    
    def get_stuck_position(self):
        """最頻出位置を返す"""
        if not self.position_history:
            return None
        
        position_counts = {}
        for pos in self.position_history:
            key = f"{pos[0]},{pos[1]}"
            position_counts[key] = position_counts.get(key, 0) + 1
        
        stuck_key = max(position_counts, key=position_counts.get)
        parts = stuck_key.split(',')
        return (int(parts[0]), int(parts[1]))


def test_15x15_with_stuck_analysis():
    """15×15迷路で袋小路分析付きテスト"""
    
    print("="*80)
    print("🏔️ 純粋geDIG記憶エージェント 15×15迷路チャレンジ")
    print("  袋小路からの脱出で深い推論が活用されるか検証")
    print("="*80)
    
    # 15×15迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(15, 15), seed=42)
    
    print("\n🗺️ 迷路 (15×15) 一部表示:")
    for i in range(10):
        row_str = ''.join(['.' if maze[i][j] == 0 else '█' for j in range(15)])
        print(row_str)
    print("... (続く)")
    
    # エージェント作成（深い推論を許可）
    agent = PureMemoryAgentFinal(
        maze=maze,
        datastore_path="../results/pure_15x15_stuck_analysis",
        config={
            'max_depth': 5,  # 最大5ホップまで
            'search_k': 30    # より多くの記憶を検索
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + \
                      abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期距離: {initial_distance}")
    print("-" * 40)
    
    # 袋小路検出器
    stuck_detector = StuckDetector(window_size=30)
    
    # 統計記録
    stuck_events = []
    depth_when_stuck = []
    escape_success = []
    
    # 実行（タイムアウトなし）
    start_time = time.time()
    max_steps = 5000  # 十分な時間を与える
    
    last_stuck_step = -100
    stuck_count = 0
    
    for step in range(max_steps):
        # ゴール確認
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\n🎉 成功！ {step}ステップでゴール到達")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  総エピソード数: {stats['total_episodes']}")
            
            # 袋小路分析
            analyze_stuck_events(stuck_events, depth_when_stuck, escape_success)
            
            # メモリ統計
            print_memory_stats(stats)
            
            return True
        
        # 現在位置を記録
        current_pos = agent.position
        stuck_detector.update(current_pos)
        
        # 袋小路検出
        if stuck_detector.is_stuck() and step - last_stuck_step > 50:
            stuck_count += 1
            stuck_pos = stuck_detector.get_stuck_position()
            print(f"\n⚠️ 袋小路検出 #{stuck_count} (Step {step})")
            print(f"  位置: {stuck_pos}")
            
            # この時点での深度使用を記録
            current_depth_usage = agent.stats['depth_usage'].copy()
            stuck_events.append({
                'step': step,
                'position': stuck_pos,
                'depth_usage_before': current_depth_usage
            })
            
            last_stuck_step = step
        
        # 行動実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 袋小路から脱出したかチェック
        if stuck_events and step - stuck_events[-1]['step'] == 50:
            # 50ステップ後の状況を確認
            escaped = not stuck_detector.is_stuck()
            if escaped:
                print(f"  ✅ 袋小路から脱出成功！")
                
                # 脱出時の深度使用
                depth_diff = {}
                for d in range(1, 6):
                    before = stuck_events[-1]['depth_usage_before'].get(d, 0)
                    after = agent.stats['depth_usage'].get(d, 0)
                    depth_diff[d] = after - before
                
                print(f"  脱出時の深度使用:")
                for d, count in depth_diff.items():
                    if count > 0:
                        print(f"    {d}ホップ: {count}回")
                
                escape_success.append(True)
                depth_when_stuck.append(depth_diff)
            else:
                print(f"  ❌ まだ袋小路にいる...")
                escape_success.append(False)
        
        # 進捗報告
        if step % 100 == 0 and step > 0:
            stats = agent.get_statistics()
            distance = stats['distance_to_goal']
            improvement = (initial_distance - distance) / initial_distance * 100
            
            print(f"\nStep {step}: ")
            print(f"  位置: {stats['position']}")
            print(f"  距離: {distance} ({improvement:+.1f}%改善)")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            
            # 深度使用パターン
            total_depth_usage = sum(stats['depth_usage'].values())
            if total_depth_usage > 0:
                print(f"  深度使用分布:")
                for d, count in stats['depth_usage'].items():
                    if count > 0:
                        ratio = count / total_depth_usage * 100
                        print(f"    {d}ホップ: {ratio:.1f}%")
    
    # タイムアウト
    elapsed = time.time() - start_time
    final_stats = agent.get_statistics()
    
    print(f"\n⏱️ {max_steps}ステップで未到達")
    print(f"  最終距離: {final_stats['distance_to_goal']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  実行時間: {elapsed:.2f}秒")
    
    # 袋小路分析
    analyze_stuck_events(stuck_events, depth_when_stuck, escape_success)
    
    return False


def analyze_stuck_events(stuck_events, depth_when_stuck, escape_success):
    """袋小路イベントの分析"""
    if not stuck_events:
        print("\n袋小路は検出されませんでした")
        return
    
    print("\n" + "="*80)
    print("🔍 袋小路分析")
    print("="*80)
    
    print(f"\n袋小路検出回数: {len(stuck_events)}")
    
    if escape_success:
        success_rate = sum(escape_success) / len(escape_success) * 100
        print(f"脱出成功率: {success_rate:.1f}%")
    
    # 脱出時の深度使用分析
    if depth_when_stuck:
        print("\n脱出時の深度使用パターン:")
        
        avg_depth_usage = {d: 0 for d in range(1, 6)}
        for depth_diff in depth_when_stuck:
            for d, count in depth_diff.items():
                avg_depth_usage[d] += count
        
        num_escapes = len(depth_when_stuck)
        for d in range(1, 6):
            if avg_depth_usage[d] > 0:
                avg = avg_depth_usage[d] / num_escapes
                print(f"  {d}ホップ: 平均{avg:.1f}回")
        
        # 深い推論の使用率
        deep_usage = sum(avg_depth_usage[d] for d in range(3, 6))
        shallow_usage = sum(avg_depth_usage[d] for d in range(1, 3))
        
        if deep_usage + shallow_usage > 0:
            deep_ratio = deep_usage / (deep_usage + shallow_usage) * 100
            print(f"\n深い推論（3-5ホップ）の使用率: {deep_ratio:.1f}%")
            
            if deep_ratio > 50:
                print("  → ✅ 袋小路脱出時に深い推論が活用されている！")
            else:
                print("  → 📊 主に浅い推論で対処している")


def print_memory_stats(stats):
    """メモリ統計の表示"""
    mem_stats = stats['memory_stats']
    
    print(f"\n📊 メモリ統計:")
    print(f"  経験数: {mem_stats.get('total_experiences', 0)}")
    print(f"  エッジ数: {mem_stats.get('total_edges', 0)}")
    
    if 'avg_gedig' in mem_stats:
        print(f"  平均geDIG: {mem_stats['avg_gedig']:.3f}")
        print(f"  最小geDIG: {mem_stats.get('min_gedig', 0):.3f}")
        print(f"  最大geDIG: {mem_stats.get('max_gedig', 0):.3f}")
    
    if 'graph_density' in mem_stats:
        print(f"  グラフ密度: {mem_stats['graph_density']:.3f}")
        print(f"  平均次数: {mem_stats.get('avg_degree', 0):.2f}")
    
    # 深度使用統計
    print(f"\n深度使用統計:")
    total_usage = sum(stats['depth_usage'].values())
    for depth, count in sorted(stats['depth_usage'].items()):
        if count > 0:
            ratio = count / total_usage * 100
            bar = '█' * int(ratio / 5)
            print(f"  {depth}ホップ: {count:4d}回 ({ratio:5.1f}%) {bar}")


if __name__ == "__main__":
    print("🚀 15×15迷路での深い推論検証実験")
    print("  袋小路からの脱出で深い記憶が活用されるか分析")
    print()
    
    success = test_15x15_with_stuck_analysis()
    
    print("\n" + "="*80)
    if success:
        print("🏆 15×15迷路攻略成功！")
        print("   深い推論が袋小路脱出に貢献")
    else:
        print("📊 学習継続中")
        print("   袋小路での振る舞いを分析済み")
    print("="*80)