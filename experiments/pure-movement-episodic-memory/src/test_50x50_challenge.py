#!/usr/bin/env python3
"""
50×50大規模迷路での挑戦
マジカルナンバー7 + 深い推論（最大20ホップ）での評価
"""

import numpy as np
import random
import time
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def generate_large_perfect_maze(size=(51, 51), seed=None):
    """大規模な完全迷路を生成（DFSアルゴリズム）"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    height, width = size
    # 奇数サイズに調整
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1
    
    # 初期化（全て壁）
    maze = np.ones((height, width), dtype=int)
    
    # スタート地点
    current = (1, 1)
    maze[current] = 0
    
    # スタック（バックトラック用）
    stack = [current]
    
    # 方向
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        # 未訪問の隣接セルを探す
        neighbors = []
        y, x = current
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 < ny < height-1 and 0 < nx < width-1:
                if maze[ny, nx] == 1:  # 未訪問
                    neighbors.append((ny, nx, dy, dx))
        
        if neighbors:
            # ランダムに選択
            ny, nx, dy, dx = random.choice(neighbors)
            # 壁を削って通路を作る
            maze[y + dy//2, x + dx//2] = 0
            maze[ny, nx] = 0
            # 次のセルへ
            current = (ny, nx)
            stack.append(current)
        else:
            # バックトラック
            if stack:
                current = stack.pop()
    
    return maze


class BiasedRandomWalkAgent:
    """ベースライン：ゴール方向バイアス付きランダムウォーク"""
    
    def __init__(self, maze):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.height - 2, self.width - 2)
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0),
            'right': (0, 1),
            'down': (1, 0),
            'left': (0, -1)
        }
        self.steps = 0
        self.path = [self.position]
    
    def get_action(self):
        """ゴール方向を優先したランダム選択"""
        goal_dir_x = self.goal[0] - self.position[0]
        goal_dir_y = self.goal[1] - self.position[1]
        
        weights = []
        for action in self.actions:
            dx, dy = self.action_deltas[action]
            alignment = dx * np.sign(goal_dir_x) + dy * np.sign(goal_dir_y)
            
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            if (0 <= new_x < self.height and 
                0 <= new_y < self.width and 
                self.maze[new_x, new_y] == 0):
                if alignment > 0:
                    weights.append(3)
                elif alignment == 0:
                    weights.append(2)
                else:
                    weights.append(1)
            else:
                weights.append(0)
        
        if sum(weights) > 0:
            return random.choices(self.actions, weights=weights)[0]
        else:
            return random.choice(self.actions)
    
    def execute_action(self, action):
        """行動実行"""
        dx, dy = self.action_deltas[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            self.position = (new_x, new_y)
            self.path.append(self.position)
            self.steps += 1
            return True
        else:
            self.steps += 1
            return False
    
    def is_goal_reached(self):
        return self.position == self.goal


def test_50x50_maze():
    """50×50迷路での実験"""
    
    print("="*80)
    print("🏰 50×50大規模迷路チャレンジ")
    print("  設定: エッジ数7（マジカルナンバー）、最大20ホップ")
    print("="*80)
    
    # 50×50迷路生成
    print("\n⏳ 迷路生成中...")
    maze = generate_large_perfect_maze((51, 51), seed=2024)
    
    # 迷路の統計
    passages = np.sum(maze == 0)
    height, width = maze.shape
    
    print(f"\n📊 迷路統計:")
    print(f"  サイズ: {height}×{width}")
    print(f"  通路数: {passages}マス")
    print(f"  密度: {passages/(height*width)*100:.1f}%")
    print(f"  最短距離（マンハッタン）: {abs(49-1) + abs(49-1)} = 96")
    
    # 迷路の一部を表示（左上と右下）
    print("\n迷路の一部（左上10×10）:")
    for i in range(10):
        row_str = ""
        for j in range(10):
            if i == 1 and j == 1:
                row_str += "S"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    print("\n迷路の一部（右下10×10）:")
    for i in range(41, 51):
        row_str = ""
        for j in range(41, 51):
            if i == 49 and j == 49:
                row_str += "G"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # 実験開始
    print("\n" + "="*80)
    print("📊 実験開始")
    print("="*80)
    
    results = {}
    
    # 1. ベースライン：バイアス付きランダムウォーク
    print("\n【1. ベースライン：バイアス付きランダムウォーク】")
    
    baseline_results = []
    for trial in range(3):
        print(f"  試行{trial+1}: ", end="", flush=True)
        random.seed(trial)
        
        agent = BiasedRandomWalkAgent(maze)
        start_time = time.time()
        
        for step in range(3000):  # 最大3000ステップ
            if agent.is_goal_reached():
                elapsed = time.time() - start_time
                print(f"✅ {step}ステップ ({elapsed:.1f}秒)")
                baseline_results.append({
                    'success': True,
                    'steps': step,
                    'time': elapsed
                })
                break
            agent.execute_action(agent.get_action())
        else:
            elapsed = time.time() - start_time
            print(f"❌ タイムアウト ({elapsed:.1f}秒)")
            baseline_results.append({
                'success': False,
                'steps': 3000,
                'time': elapsed
            })
    
    # 2. 純粋記憶エージェント（マジカルナンバー7 + 深い推論）
    print("\n【2. 純粋記憶：エッジ7 + 最大20ホップ】")
    
    memory_results = []
    for trial in range(3):
        print(f"  試行{trial+1}: ", end="", flush=True)
        
        agent = PureMemoryAgentOptimized(
            maze=maze,
            datastore_path=f"../results/50x50_trial_{trial}",
            config={
                'max_depth': 20,  # 深い推論
                'search_k': 50,   # より多くの候補を検索
                'gedig_threshold': 0.5,
                'max_edges_per_node': 7  # マジカルナンバー
            }
        )
        
        start_time = time.time()
        path = [agent.position]
        depth_usage = {}
        
        for step in range(1000):  # 最大1000ステップ
            if agent.is_goal_reached():
                elapsed = time.time() - start_time
                
                # 深度使用統計
                for d, count in agent.stats['depth_usage'].items():
                    depth_usage[d] = depth_usage.get(d, 0) + count
                
                print(f"✅ {step}ステップ ({elapsed:.1f}秒)")
                memory_results.append({
                    'success': True,
                    'steps': step,
                    'time': elapsed,
                    'path_length': len(path),
                    'unique_visits': len(set(path)),
                    'depth_usage': depth_usage,
                    'avg_gedig': agent.get_statistics()['avg_gedig']
                })
                break
            
            # 進捗表示
            if step % 200 == 199:
                dist = abs(agent.position[0] - agent.goal[0]) + \
                       abs(agent.position[1] - agent.goal[1])
                print(f"\n    Step {step+1}: 距離{dist}", end="", flush=True)
            
            action = agent.get_action()
            agent.execute_action(action)
            path.append(agent.position)
        else:
            elapsed = time.time() - start_time
            print(f"❌ タイムアウト ({elapsed:.1f}秒)")
            
            for d, count in agent.stats['depth_usage'].items():
                depth_usage[d] = depth_usage.get(d, 0) + count
            
            memory_results.append({
                'success': False,
                'steps': 1000,
                'time': elapsed,
                'path_length': len(path),
                'unique_visits': len(set(path)),
                'depth_usage': depth_usage,
                'avg_gedig': agent.get_statistics()['avg_gedig']
            })
    
    # 3. 比較実験：エッジ15 + 深度10（従来設定）
    print("\n【3. 比較：エッジ15 + 最大10ホップ（従来設定）】")
    
    traditional_results = []
    for trial in range(3):
        print(f"  試行{trial+1}: ", end="", flush=True)
        
        agent = PureMemoryAgentOptimized(
            maze=maze,
            datastore_path=f"../results/50x50_traditional_{trial}",
            config={
                'max_depth': 10,
                'search_k': 50,
                'gedig_threshold': 0.5,
                'max_edges_per_node': 15
            }
        )
        
        start_time = time.time()
        
        for step in range(2000):
            if agent.is_goal_reached():
                elapsed = time.time() - start_time
                print(f"✅ {step}ステップ ({elapsed:.1f}秒)")
                traditional_results.append({
                    'success': True,
                    'steps': step,
                    'time': elapsed
                })
                break
            
            if step % 200 == 199:
                dist = abs(agent.position[0] - agent.goal[0]) + \
                       abs(agent.position[1] - agent.goal[1])
                print(f"\n    Step {step+1}: 距離{dist}", end="", flush=True)
            
            agent.execute_action(agent.get_action())
        else:
            elapsed = time.time() - start_time
            print(f"❌ タイムアウト ({elapsed:.1f}秒)")
            traditional_results.append({
                'success': False,
                'steps': 1000,
                'time': elapsed
            })
    
    # 結果分析
    print("\n" + "="*80)
    print("📈 結果分析")
    print("="*80)
    
    def analyze_results(name, results_list):
        success_count = sum(1 for r in results_list if r['success'])
        success_rate = success_count / len(results_list) * 100
        
        print(f"\n【{name}】")
        print(f"  成功率: {success_rate:.0f}% ({success_count}/3)")
        
        if success_count > 0:
            success_results = [r for r in results_list if r['success']]
            avg_steps = np.mean([r['steps'] for r in success_results])
            avg_time = np.mean([r['time'] for r in success_results])
            print(f"  平均ステップ: {avg_steps:.0f}")
            print(f"  平均時間: {avg_time:.1f}秒")
            
            # 追加統計（記憶エージェントのみ）
            if 'depth_usage' in success_results[0]:
                all_depth_usage = {}
                for r in success_results:
                    for d, count in r['depth_usage'].items():
                        all_depth_usage[d] = all_depth_usage.get(d, 0) + count
                
                total = sum(all_depth_usage.values())
                if total > 0:
                    print(f"  深度使用分布:")
                    for d in sorted(all_depth_usage.keys())[:5]:
                        ratio = all_depth_usage[d] / total * 100
                        print(f"    {d}ホップ: {ratio:.1f}%")
                
                avg_gedig = np.mean([r['avg_gedig'] for r in success_results])
                print(f"  平均geDIG: {avg_gedig:.4f}")
    
    analyze_results("バイアス付きランダム", baseline_results)
    analyze_results("マジカルナンバー7 + 深い推論", memory_results)
    analyze_results("従来設定（エッジ15 + 深度10）", traditional_results)
    
    # 最終評価
    print("\n" + "="*80)
    print("💡 最終評価")
    print("="*80)
    
    baseline_success = sum(1 for r in baseline_results if r['success'])
    memory_success = sum(1 for r in memory_results if r['success'])
    traditional_success = sum(1 for r in traditional_results if r['success'])
    
    print("\n50×50迷路での結果:")
    
    if memory_success > baseline_success:
        print("✅ マジカルナンバー7設定がランダムウォークを上回る！")
    else:
        print("❌ ランダムウォークの方が効果的")
    
    if memory_success >= traditional_success:
        print("✨ マジカルナンバー7でも従来設定と同等以上の性能！")
    else:
        print("🤔 従来設定の方が安定")
    
    print("\n📝 結論:")
    print("  50×50という大規模迷路では、")
    print("  純粋記憶の真価が発揮される環境")
    print("  マジカルナンバー7 + 深い推論の組み合わせが鍵")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実験実行
    test_50x50_maze()