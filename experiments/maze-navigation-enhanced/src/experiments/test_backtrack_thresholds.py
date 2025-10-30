#!/usr/bin/env python3
"""
バックトラック閾値設計の検証実験
Different threshold combinations to trigger backtracking
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
import random


def generate_challenging_maze(size: int, seed: int = None) -> np.ndarray:
    """再帰的バックトラッキングで複雑な迷路を生成"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    maze = np.ones((size, size), dtype=int)
    
    def carve(x, y):
        maze[y, x] = 0
    
    # Start carving from (1, 1)
    carve(1, 1)
    
    def neighbors(cx, cy):
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                yield nx, ny, dx, dy
    
    stack = [(1, 1)]
    visited = {stack[0]}
    
    while stack:
        x, y = stack[-1]
        nbs = [(nx, ny, dx, dy) for nx, ny, dx, dy in neighbors(x, y) if (nx, ny) not in visited]
        
        if not nbs:
            stack.pop()
            continue
        
        nx, ny, dx, dy = random.choice(nbs)
        maze[y + dy // 2, x + dx // 2] = 0
        maze[ny, nx] = 0
        visited.add((nx, ny))
        stack.append((nx, ny))
    
    # 意図的に行き止まりを増やす（バックトラック誘発）
    # いくつかの通路を塞ぐ
    for _ in range(size // 3):
        x = random.randint(2, size - 3)
        y = random.randint(2, size - 3)
        if maze[y, x] == 0:
            # 周囲の開いているセルをカウント
            open_count = sum(1 for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]
                           if 0 <= y+dy < size and 0 <= x+dx < size and maze[y+dy, x+dx] == 0)
            # 通路の接続点でない場合のみ塞ぐ
            if open_count <= 2:
                maze[y, x] = 1
    
    # ゴールへの最低限のパスを確保
    maze[size - 2, size - 2] = 0
    maze[size - 3, size - 2] = 0
    maze[size - 2, size - 3] = 0
    
    return maze


def run_threshold_experiment(
    maze_size: int,
    seed: int,
    wiring_strategy: str,
    gedig_threshold: float,
    backtrack_threshold: float,
    max_steps: int = 1000
) -> Dict:
    """特定の閾値組み合わせで実験を実行"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    # チャレンジング迷路を生成
    maze = generate_challenging_maze(maze_size, seed)
    
    start = (1, 1)
    goal = (maze_size - 2, maze_size - 2)
    
    # Navigator作成
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=wiring_strategy,
        gedig_threshold=gedig_threshold,
        backtrack_threshold=backtrack_threshold,
        simple_mode=True,
        backtrack_debounce=True,
        use_escalation=True
    )
    
    # ナビゲーション実行
    start_time = time.time()
    path = []
    backtrack_events = []
    dead_end_events = []
    gedig_history = []
    
    for step in range(max_steps):
        action = nav.step()
        path.append(nav.current_pos)
        
        # geDIG値を記録
        if hasattr(nav, 'gedig_history') and nav.gedig_history:
            gedig_history = nav.gedig_history.copy()
        
        # イベント追跡
        events = getattr(nav, 'event_log', [])
        for event in events:
            if event.get('type') == 'backtrack_trigger' and event not in backtrack_events:
                backtrack_events.append({
                    'step': step,
                    'position': nav.current_pos,
                    'gedig_value': gedig_history[-1] if gedig_history else None
                })
            elif event.get('type') == 'dead_end_detected' and event not in dead_end_events:
                dead_end_events.append({
                    'step': step,
                    'position': nav.current_pos
                })
        
        if nav.current_pos == goal:
            break
    
    end_time = time.time()
    
    # 結果収集
    result = {
        'maze_size': maze_size,
        'seed': seed,
        'wiring_strategy': wiring_strategy,
        'gedig_threshold': gedig_threshold,
        'backtrack_threshold': backtrack_threshold,
        'goal_reached': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / len(set(path)) if set(path) else 0,
        'time_seconds': end_time - start_time,
        'backtrack_count': len(backtrack_events),
        'dead_end_count': len(dead_end_events),
        'backtrack_events': backtrack_events[:5],  # 最初の5つ
        'gedig_stats': {
            'mean': np.mean(gedig_history) if gedig_history else 0,
            'min': np.min(gedig_history) if gedig_history else 0,
            'max': np.max(gedig_history) if gedig_history else 0,
            'below_bt_threshold': sum(1 for g in gedig_history if g < backtrack_threshold) if gedig_history else 0
        }
    }
    
    return result


def test_threshold_combinations():
    """様々な閾値組み合わせをテスト"""
    
    print("=" * 80)
    print("バックトラック閾値設計検証実験")
    print("=" * 80)
    
    results = []
    
    # テストする閾値の組み合わせ
    threshold_combinations = [
        # (gedig_threshold, backtrack_threshold, description)
        (-0.05, -0.1, "緩い設定（バックトラック多発想定）"),
        (-0.1, -0.2, "中間設定"),
        (-0.15, -0.3, "デフォルト設定"),
        (-0.2, -0.4, "厳しい設定（バックトラック少）"),
        (-0.25, -0.5, "非常に厳しい設定"),
    ]
    
    maze_size = 15
    seeds = [42, 123, 456]
    
    for gedig_th, bt_th, description in threshold_combinations:
        print(f"\n{description}")
        print(f"  geDIG閾値: {gedig_th}, バックトラック閾値: {bt_th}")
        print("-" * 60)
        
        strategy_results = {'simple': [], 'gedig_optimized': []}
        
        for seed in seeds:
            print(f"  Seed {seed}:")
            
            # Simple戦略
            result_simple = run_threshold_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='simple',
                gedig_threshold=gedig_th,
                backtrack_threshold=bt_th,
                max_steps=2000
            )
            strategy_results['simple'].append(result_simple)
            
            print(f"    Simple:          {'✓' if result_simple['goal_reached'] else '✗'} "
                  f"steps={result_simple['steps']:>4}, "
                  f"BT={result_simple['backtrack_count']:>2}, "
                  f"DE={result_simple['dead_end_count']:>2}")
            
            # geDIG最適化戦略
            result_gedig = run_threshold_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='gedig_optimized',
                gedig_threshold=gedig_th,
                backtrack_threshold=bt_th,
                max_steps=2000
            )
            strategy_results['gedig_optimized'].append(result_gedig)
            
            gedig_info = f"(min={result_gedig['gedig_stats']['min']:.3f})"
            print(f"    geDIG_optimized: {'✓' if result_gedig['goal_reached'] else '✗'} "
                  f"steps={result_gedig['steps']:>4}, "
                  f"BT={result_gedig['backtrack_count']:>2}, "
                  f"DE={result_gedig['dead_end_count']:>2} {gedig_info}")
            
            # バックトラック差分
            bt_diff = result_gedig['backtrack_count'] - result_simple['backtrack_count']
            if bt_diff != 0:
                print(f"      → バックトラック差: {bt_diff:+d}")
        
        # 統計サマリー
        for strategy in ['simple', 'gedig_optimized']:
            res_list = strategy_results[strategy]
            avg_bt = sum(r['backtrack_count'] for r in res_list) / len(res_list)
            avg_steps = sum(r['steps'] for r in res_list if r['goal_reached']) / max(1, sum(1 for r in res_list if r['goal_reached']))
            success_rate = sum(1 for r in res_list if r['goal_reached']) / len(res_list) * 100
            
            print(f"    {strategy:15} 平均: BT={avg_bt:.1f}, steps={avg_steps:.0f}, 成功率={success_rate:.0f}%")
        
        results.extend(strategy_results['simple'])
        results.extend(strategy_results['gedig_optimized'])
    
    # 結果保存
    os.makedirs('results/backtrack_threshold_test', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/backtrack_threshold_test/results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n結果を保存: {output_file}")
    
    # 最適な閾値の提案
    print("\n" + "=" * 80)
    print("推奨閾値設定")
    print("=" * 80)
    
    # バックトラックが適度に発生する組み合わせを探す
    best_combination = None
    best_score = float('inf')
    
    for res in results:
        if res['wiring_strategy'] == 'gedig_optimized' and res['goal_reached']:
            # バックトラックが1-5回の範囲が理想的
            if 1 <= res['backtrack_count'] <= 5:
                score = abs(res['backtrack_count'] - 3) + res['redundancy']
                if score < best_score:
                    best_score = score
                    best_combination = res
    
    if best_combination:
        print(f"最適な設定:")
        print(f"  geDIG閾値: {best_combination['gedig_threshold']}")
        print(f"  バックトラック閾値: {best_combination['backtrack_threshold']}")
        print(f"  平均バックトラック: {best_combination['backtrack_count']}回")
        print(f"  ステップ効率: {best_combination['steps']}歩")


if __name__ == '__main__':
    test_threshold_combinations()