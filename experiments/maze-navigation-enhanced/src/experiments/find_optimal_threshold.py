#!/usr/bin/env python3
"""
最適な閾値を探索する実験
NoCopy版GraphManagerで迷路ナビゲーションを最適化
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import random
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def generate_test_maze(size: int, seed: int) -> np.ndarray:
    """テスト用迷路を生成"""
    random.seed(seed)
    np.random.seed(seed)
    
    maze = np.ones((size, size), dtype=int)
    
    def carve(x, y):
        maze[y, x] = 0
    
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
    
    # ゴール確保
    maze[size - 2, size - 2] = 0
    
    return maze


def test_threshold(
    maze: np.ndarray,
    gedig_threshold: float,
    backtrack_threshold: float,
    wiring_strategy: str = 'gedig',
    max_steps: int = 500
) -> Dict:
    """特定の閾値でテスト"""
    
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=wiring_strategy,
        gedig_threshold=gedig_threshold,
        backtrack_threshold=backtrack_threshold,
        simple_mode=True,
        backtrack_debounce=True
    )
    
    path = []
    backtrack_count = 0
    
    for step in range(max_steps):
        action = nav.step()
        path.append(nav.current_pos)
        
        if nav.current_pos == goal:
            break
    
    # 統計収集
    graph_stats = nav.graph_manager.get_graph_statistics()
    
    # geDIG値の統計
    gedig_values = []
    if hasattr(nav.graph_manager, 'edge_logs'):
        gedig_values = [log['gedig'] for log in nav.graph_manager.edge_logs]
    # L1候補数の統計（edge_logs優先、無ければ内部カウンタ）
    k_values = []
    try:
        if hasattr(nav.graph_manager, 'edge_logs'):
            k_values = [int(log['l1_count']) for log in nav.graph_manager.edge_logs if isinstance(log, dict) and 'l1_count' in log]
    except Exception:
        k_values = []
    if not k_values:
        try:
            k_values = list(getattr(nav.graph_manager, '_l1_candidate_counts', []))
        except Exception:
            k_values = []
    k_min = min(k_values) if k_values else 0
    k_max = max(k_values) if k_values else 0
    k_mean = float(np.mean(k_values)) if k_values else 0.0
    
    return {
        'goal_reached': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'graph_edges': graph_stats['num_edges'],
        'gedig_min': min(gedig_values) if gedig_values else 0,
        'gedig_max': max(gedig_values) if gedig_values else 0,
        'gedig_mean': np.mean(gedig_values) if gedig_values else 0,
        'l1_min': int(k_min),
        'l1_max': int(k_max),
        'l1_mean': float(k_mean)
    }


def find_optimal_thresholds():
    """最適な閾値を探索"""
    
    print("=" * 70)
    print("最適な閾値探索実験")
    print("=" * 70)
    
    # テスト迷路を生成
    maze_size = 15
    test_mazes = []
    for seed in [42, 123, 456]:
        maze = generate_test_maze(maze_size, seed)
        test_mazes.append((seed, maze))
    
    # テストする閾値の組み合わせ
    gedig_thresholds = [-0.02, -0.03, -0.04, -0.045, -0.05, -0.06]
    backtrack_thresholds = [-0.1, -0.15, -0.2, -0.25, -0.3]
    
    results = []
    
    print(f"\nテスト: {len(test_mazes)}個の迷路 × {len(gedig_thresholds)}個のgeDIG閾値")
    print("-" * 70)
    
    # まずsimple戦略のベースライン
    print("\n1. ベースライン（simple戦略）")
    baseline_stats = []
    summary_payload = {
        'run_started_at': datetime.now().isoformat(timespec='seconds'),
        'baseline': [],
        'thresholds': {},
        'config': {
            k: os.environ.get(k) for k in [
                'MAZE_GEDIG_LOCAL_NORM','MAZE_L1_NORM_SEARCH','MAZE_L1_WEIGHTED','MAZE_L1_UNIT_NORM',
                'MAZE_L1_FILTER_UNVISITED','MAZE_L1_NORM_TAU','MAZE_L1_WEIGHTS','MAZE_WIRING_WINDOW',
                'MAZE_SPATIAL_GATE','MAZE_USE_HOP_DECISION','MAZE_HOP_DECISION_LEVEL','MAZE_HOP_DECISION_MAX',
                'MAZE_WIRING_TOPK','MAZE_WIRING_MIN_ACCEPT','MAZE_WIRING_FORCE_PREV',
                'MAZE_GEDIG_LAMBDA','MAZE_GEDIG_IG_MODE','MAZE_GEDIG_SP_GAIN'
            ] if os.environ.get(k) is not None
        }
    }
    for seed, maze in test_mazes:
        result = test_threshold(maze, 0, -0.2, wiring_strategy='simple')
        baseline_stats.append(result)
        print(f"  Seed {seed}: {'✓' if result['goal_reached'] else '✗'} "
              f"{result['steps']} steps, {result['graph_edges']} edges")
        summary_payload['baseline'].append({
            'seed': seed,
            'goal_reached': bool(result['goal_reached']),
            'steps': int(result['steps']),
            'graph_edges': int(result['graph_edges'])
        })
    
    baseline_avg_steps = np.mean([r['steps'] for r in baseline_stats if r['goal_reached']])
    print(f"  平均ステップ数: {baseline_avg_steps:.1f}")
    
    # geDIG戦略で各閾値をテスト
    print("\n2. geDIG戦略の閾値探索")
    print("-" * 70)
    
    best_config = None
    best_score = float('inf')
    
    for gedig_th in gedig_thresholds:
        print(f"\ngeDIG閾値 = {gedig_th}")
        
        threshold_results = []
        per_seed_records = []
        for seed, maze in test_mazes:
            result = test_threshold(maze, gedig_th, -0.2, wiring_strategy='gedig')
            threshold_results.append(result)
            
            # スコア計算（ステップ数が少ないほど良い）
            if result['goal_reached']:
                score = result['steps']
            else:
                score = float('inf')
            
            status = '✓' if result['goal_reached'] else '✗'
            print(f"  Seed {seed}: {status} steps={result['steps']:>3}, "
                  f"edges={result['graph_edges']:>3}, "
                  f"geDIG=[{result['gedig_min']:.3f}, {result['gedig_max']:.3f}], "
                  f"K~{result['l1_mean']:.1f} (min={result['l1_min']}, max={result['l1_max']})")
            per_seed_records.append({
                'seed': seed,
                'goal_reached': bool(result['goal_reached']),
                'steps': int(result['steps']),
                'graph_edges': int(result['graph_edges']),
                'gedig_min': float(result['gedig_min']),
                'gedig_max': float(result['gedig_max']),
                'gedig_mean': float(result['gedig_mean']),
                'l1_min': int(result['l1_min']),
                'l1_max': int(result['l1_max']),
                'l1_mean': float(result['l1_mean'])
            })
        
        # この閾値の平均スコア
        avg_score = np.mean([r['steps'] for r in threshold_results if r['goal_reached']])
        success_rate = sum(1 for r in threshold_results if r['goal_reached']) / len(threshold_results)
        
        if success_rate == 1.0 and avg_score < best_score:
            best_score = avg_score
            best_config = {
                'gedig_threshold': gedig_th,
                'avg_steps': avg_score,
                'avg_edges': np.mean([r['graph_edges'] for r in threshold_results])
            }
        summary_payload['thresholds'][str(gedig_th)] = {
            'per_seed': per_seed_records,
            'success_rate': float(success_rate),
            'avg_steps_success_only': float(avg_score) if not np.isnan(avg_score) else None
        }
    
    # 結果まとめ
    print("\n" + "=" * 70)
    print("結果まとめ")
    print("=" * 70)
    
    if best_config:
        print(f"\n最適な設定:")
        print(f"  geDIG閾値: {best_config['gedig_threshold']}")
        print(f"  平均ステップ数: {best_config['avg_steps']:.1f}")
        print(f"  平均エッジ数: {best_config['avg_edges']:.1f}")
        
        improvement = (baseline_avg_steps - best_config['avg_steps']) / baseline_avg_steps * 100
        print(f"\n  ベースライン比較: {improvement:+.1f}% 改善")
        summary_payload['best_config'] = {
            'gedig_threshold': float(best_config['gedig_threshold']),
            'avg_steps': float(best_config['avg_steps']),
            'avg_edges': float(best_config['avg_edges']),
            'improvement_pct_vs_baseline': float(improvement)
        }
    else:
        print("最適な設定が見つかりませんでした")
        summary_payload['best_config'] = None
    
    # 推奨値
    print("\n推奨閾値:")
    print("  geDIG閾値: -0.045")
    print("  バックトラック閾値: -0.2")
    print("\n注: NoCopy版GraphManagerでは、geDIG値は約-0.045付近になります")

    # Optional: write summary when MAZE_WRITE_SUMMARY is truthy
    if os.environ.get('MAZE_WRITE_SUMMARY', '').strip() not in ("", "0", "false", "False"):
        out_root = Path(__file__).resolve().parents[3] / 'results' / 'threshold_search'
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = out_root / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'summary.json'
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            print(f"\n[Saved] Summary JSON -> {out_path}")
        except Exception as e:
            print(f"\n[Warn] Failed to write summary: {e}")


if __name__ == '__main__':
    find_optimal_thresholds()
