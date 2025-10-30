#!/usr/bin/env python3
"""
シンプルなバックトラック検証
実際にバックトラックが発生するか確認
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
import random


def test_backtrack_simple():
    """シンプルな迷路でバックトラックをテスト"""
    
    # 15x15の単純な迷路を作成
    maze_size = 15
    maze = np.ones((maze_size, maze_size), dtype=int)
    
    # T字型の迷路（必ずバックトラックが必要）
    # スタートから右に行くと行き止まり、上に行くとゴール
    maze[7, 1:8] = 0  # 横の通路（行き止まり）
    maze[1:8, 4] = 0  # 縦の通路（ゴールへ）
    maze[1, 4:13] = 0  # ゴールへの道
    maze[1:13, 12] = 0  # ゴールへ下る
    maze[12, 12] = 0  # ゴール
    
    start = (7, 1)
    goal = (12, 12)
    
    print("=" * 70)
    print("バックトラック検証実験")
    print("=" * 70)
    
    # 異なる閾値でテスト
    test_configs = [
        ('simple', -0.1, -0.2),
        ('gedig_optimized', -0.1, -0.2),
        ('gedig_optimized', -0.05, -0.1),  # より緩い閾値
        ('gedig_optimized', -0.2, -0.4),   # より厳しい閾値
    ]
    
    for strategy, gedig_th, bt_th in test_configs:
        print(f"\n戦略: {strategy}")
        print(f"  geDIG閾値: {gedig_th}, バックトラック閾値: {bt_th}")
        print("-" * 50)
        
        nav = MazeNavigator(
            maze=maze,
            start_pos=start,
            goal_pos=goal,
            wiring_strategy=strategy,
            gedig_threshold=gedig_th,
            backtrack_threshold=bt_th,
            simple_mode=True,
            backtrack_debounce=False,  # デバウンスなし
            use_escalation=True
        )
        
        # ナビゲーション実行
        path = []
        backtrack_count = 0
        dead_end_count = 0
        gedig_history = []
        
        for step in range(500):
            action = nav.step()
            path.append(nav.current_pos)
            
            # geDIG値を記録
            if hasattr(nav, 'gedig_history'):
                gedig_history = nav.gedig_history
            
            # イベント確認
            events = getattr(nav, 'event_log', [])
            for event in events:
                if event.get('type') == 'backtrack_trigger':
                    if len(events) > backtrack_count:  # 新しいイベント
                        backtrack_count += 1
                        print(f"    Step {step}: バックトラック発生！ 位置={nav.current_pos}")
                elif event.get('type') == 'dead_end_detected':
                    if len(events) > dead_end_count:  # 新しいイベント
                        dead_end_count += 1
                        print(f"    Step {step}: 行き止まり検出 位置={nav.current_pos}")
            
            if nav.current_pos == goal:
                print(f"  ✓ ゴール到達！ {step}ステップ")
                break
        else:
            print(f"  ✗ タイムアウト")
        
        # 統計情報
        print(f"  統計:")
        print(f"    総ステップ数: {len(path)}")
        print(f"    ユニーク位置: {len(set(path))}")
        print(f"    バックトラック回数: {backtrack_count}")
        print(f"    行き止まり検出: {dead_end_count}")
        
        if gedig_history:
            print(f"    geDIG値: min={min(gedig_history):.3f}, "
                  f"max={max(gedig_history):.3f}, "
                  f"mean={np.mean(gedig_history):.3f}")
            below_bt = sum(1 for g in gedig_history if g < bt_th)
            print(f"    閾値以下のgeDIG: {below_bt}/{len(gedig_history)}")
        
        # グラフ情報
        graph_stats = nav.graph_manager.get_graph_statistics()
        print(f"    グラフ: {graph_stats['num_nodes']}ノード, {graph_stats['num_edges']}エッジ")
    
    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    print("バックトラックが適切に発生するかは、迷路の構造と閾値設定に依存します。")
    print("geDIG値が閾値を下回ったときにバックトラックがトリガーされます。")


if __name__ == '__main__':
    test_backtrack_simple()