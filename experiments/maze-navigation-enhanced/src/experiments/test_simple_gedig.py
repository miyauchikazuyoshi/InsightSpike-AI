#!/usr/bin/env python3
"""
シンプルなgeDIG閾値テスト
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def create_simple_dead_end_maze():
    """シンプルな袋小路付き迷路"""
    maze = np.ones((9, 9), dtype=int)
    
    # メイン通路（縦）
    for y in range(1, 8):
        maze[y, 4] = 0
    
    # 横通路（ゴールへ）
    for x in range(4, 8):
        maze[1, x] = 0
    
    # 袋小路（左側）
    for x in range(1, 4):
        maze[4, x] = 0
    
    return maze


def main():
    print("="*60)
    print("SIMPLE geDIG THRESHOLD TEST")
    print("="*60)
    
    # 迷路作成
    maze = create_simple_dead_end_maze()
    start_pos = (4, 7)
    goal_pos = (7, 1)
    
    print("\nテスト迷路:")
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
                row += "· "
        print(row)
    
    # デフォルト重み
    weights = np.array([
        1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0
    ])
    
    # テスト1: 閾値なし（バックトラックなし）
    print("\n" + "-"*60)
    print("Test 1: バックトラック閾値 = -1.0 (実質無効)")
    
    navigator1 = MazeNavigator(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        weights=weights,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=-1.0,  # 非常に低い値（バックトラックしない）
        wiring_strategy='simple'
    )
    
    success1 = navigator1.run(max_steps=50)
    stats1 = navigator1.get_statistics()
    
    print(f"  結果: {'成功' if success1 else '失敗'}")
    print(f"  ステップ数: {navigator1.step_count}")
    print(f"  ユニーク位置: {stats1['unique_positions']}")
    
    # geDIG履歴の最小値を確認
    if navigator1.gedig_history:
        min_gedig = min(navigator1.gedig_history)
        print(f"  最小geDIG値: {min_gedig:.4f}")
    
    # テスト2: 適切な閾値
    print("\n" + "-"*60)
    print("Test 2: バックトラック閾値 = -0.2")
    
    navigator2 = MazeNavigator(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        weights=weights,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=-0.2,
        wiring_strategy='simple'
    )
    
    success2 = navigator2.run(max_steps=50)
    stats2 = navigator2.get_statistics()
    
    print(f"  結果: {'成功' if success2 else '失敗'}")
    print(f"  ステップ数: {navigator2.step_count}")
    print(f"  ユニーク位置: {stats2['unique_positions']}")
    
    # バックトラックイベントを確認
    backtrack_events = [e for e in navigator2.event_log if e['type'] == 'backtrack_trigger']
    print(f"  バックトラック回数: {len(backtrack_events)}")
    
    if backtrack_events:
        print("  バックトラックイベント:")
        for event in backtrack_events[:3]:  # 最初の3個
            print(f"    Step {event['step']}: {event['message']}")
    
    # 分析
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("\n閾値の効果:")
    if success1 and success2:
        if navigator2.step_count < navigator1.step_count:
            print(f"✓ 適切な閾値により{navigator1.step_count - navigator2.step_count}ステップ短縮")
        else:
            print("- 両方成功（閾値の効果は限定的）")
    elif not success1 and success2:
        print("✓ 閾値設定により成功（バックトラックが有効）")
    elif success1 and not success2:
        print("✗ 閾値が厳しすぎる可能性")
    else:
        print("✗ 両方失敗（迷路が難しすぎる可能性）")
    
    # 推奨
    print("\n推奨事項:")
    if navigator1.gedig_history:
        gedig_values = navigator1.gedig_history
        negative_values = [v for v in gedig_values if v < 0]
        if negative_values:
            avg_negative = np.mean(negative_values)
            print(f"  検出された負のgeDIG平均: {avg_negative:.4f}")
            print(f"  推奨閾値: {avg_negative * 0.8:.4f}～{avg_negative * 1.2:.4f}")
        else:
            print("  負のgeDIG値が検出されませんでした")
            print("  この迷路では袋小路での短絡が発生していない可能性があります")


if __name__ == "__main__":
    main()