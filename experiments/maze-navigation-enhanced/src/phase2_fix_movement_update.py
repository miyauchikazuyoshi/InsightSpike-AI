#!/usr/bin/env python3
"""
移動時の訪問回数更新の正確な実装
================================
"""

import numpy as np

def simulate_correct_updates():
    """正しい訪問回数更新をシミュレート"""
    
    print("="*80)
    print("CORRECT VISIT COUNT UPDATE SIMULATION")
    print("="*80)
    
    # 経路
    path = [
        (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), (5, 4), (5, 3),
        (4, 3), (3, 3), (2, 3), (1, 3)
    ]
    
    episodes = {}
    visited_positions = set()
    
    print("\n各ステップでの処理:")
    print("-"*60)
    
    for step, pos in enumerate(path):
        visited_positions.add(pos)
        print(f"\nStep {step}: 位置 {pos}")
        
        # 1. 観測フェーズ
        print("  観測:")
        for d, (dx, dy) in [('N', (0,-1)), ('S', (0,1)), ('E', (1,0)), ('W', (-1,0))]:
            next_pos = (pos[0]+dx, pos[1]+dy)
            key = (pos, d)
            
            if key not in episodes:
                # 新規エピソード作成
                initial_visits = 1 if next_pos in visited_positions else 0
                episodes[key] = {
                    'visits': initial_visits,
                    'next_pos': next_pos,
                    'created_at': step
                }
                if initial_visits > 0:
                    print(f"    {pos}→{d} to {next_pos}: visits={initial_visits} (既訪問)")
                else:
                    print(f"    {pos}→{d} to {next_pos}: visits={initial_visits} (新規)")
        
        # 2. 移動フェーズ
        if step < len(path) - 1:
            next_pos = path[step + 1]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            
            # 選択した方向を判定
            if dx == 1:
                selected_dir = 'E'
            elif dx == -1:
                selected_dir = 'W'
            elif dy == 1:
                selected_dir = 'S'
            elif dy == -1:
                selected_dir = 'N'
            else:
                selected_dir = None
            
            if selected_dir:
                key = (pos, selected_dir)
                if key in episodes:
                    episodes[key]['visits'] += 1
                    print(f"  移動: {selected_dir}を選択 → visits={episodes[key]['visits']}")
    
    # Step 10時点での重要なエピソード
    print("\n" + "="*60)
    print("Step 10時点での重要なエピソード:")
    print("-"*60)
    
    important_episodes = [
        ((1, 3), 'E'),  # 現在位置から東
        ((2, 3), 'W'),  # 来た道を戻る
        ((5, 3), 'N'),  # 未探索の上
        ((5, 3), 'S'),  # 既訪問の下
        ((5, 3), 'W'),  # 左分岐へ
    ]
    
    for key in important_episodes:
        if key in episodes:
            ep = episodes[key]
            print(f"{key[0]}→{key[1]} to {ep['next_pos']}: visits={ep['visits']} "
                  f"(created at Step {ep['created_at']})")
    
    print("\n💡 正しい状態:")
    print("  - (2,3)→W: visits=1 (Step 9で選択済み)")
    print("  - (5,3)→N: visits=0 (未選択)")
    print("  - (1,3)→E: visits=1 (Step 10で作成、既訪問として初期化)")

if __name__ == "__main__":
    simulate_correct_updates()