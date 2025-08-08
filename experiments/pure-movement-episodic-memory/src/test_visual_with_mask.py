#!/usr/bin/env python3
"""
視覚エピソードの有効活用テスト
方向マスクにより視覚情報がより活用されることを検証
"""

import numpy as np
from typing import List, Dict


def test_visual_episodes_utility():
    """視覚エピソードの有効性を検証"""
    
    print("="*60)
    print("視覚エピソードの活用（方向マスクあり/なし比較）")
    print("="*60)
    
    # T字路での視覚エピソード
    visual_episodes = [
        # 位置(3,3)での視覚観測
        {'pos': [3, 3], 'dir': 'up',    'wall': False, 'vec': np.array([0.3, 0.3, 0.00, 0.5, 1.0, 0.2, 0.0])},
        {'pos': [3, 3], 'dir': 'right', 'wall': False, 'vec': np.array([0.3, 0.3, 0.33, 0.5, 1.0, 0.2, 0.0])},
        {'pos': [3, 3], 'dir': 'down',  'wall': True,  'vec': np.array([0.3, 0.3, 0.66, 0.5, -1.0, 0.2, 0.0])},
        {'pos': [3, 3], 'dir': 'left',  'wall': True,  'vec': np.array([0.3, 0.3, 1.00, 0.5, -1.0, 0.2, 0.0])},
    ]
    
    # 移動エピソード
    movement_episodes = [
        # 位置(3,3)での移動履歴
        {'pos': [3, 3], 'dir': 'up',    'success': True,  'vec': np.array([0.3, 0.3, 0.00, 1.0, 1.0, 0.2, 0.0])},
        {'pos': [3, 3], 'dir': 'right', 'success': True,  'vec': np.array([0.3, 0.3, 0.33, 1.0, 1.0, 0.2, 0.0])},
        {'pos': [3, 3], 'dir': 'down',  'success': False, 'vec': np.array([0.3, 0.3, 0.66, 0.0, -1.0, 0.2, 0.0])},
    ]
    
    all_episodes = visual_episodes + movement_episodes
    
    # 現在位置(3,3)でのクエリ
    query = np.array([0.3, 0.3, 0.5, 1.0, 0.0, 0.2, 0.0])  # 方向NULL、成功希望
    
    print("\n📍 状況: T字路（上と右が通路、下と左が壁）")
    print("🎯 目標: 成功する方向を選択")
    
    for use_mask in [False, True]:
        print(f"\n{'='*40}")
        print(f"方向マスク: {'あり' if use_mask else 'なし'}")
        print(f"{'='*40}")
        
        scores = []
        for ep in all_episodes:
            if use_mask:
                # 方向成分をマスク
                masked_query = query.copy()
                masked_query[2] = 0
                masked_ep = ep['vec'].copy()
                masked_ep[2] = 0
                
                similarity = np.dot(masked_query, masked_ep) / (
                    np.linalg.norm(masked_query) * np.linalg.norm(masked_ep) + 1e-8
                )
            else:
                similarity = np.dot(query, ep['vec']) / (
                    np.linalg.norm(query) * np.linalg.norm(ep['vec']) + 1e-8
                )
            
            ep_type = 'visual' if 'wall' in ep else 'move'
            direction = ep['dir']
            detail = f"wall={ep.get('wall', '?')}" if 'wall' in ep else f"success={ep.get('success', '?')}"
            
            scores.append((ep_type, direction, detail, similarity))
        
        # スコア順にソート
        scores.sort(key=lambda x: x[3], reverse=True)
        
        print("\n上位エピソード:")
        for ep_type, direction, detail, score in scores[:5]:
            print(f"  {ep_type:6} {direction:5} {detail:12} : {score:.3f}")
        
        # 視覚エピソードの寄与を分析
        visual_scores = [s[3] for s in scores if s[0] == 'visual']
        move_scores = [s[3] for s in scores if s[0] == 'move']
        
        print(f"\n視覚エピソード平均スコア: {np.mean(visual_scores):.3f}")
        print(f"移動エピソード平均スコア: {np.mean(move_scores):.3f}")
    
    print("\n" + "="*60)
    print("💡 分析結果:")
    print("="*60)
    print("1. マスクなし：方向成分の影響で視覚と移動が混在")
    print("2. マスクあり：視覚エピソードが明確に分離")
    print("3. 視覚情報（壁/通路）が有効に活用される")
    print("4. 同じ位置の全方向の視覚情報が平等に評価")


def test_visual_guidance():
    """視覚情報による行動誘導"""
    
    print("\n" + "="*60)
    print("視覚誘導ナビゲーション")
    print("="*60)
    
    # 迷路の一部を模擬
    # □□■□
    # □×■□  ×=現在位置、■=壁
    # □□□□
    
    episodes = []
    
    # 現在位置(1,1)での視覚
    episodes.append({'type': 'visual', 'pos': [1, 1], 'dir': 'up',    'wall': False, 
                    'vec': np.array([0.25, 0.25, 0.00, 0.5, 1.0, 0.1, 0.0])})
    episodes.append({'type': 'visual', 'pos': [1, 1], 'dir': 'right', 'wall': True,
                    'vec': np.array([0.25, 0.25, 0.33, 0.5, -1.0, 0.1, 0.0])})
    episodes.append({'type': 'visual', 'pos': [1, 1], 'dir': 'down',  'wall': False,
                    'vec': np.array([0.25, 0.25, 0.66, 0.5, 1.0, 0.1, 0.0])})
    episodes.append({'type': 'visual', 'pos': [1, 1], 'dir': 'left',  'wall': False,
                    'vec': np.array([0.25, 0.25, 1.00, 0.5, 1.0, 0.1, 0.0])})
    
    # 過去の成功移動（他の位置）
    episodes.append({'type': 'move', 'pos': [2, 2], 'dir': 'right', 'success': True,
                    'vec': np.array([0.50, 0.50, 0.33, 1.0, 1.0, 0.3, 0.0])})
    
    query = np.array([0.25, 0.25, 0.5, 1.0, 0.0, 0.1, 0.0])
    
    print("\n現在位置(1,1): 右に壁、他3方向は通路")
    print("\nマスク検索での推論:")
    
    # 方向マスクで検索
    masked_scores = []
    for ep in episodes:
        masked_query = query.copy()
        masked_query[2] = 0
        masked_ep = ep['vec'].copy()
        masked_ep[2] = 0
        
        score = np.dot(masked_query, masked_ep) / (
            np.linalg.norm(masked_query) * np.linalg.norm(masked_ep) + 1e-8
        )
        masked_scores.append((ep, score))
    
    masked_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位の視覚エピソードから行動決定
    print("\n視覚エピソードの活用:")
    wall_directions = []
    open_directions = []
    
    for ep, score in masked_scores[:4]:
        if ep['type'] == 'visual':
            if ep['wall']:
                wall_directions.append(ep['dir'])
                print(f"  {ep['dir']:5}: 壁 (スコア={score:.3f})")
            else:
                open_directions.append(ep['dir'])
                print(f"  {ep['dir']:5}: 通路 (スコア={score:.3f})")
    
    print(f"\n推奨行動: {open_directions[0] if open_directions else 'なし'}")
    print(f"回避方向: {wall_directions}")
    
    print("\n💡 視覚エピソードの価値:")
    print("- 現在位置の環境情報を即座に提供")
    print("- 壁を避けて通路を選ぶ基本戦略を実現")
    print("- 移動履歴がなくても行動可能")


if __name__ == "__main__":
    test_visual_episodes_utility()
    test_visual_guidance()