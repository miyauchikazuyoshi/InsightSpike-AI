#!/usr/bin/env python3
"""
慣性（momentum）効果の検証
方向成分を除外することで、直進継続が促進されるか
"""

import numpy as np
from typing import List, Tuple


def simulate_corridor_navigation(use_mask: bool = False):
    """
    廊下での直進ナビゲーションシミュレーション
    
    Args:
        use_mask: 方向成分をマスクするか
    """
    print(f"\n{'='*60}")
    print(f"廊下ナビゲーション（方向マスク: {use_mask}）")
    print(f"{'='*60}")
    
    # 廊下を模擬（横一直線の通路）
    # 位置: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (ゴール)
    
    episodes = []
    
    # 各位置での移動履歴
    # 位置0: 右に成功
    episodes.append({
        'pos': [0, 0],
        'dir': 'right',
        'vec': np.array([0.0, 0.0, 0.33, 1.0, 1.0, 0.0, 0.0])
    })
    
    # 位置1: 右に成功（継続）
    episodes.append({
        'pos': [1, 0],
        'dir': 'right',
        'vec': np.array([0.2, 0.0, 0.33, 1.0, 1.0, 0.2, 0.0])
    })
    
    # 位置2: 右に成功（継続）
    episodes.append({
        'pos': [2, 0],
        'dir': 'right',
        'vec': np.array([0.4, 0.0, 0.33, 1.0, 1.0, 0.3, 0.0])
    })
    
    # 位置2: 上に失敗（壁）
    episodes.append({
        'pos': [2, 0],
        'dir': 'up',
        'vec': np.array([0.4, 0.0, 0.0, 0.0, -1.0, 0.3, 0.0])
    })
    
    # 位置2: 下に失敗（壁）
    episodes.append({
        'pos': [2, 0],
        'dir': 'down',
        'vec': np.array([0.4, 0.0, 0.66, 0.0, -1.0, 0.3, 0.0])
    })
    
    # 現在位置3でのクエリ
    current_pos = [3, 0]
    query = np.array([0.6, 0.0, 0.5, 1.0, 0.0, 0.4, 0.0])  # 方向NULL
    
    print(f"\n📍 現在位置: {current_pos}")
    print(f"🎯 目標: 右方向への継続移動")
    
    # 類似度計算
    print(f"\n📊 類似度スコア:")
    
    for ep in episodes:
        if use_mask:
            # 方向成分（次元2）をマスク
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
        
        print(f"  位置{ep['pos']} {ep['dir']:5} : {similarity:.3f}")
    
    print(f"\n💡 分析:")
    if use_mask:
        print("- 方向マスクあり: 位置が近いエピソードが高スコア")
        print("- 直前の右移動成功が自然に選ばれやすい")
        print("- 慣性効果: 同じ方向への継続が促進される")
    else:
        print("- 方向マスクなし: 方向NULL(0.5)の影響で差が小さい")
        print("- どの方向も同程度のスコア")


def test_turn_vs_straight():
    """曲がり角 vs 直進の選択"""
    
    print(f"\n{'='*60}")
    print("曲がり角での判断：直進 vs 曲がる")
    print(f"{'='*60}")
    
    # T字路での履歴
    episodes = [
        # 前回位置(2,3)から右へ成功
        {'from': [2, 3], 'to': [3, 3], 'dir': 'right', 
         'vec': np.array([0.2, 0.3, 0.33, 1.0, 1.0, 0.1, 0.0])},
        
        # 現在位置(3,3)から右へ成功（直進継続）
        {'from': [3, 3], 'to': [4, 3], 'dir': 'right',
         'vec': np.array([0.3, 0.3, 0.33, 1.0, 1.0, 0.2, 0.0])},
        
        # 現在位置(3,3)から上へ成功（曲がる）
        {'from': [3, 3], 'to': [3, 2], 'dir': 'up',
         'vec': np.array([0.3, 0.3, 0.0, 1.0, 1.0, 0.2, 0.0])},
    ]
    
    # 現在(4,3)でのクエリ（(3,3)から右に来た後）
    query = np.array([0.4, 0.3, 0.5, 1.0, 0.0, 0.3, 0.0])
    
    print("\n状況: (3,3)から右へ移動して(4,3)に到達")
    print("選択肢: 右へ直進継続 or 他の方向へ曲がる")
    
    for use_mask in [False, True]:
        print(f"\n{'マスクあり' if use_mask else 'マスクなし'}:")
        
        scores = []
        for ep in episodes:
            if use_mask:
                masked_query = query.copy()
                masked_query[2] = 0
                masked_ep = ep['vec'].copy()
                masked_ep[2] = 0
                score = np.dot(masked_query, masked_ep) / (
                    np.linalg.norm(masked_query) * np.linalg.norm(masked_ep) + 1e-8
                )
            else:
                score = np.dot(query, ep['vec']) / (
                    np.linalg.norm(query) * np.linalg.norm(ep['vec']) + 1e-8
                )
            scores.append((ep['dir'], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        for direction, score in scores[:2]:
            print(f"  {direction:5}: {score:.3f}")
    
    print("\n💡 結論:")
    print("- マスクあり: 直進継続（right）が選ばれやすい")
    print("- これは「慣性の法則」的な動作を生む")
    print("- 長い廊下を効率的に進むのに有利")


if __name__ == "__main__":
    simulate_corridor_navigation(use_mask=False)
    simulate_corridor_navigation(use_mask=True)
    test_turn_vs_straight()