#!/usr/bin/env python3
"""
正しい訪問回数でのノルム検索
============================
"""

import numpy as np

def correct_norm_search():
    """正しい訪問回数でノルム検索"""
    
    print("="*80)
    print("CORRECT NORM SEARCH at Step 10")
    print("="*80)
    
    # 重み
    weights = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0])
    w, h = 11, 11
    
    def create_vector(pos, dir, is_wall, visits):
        direction_map = {'N': (0,-1), 'S': (0,1), 'E': (1,0), 'W': (-1,0)}
        dx, dy = direction_map.get(dir, (0,0))
        return np.array([
            pos[0]/w, pos[1]/h,
            dx, dy,
            -1.0 if is_wall else 1.0,
            np.log1p(visits),
            0.0, 0.0
        ])
    
    # Step 10時点での正しいエピソード状態
    episodes = [
        # 現在位置から
        {'pos': (1,3), 'dir': 'E', 'next': (2,3), 'visits': 1, 'wall': False},  # 既訪問
        
        # 来た道
        {'pos': (2,3), 'dir': 'W', 'next': (1,3), 'visits': 1, 'wall': False},  # 選択済み！
        {'pos': (2,3), 'dir': 'E', 'next': (3,3), 'visits': 1, 'wall': False},  # 既訪問
        
        # さらに前
        {'pos': (3,3), 'dir': 'W', 'next': (2,3), 'visits': 1, 'wall': False},  # 選択済み
        {'pos': (3,3), 'dir': 'E', 'next': (4,3), 'visits': 1, 'wall': False},  # 既訪問
        
        {'pos': (4,3), 'dir': 'W', 'next': (3,3), 'visits': 1, 'wall': False},  # 選択済み
        {'pos': (4,3), 'dir': 'E', 'next': (5,3), 'visits': 1, 'wall': False},  # 既訪問
        
        # 分岐点
        {'pos': (5,3), 'dir': 'N', 'next': (5,2), 'visits': 0, 'wall': False},  # 未選択！
        {'pos': (5,3), 'dir': 'S', 'next': (5,4), 'visits': 1, 'wall': False},  # 既訪問
        {'pos': (5,3), 'dir': 'W', 'next': (4,3), 'visits': 1, 'wall': False},  # 選択済み
        
        # 縦通路
        {'pos': (5,4), 'dir': 'N', 'next': (5,3), 'visits': 1, 'wall': False},  # 選択済み
        {'pos': (5,4), 'dir': 'S', 'next': (5,5), 'visits': 1, 'wall': False},  # 既訪問
        
        {'pos': (5,5), 'dir': 'N', 'next': (5,4), 'visits': 1, 'wall': False},  # 選択済み
        {'pos': (5,5), 'dir': 'S', 'next': (5,6), 'visits': 1, 'wall': False},  # 既訪問
    ]
    
    # クエリ（(1,3)から探索）
    query = create_vector((1,3), '', False, 0)
    query[4] = 1.0  # 通路を探す
    query_weighted = query * weights
    
    # 距離計算
    distances = []
    for ep in episodes:
        if not ep['wall']:
            ep_vec = create_vector(ep['pos'], ep['dir'], ep['wall'], ep['visits'])
            ep_weighted = ep_vec * weights
            dist = np.linalg.norm(query_weighted - ep_weighted)
            distances.append((dist, ep))
    
    distances.sort(key=lambda x: x[0])
    
    print("\nTop 10 Nearest Episodes (正しい訪問回数):")
    print("-"*60)
    for i, (dist, ep) in enumerate(distances[:10]):
        marker = ""
        if ep['pos'] == (5,3) and ep['dir'] == 'N':
            marker = " ← 未探索の上方向！visits=0"
        elif ep['pos'] == (2,3) and ep['dir'] == 'W':
            marker = " ← 来た道を戻る方向 visits=1"
        
        print(f"  {i+1:2d}. dist={dist:.4f}: {ep['pos']}→{ep['dir']} to {ep['next']}, "
              f"visits={ep['visits']}{marker}")
    
    # 距離の内訳を詳細分析
    print("\n距離の詳細分析:")
    print("-"*60)
    
    # (5,3)→N と (2,3)→W を比較
    ep_53n = {'pos': (5,3), 'dir': 'N', 'next': (5,2), 'visits': 0, 'wall': False}
    ep_23w = {'pos': (2,3), 'dir': 'W', 'next': (1,3), 'visits': 1, 'wall': False}
    
    for name, ep in [("(5,3)→N (未探索)", ep_53n), ("(2,3)→W (来た道)", ep_23w)]:
        ep_vec = create_vector(ep['pos'], ep['dir'], ep['wall'], ep['visits'])
        ep_weighted = ep_vec * weights
        diff = query_weighted - ep_weighted
        dist = np.linalg.norm(diff)
        
        print(f"\n{name}:")
        print(f"  総距離: {dist:.4f}")
        print("  成分別寄与:")
        
        components = ['pos_x', 'pos_y', 'dir_x', 'dir_y', 'wall', 'visits', 'result', 'goal']
        for i, comp in enumerate(components):
            if weights[i] > 0:
                contrib = diff[i]**2
                if contrib > 0.001:
                    print(f"    {comp}: weight={weights[i]:.1f}, diff={diff[i]:.4f}, "
                          f"contrib={contrib:.4f} ({contrib/dist**2*100:.1f}%)")
    
    print("\n💡 結論:")
    print("  - (5,3)→N は visits=0 で距離に有利")
    print("  - (2,3)→W は visits=1 で距離に不利")
    print("  - でも位置の差（4マス vs 1マス）が大きすぎる")
    print("  - 訪問回数の重みを上げれば(5,3)→Nが1位になる可能性")

if __name__ == "__main__":
    correct_norm_search()