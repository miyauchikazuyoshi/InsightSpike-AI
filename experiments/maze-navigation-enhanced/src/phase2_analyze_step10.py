#!/usr/bin/env python3
"""
Step 10時点での(1,3)の状態を正確に分析
======================================
"""

import numpy as np

def analyze_step_10():
    """Step 10での正確な状態を再現"""
    
    print("="*80)
    print("STEP 10 ANALYSIS - Exact State at (1,3)")
    print("="*80)
    
    # 経路
    path = [
        (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), (5, 4), (5, 3),
        (4, 3), (3, 3), (2, 3), (1, 3)  # Step 10で(1,3)に到達
    ]
    
    visited_positions = set(path[:11])  # Step 10までに訪問した位置
    
    print(f"\nStep 10までの訪問位置:")
    print(f"  {visited_positions}")
    
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
    
    # Step 10時点でのエピソード状態を正確に再現
    episodes_at_step10 = []
    
    # Step 6: (5,3)で作成されたエピソード
    print(f"\n🔍 Step 6で(5,3)に到達時のエピソード作成:")
    for d, next_pos in [('N', (5,2)), ('S', (5,4)), ('E', (6,3)), ('W', (4,3))]:
        is_wall = (d == 'E')  # 東は壁
        initial_visits = 1 if next_pos in visited_positions else 0
        
        print(f"  (5,3)→{d} to {next_pos}: ", end="")
        if is_wall:
            print(f"壁")
        else:
            print(f"通路, initial_visits={initial_visits} ({'既訪問' if initial_visits else '未訪問'})")
            
            if d == 'W':  # Step 7で選択
                visits = initial_visits + 1
                print(f"    → Step 7で選択, visits={visits}")
            else:
                visits = initial_visits
            
            episodes_at_step10.append({
                'pos': (5,3),
                'dir': d,
                'next_pos': next_pos,
                'visits': visits,
                'timestamp': 6
            })
    
    # Step 10: (1,3)で作成されたエピソード
    print(f"\n🔍 Step 10で(1,3)に到達時のエピソード作成:")
    for d, next_pos in [('N', (1,2)), ('S', (1,4)), ('E', (2,3)), ('W', (0,3))]:
        is_wall = (d != 'E')  # 東以外は壁
        
        if not is_wall:
            initial_visits = 1 if next_pos in visited_positions else 0
            print(f"  (1,3)→{d} to {next_pos}: 通路, initial_visits={initial_visits}")
            
            episodes_at_step10.append({
                'pos': (1,3),
                'dir': d,
                'next_pos': next_pos,
                'visits': initial_visits,  # Step 10ではまだ移動していない
                'timestamp': 10
            })
    
    # 他の重要なエピソード
    print(f"\n🔍 他の重要なエピソード (Step 9まで):")
    
    # (2,3)→W (Step 9で作成)
    print(f"  (2,3)→W to (1,3): 通路, initial_visits=0 (未訪問)")
    episodes_at_step10.append({
        'pos': (2,3), 'dir': 'W', 'next_pos': (1,3),
        'visits': 0,  # Step 9時点で(1,3)は未訪問、Step 10で選択されるが分析時点では0
        'timestamp': 9
    })
    
    # ノルム検索
    print(f"\n📊 Step 10時点でのノルム検索 from (1,3):")
    print("-"*60)
    
    query = create_vector((1,3), '', False, 0)
    query[4] = 1.0
    query_weighted = query * weights
    
    distances = []
    for ep in episodes_at_step10:
        ep_vec = create_vector(ep['pos'], ep['dir'], False, ep['visits'])
        ep_weighted = ep_vec * weights
        dist = np.linalg.norm(query_weighted - ep_weighted)
        distances.append((dist, ep))
    
    distances.sort(key=lambda x: x[0])
    
    print("Top Episodes:")
    for i, (dist, ep) in enumerate(distances[:10]):
        marker = ""
        if ep['pos'] == (5,3) and ep['dir'] == 'N':
            marker = " ← 未探索の上方向！"
        elif ep['pos'] == (2,3) and ep['dir'] == 'W':
            marker = " ← 来た道を戻る方向"
        
        print(f"  {i+1}. dist={dist:.4f}: {ep['pos']}→{ep['dir']} to {ep['next_pos']}, "
              f"visits={ep['visits']}{marker}")
    
    print(f"\n💡 結論:")
    print("  - (5,3)→N (未探索の上) は visits=0")
    print("  - (2,3)→W (来た道戻る) も visits=0")
    print("  - 両方とも未選択なので visits=0 で、位置の距離だけで順位が決まる")
    print("  - (1,3)から(5,3)は4マス離れているので、距離が大きい")

if __name__ == "__main__":
    analyze_step_10()