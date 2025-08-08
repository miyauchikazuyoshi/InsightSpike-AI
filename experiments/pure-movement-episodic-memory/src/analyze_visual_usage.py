#!/usr/bin/env python3
"""
視覚情報の利用状況を分析
実際にどう使われているか確認
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def analyze_visual_usage():
    """視覚情報の利用を詳細分析"""
    
    print("="*60)
    print("👁️ 視覚情報の利用分析")
    print("="*60)
    
    # 簡単な迷路で分析
    maze = np.array([
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],  # 直線廊下
        [1,1,1,1,1,0,1],  # ゴールへの道
        [1,0,0,0,0,0,1],  # 袋小路
        [1,1,1,1,1,1,1]
    ])
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 1 and j == 5:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/visual_analysis",
        config={
            'max_depth': 3,
            'search_k': 20,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 7
        }
    )
    
    print("\n初期状態:")
    print(f"  位置: {agent.position}")
    print(f"  ゴール: {agent.goal}")
    
    # 最初の行動決定時の詳細
    print("\n" + "="*60)
    print("1ステップ目の詳細分析")
    print("="*60)
    
    # 視覚観測前のエピソード数
    before_count = len(agent.experience_metadata)
    
    # get_actionを呼ぶ（視覚観測が追加される）
    action = agent.get_action()
    
    # 視覚観測後のエピソード数
    after_count = len(agent.experience_metadata)
    
    print(f"\n視覚観測の追加:")
    print(f"  観測前エピソード数: {before_count}")
    print(f"  観測後エピソード数: {after_count}")
    print(f"  追加された視覚観測: {after_count - before_count}")
    
    # 追加された視覚エピソードを確認
    print("\n追加された視覚エピソード:")
    for i in range(before_count, after_count):
        meta = agent.experience_metadata[i]
        if meta['type'] == 'visual':
            print(f"  {i}: 位置{meta['pos']}, 方向={meta['direction']}, "
                  f"壁={'あり' if meta['is_wall'] else 'なし'}")
    
    print(f"\n選択された行動: {action}")
    
    # 数ステップ実行して視覚情報の蓄積を確認
    print("\n" + "="*60)
    print("5ステップ実行")
    print("="*60)
    
    for step in range(5):
        action = agent.get_action()
        success = agent.execute_action(action)
        
        print(f"\nStep {step+1}:")
        print(f"  行動: {action}")
        print(f"  結果: {'成功' if success else '壁衝突'}")
        print(f"  現在位置: {agent.position}")
        
        # 視覚エピソードの統計
        visual_count = sum(1 for m in agent.experience_metadata if m['type'] == 'visual')
        movement_count = sum(1 for m in agent.experience_metadata if m['type'] == 'movement')
        
        print(f"  エピソード統計:")
        print(f"    視覚: {visual_count}")
        print(f"    移動: {movement_count}")
        print(f"    合計: {len(agent.experience_metadata)}")
    
    # クエリと類似エピソードの関係を分析
    print("\n" + "="*60)
    print("クエリと検索結果の分析")
    print("="*60)
    
    query = agent._create_task_query()
    print(f"\nタスククエリ:")
    print(f"  位置: ({query[0]:.2f}, {query[1]:.2f})")
    print(f"  方向: {query[2]:.2f}")
    print(f"  成功希望: {query[3]:.2f}")
    print(f"  通路選好: {query[4]:.2f}")
    
    # 類似度検索
    distances, indices = agent.vector_index.search(
        query.reshape(1, -1),
        k=10
    )
    
    print("\n類似エピソード Top 10:")
    visual_in_top = 0
    movement_in_top = 0
    
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if 0 <= idx < len(agent.experience_metadata):
            meta = agent.experience_metadata[idx]
            episode_type = meta['type']
            
            if episode_type == 'visual':
                visual_in_top += 1
                marker = "👁️"
            else:
                movement_in_top += 1
                marker = "🚶"
            
            print(f"  {rank+1}. {marker} {episode_type:8s} "
                  f"位置{meta.get('pos', '?'):8s} "
                  f"壁={'あり' if meta.get('is_wall', False) else 'なし':4s} "
                  f"類似度={dist:.3f}")
    
    print(f"\nTop 10内の分布:")
    print(f"  視覚エピソード: {visual_in_top}/10")
    print(f"  移動エピソード: {movement_in_top}/10")
    
    # メッセージパッシング後の結果
    print("\n" + "="*60)
    print("メッセージパッシング後の影響")
    print("="*60)
    
    # 深度3でメッセージパッシング
    aggregated = agent._message_passing(indices[0][:10].tolist(), 3)
    
    print(f"\n集約されたベクトル:")
    print(f"  位置: ({aggregated[0]:.2f}, {aggregated[1]:.2f})")
    print(f"  方向: {aggregated[2]:.2f}")
    print(f"  成功/失敗: {aggregated[3]:.2f}")
    print(f"  壁情報: {aggregated[4]:.2f}")
    
    # 最終的な行動選択への影響
    print("\n最終的な行動選択プロセス:")
    print("  1. 視覚観測で4方向の壁情報を取得")
    print("  2. クエリで類似エピソードを検索")
    print("  3. メッセージパッシングで情報集約")
    print("  4. 集約結果に最も近いエピソードを選択")
    print("  5. そのエピソードの行動を実行")
    
    # 問題点の分析
    print("\n" + "="*60)
    print("💭 視覚情報利用の問題点")
    print("="*60)
    
    print("""
    1. 視覚情報は収集されている ✅
       - 毎ステップ4方向を観測
       - エピソードとして記録
    
    2. でも活用が不十分 ❌
       - 類似度検索で視覚エピソードが選ばれにくい
       - メッセージパッシングで情報が混ざる
       - 壁情報が行動選択に直接反映されない
    
    3. 根本的な問題
       - 「この方向は壁」という情報があっても
       - 「だから逆方向へ行く」という推論ができない
       - ベクトル平均では方向の反転が表現できない
    """)


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 分析実行
    analyze_visual_usage()