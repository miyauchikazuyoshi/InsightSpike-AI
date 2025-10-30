#!/usr/bin/env python3
"""
シンプルな壁衝突テスト
ソフトマックスで壁を選ぶ確率を実証
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def softmax_probabilities(similarities, temperature):
    """ソフトマックス確率を計算"""
    scaled = similarities / temperature
    exp_values = np.exp(scaled - np.max(scaled))
    return exp_values / np.sum(exp_values)


def simulate_wall_selection():
    """壁選択のシミュレーション"""
    
    print("="*60)
    print("WALL SELECTION PROBABILITY SIMULATION")
    print("="*60)
    
    # 4方向の類似度（仮定）
    # 北: 壁, 東: 未訪問通路, 南: 訪問済み通路, 西: 壁
    similarities = {
        'North (wall)': -0.5,     # 壁は類似度低い
        'East (unvisited)': 0.8,  # 未訪問は類似度高い
        'South (visited)': 0.2,   # 訪問済みは中程度
        'West (wall)': -0.4       # 壁は類似度低い
    }
    
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    print("\n各温度での選択確率:")
    print("-"*40)
    
    results = {}
    
    for temp in temperatures:
        sims = np.array(list(similarities.values()))
        probs = softmax_probabilities(sims, temp)
        
        prob_dict = dict(zip(similarities.keys(), probs))
        results[temp] = prob_dict
        
        print(f"\n温度 T={temp}:")
        for direction, prob in prob_dict.items():
            print(f"  {direction:20s}: {prob:6.1%}")
        
        wall_prob = prob_dict['North (wall)'] + prob_dict['West (wall)']
        print(f"  → 壁選択確率合計: {wall_prob:6.1%}")
    
    return results


def visualize_temperature_effect():
    """温度の影響を可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 温度による確率変化
    ax = axes[0, 0]
    
    temperatures = np.linspace(0.1, 3.0, 100)
    
    # 固定の類似度差
    sim_unvisited = 0.8   # 未訪問通路
    sim_visited = 0.2     # 訪問済み通路
    sim_wall = -0.5       # 壁
    
    probs_unvisited = []
    probs_visited = []
    probs_wall = []
    
    for temp in temperatures:
        sims = np.array([sim_unvisited, sim_visited, sim_wall])
        probs = softmax_probabilities(sims, temp)
        probs_unvisited.append(probs[0])
        probs_visited.append(probs[1])
        probs_wall.append(probs[2])
    
    ax.plot(temperatures, probs_unvisited, 'g-', label='Unvisited', linewidth=2)
    ax.plot(temperatures, probs_visited, 'b-', label='Visited', linewidth=2)
    ax.plot(temperatures, probs_wall, 'r-', label='Wall', linewidth=2)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Selection Probability')
    ax.set_title('Temperature Effect on Direction Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.33, color='k', linestyle='--', alpha=0.3, label='Uniform')
    
    # 2. 袋小路での確率分布
    ax = axes[0, 1]
    
    # 袋小路の状況: 3方向が壁、1方向のみ通路（来た道）
    temps = [0.1, 0.5, 1.0, 2.0]
    x_pos = np.arange(len(temps))
    
    wall_probs = []
    passage_probs = []
    
    for temp in temps:
        # 3つの壁と1つの通路
        sims = np.array([-0.5, -0.5, -0.5, 0.3])  # 3壁 + 1通路（来た道）
        probs = softmax_probabilities(sims, temp)
        wall_prob = np.sum(probs[:3])  # 3つの壁の確率合計
        passage_prob = probs[3]
        
        wall_probs.append(wall_prob)
        passage_probs.append(passage_prob)
    
    width = 0.35
    ax.bar(x_pos - width/2, wall_probs, width, label='Wall (×3)', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, passage_probs, width, label='Passage (×1)', color='green', alpha=0.7)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Total Probability')
    ax.set_title('Probability in Dead-end (3 walls, 1 passage)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'T={t}' for t in temps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 実際の迷路での例
    ax = axes[1, 0]
    
    # 5x5の小さな迷路例
    maze = np.ones((5, 5))
    maze[1:4, 2] = 0  # 縦通路
    maze[2, 1:4] = 0  # 横通路
    
    for y in range(5):
        for x in range(5):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax.add_patch(rect)
    
    # エージェント位置
    agent_pos = (2, 2)
    ax.plot(agent_pos[0], agent_pos[1], 'bo', markersize=15)
    
    # 各方向の確率を表示（T=0.5の場合）
    directions = {
        (2, 1): 0.35,  # 北（通路）
        (3, 2): 0.10,  # 東（壁）
        (2, 3): 0.45,  # 南（通路、未訪問）
        (1, 2): 0.10   # 西（通路、訪問済み）
    }
    
    for pos, prob in directions.items():
        ax.annotate(f'{prob:.0%}', xy=pos, ha='center', va='center',
                   color='white' if maze[pos[1], pos[0]] == 1 else 'blue',
                   fontweight='bold', fontsize=10)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('Example: Probabilities at Junction (T=0.5)')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 4. 理論的な分析
    ax = axes[1, 1]
    ax.axis('off')
    
    text = """重要な発見:

1. チート実装の問題:
   - 壁を除外してからソフトマックス
   - 完璧な知識を前提
   - 探索の不確実性を無視

2. 正しい実装:
   - 全方向でソフトマックス計算
   - 壁も選択される可能性
   - 温度で探索度を制御

3. 温度の影響:
   - T↓: 最良選択に集中（壁選択↓）
   - T↑: 均等に近づく（壁選択↑）
   
4. 袋小路での挙動:
   - 3方向が壁でも選択確率あり
   - 高温度では50%以上が壁選択
   - 学習により回避が進む"""
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top')
    ax.set_title('Theoretical Analysis')
    
    plt.suptitle('Wall Selection Probability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    # シミュレーション実行
    results = simulate_wall_selection()
    
    # 可視化
    fig = visualize_temperature_effect()
    
    # 保存
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/wall_selection_theory_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    
    plt.show()
    
    # 結論
    print("\n" + "="*60)
    print("結論")
    print("="*60)
    print("""
現在の実装は「チート」していました:
- 壁を最初から除外してソフトマックス計算
- 実際の探索では壁の存在を完全には知らないはず
- 温度パラメータの意味が薄れる

正しい実装では:
- 温度0.5で約10-20%の確率で壁を選択
- 温度1.0で約20-30%の確率で壁を選択
- 温度2.0で約30-40%の確率で壁を選択

これが本来の探索的挙動です！
""")


if __name__ == "__main__":
    main()