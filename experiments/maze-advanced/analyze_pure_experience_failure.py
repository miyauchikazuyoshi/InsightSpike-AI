#!/usr/bin/env python3
"""
純粋経験学習の失敗分析
====================

なぜ純粋な類似性ベースの学習が迷路解決に失敗するのかを分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

def load_json_results(filepath: str) -> Dict:
    """JSONファイルから結果を読み込む"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_path_pattern(path: List[List[int]]) -> Dict:
    """経路パターンを分析"""
    # 位置の訪問回数
    position_counts = Counter(tuple(pos) for pos in path)
    
    # 最も頻繁に訪問された位置
    most_visited = position_counts.most_common(5)
    
    # ループ検出
    loops = []
    for i in range(len(path) - 2):
        for j in range(i + 2, len(path)):
            if path[i] == path[j]:
                loop_length = j - i
                loops.append({
                    'start_index': i,
                    'end_index': j,
                    'length': loop_length,
                    'position': path[i]
                })
    
    # 一意の位置数
    unique_positions = len(set(tuple(pos) for pos in path))
    
    return {
        'total_steps': len(path),
        'unique_positions': unique_positions,
        'coverage_ratio': unique_positions / len(path),
        'most_visited': most_visited,
        'loops': loops[:10],  # 最初の10個のループ
        'position_counts': dict(position_counts)
    }

def analyze_decision_pattern(decision_log: List[Dict]) -> Dict:
    """意思決定パターンを分析"""
    if not decision_log:
        return {}
    
    # 行動選択の分布
    action_counts = Counter(log['selected_action'] for log in decision_log)
    
    # 類似度の統計
    similarities = []
    for log in decision_log:
        if 'action_similarities' in log:
            for action, sim in log['action_similarities'].items():
                if sim != float('-inf'):
                    similarities.append(sim)
    
    # 初期の決定を詳細分析
    initial_decisions = decision_log[:5]
    
    return {
        'action_distribution': dict(action_counts),
        'similarity_stats': {
            'mean': np.mean(similarities) if similarities else 0,
            'std': np.std(similarities) if similarities else 0,
            'min': min(similarities) if similarities else 0,
            'max': max(similarities) if similarities else 0
        },
        'initial_decisions': initial_decisions
    }

def visualize_analysis(maze_size: int, path: List[List[int]], analysis: Dict):
    """分析結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 訪問頻度ヒートマップ
    ax = axes[0, 0]
    visit_map = np.zeros((maze_size, maze_size))
    for pos in path:
        visit_map[pos[1], pos[0]] += 1
    
    im = ax.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax.set_title('Visit Frequency Heatmap')
    plt.colorbar(im, ax=ax)
    
    # スタートとゴールを表示
    ax.plot(0, 0, 'go', markersize=10, label='Start')
    ax.plot(maze_size-1, maze_size-1, 'r*', markersize=15, label='Goal')
    ax.legend()
    
    # 2. 行動分布
    ax = axes[0, 1]
    if 'action_distribution' in analysis['decision']:
        actions = list(analysis['decision']['action_distribution'].keys())
        counts = list(analysis['decision']['action_distribution'].values())
        ax.bar(actions, counts)
        ax.set_title('Action Distribution')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
    
    # 3. ループ分析
    ax = axes[1, 0]
    if analysis['path']['loops']:
        loop_lengths = [loop['length'] for loop in analysis['path']['loops']]
        ax.hist(loop_lengths, bins=20, alpha=0.7)
        ax.set_title('Loop Length Distribution')
        ax.set_xlabel('Loop Length (steps)')
        ax.set_ylabel('Count')
    
    # 4. 問題の要約
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    === 失敗の要因分析 ===
    
    1. 探索の偏り:
       - 総ステップ数: {analysis['path']['total_steps']}
       - 訪問した一意の位置: {analysis['path']['unique_positions']}
       - カバレッジ率: {analysis['path']['coverage_ratio']:.2%}
    
    2. ループの形成:
       - 検出されたループ数: {len(analysis['path']['loops'])}
       - 最頻訪問位置: {analysis['path']['most_visited'][0] if analysis['path']['most_visited'] else 'N/A'}
    
    3. 決定の問題:
       - 初期位置での停滞
       - 同じパターンの繰り返し
       - 新規探索の欠如
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def main():
    """メイン分析関数"""
    print("=== 純粋経験学習の失敗分析 ===\n")
    
    # 5x5迷路の結果を分析
    print("5x5 Maze Analysis:")
    result_5x5 = load_json_results('results/pure_experience_5x5_20250802_025002.json')
    
    path_analysis_5 = analyze_path_pattern(result_5x5['path'])
    decision_analysis_5 = analyze_decision_pattern(result_5x5.get('decision_log', []))
    
    print(f"- Total steps: {path_analysis_5['total_steps']}")
    print(f"- Unique positions: {path_analysis_5['unique_positions']}")
    print(f"- Coverage ratio: {path_analysis_5['coverage_ratio']:.2%}")
    print(f"- Most visited position: {path_analysis_5['most_visited'][0]}")
    print(f"- Number of loops detected: {len(path_analysis_5['loops'])}")
    
    # 失敗の根本原因
    print("\n=== 失敗の根本原因 ===")
    print("1. **初期バイアス**: 最初の経験（ゴールエピソード）への過度の依存")
    print("2. **探索不足**: 新しい領域への移動インセンティブの欠如")
    print("3. **局所最適**: 類似性が高い既知の位置に留まる傾向")
    print("4. **フィードバック不足**: 進捗を評価する仕組みがない")
    
    # 改善提案
    print("\n=== 改善提案 ===")
    print("1. **新規性ボーナス**: 未訪問位置への移動を促進")
    print("2. **方向性の連続性**: 同じ方向への継続的な移動を支援")
    print("3. **ドーナツサーチ**: 適度な距離の探索を促進")
    print("4. **ε-greedy戦略**: ランダムな探索と活用のバランス")
    
    # 可視化
    analysis = {
        'path': path_analysis_5,
        'decision': decision_analysis_5
    }
    
    fig = visualize_analysis(5, result_5x5['path'], analysis)
    plt.savefig('results/pure_experience_failure_analysis.png', dpi=150)
    plt.close()
    
    print("\n分析結果を 'results/pure_experience_failure_analysis.png' に保存しました")
    
    # 詳細な失敗パターン
    print("\n=== 詳細な失敗パターン ===")
    if path_analysis_5['loops']:
        print(f"最初のループ: 位置 {path_analysis_5['loops'][0]['position']} で "
              f"{path_analysis_5['loops'][0]['length']} ステップのループ")
    
    # 行動の偏り
    if decision_analysis_5 and 'action_distribution' in decision_analysis_5:
        print("\n行動の分布:")
        for action, count in decision_analysis_5['action_distribution'].items():
            print(f"  {action}: {count}回")

if __name__ == "__main__":
    main()