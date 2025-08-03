#!/usr/bin/env python3
"""
エピソード記憶の「51%攻撃」問題の分析
=====================================

訪問回数が多い場所のエピソードが支配的になる問題を検証
"""

import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

def analyze_episode_dominance(json_file: str) -> Dict:
    """エピソードの支配度を分析"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    path = data['path']
    decision_log = data.get('decision_log', [])
    
    # 位置ごとの訪問回数
    position_visits = Counter(tuple(pos) for pos in path)
    
    # 最も訪問された位置
    most_visited = position_visits.most_common(5)
    total_steps = len(path)
    
    # 支配度の計算（最も訪問された位置の割合）
    dominance_ratio = most_visited[0][1] / total_steps if most_visited else 0
    
    # ループパターンの検出
    loop_patterns = []
    for i in range(len(path) - 4):
        segment = path[i:i+5]
        if segment[0] == segment[4] and segment[1] == segment[3]:
            loop_patterns.append({
                'pattern': segment,
                'start_index': i
            })
    
    # 決定ログから類似度の偏りを分析
    similarity_bias = []
    if decision_log:
        for log in decision_log[:10]:  # 最初の10ステップ
            if 'action_similarities' in log:
                similarities = log['action_similarities']
                # 各行動の類似度の差
                values = [v for v in similarities.values() if isinstance(v, (int, float))]
                if values:
                    max_sim = max(values)
                    min_sim = min(v for v in values if v != float('-inf'))
                    similarity_bias.append({
                        'step': log['step'],
                        'position': log['position'],
                        'max_similarity': max_sim,
                        'similarity_spread': max_sim - min_sim,
                        'selected': log.get('selected_action', 'N/A')
                    })
    
    return {
        'total_steps': total_steps,
        'unique_positions': len(set(tuple(pos) for pos in path)),
        'most_visited_positions': most_visited,
        'dominance_ratio': dominance_ratio,
        'loop_patterns': loop_patterns[:5],  # 最初の5つのループ
        'similarity_bias': similarity_bias
    }

def main():
    """メイン分析"""
    print("=== エピソード記憶の「51%攻撃」問題分析 ===\n")
    
    # 10x10迷路の結果を分析
    result = analyze_episode_dominance('results/pure_5episodes_10x10_20250802_031650.json')
    
    print(f"総ステップ数: {result['total_steps']}")
    print(f"訪問した一意の位置数: {result['unique_positions']}")
    print(f"\n最頻訪問位置:")
    for pos, count in result['most_visited_positions']:
        ratio = count / result['total_steps'] * 100
        print(f"  位置 {pos}: {count}回 ({ratio:.1f}%)")
    
    print(f"\n支配度（最頻訪問位置の割合）: {result['dominance_ratio']*100:.1f}%")
    
    if result['dominance_ratio'] > 0.3:
        print("⚠️ 特定位置が30%以上を占めており、「51%攻撃」状態に近い")
    
    print(f"\n検出されたループパターン: {len(result['loop_patterns'])}個")
    if result['loop_patterns']:
        first_loop = result['loop_patterns'][0]
        print(f"最初のループ: {first_loop['pattern']}")
    
    print("\n=== 問題の本質 ===")
    print("1. 訪問回数が多い場所のエピソードが蓄積")
    print("2. 類似度検索がそれらのエピソードに偏る")
    print("3. 同じ場所に留まる自己強化ループが発生")
    print("4. 新しい探索が阻害される")
    
    print("\n=== 解決策の提案 ===")
    print("1. エピソードの重複を制限（同じ状態-行動ペアは1つまで）")
    print("2. 時間減衰（古いエピソードの重みを下げる）")
    print("3. エピソード数の上限設定")
    print("4. 多様性を保つサンプリング")

if __name__ == "__main__":
    main()