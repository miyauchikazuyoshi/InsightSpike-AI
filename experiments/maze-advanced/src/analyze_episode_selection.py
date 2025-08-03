#!/usr/bin/env python3
"""
エピソード選択の分析
===================
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_episode_selection(json_file: str):
    """エピソード選択パターンを分析"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== {data['maze_size']}x{data['maze_size']} Maze Episode Analysis ===")
    print(f"Total steps: {data['total_steps']}")
    print(f"Success: {data['success']}")
    print(f"Total episodes: {len(data['episode_records'])}")
    
    # 各ステップでの類似エピソード選択を分析
    print("\n--- Episode Selection Pattern ---")
    
    goal_references = 0
    wall_avoidances = 0
    exploration_bonuses = 0
    
    for i, record in enumerate(data['episode_records'][:10]):  # 最初の10ステップ
        print(f"\nStep {record['step']}: Position {record['position']} → {record['action_name']}")
        
        # 最も類似したエピソード
        if record['similar_episodes']:
            best_similar = record['similar_episodes'][0]
            print(f"  Most similar: {best_similar['episode_id']} (similarity: {best_similar['similarity']:.3f})")
            
            if best_similar['episode_id'] == 'GOAL_EPISODE':
                goal_references += 1
            
        # アクションスコア
        print(f"  Action scores: {record['action_scores']}")
        print(f"  Selected: {record['action_name']} → {record['result']}")
    
    # メモリ成長の可視化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. メモリノード数の増加
    steps = [r['step'] for r in data['episode_records']]
    memory_counts = [r['memory_node_count'] for r in data['episode_records']]
    
    ax1.plot(steps, memory_counts, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Memory Node Count')
    ax1.set_title(f'Memory Growth in {data["maze_size"]}x{data["maze_size"]} Maze')
    ax1.grid(True, alpha=0.3)
    
    # 2. 各ステップでの最高類似度
    max_similarities = []
    for record in data['episode_records']:
        if record['similar_episodes']:
            max_sim = max(ep['similarity'] for ep in record['similar_episodes'])
            max_similarities.append(max_sim)
        else:
            max_similarities.append(0)
    
    ax2.plot(steps, max_similarities, 'g-', linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='High Similarity Threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Max Similarity to Past Episodes')
    ax2.set_title('Learning from Past Experience')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ファイル名から保存
    base_name = Path(json_file).stem
    output_file = f'results/{base_name}_analysis.png'
    plt.savefig(output_file)
    plt.close()
    
    print(f"\n--- Summary ---")
    print(f"Goal episode references: {goal_references}")
    print(f"Average max similarity: {np.mean(max_similarities):.3f}")
    print(f"Analysis saved to: {output_file}")
    
    return data


def create_episode_selection_summary():
    """複数の実験結果をまとめて表示"""
    
    # 10x10と20x20の結果を探す
    results_dir = Path('results')
    json_files = sorted(results_dir.glob('maze_*_episodes_*.json'))
    
    if not json_files:
        print("No episode JSON files found!")
        return
    
    for json_file in json_files[-2:]:  # 最新の2つ
        analyze_episode_selection(str(json_file))


if __name__ == "__main__":
    create_episode_selection_summary()