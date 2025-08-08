#!/usr/bin/env python3
"""
深度とgeDIG値の関係を詳細分析
深い推論でgeDIG値がどう変化するか
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized
from test_true_perfect_maze import generate_perfect_maze_dfs


def analyze_gedig_by_depth():
    """深度別のgeDIG値を詳細分析"""
    
    print("="*70)
    print("📊 深度とgeDIG値の関係分析")
    print("="*70)
    
    # 11×11の完全迷路
    maze = generate_perfect_maze_dfs((11, 11), seed=42)
    
    # 異なる深度設定で実験
    depths = [3, 5, 7, 10, 15, 20]
    results = {}
    
    for max_depth in depths:
        print(f"\n【最大深度: {max_depth}ホップ】")
        
        # エージェント作成（エッジ数7固定）
        agent = PureMemoryAgentOptimized(
            maze=maze,
            datastore_path=f"../results/gedig_depth_{max_depth}",
            config={
                'max_depth': max_depth,
                'search_k': 30,
                'gedig_threshold': 0.5,
                'max_edges_per_node': 7
            }
        )
        
        # 100ステップ実行してgeDIG値を収集
        gedig_by_step = []
        gedig_by_actual_depth = {d: [] for d in range(1, max_depth+1)}
        
        for step in range(100):
            if agent.is_goal_reached():
                print(f"  ✅ {step}ステップでゴール到達")
                break
            
            # 現在のgeDIG値を記録
            if agent.stats['gedig_values']:
                current_gedig = agent.stats['gedig_values'][-1] if agent.stats['gedig_values'] else 0
                gedig_by_step.append(current_gedig)
            
            # 行動実行
            action = agent.get_action()
            agent.execute_action(action)
            
            # 実際に使用された深度を記録
            actual_depth = agent._select_depth_by_gedig()
            if agent.stats['gedig_values']:
                recent_gedig = np.mean(agent.stats['gedig_values'][-10:])
                gedig_by_actual_depth[actual_depth].append(recent_gedig)
        
        # 統計計算
        stats = agent.get_statistics()
        
        results[max_depth] = {
            'success': agent.is_goal_reached(),
            'steps': step,
            'avg_gedig': stats['avg_gedig'],
            'gedig_history': gedig_by_step,
            'gedig_by_depth': gedig_by_actual_depth,
            'depth_usage': stats['depth_usage'],
            'final_gedig': agent.stats['gedig_values'][-1] if agent.stats['gedig_values'] else None
        }
        
        print(f"  平均geDIG: {stats['avg_gedig']:.4f}")
        print(f"  最終geDIG: {results[max_depth]['final_gedig']:.4f}" if results[max_depth]['final_gedig'] else "  最終geDIG: N/A")
        
        # 実際の深度使用分布
        total_usage = sum(stats['depth_usage'].values())
        if total_usage > 0:
            print(f"  深度使用分布:")
            for d in sorted(stats['depth_usage'].keys())[:5]:
                usage = stats['depth_usage'][d]
                if usage > 0:
                    print(f"    {d}ホップ: {usage/total_usage*100:.1f}%")
    
    # 分析結果
    print("\n" + "="*70)
    print("📈 分析結果")
    print("="*70)
    
    print("\n【最大深度とgeDIG値の関係】")
    print("最大深度 | 成功 | 平均geDIG | 最終geDIG | 主要使用深度")
    print("-" * 60)
    
    for depth in depths:
        r = results[depth]
        success = "✅" if r['success'] else "❌"
        avg_gedig = r['avg_gedig']
        final_gedig = r['final_gedig'] if r['final_gedig'] else 0
        
        # 最も使用された深度
        if r['depth_usage']:
            main_depth = max(r['depth_usage'].items(), key=lambda x: x[1])[0]
        else:
            main_depth = 1
        
        print(f"  {depth:2d}    |  {success}  | {avg_gedig:8.4f} | {final_gedig:8.4f} | {main_depth}ホップ")
    
    # geDIG値の変化をプロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 各深度でのgeDIG推移
    ax = axes[0, 0]
    for depth in [5, 10, 15]:
        if depth in results and results[depth]['gedig_history']:
            history = results[depth]['gedig_history'][:50]  # 最初の50ステップ
            ax.plot(history, label=f'深度{depth}', alpha=0.7)
    ax.set_xlabel('ステップ')
    ax.set_ylabel('geDIG値')
    ax.set_title('深度別geDIG値の推移')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='ゼロライン')
    
    # 2. 平均geDIG vs 最大深度
    ax = axes[0, 1]
    avg_gedigs = [results[d]['avg_gedig'] for d in depths]
    colors = ['green' if results[d]['success'] else 'red' for d in depths]
    bars = ax.bar(depths, avg_gedigs, color=colors, alpha=0.6)
    ax.set_xlabel('最大深度')
    ax.set_ylabel('平均geDIG値')
    ax.set_title('最大深度と平均geDIG値の関係')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 成功/失敗をラベル
    for i, (depth, bar) in enumerate(zip(depths, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                '✅' if results[depth]['success'] else '❌',
                ha='center', va='bottom' if height > 0 else 'top')
    
    # 3. 実際の深度使用とgeDIG
    ax = axes[1, 0]
    depth_15_result = results[15]  # 深度15の結果を詳細分析
    if depth_15_result['gedig_by_depth']:
        actual_depths = []
        mean_gedigs = []
        
        for d, values in depth_15_result['gedig_by_depth'].items():
            if values:
                actual_depths.append(d)
                mean_gedigs.append(np.mean(values))
        
        if actual_depths:
            ax.scatter(actual_depths, mean_gedigs, s=100, alpha=0.6)
            ax.plot(actual_depths, mean_gedigs, 'b--', alpha=0.3)
    
    ax.set_xlabel('実際に使用された深度')
    ax.set_ylabel('平均geDIG値')
    ax.set_title('深度15設定での実深度とgeDIG（適応的選択）')
    ax.grid(True, alpha=0.3)
    
    # 4. geDIG値の分布
    ax = axes[1, 1]
    for depth in [5, 10, 15]:
        if depth in results and results[depth]['gedig_history']:
            history = results[depth]['gedig_history']
            if history:
                ax.hist(history, bins=20, alpha=0.5, label=f'深度{depth}')
    
    ax.set_xlabel('geDIG値')
    ax.set_ylabel('頻度')
    ax.set_title('geDIG値の分布')
    ax.legend()
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    plt.suptitle('深度とgeDIG値の関係分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/gedig_depth_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ グラフ保存: results/gedig_depth_analysis.png")
    
    # 洞察
    print("\n" + "="*70)
    print("💡 洞察")
    print("="*70)
    
    # geDIG値の傾向を分析
    gedig_trend = np.corrcoef(depths, avg_gedigs)[0, 1]
    
    print(f"\n相関係数（深度 vs geDIG）: {gedig_trend:.3f}")
    
    if gedig_trend < -0.3:
        print("  → 深度が上がるとgeDIG値が改善（より負に）")
        print("  → 深い推論が情報利得を増大させる")
    elif gedig_trend > 0.3:
        print("  → 深度が上がるとgeDIG値が悪化")
        print("  → 深すぎる推論はノイズを増幅")
    else:
        print("  → 深度とgeDIG値に明確な相関なし")
        print("  → 最適深度が存在する可能性")
    
    # 適応的深度選択の分析
    print("\n🔍 適応的深度選択の挙動:")
    print("  geDIG < -0.3 → 深度5を選択")
    print("  geDIG < 0.0 → 深度4を選択")
    print("  geDIG < 0.3 → 深度3を選択")
    print("  geDIG < 0.5 → 深度2を選択")
    print("  geDIG ≥ 0.5 → 深度1を選択")
    
    # 最適設定の提案
    best_depth = min(depths, key=lambda d: abs(results[d]['avg_gedig']) if results[d]['success'] else float('inf'))
    
    print(f"\n📝 最適設定:")
    print(f"  推奨最大深度: {best_depth}")
    print(f"  平均geDIG: {results[best_depth]['avg_gedig']:.4f}")
    print(f"  理由: 成功率と学習品質のバランス")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 分析実行
    analyze_gedig_by_depth()