#!/usr/bin/env python3
"""
PyG geDIG実験結果分析
===================
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 実験結果（出力から抽出）
results = {
    "PyG geDIG": {
        "relevance_score": 0.327,
        "relevance_std": 0.434,
        "latency": 5.0,
        "embedding_time": 1.9
    },
    "Original geDIG": {
        "relevance_score": 0.035,
        "relevance_std": 0.131,
        "latency": 0.9,
        "embedding_time": 0.3
    },
    "TF-IDF": {
        "relevance_score": 0.538,
        "relevance_std": 0.481,
        "latency": 2.1,
        "embedding_time": 0.0
    },
    "Sentence-BERT": {
        "relevance_score": 0.633,
        "relevance_std": 0.442,
        "latency": 45.2,
        "embedding_time": 4.5
    }
}

# 統計的有意性
statistical_results = {
    "PyG geDIG vs TF-IDF": {
        "improvement": -39.3,
        "p_value": 0.000876,
        "cohen_d": -0.345,
        "significant": True
    },
    "Original geDIG vs TF-IDF": {
        "improvement": -93.5,
        "p_value": 0.000000,
        "cohen_d": -1.032,
        "significant": True
    },
    "Sentence-BERT vs TF-IDF": {
        "improvement": 17.6,
        "p_value": 0.088451,
        "cohen_d": 0.173,
        "significant": False
    }
}

def analyze_pyg_results():
    """PyG geDIG結果の詳細分析"""
    
    print("🧠 PyG geDIG実験結果分析（550問）")
    print("=" * 60)
    
    # 1. 性能ランキング
    print("\n📊 性能ランキング（Relevance Score）:")
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['relevance_score'], reverse=True)
    
    for rank, (method, metrics) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method}: {metrics['relevance_score']:.3f} ± {metrics['relevance_std']:.3f}")
    
    # 2. 速度ランキング
    print("\n⚡ 速度ランキング（Query Latency）:")
    sorted_speed = sorted(results.items(), key=lambda x: x[1]['latency'])
    
    for rank, (method, metrics) in enumerate(sorted_speed, 1):
        print(f"   {rank}. {method}: {metrics['latency']:.1f}ms")
    
    # 3. PyG geDIG特別分析
    print("\n🧠 PyG geDIG詳細分析:")
    pyg_score = results["PyG geDIG"]["relevance_score"]
    original_score = results["Original geDIG"]["relevance_score"]
    
    pyg_improvement = (pyg_score - original_score) / original_score * 100
    print(f"   PyG vs Original改善率: +{pyg_improvement:.1f}%")
    print(f"   PyGはOriginalの{pyg_score/original_score:.1f}倍の性能")
    
    # 4. 統計的有意性
    print("\n📊 統計的有意性サマリ:")
    for comparison, stats in statistical_results.items():
        significance = "✅ 有意" if stats["significant"] else "❌ 非有意"
        print(f"   {comparison}:")
        print(f"      改善率: {stats['improvement']:+.1f}%")
        print(f"      p値: {stats['p_value']:.6f}")
        print(f"      効果量: {stats['cohen_d']:.3f}")
        print(f"      結果: {significance}")
    
    # 5. 効率性分析
    print("\n⚡ 効率性分析（精度/速度比）:")
    for method, metrics in results.items():
        efficiency = metrics['relevance_score'] / metrics['latency'] * 1000
        print(f"   {method}: {efficiency:.2f} (score/ms × 1000)")
    
    # 可視化
    create_detailed_visualization()

def create_detailed_visualization():
    """詳細な可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    colors = ['gold', 'lightblue', 'lightgreen', 'coral']
    
    # 1. 性能比較（エラーバー付き）
    scores = [results[m]['relevance_score'] for m in methods]
    stds = [results[m]['relevance_std'] for m in methods]
    
    bars1 = ax1.bar(methods, scores, yerr=stds, capsize=5, color=colors)
    ax1.set_ylabel('Relevance Score')
    ax1.set_title('Retrieval Performance Comparison (550 Questions)')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. レイテンシ比較（対数スケール）
    latencies = [results[m]['latency'] for m in methods]
    bars2 = ax2.bar(methods, latencies, color=colors)
    ax2.set_ylabel('Query Latency (ms)')
    ax2.set_title('Response Time Comparison')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, latency in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{latency:.1f}', ha='center', va='bottom')
    
    # 3. PyG vs Original geDIG比較
    gedig_methods = ['PyG geDIG', 'Original geDIG']
    gedig_scores = [results[m]['relevance_score'] for m in gedig_methods]
    gedig_latencies = [results[m]['latency'] for m in gedig_methods]
    
    x = np.arange(len(gedig_methods))
    width = 0.35
    
    bars3_1 = ax3.bar(x - width/2, gedig_scores, width, label='Relevance Score', color='gold')
    ax3_2 = ax3.twinx()
    bars3_2 = ax3_2.bar(x + width/2, gedig_latencies, width, label='Latency (ms)', color='lightblue')
    
    ax3.set_xlabel('geDIG Variants')
    ax3.set_ylabel('Relevance Score', color='gold')
    ax3_2.set_ylabel('Latency (ms)', color='lightblue')
    ax3.set_title('PyG vs Original geDIG Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gedig_methods)
    ax3.tick_params(axis='y', labelcolor='gold')
    ax3_2.tick_params(axis='y', labelcolor='lightblue')
    
    # 4. 効率性マトリックス
    scores_all = [results[m]['relevance_score'] for m in methods]
    latencies_all = [results[m]['latency'] for m in methods]
    
    for i, (method, score, latency) in enumerate(zip(methods, scores_all, latencies_all)):
        ax4.scatter(latency, score, s=300, c=colors[i], label=method, 
                   alpha=0.7, edgecolors='black', linewidth=2)
        ax4.annotate(method, (latency, score), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Query Latency (ms)')
    ax4.set_ylabel('Relevance Score')
    ax4.set_title('Efficiency Matrix (Upper-Left is Better)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(latencies_all) * 1.2)
    ax4.set_ylim(0, max(scores_all) * 1.1)
    
    # 最適領域を強調
    ax4.axvspan(0, 5, alpha=0.1, color='green', label='Fast Zone')
    ax4.axhspan(0.5, 1.0, alpha=0.1, color='green', label='High Accuracy Zone')
    
    plt.tight_layout()
    plt.savefig('pyg_gedig_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📈 可視化保存: pyg_gedig_analysis.png")

def conclusions():
    """最終結論"""
    
    print("\n🎯 PyG geDIG実験最終結論:")
    print("=" * 60)
    
    print("\n✅ **主要発見:**")
    print("1. PyG geDIGはOriginal geDIGの**9.3倍**の性能向上")
    print("2. Sentence-BERTが最高性能（0.633）を達成")
    print("3. PyG geDIG（0.327）はTF-IDF（0.538）に及ばず")
    print("4. 統計的に有意な差を確認（p < 0.001）")
    
    print("\n⚡ **速度分析:**")
    print("1. Original geDIG: 0.9ms（最速）")
    print("2. TF-IDF: 2.1ms")
    print("3. PyG geDIG: 5.0ms（中速）")
    print("4. Sentence-BERT: 45.2ms（最遅）")
    
    print("\n🧠 **PyG geDIGの意義:**")
    print("1. グラフニューラルネットワークの可能性を実証")
    print("2. CPU環境でも実用的な速度（5ms）")
    print("3. GPU環境では更なる高速化が期待")
    print("4. アーキテクチャ改善で精度向上の余地大")
    
    print("\n🚀 **今後の改善提案:**")
    print("1. Graph Attention Networks (GAT) の導入")
    print("2. 事前学習済みグラフ表現の活用")
    print("3. ΔGED×ΔIG計算の最適化")
    print("4. マルチスケールグラフ特徴の統合")

if __name__ == "__main__":
    analyze_pyg_results()
    conclusions()