#!/usr/bin/env python3
"""
Implementation of Emergence Metrics for InsightSpike-AI
創発性を定量的に評価する指標の実装
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
from typing import List, Dict, Tuple, Any

class EmergenceMetrics:
    """創発性評価メトリクスの実装"""
    
    def __init__(self):
        self.history = []
        
    def calculate_structural_emergence(self, 
                                     graph_before: nx.Graph, 
                                     graph_after: nx.Graph) -> Dict[str, float]:
        """構造的創発性の計算"""
        
        # 1. Edge Surprise Score (新規エッジの意外性)
        new_edges = set(graph_after.edges()) - set(graph_before.edges())
        edge_surprise = 0
        
        if len(new_edges) > 0:
            # 既存の構造から新規エッジの確率を推定
            degree_before = dict(graph_before.degree())
            total_possible = len(graph_before.nodes()) * (len(graph_before.nodes()) - 1) / 2
            
            for u, v in new_edges:
                if u in degree_before and v in degree_before:
                    # 次数積に基づく接続確率
                    prob = (degree_before[u] * degree_before[v]) / total_possible
                    edge_surprise += -np.log(max(prob, 1e-10))
            
            edge_surprise /= len(new_edges)
        
        # 2. Clustering Evolution (クラスタリング係数の変化)
        clustering_before = nx.average_clustering(graph_before)
        clustering_after = nx.average_clustering(graph_after)
        clustering_delta = clustering_after - clustering_before
        
        # 3. Hub Emergence (新しいハブの出現)
        centrality_before = nx.betweenness_centrality(graph_before)
        centrality_after = nx.betweenness_centrality(graph_after)
        
        # 中心性が大幅に増加したノードを検出
        hub_emergence = 0
        for node in centrality_after:
            if node in centrality_before:
                delta = centrality_after[node] - centrality_before[node]
                if delta > 0.1:  # 閾値
                    hub_emergence += delta
        
        # 4. Modularity Change (モジュラリティの変化)
        communities_before = list(nx.community.greedy_modularity_communities(graph_before))
        communities_after = list(nx.community.greedy_modularity_communities(graph_after))
        
        modularity_before = nx.community.modularity(graph_before, communities_before)
        modularity_after = nx.community.modularity(graph_after, communities_after)
        modularity_delta = modularity_after - modularity_before
        
        return {
            'edge_surprise': edge_surprise,
            'clustering_evolution': clustering_delta,
            'hub_emergence': hub_emergence,
            'modularity_change': modularity_delta,
            'structural_score': (edge_surprise + abs(clustering_delta) + hub_emergence + abs(modularity_delta)) / 4
        }
    
    def calculate_semantic_emergence(self, 
                                   embeddings_before: np.ndarray,
                                   embeddings_after: np.ndarray,
                                   texts: List[str]) -> Dict[str, float]:
        """意味的創発性の計算"""
        
        # 1. Semantic Drift (意味空間の移動)
        centroid_before = np.mean(embeddings_before, axis=0)
        centroid_after = np.mean(embeddings_after, axis=0)
        semantic_drift = 1 - cosine(centroid_before, centroid_after)
        
        # 2. Concept Diversity (概念の多様性増加)
        diversity_before = np.mean(np.std(embeddings_before, axis=0))
        diversity_after = np.mean(np.std(embeddings_after, axis=0))
        diversity_increase = (diversity_after - diversity_before) / diversity_before
        
        # 3. Semantic Coherence (意味的一貫性)
        # 新しく追加されたテキスト間の一貫性
        if len(embeddings_after) > len(embeddings_before):
            new_embeddings = embeddings_after[len(embeddings_before):]
            coherence = np.mean(cosine_similarity(new_embeddings))
        else:
            coherence = 0
        
        # 4. Contextual Novelty (文脈的新規性)
        # 新しい概念が既存概念とどれだけ異なるか
        novelty = 0
        if len(embeddings_after) > len(embeddings_before):
            new_embeddings = embeddings_after[len(embeddings_before):]
            old_embeddings = embeddings_before
            
            for new_emb in new_embeddings:
                max_sim = np.max(cosine_similarity([new_emb], old_embeddings))
                novelty += (1 - max_sim)
            
            novelty /= len(new_embeddings)
        
        return {
            'semantic_drift': semantic_drift,
            'diversity_increase': diversity_increase,
            'coherence': coherence,
            'contextual_novelty': novelty,
            'semantic_score': (semantic_drift + diversity_increase + coherence + novelty) / 4
        }
    
    def calculate_information_emergence(self,
                                      graph_before: nx.Graph,
                                      graph_after: nx.Graph,
                                      embeddings_after: np.ndarray) -> Dict[str, float]:
        """情報理論的創発性の計算"""
        
        # 1. Graph Entropy Change (グラフエントロピーの変化)
        def graph_entropy(G):
            degrees = [d for n, d in G.degree()]
            if sum(degrees) == 0:
                return 0
            degree_dist = np.array(degrees) / sum(degrees)
            return entropy(degree_dist)
        
        entropy_before = graph_entropy(graph_before)
        entropy_after = graph_entropy(graph_after)
        entropy_change = entropy_after - entropy_before
        
        # 2. Compression Potential (圧縮可能性)
        # より構造化されたデータはより圧縮可能
        if len(embeddings_after) > 0:
            # SVDによる圧縮可能性の評価
            U, s, Vt = np.linalg.svd(embeddings_after, full_matrices=False)
            # 特異値の減衰率
            compression_potential = 1 - (s[-1] / s[0]) if len(s) > 0 else 0
        else:
            compression_potential = 0
        
        # 3. Mutual Information Increase (相互情報量の増加)
        # 簡略化: ノード間の接続パターンの情報量
        adj_before = nx.adjacency_matrix(graph_before).todense()
        adj_after = nx.adjacency_matrix(graph_after).todense()
        
        # 接続パターンのエントロピー変化
        mi_change = np.abs(entropy(adj_after.flatten()) - entropy(adj_before.flatten()))
        
        return {
            'entropy_change': entropy_change,
            'compression_potential': compression_potential,
            'mutual_info_change': mi_change,
            'information_score': (abs(entropy_change) + compression_potential + mi_change) / 3
        }
    
    def calculate_emergence_spike(self, 
                                current_metrics: Dict[str, float],
                                threshold: float = 2.0) -> bool:
        """創発スパイクの検出"""
        
        if len(self.history) < 3:
            self.history.append(current_metrics)
            return False
        
        # 移動平均と標準偏差
        recent_scores = [h['total_score'] for h in self.history[-10:]]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        # 現在のスコアが平均+2σを超えたらスパイク
        current_total = current_metrics['total_score']
        is_spike = current_total > mean_score + threshold * std_score
        
        self.history.append(current_metrics)
        
        return is_spike
    
    def calculate_total_emergence(self,
                                graph_before: nx.Graph,
                                graph_after: nx.Graph,
                                embeddings_before: np.ndarray,
                                embeddings_after: np.ndarray,
                                texts: List[str]) -> Dict[str, Any]:
        """総合的な創発性スコアの計算"""
        
        # 各カテゴリの創発性を計算
        structural = self.calculate_structural_emergence(graph_before, graph_after)
        semantic = self.calculate_semantic_emergence(embeddings_before, embeddings_after, texts)
        information = self.calculate_information_emergence(graph_before, graph_after, embeddings_after)
        
        # 総合スコア
        total_score = (structural['structural_score'] + 
                      semantic['semantic_score'] + 
                      information['information_score']) / 3
        
        # 結果をまとめる
        results = {
            'structural': structural,
            'semantic': semantic,
            'information': information,
            'total_score': total_score,
            'timestamp': len(self.history)
        }
        
        # スパイク検出
        is_spike = self.calculate_emergence_spike(results)
        results['is_emergence_spike'] = is_spike
        
        return results


def analyze_insightspike_emergence(agent, new_texts: List[str]):
    """InsightSpike-AIの創発性を分析"""
    
    metrics = EmergenceMetrics()
    
    # 初期状態を記録
    initial_state = agent.get_memory_graph_state()
    initial_graph = nx.Graph()  # 実際のグラフに変換
    initial_embeddings = []  # 実際の埋め込みを取得
    
    # 新しいテキストを追加しながら創発性を測定
    for i, text in enumerate(new_texts):
        # 現在の状態を保存
        before_graph = initial_graph.copy()
        before_embeddings = initial_embeddings.copy()
        
        # 新しいエピソードを追加
        result = agent.add_episode_with_graph_update(text)
        
        # 更新後の状態を取得
        after_state = agent.get_memory_graph_state()
        after_graph = nx.Graph()  # 更新されたグラフ
        after_embeddings = []  # 更新された埋め込み
        
        # 創発性を計算
        emergence = metrics.calculate_total_emergence(
            before_graph, after_graph,
            np.array(before_embeddings), np.array(after_embeddings),
            new_texts[:i+1]
        )
        
        if emergence['is_emergence_spike']:
            print(f"🎯 Emergence Spike detected at step {i}!")
            print(f"   Total score: {emergence['total_score']:.3f}")
            print(f"   Structural: {emergence['structural']['structural_score']:.3f}")
            print(f"   Semantic: {emergence['semantic']['semantic_score']:.3f}")
            print(f"   Information: {emergence['information']['information_score']:.3f}")
    
    return metrics.history


if __name__ == "__main__":
    # デモンストレーション
    print("=== Emergence Metrics Demo ===\n")
    
    # サンプルグラフの作成
    G1 = nx.karate_club_graph()
    G2 = G1.copy()
    G2.add_edges_from([(0, 10), (5, 15), (20, 30)])  # 新しいエッジを追加
    
    # サンプル埋め込み
    embeddings1 = np.random.randn(34, 128)
    embeddings2 = np.vstack([embeddings1, np.random.randn(5, 128)])
    
    # メトリクスの計算
    metrics = EmergenceMetrics()
    result = metrics.calculate_total_emergence(
        G1, G2, embeddings1, embeddings2, 
        ["text1", "text2", "text3", "text4", "text5"]
    )
    
    print("Emergence Analysis Results:")
    print(f"Total Emergence Score: {result['total_score']:.3f}")
    print(f"- Structural: {result['structural']['structural_score']:.3f}")
    print(f"- Semantic: {result['semantic']['semantic_score']:.3f}")
    print(f"- Information: {result['information']['information_score']:.3f}")
    print(f"Spike Detected: {result['is_emergence_spike']}")