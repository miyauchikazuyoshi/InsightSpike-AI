"""
Refined GraphManager with better incremental geDIG calculation.
本物のgeDIG計算に近い、でも高速な実装。
"""

from typing import List, Optional, Dict, Any, Set, Tuple
import networkx as nx
import numpy as np
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


class RefinedGraphManager:
    """洗練されたインクリメンタルgeDIG計算を持つGraphManager"""
    
    def __init__(self, gedig_evaluator: Optional[GeDIGEvaluator] = None):
        self.graph = nx.Graph()
        self.gedig_evaluator = gedig_evaluator or GeDIGEvaluator()
        self._gedig_cache: Dict[tuple, float] = {}
        self.edge_logs: List[Dict[str, Any]] = []
        self.graph_history: List[nx.Graph] = []
        self.edge_creation_log = []
        
        # 洗練化のための追加状態
        self._node_vectors: Dict[int, np.ndarray] = {}  # ノードのベクトル保存
        self._local_structures: Dict[int, Dict] = {}  # ローカル構造キャッシュ
        self._community_map: Dict[int, int] = {}  # コミュニティ割り当て
        self._last_ged = 0.0  # 前回のGED値
        self._last_ig = 0.0  # 前回のIG値
    
    def add_episode_node(self, episode: Episode) -> None:
        """Add episode node to graph with vector caching."""
        self.graph.add_node(
            episode.episode_id,
            position=episode.position,
            timestamp=episode.timestamp
        )
        # ベクトルを保存（IG計算用）
        if hasattr(episode, 'vector') and episode.vector is not None:
            self._node_vectors[episode.episode_id] = episode.vector
    
    def wire_edges(self, episodes: List[Episode], strategy: str = 'refined_gedig') -> None:
        """Wire episodes with specified strategy."""
        if strategy == 'refined_gedig':
            self._wire_with_refined_gedig(episodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _wire_with_refined_gedig(
        self,
        episodes: List[Episode],
        threshold: float = -0.15,
        use_real_gedig_sampling: bool = True
    ) -> None:
        """
        洗練されたgeDIG配線戦略
        
        Key improvements:
        1. 実際のgeDIG計算を定期的にサンプリング
        2. ローカル構造の変化を考慮
        3. 情報利得（IG）の近似計算
        4. コミュニティ構造の考慮
        """
        if len(episodes) < 2:
            return
        
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        
        # 実際のgeDIG値を定期的に計算して校正
        calibration_interval = 10
        real_gedig_samples = []
        
        for i in range(1, len(sorted_episodes)):
            current = sorted_episodes[i]
            
            best_connection = None
            best_gedig = None
            
            # 探索範囲を限定
            search_limit = min(i, 5 + i // 10)
            
            for j in range(max(0, i - search_limit), i):
                other = sorted_episodes[j]
                
                cache_key = (current.episode_id, other.episode_id)
                
                # キャッシュチェック
                if cache_key in self._gedig_cache:
                    gedig_value = self._gedig_cache[cache_key]
                else:
                    # 校正：10回に1回は本物のgeDIG計算
                    if use_real_gedig_sampling and len(real_gedig_samples) % calibration_interval == 0:
                        # 本物のgeDIG計算（遅いが正確）
                        temp_graph = self.graph.copy()
                        temp_graph.add_edge(current.episode_id, other.episode_id)
                        real_gedig = self.gedig_evaluator.calculate(self.graph, temp_graph)
                        
                        # 近似計算も実行
                        approx_gedig = self._calculate_refined_gedig(
                            current.episode_id,
                            other.episode_id
                        )
                        
                        # 校正係数を更新
                        if approx_gedig != 0:
                            calibration_factor = real_gedig / approx_gedig
                            real_gedig_samples.append((real_gedig, approx_gedig, calibration_factor))
                        
                        gedig_value = real_gedig
                    else:
                        # 洗練された近似計算
                        gedig_value = self._calculate_refined_gedig(
                            current.episode_id,
                            other.episode_id
                        )
                        
                        # 校正係数で補正
                        if real_gedig_samples:
                            avg_calibration = np.mean([s[2] for s in real_gedig_samples[-5:]])
                            gedig_value *= avg_calibration
                    
                    self._gedig_cache[cache_key] = gedig_value
                
                # ベストな接続を追跡
                if best_gedig is None or gedig_value < best_gedig:
                    best_gedig = gedig_value
                    best_connection = other.episode_id
            
            # 閾値チェックして接続
            if best_connection is not None and best_gedig is not None and best_gedig <= threshold:
                self.graph.add_edge(current.episode_id, best_connection)
                self.edge_logs.append({
                    'from': current.episode_id,
                    'to': best_connection,
                    'gedig': best_gedig,
                    'threshold': threshold
                })
                
                # ローカル構造を更新
                self._update_local_structures(current.episode_id, best_connection)
    
    def _calculate_refined_gedig(self, node1: int, node2: int) -> float:
        """
        洗練されたインクリメンタルgeDIG計算
        
        geDIG = ΔGED - λ * ΔIG の近似
        """
        # 1. ΔGED（グラフ編集距離の変化）を近似
        delta_ged = self._estimate_delta_ged(node1, node2)
        
        # 2. ΔIG（情報利得の変化）を近似
        delta_ig = self._estimate_delta_ig(node1, node2)
        
        # 3. 結合（論文の式: F_t = ΔGED_norm - λ * ΔIG_norm）
        lambda_weight = 0.1  # IGの重み
        gedig_value = delta_ged - lambda_weight * delta_ig
        
        return gedig_value
    
    def _estimate_delta_ged(self, node1: int, node2: int) -> float:
        """
        グラフ編集距離の変化を推定
        
        考慮する要素：
        - 最短パス長の変化
        - クラスタリング係数の変化
        - 次数分布の変化
        """
        if self.graph.has_edge(node1, node2):
            return 0.0
        
        # 両ノードが存在するか確認
        if not (self.graph.has_node(node1) and self.graph.has_node(node2)):
            return -0.5  # 新規ノードの接続は良い
        
        # 1. 最短パス長の改善を評価
        try:
            current_path_length = nx.shortest_path_length(self.graph, node1, node2)
            path_improvement = (current_path_length - 1) / current_path_length
        except nx.NetworkXNoPath:
            # 接続されていないコンポーネント間の接続は非常に良い
            path_improvement = 1.0
        
        # 2. ローカルクラスタリングの変化
        neighbors1 = set(self.graph.neighbors(node1))
        neighbors2 = set(self.graph.neighbors(node2))
        common_neighbors = len(neighbors1 & neighbors2)
        total_neighbors = len(neighbors1 | neighbors2)
        
        if total_neighbors > 0:
            clustering_change = -common_neighbors / total_neighbors  # 共通隣接が多いほど負
        else:
            clustering_change = -0.5
        
        # 3. 次数バランスの考慮
        deg1 = self.graph.degree(node1)
        deg2 = self.graph.degree(node2)
        degrees = [d for n, d in self.graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 1.0
        
        # 低次数ノードの接続を優先
        degree_factor = -np.exp(-(deg1 + deg2) / (2 * max(avg_degree, 0.1)))
        
        # 統合
        delta_ged = 0.4 * path_improvement + 0.3 * clustering_change + 0.3 * degree_factor
        
        return delta_ged
    
    def _estimate_delta_ig(self, node1: int, node2: int) -> float:
        """
        情報利得の変化を推定
        
        考慮する要素：
        - ベクトル空間での類似度
        - エントロピーの変化
        - 情報の冗長性
        """
        # ベクトルが利用可能な場合
        if node1 in self._node_vectors and node2 in self._node_vectors:
            vec1 = self._node_vectors[node1]
            vec2 = self._node_vectors[node2]
            
            # コサイン類似度
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
            
            # 類似度が高い = 情報利得が少ない
            # 類似度が低い = 情報利得が多い
            delta_ig = 1.0 - abs(similarity)
        else:
            # ベクトルがない場合は構造的多様性で推定
            neighbors1 = set(self.graph.neighbors(node1)) if self.graph.has_node(node1) else set()
            neighbors2 = set(self.graph.neighbors(node2)) if self.graph.has_node(node2) else set()
            
            # Jaccard距離（非類似度）
            if neighbors1 or neighbors2:
                jaccard_distance = 1.0 - len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
                delta_ig = jaccard_distance
            else:
                delta_ig = 0.5
        
        return delta_ig
    
    def _update_local_structures(self, node1: int, node2: int) -> None:
        """ローカル構造キャッシュを更新"""
        for node in [node1, node2]:
            if self.graph.has_node(node):
                self._local_structures[node] = {
                    'degree': self.graph.degree(node),
                    'neighbors': set(self.graph.neighbors(node)),
                    'clustering': nx.clustering(self.graph, node) if self.graph.degree(node) > 1 else 0
                }
    
    # 互換性のためのメソッド
    def get_graph_snapshot(self) -> nx.Graph:
        return self.graph.copy()
    
    def save_snapshot(self) -> None:
        self.graph_history.append(self.get_graph_snapshot())
    
    def get_connected_episodes(self, episode_id: int) -> List[int]:
        if episode_id not in self.graph:
            return []
        return list(self.graph.neighbors(episode_id))
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        if stats['is_connected'] and self.graph.number_of_nodes() > 0:
            stats['diameter'] = nx.diameter(self.graph)
            stats['radius'] = nx.radius(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)
        
        if self.graph.number_of_nodes() > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats