"""
Episode graph construction and management module
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
from .episode_manager import Episode
from .gedig_evaluator import GeDIGEvaluator


class GraphManager:
    """エピソードグラフの構築と管理

    Phase0 追加: max_snapshots による履歴上限・ prune。
    今後 Phase2 で条件付き snapshot スキップと組み合わせる。
    """

    def __init__(self, gedig_evaluator: Optional[GeDIGEvaluator] = None, max_snapshots: Optional[int] = None):
        """Args:
            gedig_evaluator: geDIG評価器
            max_snapshots: 履歴上限 (None=無制限)
        """
        self.graph = nx.Graph()
        self.gedig_evaluator = gedig_evaluator or GeDIGEvaluator()
        self.graph_history: list[nx.Graph] = []  # スナップショット履歴
        self.edge_creation_log = []  # エッジ作成ログ
        self.max_snapshots = max_snapshots
    
    def add_episode_node(self, episode: Episode) -> None:
        """
        エピソードをノードとしてグラフに追加
        
        Args:
            episode: 追加するエピソード
        """
        self.graph.add_node(
            episode.episode_id,
            position=episode.position,
            direction=episode.direction,
            is_wall=episode.is_wall,
            visit_count=episode.visit_count,
            timestamp=episode.timestamp
        )
    
    def wire_edges(
        self,
        episodes: List[Episode],
        strategy: str = 'simple'
    ) -> None:
        """
        エピソード間のエッジ配線
        
        Args:
            episodes: エピソードリスト
        strategy: 配線戦略 ('simple', 'temporal', 'spatial', 'gedig', 'loop_test')
        """
        if strategy == 'simple':
            self._wire_simple(episodes)
        elif strategy == 'temporal':
            self._wire_temporal(episodes)
        elif strategy == 'spatial':
            self._wire_spatial(episodes)
        elif strategy == 'gedig':
            self._wire_with_gedig(episodes)
        elif strategy == 'loop_test':
            # MazeNavigator 側で人工 loop 挿入を行うためここでは no-op (インタフェース整合用)
            return
        else:
            raise ValueError(f"Unknown wiring strategy: {strategy}")
    
    def wire_with_gedig(
        self,
        episodes: List[Episode],
        threshold: float = 0.3
    ) -> None:
        """
        Approach D: geDIG閾値に基づくエッジ配線
        
        Args:
            episodes: エピソードリスト
            threshold: geDIG閾値
        """
        self._wire_with_gedig(episodes, threshold)
    
    def get_graph_snapshot(self) -> nx.Graph:
        """
        現在のグラフのスナップショットを取得
        
        Returns:
            グラフのコピー
        """
        return self.graph.copy()
    
    def save_snapshot(self) -> None:
        """現在のグラフをスナップショット履歴に保存 (上限 prune)"""
        if self.max_snapshots is not None and self.max_snapshots <= 0:
            return  # 保存抑制
        self.graph_history.append(self.get_graph_snapshot())
        if self.max_snapshots is not None and len(self.graph_history) > self.max_snapshots:
            # 最古を削除 (単純 FIFO)。必要なら間引き戦略を後続フェーズで拡張。
            overflow = len(self.graph_history) - self.max_snapshots
            if overflow > 0:
                del self.graph_history[0:overflow]
    
    def get_connected_episodes(
        self,
        episode_id: int
    ) -> List[int]:
        """
        指定エピソードに接続されているエピソードIDを取得
        
        Args:
            episode_id: エピソードID
        
        Returns:
            接続されているエピソードIDのリスト
        """
        if episode_id not in self.graph:
            return []
        
        return list(self.graph.neighbors(episode_id))
    
    def get_shortest_path(
        self,
        source_id: int,
        target_id: int
    ) -> Optional[List[int]]:
        """
        2つのエピソード間の最短パスを取得
        
        Args:
            source_id: 開始エピソードID
            target_id: 目標エピソードID
        
        Returns:
            エピソードIDのリスト（パスが存在しない場合はNone）
        """
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """
        グラフの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        # 連結グラフの場合の追加統計
        if stats['is_connected'] and self.graph.number_of_nodes() > 0:
            stats['diameter'] = nx.diameter(self.graph)
            stats['radius'] = nx.radius(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)
        
        # 次数統計
        if self.graph.number_of_nodes() > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats
    
    def detect_communities(self) -> List[Set[int]]:
        """
        コミュニティ検出
        
        Returns:
            コミュニティ（エピソードIDの集合）のリスト
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        # Louvainアルゴリズムでコミュニティ検出
        communities = nx.community.greedy_modularity_communities(self.graph)
        return [set(community) for community in communities]
    
    def _wire_simple(self, episodes: List[Episode]) -> None:
        """
        シンプルな配線戦略：直前のエピソードと接続
        
        Args:
            episodes: エピソードリスト
        """
        for i in range(1, len(episodes)):
            self.graph.add_edge(
                episodes[i].episode_id,
                episodes[i-1].episode_id
            )
            self._log_edge_creation(
                episodes[i].episode_id,
                episodes[i-1].episode_id,
                'simple'
            )
    
    def _wire_temporal(self, episodes: List[Episode]) -> None:
        """
        時間的配線戦略：時間的に近いエピソードを接続
        
        Args:
            episodes: エピソードリスト
        """
        # タイムスタンプでソート
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        
        for i in range(len(sorted_episodes)):
            current = sorted_episodes[i]
            
            # 時間的に近い（±3ステップ）エピソードと接続
            for j in range(max(0, i-3), min(len(sorted_episodes), i+4)):
                if i != j:
                    other = sorted_episodes[j]
                    if abs(current.timestamp - other.timestamp) <= 3:
                        self.graph.add_edge(current.episode_id, other.episode_id)
                        self._log_edge_creation(
                            current.episode_id,
                            other.episode_id,
                            'temporal'
                        )
    
    def _wire_spatial(self, episodes: List[Episode]) -> None:
        """
        空間的配線戦略：同じ位置のエピソードを接続
        
        Args:
            episodes: エピソードリスト
        """
        # 位置でグループ化
        position_groups = {}
        for ep in episodes:
            if ep.position not in position_groups:
                position_groups[ep.position] = []
            position_groups[ep.position].append(ep)
        
        # 同じ位置のエピソード間にエッジを作成
        for position, group in position_groups.items():
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    self.graph.add_edge(
                        group[i].episode_id,
                        group[j].episode_id
                    )
                    self._log_edge_creation(
                        group[i].episode_id,
                        group[j].episode_id,
                        'spatial'
                    )
    
    def _wire_with_gedig(
        self,
        episodes: List[Episode],
        threshold: float = 0.3
    ) -> None:
        """
        Approach D: geDIG閾値に基づく配線
        
        Args:
            episodes: エピソードリスト
            threshold: geDIG閾値
        """
        if len(episodes) < 2:
            return
        
        # タイムスタンプでソート
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        
        for i in range(1, len(sorted_episodes)):
            current = sorted_episodes[i]
            
            # 現在のグラフのスナップショット
            g_before = self.get_graph_snapshot()
            
            # 仮エッジを追加してgeDIG値を計算
            best_connection = None
            best_gedig = None
            
            # 過去のエピソードとの接続を評価
            for j in range(max(0, i-10), i):  # 最大10個前まで評価
                other = sorted_episodes[j]
                
                # 仮グラフで評価
                temp_graph = g_before.copy()
                temp_graph.add_edge(current.episode_id, other.episode_id)
                
                # geDIG値計算
                gedig_value = self.gedig_evaluator.calculate(g_before, temp_graph)
                
                # 最小のgeDIG値を選択（構造統合と情報整理の最大化）
                # geDIG値は負が良い（論文の定義に従う）
                if best_gedig is None or gedig_value < best_gedig:
                    best_gedig = gedig_value
                    best_connection = other.episode_id
            
            # 閾値以下（負の方向に十分改善）であれば接続
            # 注：geDIG値は負が良い（構造統合と情報整理）ので、threshold=-0.3のような負の値を使用
            # thresholdが既に負の値として渡される場合はそのまま使用
            threshold_value = threshold if threshold < 0 else -abs(threshold)
            if best_connection is not None and best_gedig is not None and best_gedig <= threshold_value:
                self.graph.add_edge(current.episode_id, best_connection)
                self._log_edge_creation(
                    current.episode_id,
                    best_connection,
                    f'gedig (value={best_gedig:.3f})'
                )
    
    def _log_edge_creation(
        self,
        source: int,
        target: int,
        strategy: str
    ) -> None:
        """
        エッジ作成をログに記録
        
        Args:
            source: ソースエピソードID
            target: ターゲットエピソードID
            strategy: 使用した戦略
        """
        self.edge_creation_log.append({
            'source': source,
            'target': target,
            'strategy': strategy,
            'graph_size': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges()
        })