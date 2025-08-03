#!/usr/bin/env python3
"""
GeDIG Sleep Navigator
=====================

GED/IGベースの睡眠サイクルを実装したナビゲーター
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import itertools
import logging
import json
from datetime import datetime
import time
import random

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import VisualMemoryNavigator, Episode7D, generate_complex_maze

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeDIGEpisode:
    """GED/IG計算を含むエピソード"""
    episode: Episode7D
    node_id: int
    connections: Set[int]
    
    # GeDIG metrics
    ged: float = 0.0          # Graph Edit Distance
    ig: float = 0.0           # Information Gain
    c_value: float = 0.0      # Spike confidence
    
    # Access statistics
    access_count: int = 0
    last_access: int = 0
    creation_step: int = 0


class GeDIGSleepNavigator(VisualMemoryNavigator):
    """GED/IGベースの睡眠を持つナビゲーター"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        
        # GeDIGエピソード管理
        self.gedig_episodes: Dict[int, GeDIGEpisode] = {}
        self.next_node_id = 0
        
        # 睡眠パラメータ
        self.sleep_interval = 500
        self.ged_threshold = 0.3      # GEDがこれ以下なら冗長
        self.ig_threshold = 0.1       # IGがこれ以下なら価値が低い
        self.connection_limit = 15    # 接続数の上限
        
        # GeDIG計算用
        self.alpha = 0.6  # GED weight
        self.beta = 0.4   # IG weight
        
        # 睡眠履歴
        self.sleep_history = []
        self.ged_ig_history = []
    
    def add_gedig_episode(self, episode: Episode7D) -> int:
        """GeDIGエピソードを追加"""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        gedig_ep = GeDIGEpisode(
            episode=episode,
            node_id=node_id,
            connections=set(),
            creation_step=self.step_count
        )
        
        self.gedig_episodes[node_id] = gedig_ep
        self.episodes.append(episode)
        
        # 初期GED/IG計算
        self._update_ged_ig(node_id)
        
        return node_id
    
    def connect_episodes(self, id1: int, id2: int, strength: float = 1.0):
        """エピソードを接続（強度付き）"""
        if id1 != id2 and id1 in self.gedig_episodes and id2 in self.gedig_episodes:
            self.gedig_episodes[id1].connections.add(id2)
            self.gedig_episodes[id2].connections.add(id1)
            
            # GED/IGを更新
            self._update_ged_ig(id1)
            self._update_ged_ig(id2)
    
    def _calculate_ged(self, node_id: int) -> float:
        """Graph Edit Distanceを計算"""
        node = self.gedig_episodes[node_id]
        
        if not node.connections:
            return 1.0  # 孤立ノードは最大距離
        
        # 接続ノードとの平均類似度
        total_similarity = 0.0
        for conn_id in node.connections:
            if conn_id in self.gedig_episodes:
                conn_node = self.gedig_episodes[conn_id]
                # 7次元ベクトルの類似度
                similarity = self._episode_similarity(node.episode, conn_node.episode)
                total_similarity += similarity
        
        avg_similarity = total_similarity / len(node.connections)
        return 1.0 - avg_similarity  # 類似度が高いほどGEDは低い
    
    def _calculate_ig(self, node_id: int) -> float:
        """Information Gainを計算"""
        node = self.gedig_episodes[node_id]
        
        # 情報量の指標
        ig = 0.0
        
        # 1. 位置の新規性
        if node.episode.x is not None and node.episode.y is not None:
            position_visits = self.position_visits.get((node.episode.x, node.episode.y), 0)
            ig += 1.0 / (1.0 + position_visits)
        
        # 2. ゴール情報
        if node.episode.goal_or_not:
            ig += 2.0
        
        # 3. 壁情報の価値
        if node.episode.wall_or_path == 'wall' and node.episode.visit_count == 0:
            ig += 0.5  # 新しい壁情報
        
        # 4. アクセス頻度による価値
        recency = self.step_count - node.last_access
        ig *= 1.0 / (1.0 + recency * 0.001)
        
        return min(ig, 1.0)  # 0-1に正規化
    
    def _update_ged_ig(self, node_id: int):
        """GEDとIGを更新"""
        if node_id in self.gedig_episodes:
            node = self.gedig_episodes[node_id]
            node.ged = self._calculate_ged(node_id)
            node.ig = self._calculate_ig(node_id)
            
            # C値（スパイク信頼度）も計算
            node.c_value = self.alpha * (1.0 - node.ged) + self.beta * node.ig
    
    def _episode_similarity(self, ep1: Episode7D, ep2: Episode7D) -> float:
        """エピソード間の類似度"""
        score = 0.0
        count = 0
        
        # 位置の近さ
        if ep1.x is not None and ep2.x is not None:
            distance = abs(ep1.x - ep2.x) + abs(ep1.y - ep2.y)
            score += 1.0 / (1.0 + distance * 0.1)
            count += 1
        
        # 属性の一致
        if ep1.direction == ep2.direction and ep1.direction is not None:
            score += 1.0
            count += 1
        
        if ep1.wall_or_path == ep2.wall_or_path:
            score += 0.5
            count += 1
        
        return score / max(count, 1)
    
    def sleep_cycle(self):
        """GED/IGベースの睡眠サイクル"""
        print(f"\n💤 GeDIG Sleep Cycle at step {self.step_count}")
        
        initial_nodes = len(self.gedig_episodes)
        initial_edges = sum(len(n.connections) for n in self.gedig_episodes.values()) // 2
        
        # 現在の平均GED/IG
        avg_ged = np.mean([n.ged for n in self.gedig_episodes.values()])
        avg_ig = np.mean([n.ig for n in self.gedig_episodes.values()])
        
        print(f"   Before: GED={avg_ged:.3f}, IG={avg_ig:.3f}")
        
        # 1. 低価値エッジの削除（GED減少、IG最小化を考慮）
        pruned_edges = self._prune_low_value_edges()
        
        # 2. 冗長ノードの統合
        merged_nodes = self._merge_redundant_nodes()
        
        # 3. 孤立した低価値ノードの削除
        removed_nodes = self._remove_low_value_nodes()
        
        # 削除後の統計
        final_nodes = len(self.gedig_episodes)
        final_edges = sum(len(n.connections) for n in self.gedig_episodes.values()) // 2
        
        # 新しい平均GED/IG
        if self.gedig_episodes:
            new_avg_ged = np.mean([n.ged for n in self.gedig_episodes.values()])
            new_avg_ig = np.mean([n.ig for n in self.gedig_episodes.values()])
        else:
            new_avg_ged = new_avg_ig = 0.0
        
        print(f"   After:  GED={new_avg_ged:.3f}, IG={new_avg_ig:.3f}")
        print(f"   Nodes: {initial_nodes} → {final_nodes} (-{initial_nodes - final_nodes})")
        print(f"   Edges: {initial_edges} → {final_edges} (-{initial_edges - final_edges})")
        
        # 履歴を記録
        self.sleep_history.append({
            'step': self.step_count,
            'before': {'nodes': initial_nodes, 'edges': initial_edges, 'ged': avg_ged, 'ig': avg_ig},
            'after': {'nodes': final_nodes, 'edges': final_edges, 'ged': new_avg_ged, 'ig': new_avg_ig},
            'actions': {
                'pruned_edges': pruned_edges,
                'merged_nodes': merged_nodes,
                'removed_nodes': removed_nodes
            }
        })
    
    def _prune_low_value_edges(self) -> int:
        """低価値なエッジを削除（GED減少とIG最小化を考慮）"""
        pruned = 0
        
        for node_id, node in list(self.gedig_episodes.items()):
            if len(node.connections) > self.connection_limit:
                # 各接続の価値を評価
                edge_values = []
                
                for conn_id in node.connections:
                    if conn_id in self.gedig_episodes:
                        conn_node = self.gedig_episodes[conn_id]
                        
                        # エッジの価値 = 相手のC値 + 類似度ペナルティ
                        similarity = self._episode_similarity(node.episode, conn_node.episode)
                        edge_value = conn_node.c_value - similarity * 0.5  # 類似しすぎは価値が低い
                        
                        edge_values.append((conn_id, edge_value))
                
                # 価値の低いエッジから削除
                edge_values.sort(key=lambda x: x[1])
                
                edges_to_remove = len(node.connections) - self.connection_limit
                for conn_id, _ in edge_values[:edges_to_remove]:
                    node.connections.discard(conn_id)
                    if conn_id in self.gedig_episodes:
                        self.gedig_episodes[conn_id].connections.discard(node_id)
                    pruned += 1
                
                # GED/IGを再計算
                self._update_ged_ig(node_id)
        
        return pruned
    
    def _merge_redundant_nodes(self) -> int:
        """冗長なノードを統合"""
        merged = 0
        nodes_to_remove = set()
        
        # GEDが低い（類似度が高い）ノードペアを探す
        for id1, id2 in itertools.combinations(self.gedig_episodes.keys(), 2):
            if id1 in nodes_to_remove or id2 in nodes_to_remove:
                continue
            
            node1 = self.gedig_episodes[id1]
            node2 = self.gedig_episodes[id2]
            
            similarity = self._episode_similarity(node1.episode, node2.episode)
            
            # 非常に類似している場合は統合
            if similarity > 0.9:
                # IGが高い方を残す
                if node1.ig >= node2.ig:
                    keep_id, remove_id = id1, id2
                else:
                    keep_id, remove_id = id2, id1
                
                # 接続を統合
                keep_node = self.gedig_episodes[keep_id]
                remove_node = self.gedig_episodes[remove_id]
                
                for conn_id in remove_node.connections:
                    if conn_id != keep_id and conn_id in self.gedig_episodes:
                        keep_node.connections.add(conn_id)
                        self.gedig_episodes[conn_id].connections.add(keep_id)
                        self.gedig_episodes[conn_id].connections.discard(remove_id)
                
                nodes_to_remove.add(remove_id)
                merged += 1
        
        # ノードを削除
        for node_id in nodes_to_remove:
            del self.gedig_episodes[node_id]
        
        return merged
    
    def _remove_low_value_nodes(self) -> int:
        """低価値なノードを削除"""
        removed = 0
        nodes_to_remove = []
        
        for node_id, node in self.gedig_episodes.items():
            # ゴールノードは保護
            if node.episode.goal_or_not:
                continue
            
            # 削除条件：
            # 1. C値が低い
            # 2. 接続が少ない
            # 3. 最近アクセスされていない
            if (node.c_value < 0.2 and 
                len(node.connections) < 2 and
                self.step_count - node.last_access > 1000):
                nodes_to_remove.append(node_id)
        
        # 削除実行
        for node_id in nodes_to_remove:
            node = self.gedig_episodes[node_id]
            
            # 接続を切断
            for conn_id in node.connections:
                if conn_id in self.gedig_episodes:
                    self.gedig_episodes[conn_id].connections.discard(node_id)
            
            del self.gedig_episodes[node_id]
            removed += 1
        
        return removed
    
    def _search_episodes_gedig(self, queries: List[Episode7D]) -> List[Tuple[Episode7D, float]]:
        """GeDIG検索"""
        # 通常の検索
        results = super()._search_episodes(queries)
        
        # GeDIGメトリクスで再スコアリング
        gedig_results = []
        
        for episode, base_score in results:
            # 対応するGeDIGノードを探す
            for node_id, gedig_node in self.gedig_episodes.items():
                if gedig_node.episode == episode:
                    # C値を考慮したスコア
                    gedig_score = base_score * (1.0 + gedig_node.c_value)
                    gedig_results.append((episode, gedig_score))
                    
                    # アクセス記録
                    gedig_node.access_count += 1
                    gedig_node.last_access = self.step_count
                    break
        
        # スコアでソート
        gedig_results.sort(key=lambda x: x[1], reverse=True)
        
        # 上位結果間の接続を強化
        for i in range(min(5, len(gedig_results))):
            for j in range(i + 1, min(5, len(gedig_results))):
                ep1, _ = gedig_results[i]
                ep2, _ = gedig_results[j]
                
                # 対応するノードIDを探して接続
                id1 = id2 = None
                for node_id, node in self.gedig_episodes.items():
                    if node.episode == ep1:
                        id1 = node_id
                    if node.episode == ep2:
                        id2 = node_id
                
                if id1 is not None and id2 is not None:
                    self.connect_episodes(id1, id2)
        
        return gedig_results
    
    def _visualize_ged_ig_evolution(self):
        """GED/IGの変化を可視化"""
        if not self.sleep_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # データ抽出
        steps = [h['step'] for h in self.sleep_history]
        ged_before = [h['before']['ged'] for h in self.sleep_history]
        ged_after = [h['after']['ged'] for h in self.sleep_history]
        ig_before = [h['before']['ig'] for h in self.sleep_history]
        ig_after = [h['after']['ig'] for h in self.sleep_history]
        
        # 1. GEDの変化
        ax1.plot(steps, ged_before, 'b.-', label='Before Sleep', markersize=8)
        ax1.plot(steps, ged_after, 'r.-', label='After Sleep', markersize=8)
        ax1.fill_between(steps, ged_before, ged_after, alpha=0.3, color='green')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Average GED')
        ax1.set_title('Graph Edit Distance Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. IGの変化
        ax2.plot(steps, ig_before, 'b.-', label='Before Sleep', markersize=8)
        ax2.plot(steps, ig_after, 'r.-', label='After Sleep', markersize=8)
        ax2.fill_between(steps, ig_before, ig_after, alpha=0.3, color='orange')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Average IG')
        ax2.set_title('Information Gain Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ノード数とエッジ数
        nodes_before = [h['before']['nodes'] for h in self.sleep_history]
        nodes_after = [h['after']['nodes'] for h in self.sleep_history]
        edges_before = [h['before']['edges'] for h in self.sleep_history]
        edges_after = [h['after']['edges'] for h in self.sleep_history]
        
        ax3.bar(range(len(steps)), nodes_before, alpha=0.5, label='Nodes Before', color='blue')
        ax3.bar(range(len(steps)), nodes_after, alpha=0.5, label='Nodes After', color='red')
        ax3.set_xlabel('Sleep Cycle')
        ax3.set_ylabel('Count')
        ax3.set_title('Node Count Changes')
        ax3.legend()
        
        # 4. 削除アクション
        pruned = [h['actions']['pruned_edges'] for h in self.sleep_history]
        merged = [h['actions']['merged_nodes'] for h in self.sleep_history]
        removed = [h['actions']['removed_nodes'] for h in self.sleep_history]
        
        x = range(len(steps))
        width = 0.25
        ax4.bar([i - width for i in x], pruned, width, label='Pruned Edges', color='red')
        ax4.bar(x, merged, width, label='Merged Nodes', color='yellow')
        ax4.bar([i + width for i in x], removed, width, label='Removed Nodes', color='gray')
        ax4.set_xlabel('Sleep Cycle')
        ax4.set_ylabel('Count')
        ax4.set_title('Sleep Actions')
        ax4.legend()
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/ged_ig_evolution_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        
        print(f"GED/IG evolution saved to: {filename}")


def main():
    """メイン実行"""
    print("="*60)
    print("GeDIG Sleep Navigator Test")
    print("GED/IG-based sleep optimization")
    print("="*60)
    
    # シード42でテスト
    random.seed(42)
    np.random.seed(42)
    
    navigator = GeDIGSleepNavigator(maze_size=30)
    
    # カスタム実行（_search_episodesをオーバーライド）
    navigator._search_episodes = navigator._search_episodes_gedig
    
    result = navigator.solve_maze(max_steps=3000)
    
    # GED/IG進化を可視化
    navigator._visualize_ged_ig_evolution()
    
    print("\n" + "="*60)
    print("GeDIG SLEEP ANALYSIS")
    print("="*60)
    
    if result['success']:
        print("✓ Successfully solved with GeDIG sleep!")
    else:
        print("✗ Failed to solve")
    
    print(f"Steps: {result['steps']}")
    print(f"Efficiency: {result['efficiency']:.1f}%")
    print(f"Sleep cycles: {result.get('sleep_cycles', len(navigator.sleep_history))}")
    
    if navigator.sleep_history:
        print("\nSleep effectiveness:")
        initial_ged = navigator.sleep_history[0]['before']['ged']
        final_ged = navigator.sleep_history[-1]['after']['ged']
        print(f"  GED: {initial_ged:.3f} → {final_ged:.3f}")
        
        initial_ig = navigator.sleep_history[0]['before']['ig']
        final_ig = navigator.sleep_history[-1]['after']['ig']
        print(f"  IG: {initial_ig:.3f} → {final_ig:.3f}")


if __name__ == "__main__":
    main()