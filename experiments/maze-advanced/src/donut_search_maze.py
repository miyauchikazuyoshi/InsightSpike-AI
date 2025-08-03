#!/usr/bin/env python3
"""
Donut Search for Maze Exploration
=================================

ドーナツ検索（球面検索）を活用した迷路探索。
内側の既知領域を避け、外側の未知領域を探索。
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DonutSearchResult:
    """ドーナツ検索の結果"""
    inner_radius: float
    outer_radius: float
    candidates: List[Tuple[str, float]]  # (node_id, distance)
    inner_nodes: List[str]  # 内側（既知）のノード
    outer_nodes: List[str]  # 外側（遠すぎる）のノード


class DonutSearchMaze:
    """
    ドーナツ検索を使った迷路探索
    
    基本アイデア：
    - 現在位置から「近すぎず遠すぎない」領域を探索
    - 既に十分探索した領域（内側）は避ける
    - 遠すぎる領域（外側）も今は避ける
    - ドーナツ型の領域が「次に探索すべき場所」
    """
    
    def __init__(self, dimension: int = 5):
        self.dimension = dimension
        self.episode_vectors = {}  # node_id -> vector
        self.visit_counts = {}     # position -> count
        
    def add_episode(self, node_id: str, vector: np.ndarray, position: Tuple[int, int]):
        """エピソードを追加"""
        self.episode_vectors[node_id] = vector
        
        # 訪問回数を更新
        if position not in self.visit_counts:
            self.visit_counts[position] = 0
        self.visit_counts[position] += 1
        
    def donut_search(self, 
                     query_vector: np.ndarray,
                     inner_radius: float = 0.2,
                     outer_radius: float = 0.8) -> DonutSearchResult:
        """
        ドーナツ検索を実行
        
        Args:
            query_vector: 現在位置のベクトル
            inner_radius: 内側半径（これより近いノードは既知として除外）
            outer_radius: 外側半径（これより遠いノードは遠すぎるとして除外）
            
        Returns:
            ドーナツ内のノードリスト
        """
        candidates = []
        inner_nodes = []
        outer_nodes = []
        
        for node_id, vector in self.episode_vectors.items():
            # コサイン類似度ではなく、ユークリッド距離を使用
            distance = np.linalg.norm(query_vector - vector)
            
            if distance < inner_radius:
                inner_nodes.append(node_id)
            elif distance > outer_radius:
                outer_nodes.append(node_id)
            else:
                # ドーナツ内のノード
                candidates.append((node_id, distance))
                
        # 距離でソート（近い順）
        candidates.sort(key=lambda x: x[1])
        
        return DonutSearchResult(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            candidates=candidates,
            inner_nodes=inner_nodes,
            outer_nodes=outer_nodes
        )
        
    def adaptive_donut_search(self, 
                            query_vector: np.ndarray,
                            current_position: Tuple[int, int]) -> DonutSearchResult:
        """
        適応的ドーナツ検索
        
        訪問回数に基づいて半径を動的に調整
        """
        visit_count = self.visit_counts.get(current_position, 0)
        
        # 訪問回数が多いほど、より遠くを探索
        inner_radius = 0.1 + (visit_count * 0.1)  # 0.1 → 0.2 → 0.3 ...
        outer_radius = 0.5 + (visit_count * 0.2)  # 0.5 → 0.7 → 0.9 ...
        
        # 上限を設定
        inner_radius = min(inner_radius, 0.5)
        outer_radius = min(outer_radius, 1.5)
        
        return self.donut_search(query_vector, inner_radius, outer_radius)
        
    def get_exploration_direction(self, 
                                query_vector: np.ndarray,
                                current_position: Tuple[int, int]) -> Optional[int]:
        """
        ドーナツ検索に基づいて探索方向を決定
        
        Returns:
            0-3: 推奨される方向（上右下左）
            None: 推奨なし
        """
        result = self.adaptive_donut_search(query_vector, current_position)
        
        if not result.candidates:
            logger.info("No candidates in donut region")
            return None
            
        # ドーナツ内のベクトルの重心を計算
        candidate_vectors = []
        for node_id, _ in result.candidates[:5]:  # 上位5個
            candidate_vectors.append(self.episode_vectors[node_id])
            
        if candidate_vectors:
            # 重心ベクトル
            centroid = np.mean(candidate_vectors, axis=0)
            
            # 現在位置から重心への方向
            direction_vector = centroid - query_vector
            
            # X, Y成分から方向を決定
            dx = direction_vector[0]  # X成分
            dy = direction_vector[1]  # Y成分
            
            # 最も強い成分の方向を選択
            if abs(dx) > abs(dy):
                return 1 if dx > 0 else 3  # 右 or 左
            else:
                return 2 if dy > 0 else 0  # 下 or 上
                
        return None
        
    def visualize_donut(self, query_vector: np.ndarray) -> Dict:
        """ドーナツ検索の状態を可視化用に返す"""
        result = self.donut_search(query_vector)
        
        return {
            'total_episodes': len(self.episode_vectors),
            'inner_count': len(result.inner_nodes),
            'donut_count': len(result.candidates),
            'outer_count': len(result.outer_nodes),
            'inner_radius': result.inner_radius,
            'outer_radius': result.outer_radius
        }


class DonutGuidedMazeAgent:
    """ドーナツ検索を使った迷路エージェント"""
    
    def __init__(self):
        self.donut_search = DonutSearchMaze(dimension=5)
        self.step_count = 0
        
    def decide_action(self, 
                     state_vector: np.ndarray,
                     position: Tuple[int, int],
                     possible_actions: List[int]) -> int:
        """
        ドーナツ検索を使って行動を決定
        """
        # 現在のエピソードを記録
        node_id = f"step_{self.step_count}"
        self.donut_search.add_episode(node_id, state_vector, position)
        self.step_count += 1
        
        # ドーナツ検索で方向を取得
        suggested_direction = self.donut_search.get_exploration_direction(
            state_vector, position
        )
        
        # 可視化用情報
        donut_info = self.donut_search.visualize_donut(state_vector)
        logger.info(f"Donut search: {donut_info}")
        
        # 推奨方向が可能なら使用
        if suggested_direction is not None and suggested_direction in possible_actions:
            logger.info(f"Using donut search suggestion: {suggested_direction}")
            return suggested_direction
            
        # そうでなければランダムに選択
        if possible_actions:
            return np.random.choice(possible_actions)
        else:
            return 0


def test_donut_search():
    """ドーナツ検索のテスト"""
    donut = DonutSearchMaze(dimension=5)
    
    # いくつかのエピソードを追加
    positions = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    for i, pos in enumerate(positions):
        vector = np.array([
            pos[0] / 2.0,  # 正規化X
            pos[1] / 2.0,  # 正規化Y
            0.5,           # 方向
            0.0,           # 結果
            0.1            # 訪問回数
        ])
        donut.add_episode(f"ep_{i}", vector, pos)
        
    # 現在位置(1, 1)でドーナツ検索
    current_vector = np.array([0.5, 0.5, 0.5, 0.0, 0.2])
    result = donut.donut_search(current_vector, inner_radius=0.3, outer_radius=0.8)
    
    print(f"Donut search result:")
    print(f"  Inner nodes ({len(result.inner_nodes)}): {result.inner_nodes}")
    print(f"  Candidates ({len(result.candidates)}): {result.candidates}")
    print(f"  Outer nodes ({len(result.outer_nodes)}): {result.outer_nodes}")
    
    # 探索方向を取得
    direction = donut.get_exploration_direction(current_vector, (1, 1))
    print(f"  Suggested direction: {direction}")


if __name__ == "__main__":
    test_donut_search()