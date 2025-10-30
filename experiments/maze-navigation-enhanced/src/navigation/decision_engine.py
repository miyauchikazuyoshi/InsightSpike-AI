"""
Decision-making engine for action selection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.episode_manager import Episode
from core.vector_processor import VectorProcessor


class DecisionEngine:
    """行動選択ロジック"""
    
    def __init__(
        self,
        vector_processor: VectorProcessor,
        weights: Optional[np.ndarray] = None,
        temperature: float = 0.1,
        include_walls: bool = True
    ):
        """
        Args:
            vector_processor: ベクトル処理器
            weights: 重みベクトル
            temperature: ソフトマックス温度パラメータ
        """
        self.vector_processor = vector_processor
        self.temperature = temperature
        # 壁エピソード除外は Simple Mode 方針で非推奨: 強制 True + Warning
        if not include_walls:
            try:
                print("[DecisionEngine] WARNING: include_walls=False は非推奨のため強制 True に上書き (A6)")
            except Exception:
                pass
            include_walls = True
        self.include_walls = include_walls

        # デフォルト重み
        self.weights = weights if weights is not None else np.array([
            1.0,  # x座標
            1.0,  # y座標
            0.0,  # dx
            0.0,  # dy
            3.0,  # 壁フラグ（重要）
            2.0,  # 訪問回数（重要）
            0.0,  # success_outcome (デフォルト無効)
            0.0   # goal_flag (デフォルト無効)
        ])
    
    def create_query(
        self,
        position: Tuple[int, int],
        prefer_unexplored: bool = True
    ) -> np.ndarray:
        """
        現在位置からクエリベクトルを生成
        
        Args:
            position: 現在位置 (x, y)
            prefer_unexplored: 未探索を優先するか
        
        Returns:
            クエリベクトル
        """
        return self.vector_processor.create_query_vector(
            position,
            prefer_unexplored
        )
    
    def norm_search(
        self,
        query: np.ndarray,
        episodes: List[Episode],
        weights: Optional[np.ndarray] = None
    ) -> List[Tuple[float, Episode]]:
        """
        ノルム検索でエピソードをランク付け
        
        Args:
            query: クエリベクトル
            episodes: エピソードリスト
            weights: 重みベクトル（Noneの場合はデフォルト）
        
        Returns:
            (距離, エピソード)のタプルのリスト（距離順）
        """
        if weights is None:
            weights = self.weights
        
        # 重み付きクエリ
        query_weighted = self.vector_processor.apply_weights(query, weights)
        
        # 各エピソードとの距離を計算
        distances = []
        for episode in episodes:
            if not self.include_walls and episode.is_wall:
                continue  # 旧挙動: 壁除外
            
            ep_weighted = self.vector_processor.apply_weights(
                episode.vector,
                weights
            )
            
            dist = np.linalg.norm(query_weighted - ep_weighted)
            distances.append((dist, episode))
        
        # 距離でソート
        distances.sort(key=lambda x: x[0])
        
        return distances
    
    def select_action(
        self,
        episodes: Dict[str, Episode],
        temperature: Optional[float] = None,
        query: Optional[np.ndarray] = None,
        prefer_unexplored: bool = True
    ) -> Optional[str]:
        """
        エピソードから行動を選択
        
        Args:
            episodes: 方向をキーとしたエピソードの辞書
            temperature: 温度パラメータ（Noneの場合はデフォルト）
        
        Returns:
            選択された方向（'N', 'S', 'E', 'W'）または None
        """
        if temperature is None:
            temperature = self.temperature
        
        # 現在位置を推定（全エピソードで共通）
        if not episodes:
            return None
        
        first_episode = next(iter(episodes.values()))
        position = first_episode.position

        # クエリ生成（外部供給が無ければ生成）
        if query is None:
            query = self.create_query(position, prefer_unexplored=prefer_unexplored)
        
        # 有効なエピソードのみ抽出
        valid_episodes = [
            (direction, ep) for direction, ep in episodes.items()
            if (self.include_walls or not ep.is_wall)
        ]
        
        if not valid_episodes:
            return None

        # ノルム検索
        episode_list = [ep for _, ep in valid_episodes]
        ranked = self.norm_search(query, episode_list, self.weights)
        
        if not ranked:
            return None
        
        # 確率計算（ソフトマックス）
        distances = np.array([dist for dist, _ in ranked])
        scores = np.exp(-distances / temperature)
        probabilities = scores / np.sum(scores)
        
        # 確率的選択
        best_idx = np.random.choice(len(ranked), p=probabilities)
        selected_episode = ranked[best_idx][1]
        
        # 選択されたエピソードの方向を返す
        for direction, ep in valid_episodes:
            if ep.episode_id == selected_episode.episode_id:
                return direction
        
        return None
    
    def select_greedy_action(
        self,
        episodes: Dict[str, Episode]
    ) -> Optional[str]:
        """
        グリーディーに行動を選択（最も良いものを確定的に選択）
        
        Args:
            episodes: 方向をキーとしたエピソードの辞書
        
        Returns:
            選択された方向または None
        """
        if not episodes:
            return None
        
        first_episode = next(iter(episodes.values()))
        position = first_episode.position
        
        # クエリ生成
        query = self.create_query(position, prefer_unexplored=True)
        
        # 有効なエピソードのみ
        valid_episodes = [
            (direction, ep) for direction, ep in episodes.items()
            if (self.include_walls or not ep.is_wall)
        ]
        
        if not valid_episodes:
            return None
        
        # ノルム検索
        episode_list = [ep for _, ep in valid_episodes]
        ranked = self.norm_search(query, episode_list, self.weights)
        
        if not ranked:
            return None
        
        # 最良のエピソードを選択
        best_episode = ranked[0][1]
        
        # 方向を特定
        for direction, ep in valid_episodes:
            if ep.episode_id == best_episode.episode_id:
                return direction
        
        return None
    
    def analyze_options(
        self,
        episodes: Dict[str, Episode]
    ) -> Dict[str, any]:
        """
        現在の選択肢を分析
        
        Args:
            episodes: 方向をキーとしたエピソードの辞書
        
        Returns:
            分析結果の辞書
        """
        if not episodes:
            return {'error': 'No episodes provided'}
        
        first_episode = next(iter(episodes.values()))
        position = first_episode.position
        
        # クエリ生成
        query = self.create_query(position, prefer_unexplored=True)
        
        # 各方向の分析
        analysis = {
            'position': position,
            'options': {}
        }
        
        # 有効なエピソードの評価
        valid_count = 0
        for direction, episode in episodes.items():
            ep_weighted = self.vector_processor.apply_weights(
                episode.vector,
                self.weights
            )
            query_weighted = self.vector_processor.apply_weights(
                query,
                self.weights
            )
            
            distance = np.linalg.norm(query_weighted - ep_weighted)
            
            option_info = {
                'is_wall': episode.is_wall,
                'visit_count': episode.visit_count,
                'distance': float(distance)
            }
            
            if not episode.is_wall:
                valid_count += 1
                score = np.exp(-distance / self.temperature)
                option_info['score'] = float(score)
            
            analysis['options'][direction] = option_info
        
        analysis['valid_options'] = valid_count
        
        # 確率分布を計算
        if valid_count > 0:
            scores = []
            directions = []
            for direction, info in analysis['options'].items():
                if 'score' in info:
                    scores.append(info['score'])
                    directions.append(direction)
            
            scores = np.array(scores)
            probabilities = scores / np.sum(scores)
            
            for i, direction in enumerate(directions):
                analysis['options'][direction]['probability'] = float(probabilities[i])
        
        return analysis
    
    def update_weights(self, new_weights: np.ndarray) -> None:
        """
        重みベクトルを更新
        
        Args:
            new_weights: 新しい重みベクトル
        """
        self.weights = new_weights
    
    def update_temperature(self, new_temperature: float) -> None:
        """
        温度パラメータを更新
        
        Args:
            new_temperature: 新しい温度
        """
        self.temperature = new_temperature

    def set_include_walls(self, flag: bool) -> None:
        """壁エピソードを候補に含めるか切替"""
        self.include_walls = flag