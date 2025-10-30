"""
Vector processing and calculation module
"""

import numpy as np
from typing import Tuple, Optional


class VectorProcessor:
    """8次元ベクトルの生成と処理"""
    
    def __init__(self, maze_width: int, maze_height: int):
        """
        Args:
            maze_width: 迷路の幅
            maze_height: 迷路の高さ
        """
        self.width = maze_width
        self.height = maze_height
        
        # 方向マッピング
        self.direction_map = {
            'N': (0, -1),
            'S': (0, 1),
            'E': (1, 0),
            'W': (-1, 0)
        }
    
    def create_vector(
        self,
        position: Tuple[int, int],
        direction: str,
        is_wall: bool,
        visit_count: int = 0,
        *,
        success_outcome: float = 0.0,
        goal_flag: float = 0.0
    ) -> np.ndarray:
        """
        8次元ベクトルを生成
        
        Args:
            position: 現在位置 (x, y)
            direction: 方向 ('N', 'S', 'E', 'W')
            is_wall: 壁かどうか
            visit_count: 訪問回数
        
        Returns:
            8次元ベクトル [x_norm, y_norm, dx, dy, wall_flag, log_visits, success_outcome, goal_flag]
        """
        # 方向ベクトル取得
        dx, dy = self.direction_map.get(direction, (0, 0))
        
        # 8次元ベクトル構築
        vector = np.array([
            position[0] / self.width,    # 正規化されたx座標
            position[1] / self.height,   # 正規化されたy座標
            dx,                           # x方向成分
            dy,                           # y方向成分
            -1.0 if is_wall else 1.0,    # 壁フラグ
            np.log1p(visit_count),        # 訪問回数（対数スケール）
            float(success_outcome),       # 成功/失敗/未実行 指標 (-1,0,1 など)
            float(goal_flag)              # ゴール到達/接続フラグ (1 or 0 将来は近接値も)
        ], dtype=np.float32)
        
        return vector
    
    def create_query_vector(
        self,
        position: Tuple[int, int],
        prefer_unexplored: bool = True
    ) -> np.ndarray:
        """
        クエリベクトルを生成
        
        Args:
            position: 現在位置 (x, y)
            prefer_unexplored: 未探索を優先するか
        
        Returns:
            クエリ用8次元ベクトル
        """
        vector = np.array([
            position[0] / self.width,
            position[1] / self.height,
            0.0,  # 方向は中立
            0.0,  # 方向は中立
            1.0,  # 通路を探す
            0.0 if prefer_unexplored else 1.0,  # 訪問回数の好み
            0.0,  # success_outcome 中立
            0.0   # goal_flag (クエリ側でゴール方向バイアスを付けたい場合は後で 1.0 に設定)
        ], dtype=np.float32)
        
        return vector
    
    def apply_weights(
        self,
        vector: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        重みを適用
        
        Args:
            vector: 入力ベクトル
            weights: 重みベクトル（Noneの場合はデフォルト）
        
        Returns:
            重み適用後のベクトル
        """
        if weights is None:
            # デフォルト重み
            weights = np.array([
                1.0,  # x座標
                1.0,  # y座標
                0.0,  # dx（使用しない）
                0.0,  # dy（使用しない）
                3.0,  # 壁フラグ（重要）
                2.0,  # 訪問回数（重要）
                0.0,  # success_outcome (初期は探索バイアス無し)
                0.0   # goal_flag (初期は探索バイアス無し)
            ])
        
        return vector * weights
    
    def calculate_distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        2つのベクトル間の重み付きL2距離を計算
        
        Args:
            vec1: ベクトル1
            vec2: ベクトル2
            weights: 重みベクトル
        
        Returns:
            L2距離
        """
        if weights is not None:
            vec1_weighted = self.apply_weights(vec1, weights)
            vec2_weighted = self.apply_weights(vec2, weights)
        else:
            vec1_weighted = vec1
            vec2_weighted = vec2
        
        return float(np.linalg.norm(vec1_weighted - vec2_weighted))
    
    def update_visit_count(
        self,
        vector: np.ndarray,
        new_visit_count: int
    ) -> np.ndarray:
        """
        ベクトルの訪問回数を更新
        
        Args:
            vector: 元のベクトル
            new_visit_count: 新しい訪問回数
        
        Returns:
            更新されたベクトル（コピー）
        """
        updated = vector.copy()
        updated[5] = np.log1p(new_visit_count)
        return updated

    def update_success_outcome(self, vector: np.ndarray, outcome: float) -> np.ndarray:
        """成功/失敗次元を更新 (-1,0,1 など)"""
        updated = vector.copy()
        updated[6] = float(outcome)
        return updated

    def update_goal_flag(self, vector: np.ndarray, flag: float) -> np.ndarray:
        """ゴールフラグ/近接値次元を更新"""
        updated = vector.copy()
        updated[7] = float(flag)
        return updated