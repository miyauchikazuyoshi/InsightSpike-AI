"""
迷路の局所パターン検出器

迷路の各セルにおける局所構造パターン（5種類）を検出し、
構造類似度計算の基盤を提供する。
"""

from enum import IntEnum
from typing import Tuple, List, Dict, Optional
import numpy as np


class LocalPattern(IntEnum):
    """局所構造パターン"""
    DEAD_END = 0    # 行き止まり（次数1）
    CORRIDOR = 1    # 直線通路（次数2、直線）
    L_TURN = 2      # L字（次数2、90度曲がり）
    T_JUNCTION = 3  # T字路（次数3）
    CROSS = 4       # 十字路（次数4）
    UNKNOWN = -1    # 不明（壁など）


# 4方向の定義
DIRECTIONS = {
    'N': (0, -1),
    'E': (1, 0),
    'S': (0, 1),
    'W': (-1, 0),
}


class PatternDetector:
    """
    迷路の局所パターンを検出するクラス
    """

    # パターン間の構造類似度行列
    # 次数の近さと接続パターンの類似性を反映
    SIMILARITY_MATRIX = np.array([
        #  DE    CO    L     T     CR
        [1.00, 0.30, 0.30, 0.20, 0.10],  # DEAD_END
        [0.30, 1.00, 0.70, 0.50, 0.30],  # CORRIDOR
        [0.30, 0.70, 1.00, 0.60, 0.40],  # L_TURN
        [0.20, 0.50, 0.60, 1.00, 0.80],  # T_JUNCTION
        [0.10, 0.30, 0.40, 0.80, 1.00],  # CROSS
    ], dtype=np.float32)

    def __init__(self):
        pass

    def detect_pattern(
        self,
        maze: np.ndarray,
        position: Tuple[int, int]
    ) -> LocalPattern:
        """
        指定位置の局所パターンを検出

        Args:
            maze: 迷路配列（0=通路, 1=壁）
            position: (x, y) 座標

        Returns:
            LocalPattern: 検出されたパターン
        """
        x, y = position
        height, width = maze.shape

        # 壁の場合
        if maze[y, x] == 1:
            return LocalPattern.UNKNOWN

        # 4方向の開口部を調べる
        open_directions = []
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze[ny, nx] == 0:  # 通路
                    open_directions.append(direction)

        degree = len(open_directions)

        if degree == 0:
            # 孤立（通常はありえない）
            return LocalPattern.DEAD_END
        elif degree == 1:
            return LocalPattern.DEAD_END
        elif degree == 2:
            # 直線かL字かを判定
            if self._is_straight(open_directions):
                return LocalPattern.CORRIDOR
            else:
                return LocalPattern.L_TURN
        elif degree == 3:
            return LocalPattern.T_JUNCTION
        else:  # degree >= 4
            return LocalPattern.CROSS

    def _is_straight(self, directions: List[str]) -> bool:
        """2方向が直線（対向）かどうかを判定"""
        if len(directions) != 2:
            return False

        opposite_pairs = [('N', 'S'), ('E', 'W')]
        dir_set = set(directions)

        for pair in opposite_pairs:
            if dir_set == set(pair):
                return True
        return False

    def get_pattern_embedding(
        self,
        pattern: LocalPattern,
        embedding_dim: int = 4
    ) -> np.ndarray:
        """
        パターンの埋め込みベクトルを取得

        Args:
            pattern: 局所パターン
            embedding_dim: 埋め込み次元

        Returns:
            埋め込みベクトル
        """
        if pattern == LocalPattern.UNKNOWN:
            return np.zeros(embedding_dim, dtype=np.float32)

        # 特徴ベクトル: [次数正規化, 直線性, 分岐度, 開放度]
        features = {
            LocalPattern.DEAD_END: [0.25, 0.0, 0.0, 0.25],
            LocalPattern.CORRIDOR: [0.50, 1.0, 0.0, 0.50],
            LocalPattern.L_TURN: [0.50, 0.0, 0.5, 0.50],
            LocalPattern.T_JUNCTION: [0.75, 0.33, 0.67, 0.75],
            LocalPattern.CROSS: [1.00, 0.50, 1.0, 1.00],
        }

        return np.array(features[pattern], dtype=np.float32)

    def pattern_similarity(
        self,
        pattern_a: LocalPattern,
        pattern_b: LocalPattern
    ) -> float:
        """
        2つのパターン間の構造類似度

        Args:
            pattern_a: パターンA
            pattern_b: パターンB

        Returns:
            類似度 [0, 1]
        """
        if pattern_a == LocalPattern.UNKNOWN or pattern_b == LocalPattern.UNKNOWN:
            return 0.0

        return float(self.SIMILARITY_MATRIX[pattern_a, pattern_b])

    def get_neighborhood_pattern_vector(
        self,
        maze: np.ndarray,
        position: Tuple[int, int],
        radius: int = 1
    ) -> np.ndarray:
        """
        近傍のパターン分布ベクトルを取得

        Args:
            maze: 迷路配列
            position: 中心位置
            radius: 近傍半径

        Returns:
            パターン分布ベクトル [5次元: 各パターンの割合]
        """
        x, y = position
        height, width = maze.shape

        pattern_counts = np.zeros(5, dtype=np.float32)
        total = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    pattern = self.detect_pattern(maze, (nx, ny))
                    if pattern != LocalPattern.UNKNOWN:
                        pattern_counts[pattern] += 1
                        total += 1

        if total > 0:
            pattern_counts /= total

        return pattern_counts

    def compute_local_structural_similarity(
        self,
        maze: np.ndarray,
        pos_a: Tuple[int, int],
        pos_b: Tuple[int, int],
        radius: int = 1
    ) -> float:
        """
        2つの位置の局所構造類似度を計算

        Args:
            maze: 迷路配列
            pos_a: 位置A
            pos_b: 位置B
            radius: 近傍半径

        Returns:
            構造類似度 [0, 1]
        """
        vec_a = self.get_neighborhood_pattern_vector(maze, pos_a, radius)
        vec_b = self.get_neighborhood_pattern_vector(maze, pos_b, radius)

        # コサイン類似度
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class ExtendedVectorProcessor:
    """
    構造パターン情報を含む拡張ベクトルを生成
    """

    def __init__(
        self,
        width: int,
        height: int,
        include_pattern: bool = True,
        include_neighborhood: bool = False
    ):
        self.width = width
        self.height = height
        self.include_pattern = include_pattern
        self.include_neighborhood = include_neighborhood
        self.pattern_detector = PatternDetector()

        # 方向マッピング
        self.direction_map = {
            'N': (0, -1), 'S': (0, 1),
            'E': (1, 0), 'W': (-1, 0),
        }

    def get_vector_dim(self) -> int:
        """ベクトル次元を取得"""
        dim = 8  # 基本8次元
        if self.include_pattern:
            dim += 2  # パターンID + 次数
        if self.include_neighborhood:
            dim += 5  # 近傍パターン分布
        return dim

    def create_extended_vector(
        self,
        maze: np.ndarray,
        position: Tuple[int, int],
        direction: str,
        is_wall: bool,
        visit_count: int = 0,
        success_outcome: float = 0.0,
        goal_flag: float = 0.0
    ) -> np.ndarray:
        """
        拡張ベクトルを生成

        Args:
            maze: 迷路配列
            position: 現在位置
            direction: 方向
            is_wall: 壁かどうか
            visit_count: 訪問回数
            success_outcome: 成功/失敗指標
            goal_flag: ゴールフラグ

        Returns:
            拡張ベクトル
        """
        dx, dy = self.direction_map.get(direction, (0, 0))

        # 基本8次元
        base_vector = [
            position[0] / self.width,      # [0] x_norm
            position[1] / self.height,     # [1] y_norm
            dx,                             # [2] dx
            dy,                             # [3] dy
            -1.0 if is_wall else 1.0,      # [4] wall_flag
            np.log1p(visit_count),          # [5] log_visits
            float(success_outcome),         # [6] success_outcome
            float(goal_flag),               # [7] goal_flag
        ]

        # パターン情報追加
        if self.include_pattern:
            pattern = self.pattern_detector.detect_pattern(maze, position)
            pattern_id = pattern if pattern != LocalPattern.UNKNOWN else -1
            degree = self._get_degree(maze, position)
            base_vector.extend([
                pattern_id / 4.0,           # [8] pattern_id_norm
                degree / 4.0,               # [9] degree_norm
            ])

        # 近傍パターン分布追加
        if self.include_neighborhood:
            neighborhood = self.pattern_detector.get_neighborhood_pattern_vector(
                maze, position, radius=1
            )
            base_vector.extend(neighborhood.tolist())  # [10-14] pattern distribution

        return np.array(base_vector, dtype=np.float32)

    def _get_degree(
        self,
        maze: np.ndarray,
        position: Tuple[int, int]
    ) -> int:
        """位置の次数（開口部の数）を取得"""
        x, y = position
        height, width = maze.shape
        degree = 0

        for dx, dy in self.direction_map.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze[ny, nx] == 0:
                    degree += 1

        return degree


# テスト用
if __name__ == "__main__":
    # 小さなテスト迷路
    # 0=通路, 1=壁
    test_maze = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    detector = PatternDetector()

    print("=== パターン検出テスト ===")
    for y in range(test_maze.shape[0]):
        for x in range(test_maze.shape[1]):
            if test_maze[y, x] == 0:
                pattern = detector.detect_pattern(test_maze, (x, y))
                print(f"({x}, {y}): {pattern.name}")

    print("\n=== 構造類似度テスト ===")
    print(f"DEAD_END vs CORRIDOR: {detector.pattern_similarity(LocalPattern.DEAD_END, LocalPattern.CORRIDOR):.2f}")
    print(f"T_JUNCTION vs CROSS: {detector.pattern_similarity(LocalPattern.T_JUNCTION, LocalPattern.CROSS):.2f}")
    print(f"CORRIDOR vs L_TURN: {detector.pattern_similarity(LocalPattern.CORRIDOR, LocalPattern.L_TURN):.2f}")

    print("\n=== 拡張ベクトルテスト ===")
    processor = ExtendedVectorProcessor(
        width=7, height=7,
        include_pattern=True,
        include_neighborhood=True
    )
    print(f"ベクトル次元: {processor.get_vector_dim()}")

    vec = processor.create_extended_vector(
        maze=test_maze,
        position=(3, 3),
        direction='N',
        is_wall=False,
        visit_count=5
    )
    print(f"拡張ベクトル shape: {vec.shape}")
    print(f"拡張ベクトル: {vec}")
