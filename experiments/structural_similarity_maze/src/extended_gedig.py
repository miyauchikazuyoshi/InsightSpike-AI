"""
構造類似度を含む拡張geDIG

IG項に構造類似度を追加:
  ΔIG = ΔH_norm + γ・ΔSP_rel + β・ΔSS
"""

from typing import Tuple, Dict, Optional, List
import numpy as np
from dataclasses import dataclass
from pattern_detector import PatternDetector, LocalPattern, ExtendedVectorProcessor


@dataclass
class GedigMetrics:
    """geDIG計算結果"""
    delta_epc: float          # 編集経路コスト
    delta_h_norm: float       # 正規化エントロピー変化
    delta_sp_rel: float       # 相対最短路改善
    delta_ss: float           # 構造類似度変化（新規）
    delta_ig: float           # 情報利得（合計）
    f_value: float            # 統一ゲージ F
    g0: float                 # 0-hop評価値
    g_min: float              # multi-hop最小値


class ExtendedGedig:
    """
    構造類似度を含む拡張geDIG

    F = ΔEPC - λ・ΔIG
    ΔIG = ΔH_norm + γ・ΔSP_rel + β・ΔSS

    SSによる追加:
    - 局所パターン類似度: 現在位置とゴール周辺の構造パターンの類似度
    - 大域構造類似度: 探索済みグラフとゴールパスの構造類似度
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        gamma: float = 0.5,
        beta: float = 0.3,      # 構造類似度の重み（新規）
        ag_threshold: float = 0.5,
        dg_threshold: float = 0.2,
        use_local_ss: bool = True,
        use_global_ss: bool = False,
    ):
        """
        Args:
            lambda_: 情報利得の重み
            gamma: 最短路改善の重み
            beta: 構造類似度の重み（新規）
            ag_threshold: Attention Gate閾値
            dg_threshold: Decision Gate閾値
            use_local_ss: 局所パターン類似度を使用
            use_global_ss: 大域構造類似度を使用
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.ag_threshold = ag_threshold
        self.dg_threshold = dg_threshold
        self.use_local_ss = use_local_ss
        self.use_global_ss = use_global_ss

        self.pattern_detector = PatternDetector()

        # ゴール周辺の「理想的パターン」（通常はT字路や十字路が多い）
        self.goal_pattern_distribution = np.array([
            0.1,   # DEAD_END (少ない)
            0.2,   # CORRIDOR
            0.2,   # L_TURN
            0.3,   # T_JUNCTION (分岐が重要)
            0.2,   # CROSS
        ], dtype=np.float32)

    def compute_metrics(
        self,
        maze: np.ndarray,
        current_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        visited_positions: set,
        exploration_graph: Optional[Dict] = None,
    ) -> GedigMetrics:
        """
        移動に対するgeDIGメトリクスを計算

        Args:
            maze: 迷路配列
            current_pos: 現在位置
            next_pos: 次の位置
            goal_pos: ゴール位置
            visited_positions: 訪問済み位置集合
            exploration_graph: 探索グラフ（大域SS用）

        Returns:
            GedigMetrics: 計算結果
        """
        # 1. ΔEPC: 編集経路コスト（移動コスト）
        delta_epc = self._compute_delta_epc(current_pos, next_pos, visited_positions)

        # 2. ΔH_norm: エントロピー変化
        delta_h_norm = self._compute_delta_h_norm(
            maze, current_pos, next_pos, visited_positions
        )

        # 3. ΔSP_rel: 相対最短路改善
        delta_sp_rel = self._compute_delta_sp_rel(
            current_pos, next_pos, goal_pos
        )

        # 4. ΔSS: 構造類似度変化（新規）
        delta_ss = self._compute_delta_ss(
            maze, current_pos, next_pos, goal_pos, exploration_graph
        )

        # 情報利得の計算
        delta_ig = delta_h_norm + self.gamma * delta_sp_rel + self.beta * delta_ss

        # 統一ゲージ F
        f_value = delta_epc - self.lambda_ * delta_ig

        # ゲート評価値
        g0 = f_value  # 0-hop
        g_min = f_value  # 単純化（multi-hopは探索履歴から計算）

        return GedigMetrics(
            delta_epc=delta_epc,
            delta_h_norm=delta_h_norm,
            delta_sp_rel=delta_sp_rel,
            delta_ss=delta_ss,
            delta_ig=delta_ig,
            f_value=f_value,
            g0=g0,
            g_min=g_min,
        )

    def _compute_delta_epc(
        self,
        current_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        visited_positions: set,
    ) -> float:
        """編集経路コスト"""
        # 基本移動コスト
        base_cost = 1.0

        # 再訪問ペナルティ
        if next_pos in visited_positions:
            base_cost += 0.5

        # 正規化（最大コストで割る）
        return base_cost / 2.0

    def _compute_delta_h_norm(
        self,
        maze: np.ndarray,
        current_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        visited_positions: set,
    ) -> float:
        """正規化エントロピー変化"""
        # 現在位置のエントロピー（未探索方向の数に基づく）
        h_current = self._local_entropy(maze, current_pos, visited_positions)

        # 次の位置のエントロピー
        new_visited = visited_positions | {next_pos}
        h_next = self._local_entropy(maze, next_pos, new_visited)

        # エントロピー減少が正の利得
        delta_h = h_current - h_next

        # 正規化 [-1, 1]
        return np.clip(delta_h, -1.0, 1.0)

    def _local_entropy(
        self,
        maze: np.ndarray,
        position: Tuple[int, int],
        visited: set,
    ) -> float:
        """局所エントロピー（未知方向の不確実性）"""
        x, y = position
        height, width = maze.shape
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        unknown_count = 0
        passable_count = 0

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze[ny, nx] == 0:  # 通路
                    passable_count += 1
                    if (nx, ny) not in visited:
                        unknown_count += 1

        if passable_count == 0:
            return 0.0

        # 未知の割合 = エントロピーの代理
        return unknown_count / passable_count

    def _compute_delta_sp_rel(
        self,
        current_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> float:
        """相対最短路改善"""
        # マンハッタン距離
        dist_current = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
        dist_next = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])

        # 距離減少が正の利得
        improvement = dist_current - dist_next

        # 正規化
        max_dist = max(dist_current, 1)
        return improvement / max_dist

    def _compute_delta_ss(
        self,
        maze: np.ndarray,
        current_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        exploration_graph: Optional[Dict],
    ) -> float:
        """構造類似度変化（新規）"""
        delta_ss = 0.0

        if self.use_local_ss:
            # 局所パターン類似度
            ss_current = self._local_pattern_similarity_to_goal(maze, current_pos, goal_pos)
            ss_next = self._local_pattern_similarity_to_goal(maze, next_pos, goal_pos)
            delta_ss += (ss_next - ss_current)

        if self.use_global_ss and exploration_graph is not None:
            # 大域構造類似度（将来実装）
            pass

        return delta_ss

    def _local_pattern_similarity_to_goal(
        self,
        maze: np.ndarray,
        position: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> float:
        """
        位置のパターンとゴール周辺パターンの類似度

        仮説: ゴールに近い構造は「分岐が多い」傾向がある
        （行き止まりからは遠ざかりたい）
        """
        # 現在位置のパターン
        current_pattern = self.pattern_detector.detect_pattern(maze, position)

        if current_pattern == LocalPattern.UNKNOWN:
            return 0.0

        # ゴール位置のパターン
        goal_pattern = self.pattern_detector.detect_pattern(maze, goal_pos)

        # パターン類似度
        pattern_sim = self.pattern_detector.pattern_similarity(
            current_pattern, goal_pattern
        )

        # 近傍パターン分布との類似度
        neighborhood_vec = self.pattern_detector.get_neighborhood_pattern_vector(
            maze, position, radius=2
        )

        # ゴール周辺の理想的分布との類似度
        goal_sim = np.dot(neighborhood_vec, self.goal_pattern_distribution)
        goal_sim /= (np.linalg.norm(neighborhood_vec) + 1e-8)
        goal_sim /= (np.linalg.norm(self.goal_pattern_distribution) + 1e-8)

        # 距離による重み付け
        dist_to_goal = abs(position[0] - goal_pos[0]) + abs(position[1] - goal_pos[1])
        max_dist = maze.shape[0] + maze.shape[1]
        distance_factor = 1.0 - (dist_to_goal / max_dist)

        # 総合類似度
        return 0.5 * pattern_sim + 0.3 * goal_sim + 0.2 * distance_factor

    def should_explore(self, metrics: GedigMetrics) -> bool:
        """AG: 探索すべきか"""
        return metrics.g0 > self.ag_threshold

    def should_commit(self, metrics: GedigMetrics) -> bool:
        """DG: 決定すべきか"""
        return metrics.g_min < self.dg_threshold

    def evaluate_directions(
        self,
        maze: np.ndarray,
        current_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        visited_positions: set,
    ) -> Dict[str, GedigMetrics]:
        """
        4方向の評価

        Returns:
            Dict[方向, GedigMetrics]
        """
        directions = {
            'N': (0, -1),
            'E': (1, 0),
            'S': (0, 1),
            'W': (-1, 0),
        }

        results = {}
        height, width = maze.shape

        for direction, (dx, dy) in directions.items():
            nx, ny = current_pos[0] + dx, current_pos[1] + dy

            # 範囲外または壁
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if maze[ny, nx] == 1:
                continue

            next_pos = (nx, ny)
            metrics = self.compute_metrics(
                maze=maze,
                current_pos=current_pos,
                next_pos=next_pos,
                goal_pos=goal_pos,
                visited_positions=visited_positions,
            )
            results[direction] = metrics

        return results

    def select_best_direction(
        self,
        evaluations: Dict[str, GedigMetrics]
    ) -> Optional[str]:
        """
        最良の方向を選択（F値が最小の方向）
        """
        if not evaluations:
            return None

        best_direction = min(
            evaluations.keys(),
            key=lambda d: evaluations[d].f_value
        )
        return best_direction


# テスト
if __name__ == "__main__":
    # テスト迷路
    test_maze = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    start = (1, 1)
    goal = (5, 5)

    # 拡張geDIG（SSあり）
    gedig_with_ss = ExtendedGedig(
        lambda_=1.0,
        gamma=0.5,
        beta=0.3,  # 構造類似度の重み
        use_local_ss=True,
    )

    # 拡張geDIG（SSなし = ベースライン）
    gedig_baseline = ExtendedGedig(
        lambda_=1.0,
        gamma=0.5,
        beta=0.0,  # 構造類似度なし
        use_local_ss=False,
    )

    print("=== 拡張geDIG テスト ===")
    print(f"Start: {start}, Goal: {goal}")

    visited = {start}

    # 各方向の評価
    print("\n--- With Structural Similarity ---")
    evals_ss = gedig_with_ss.evaluate_directions(
        test_maze, start, goal, visited
    )
    for direction, metrics in evals_ss.items():
        print(f"{direction}: F={metrics.f_value:.3f}, ΔSS={metrics.delta_ss:.3f}, ΔIG={metrics.delta_ig:.3f}")

    best_ss = gedig_with_ss.select_best_direction(evals_ss)
    print(f"Best direction (with SS): {best_ss}")

    print("\n--- Baseline (no SS) ---")
    evals_base = gedig_baseline.evaluate_directions(
        test_maze, start, goal, visited
    )
    for direction, metrics in evals_base.items():
        print(f"{direction}: F={metrics.f_value:.3f}, ΔSS={metrics.delta_ss:.3f}, ΔIG={metrics.delta_ig:.3f}")

    best_base = gedig_baseline.select_best_direction(evals_base)
    print(f"Best direction (baseline): {best_base}")
