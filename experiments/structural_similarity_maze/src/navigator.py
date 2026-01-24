"""
構造類似度対応ナビゲータ

比較実験用:
- Baseline: 構造類似度なし
- WithSS: 構造類似度あり
"""

from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, field
import numpy as np
from extended_gedig import ExtendedGedig, GedigMetrics


@dataclass
class NavigationResult:
    """ナビゲーション結果"""
    success: bool
    steps: int
    path: List[Tuple[int, int]]
    visited_count: int
    dead_end_encounters: int
    backtrack_count: int
    metrics_history: List[Dict] = field(default_factory=list)


class MazeNavigator:
    """
    迷路ナビゲータ
    """

    def __init__(
        self,
        use_structural_similarity: bool = False,
        beta: float = 0.3,
        max_steps: int = 1000,
        lambda_: float = 1.0,
        gamma: float = 0.5,
    ):
        """
        Args:
            use_structural_similarity: 構造類似度を使用するか
            beta: 構造類似度の重み
            max_steps: 最大ステップ数
            lambda_: IG項の重み
            gamma: SP項の重み
        """
        self.use_ss = use_structural_similarity
        self.max_steps = max_steps

        self.gedig = ExtendedGedig(
            lambda_=lambda_,
            gamma=gamma,
            beta=beta if use_structural_similarity else 0.0,
            use_local_ss=use_structural_similarity,
        )

    def navigate(
        self,
        maze: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> NavigationResult:
        """
        迷路を探索

        Args:
            maze: 迷路配列 (0=通路, 1=壁)
            start: スタート位置
            goal: ゴール位置

        Returns:
            NavigationResult: 結果
        """
        current = start
        path = [current]
        visited: Set[Tuple[int, int]] = {current}
        steps = 0
        dead_end_encounters = 0
        backtrack_count = 0
        metrics_history = []

        while steps < self.max_steps:
            # ゴール到達チェック
            if current == goal:
                return NavigationResult(
                    success=True,
                    steps=steps,
                    path=path,
                    visited_count=len(visited),
                    dead_end_encounters=dead_end_encounters,
                    backtrack_count=backtrack_count,
                    metrics_history=metrics_history,
                )

            # 各方向を評価
            evaluations = self.gedig.evaluate_directions(
                maze=maze,
                current_pos=current,
                goal_pos=goal,
                visited_positions=visited,
            )

            if not evaluations:
                # 行き止まり（全方向が壁）
                dead_end_encounters += 1
                # バックトラック
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    backtrack_count += 1
                    steps += 1
                    continue
                else:
                    # スタートに戻っても動けない
                    break

            # 未訪問を優先
            unvisited_evals = {
                d: m for d, m in evaluations.items()
                if self._get_next_pos(current, d) not in visited
            }

            if unvisited_evals:
                # 未訪問の中で最良を選択
                best_dir = min(
                    unvisited_evals.keys(),
                    key=lambda d: unvisited_evals[d].f_value
                )
                best_metrics = unvisited_evals[best_dir]
            else:
                # 全て訪問済み → バックトラック
                dead_end_encounters += 1
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    backtrack_count += 1
                    steps += 1
                    continue
                else:
                    # 最良を選択（再訪問）
                    best_dir = min(
                        evaluations.keys(),
                        key=lambda d: evaluations[d].f_value
                    )
                    best_metrics = evaluations[best_dir]

            # 移動
            next_pos = self._get_next_pos(current, best_dir)
            current = next_pos
            path.append(current)
            visited.add(current)
            steps += 1

            # メトリクス記録
            metrics_history.append({
                'step': steps,
                'position': current,
                'direction': best_dir,
                'f_value': best_metrics.f_value,
                'delta_ss': best_metrics.delta_ss,
                'delta_ig': best_metrics.delta_ig,
            })

        # 最大ステップ到達（失敗）
        return NavigationResult(
            success=False,
            steps=steps,
            path=path,
            visited_count=len(visited),
            dead_end_encounters=dead_end_encounters,
            backtrack_count=backtrack_count,
            metrics_history=metrics_history,
        )

    def _get_next_pos(
        self,
        current: Tuple[int, int],
        direction: str
    ) -> Tuple[int, int]:
        """方向から次の位置を取得"""
        dx_dy = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
        dx, dy = dx_dy[direction]
        return (current[0] + dx, current[1] + dy)


def generate_maze(size: int, seed: int = 42) -> np.ndarray:
    """
    迷路を生成（再帰的分割法の簡易版）

    Args:
        size: 迷路サイズ（奇数推奨）
        seed: 乱数シード

    Returns:
        迷路配列 (0=通路, 1=壁)
    """
    np.random.seed(seed)

    # 全て壁で初期化
    maze = np.ones((size, size), dtype=np.int32)

    def carve(x: int, y: int):
        """再帰的に通路を掘る"""
        maze[y, x] = 0
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        np.random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1:
                if maze[ny, nx] == 1:
                    # 壁を掘る
                    maze[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

    # スタート位置から掘り始める
    carve(1, 1)

    return maze


def print_maze(
    maze: np.ndarray,
    path: List[Tuple[int, int]] = None,
    start: Tuple[int, int] = None,
    goal: Tuple[int, int] = None
):
    """迷路を表示"""
    height, width = maze.shape
    path_set = set(path) if path else set()

    for y in range(height):
        row = ""
        for x in range(width):
            pos = (x, y)
            if pos == start:
                row += "S "
            elif pos == goal:
                row += "G "
            elif pos in path_set:
                row += "* "
            elif maze[y, x] == 1:
                row += "# "
            else:
                row += ". "
        print(row)


# テスト
if __name__ == "__main__":
    # 小さな迷路でテスト
    size = 15
    maze = generate_maze(size, seed=42)
    start = (1, 1)
    goal = (size - 2, size - 2)

    print(f"=== {size}x{size} 迷路テスト ===")
    print(f"Start: {start}, Goal: {goal}")
    print()

    # Baseline
    print("--- Baseline (no SS) ---")
    nav_base = MazeNavigator(use_structural_similarity=False)
    result_base = nav_base.navigate(maze, start, goal)
    print(f"Success: {result_base.success}")
    print(f"Steps: {result_base.steps}")
    print(f"Visited: {result_base.visited_count}")
    print(f"Dead-ends: {result_base.dead_end_encounters}")
    print(f"Backtracks: {result_base.backtrack_count}")

    print()

    # With SS
    print("--- With Structural Similarity ---")
    nav_ss = MazeNavigator(use_structural_similarity=True, beta=0.3)
    result_ss = nav_ss.navigate(maze, start, goal)
    print(f"Success: {result_ss.success}")
    print(f"Steps: {result_ss.steps}")
    print(f"Visited: {result_ss.visited_count}")
    print(f"Dead-ends: {result_ss.dead_end_encounters}")
    print(f"Backtracks: {result_ss.backtrack_count}")

    print()

    # 改善率
    if result_base.steps > 0:
        improvement = (result_base.steps - result_ss.steps) / result_base.steps * 100
        print(f"=== 改善率: {improvement:.1f}% ===")

    print()
    print("=== 迷路とパス (With SS) ===")
    print_maze(maze, result_ss.path, start, goal)
