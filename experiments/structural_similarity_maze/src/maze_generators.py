"""
様々なタイプの迷路生成器

- 完全木（行き止まり少ない）
- ループあり迷路（複数経路）
- 行き止まり多い迷路
"""

import numpy as np
from typing import Tuple, List
import random


def generate_perfect_maze(size: int, seed: int = 42) -> np.ndarray:
    """
    完全木迷路（従来の再帰的分割法）
    行き止まりが少ない
    """
    np.random.seed(seed)
    random.seed(seed)

    maze = np.ones((size, size), dtype=np.int32)

    def carve(x: int, y: int):
        maze[y, x] = 0
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1:
                if maze[ny, nx] == 1:
                    maze[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

    carve(1, 1)
    return maze


def generate_dead_end_maze(
    size: int,
    seed: int = 42,
    dead_end_ratio: float = 0.3
) -> np.ndarray:
    """
    行き止まりが多い迷路

    Args:
        size: 迷路サイズ
        seed: 乱数シード
        dead_end_ratio: 行き止まりになる確率

    Returns:
        迷路配列
    """
    np.random.seed(seed)
    random.seed(seed)

    # まず完全木を作る
    maze = generate_perfect_maze(size, seed)

    # 一部の通路を塞いで行き止まりを作る
    height, width = maze.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 0:  # 通路
                # 隣接する通路の数を数える
                neighbors = []
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = x + dx, y + dy
                    if maze[ny, nx] == 0:
                        neighbors.append((nx, ny))

                # 3つ以上の隣接通路がある場合、一部を塞ぐ
                if len(neighbors) >= 3 and random.random() < dead_end_ratio:
                    # ランダムに1つの隣接を塞ぐ（ただしスタート・ゴール方向は除く）
                    safe_neighbors = [
                        n for n in neighbors
                        if n != (1, 1) and n != (size - 2, size - 2)
                    ]
                    if safe_neighbors:
                        block_x, block_y = random.choice(safe_neighbors)
                        # 塞いでも連結性が保たれるか簡易チェック
                        maze[block_y, block_x] = 1

    # 連結性を確認し、必要なら修復
    maze = ensure_connectivity(maze, (1, 1), (size - 2, size - 2))

    return maze


def generate_branching_maze(
    size: int,
    seed: int = 42,
    branch_probability: float = 0.4
) -> np.ndarray:
    """
    分岐と行き止まりが多い迷路
    Prim's algorithm変種

    Args:
        size: 迷路サイズ
        seed: 乱数シード
        branch_probability: 分岐を作る確率

    Returns:
        迷路配列
    """
    np.random.seed(seed)
    random.seed(seed)

    maze = np.ones((size, size), dtype=np.int32)

    # スタート
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 0

    # フロンティア（壁の候補）
    frontiers = []

    def add_frontiers(x, y):
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1:
                if maze[ny, nx] == 1:
                    frontiers.append((nx, ny, x, y))

    add_frontiers(start_x, start_y)

    while frontiers:
        # ランダムにフロンティアを選択
        idx = random.randint(0, len(frontiers) - 1)
        fx, fy, px, py = frontiers.pop(idx)

        if maze[fy, fx] == 1:  # まだ壁なら
            # 通路にする
            maze[fy, fx] = 0
            maze[(fy + py) // 2, (fx + px) // 2] = 0

            # 新しいフロンティアを追加
            add_frontiers(fx, fy)

            # 確率的に追加の分岐を作らない（行き止まりを増やす）
            if random.random() > branch_probability:
                # フロンティアの一部を削除
                if len(frontiers) > 2:
                    remove_count = random.randint(1, min(3, len(frontiers) - 1))
                    for _ in range(remove_count):
                        if frontiers:
                            frontiers.pop(random.randint(0, len(frontiers) - 1))

    # ゴールへの経路を確保
    maze = ensure_connectivity(maze, (1, 1), (size - 2, size - 2))

    return maze


def ensure_connectivity(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> np.ndarray:
    """
    スタートからゴールへの経路を確保

    Args:
        maze: 迷路配列
        start: スタート位置
        goal: ゴール位置

    Returns:
        修正された迷路
    """
    from collections import deque

    height, width = maze.shape
    maze = maze.copy()

    # BFSで到達可能か確認
    def is_reachable():
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if maze[ny, nx] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    # 到達不能なら経路を掘る
    if not is_reachable():
        # 単純にゴールまで直線的に掘る
        x, y = start
        gx, gy = goal

        while (x, y) != (gx, gy):
            maze[y, x] = 0

            if x < gx:
                x += 1
            elif x > gx:
                x -= 1
            elif y < gy:
                y += 1
            elif y > gy:
                y -= 1

        maze[gy, gx] = 0

    return maze


def count_dead_ends(maze: np.ndarray) -> int:
    """行き止まりの数を数える"""
    height, width = maze.shape
    dead_ends = 0

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 0:  # 通路
                neighbors = 0
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    if maze[y + dy, x + dx] == 0:
                        neighbors += 1
                if neighbors == 1:
                    dead_ends += 1

    return dead_ends


def print_maze_stats(maze: np.ndarray, name: str):
    """迷路の統計を表示"""
    height, width = maze.shape
    passages = np.sum(maze == 0)
    dead_ends = count_dead_ends(maze)
    print(f"{name}: {width}x{height}, passages={passages}, dead_ends={dead_ends}")


# テスト
if __name__ == "__main__":
    size = 21

    print("=== 迷路生成テスト ===\n")

    # 完全木
    maze_perfect = generate_perfect_maze(size, seed=42)
    print_maze_stats(maze_perfect, "Perfect (tree)")

    # 行き止まり多い
    maze_dead = generate_dead_end_maze(size, seed=42, dead_end_ratio=0.4)
    print_maze_stats(maze_dead, "Dead-end heavy")

    # 分岐迷路
    maze_branch = generate_branching_maze(size, seed=42, branch_probability=0.3)
    print_maze_stats(maze_branch, "Branching")

    print("\n=== 行き止まりの視覚化（Dead-end heavy） ===")
    for y in range(size):
        row = ""
        for x in range(size):
            if maze_dead[y, x] == 1:
                row += "# "
            else:
                # 行き止まりか確認
                neighbors = sum(
                    1 for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                    if 0 <= y + dy < size and 0 <= x + dx < size and maze_dead[y + dy, x + dx] == 0
                )
                if neighbors == 1:
                    row += "D "  # Dead-end
                else:
                    row += ". "
        print(row)
