"""
Branch detection and management module
"""

import numpy as np
from typing import Tuple, Set, Optional, Dict, List


class BranchDetector:
    """分岐点の検出と管理 (A/B 対応: 分岐完了判定緩和 + 進捗トラッキング)

    A: dead-end から戻り分岐起点に再達し *未訪問方向が無い* ことを条件に完了。
       入口 open_neighbors を保持し、visited_positions に含まれるかを都度評価。
    B: 分岐ごとの進捗 (探索済み出口数 / 総出口数, exploring/completed) を branch_history イベントとして記録。
    """
    
    def __init__(self, backtrack_threshold: float = -0.2):
        self.backtrack_threshold = backtrack_threshold
        # 全体管理
        self.branch_points: Set[Tuple[int, int]] = set()
        self.current_branch_entry: Optional[Tuple[int, int]] = None
        self.branch_positions: Set[Tuple[int, int]] = set()
        # 分岐状態: entry_pos -> state dict
        self.branch_states: Dict[Tuple[int, int], Dict[str, any]] = {}
        # 履歴
        self.branch_history: List[Dict] = []
        self.completed_branches: List[Dict] = []
    
    def detect_branch_entry(
        self,
        position: Tuple[int, int],
        maze: np.ndarray
    ) -> bool:
        """
        分岐点への進入を検出
        
        Args:
            position: 現在位置 (x, y)
            maze: 迷路配列（0: 通路、1: 壁）
        
        Returns:
            分岐点に進入したかどうか
        """
        # 既に分岐探索中の場合
        if self.current_branch_entry is not None:
            return False
        
        # 利用可能な方向数を計算
        available_directions = self._count_available_directions(position, maze)
        
        # 3方向以上なら分岐点
        if available_directions >= 3:
            self.current_branch_entry = position
            self.branch_positions = {position}
            self.branch_points.add(position)
            # 入口からの開放隣接セルを列挙
            open_neighbors = self._list_open_neighbors(position, maze)
            self.branch_states[position] = {
                'open_neighbors': open_neighbors,  # set of (x,y)
                'explored_neighbors': set(),
                'status': 'exploring',
                'steps_in_branch': 0
            }
            self.branch_history.append({
                'type': 'entry',
                'position': position,
                'available_directions': available_directions,
                'open_neighbors': list(open_neighbors)
            })
            return True
        
        return False
    
    def detect_branch_completion(
        self,
        position: Tuple[int, int],
        maze: Optional[np.ndarray] = None,
        visited_positions: Optional[Set[Tuple[int, int]]] = None
    ) -> bool:
        """分岐探索完了を検出 (A)

        完了条件 (緩和):
          1. 分岐探索中 (current_branch_entry != None)
          2. 分岐入口に再到達
          3. 入口 open_neighbors が全て visited_positions に含まれる
        旧条件 (branch_positions > 3) は補助指標として参考に残しつつ無視。
        """
        if self.current_branch_entry is None:
            return False
        # 進捗更新
        self.branch_positions.add(position)
        entry = self.current_branch_entry
        state = self.branch_states.get(entry)
        if state is not None and visited_positions is not None:
            newly_explored = set()
            for npos in state['open_neighbors']:
                if npos in visited_positions and npos not in state['explored_neighbors']:
                    state['explored_neighbors'].add(npos)
                    newly_explored.add(npos)
            if newly_explored:
                # B: 進捗イベント
                self.branch_history.append({
                    'type': 'progress',
                    'entry': entry,
                    'explored': len(state['explored_neighbors']),
                    'total': len(state['open_neighbors']),
                    'newly_explored': list(newly_explored)
                })
        # ステップ数カウント
        if state is not None:
            state['steps_in_branch'] += 1
        # 完了判定
        if position == entry:
            if state is not None and state['open_neighbors'] <= state['explored_neighbors']:
                # 完了
                self.completed_branches.append({
                    'entry': entry,
                    'positions': self.branch_positions.copy(),
                    'size': len(self.branch_positions),
                    'open_neighbor_count': len(state['open_neighbors']),
                    'steps_in_branch': state['steps_in_branch']
                })
                state['status'] = 'completed'
                self.branch_history.append({
                    'type': 'completion',
                    'position': entry,
                    'explored_positions': len(self.branch_positions),
                    'open_neighbors': len(state['open_neighbors'])
                })
                # リセット (現在の探索のみ)
                self.current_branch_entry = None
                self.branch_positions = set()
                return True
        return False
    
    def should_backtrack(
        self,
        gedig_value: float,
        threshold: Optional[float] = None
    ) -> bool:
        """
        バックトラックすべきか判定
        
        Args:
            gedig_value: 現在のgeDIG値
            threshold: 閾値（Noneの場合はデフォルト）
        
        Returns:
            バックトラックすべきかどうか
        """
        if threshold is None:
            threshold = self.backtrack_threshold
        
        return gedig_value < threshold
    
    def is_at_branch_point(
        self,
        position: Tuple[int, int]
    ) -> bool:
        """
        指定位置が分岐点かどうか判定
        
        Args:
            position: 位置
        
        Returns:
            分岐点かどうか
        """
        return position in self.branch_points
    
    def is_exploring_branch(self) -> bool:
        """
        現在分岐を探索中かどうか
        
        Returns:
            分岐探索中かどうか
        """
        return self.current_branch_entry is not None
    
    def get_current_branch_info(self) -> Optional[Dict]:
        """
        現在探索中の分岐情報を取得
        
        Returns:
            分岐情報の辞書（探索中でない場合はNone）
        """
        if self.current_branch_entry is None:
            return None
        
        return {
            'entry': self.current_branch_entry,
            'positions_explored': len(self.branch_positions),
            'current_positions': self.branch_positions.copy()
        }
    
    def get_branch_statistics(self) -> Dict:
        """
        分岐統計を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            'total_branch_points': len(self.branch_points),
            'completed_branches': len(self.completed_branches),
            'currently_exploring': self.is_exploring_branch()
        }
        
        if self.completed_branches:
            sizes = [b['size'] for b in self.completed_branches]
            stats['average_branch_size'] = np.mean(sizes)
            stats['max_branch_size'] = max(sizes)
            stats['min_branch_size'] = min(sizes)
        
        return stats
    
    def mark_dead_end(self, position: Tuple[int, int]) -> None:
        """
        行き止まりをマーク
        
        Args:
            position: 行き止まりの位置
        """
        self.branch_history.append({
            'type': 'dead_end',
            'position': position
        })
    
    def suggest_backtrack_target(
        self,
        current_position: Tuple[int, int],
        visited_positions: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        バックトラック先を提案
        
        Args:
            current_position: 現在位置
            visited_positions: 訪問済み位置の集合
        
        Returns:
            バックトラック先の位置（提案がない場合はNone）
        """
        # 最も近い未完了の分岐点を探す
        unfinished_branches = []
        
        for branch_point in self.branch_points:
            # その分岐点が完了済みか確認
            is_completed = any(
                b['entry'] == branch_point
                for b in self.completed_branches
            )
            
            if not is_completed:
                # マンハッタン距離を計算
                distance = abs(current_position[0] - branch_point[0]) + \
                          abs(current_position[1] - branch_point[1])
                unfinished_branches.append((distance, branch_point))
        
        if unfinished_branches:
            # 最も近い未完了分岐点を返す
            unfinished_branches.sort(key=lambda x: x[0])
            return unfinished_branches[0][1]
        
        return None
    
    def reset(self) -> None:
        """状態をリセット"""
        self.branch_points.clear()
        self.current_branch_entry = None
        self.branch_positions.clear()
        self.branch_history.clear()
        self.completed_branches.clear()
        self.branch_states.clear()
    
    def _count_available_directions(
        self,
        position: Tuple[int, int],
        maze: np.ndarray
    ) -> int:
        """
        利用可能な方向数を計算（内部メソッド）
        
        Args:
            position: 位置
            maze: 迷路配列
        
        Returns:
            利用可能な方向数
        """
        x, y = position
        h, w = maze.shape
        count = 0
        
        # 4方向をチェック
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                count += 1
        
        return count

    def _list_open_neighbors(
        self,
        position: Tuple[int, int],
        maze: np.ndarray
    ) -> Set[Tuple[int, int]]:
        """入口位置からの開放隣接セル集合を返す"""
        x, y = position
        h, w = maze.shape
        opens: Set[Tuple[int, int]] = set()
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and maze[ny_, nx_] == 0:
                opens.add((nx_, ny_))
        return opens