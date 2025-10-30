"""
Episode lifecycle management module
"""

import numpy as np
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass, field
from .vector_processor import VectorProcessor


@dataclass
class Episode:
    """エピソードデータクラス"""
    position: Tuple[int, int]
    direction: str
    vector: np.ndarray
    is_wall: bool
    visit_count: int = 0
    # 方向ごとの観測回数（現在位置でこの方向を観測した回数）
    observation_count: int = 0
    episode_id: int = 0
    timestamp: int = 0
    # 新規: 成功/失敗/未実行 指標 (-1,0,1 等)
    success_outcome: float = 0.0
    # 新規: ゴールフラグ (到達:1 / 未到達:0) あるいは将来近接値
    goal_flag: float = 0.0
    # Phase0: 重み適用後ベクトルキャッシュ (weights のバージョン識別用フィールド)
    _weighted_vector: Optional[np.ndarray] = field(default=None, repr=False)
    _weights_version: Optional[int] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'position': self.position,
            'direction': self.direction,
            'is_wall': self.is_wall,
            'visit_count': self.visit_count,
            'observation_count': self.observation_count,
            'episode_id': self.episode_id,
            'timestamp': self.timestamp
        }

    # キャッシュ管理ヘルパ (Navigator 側から利用予定)
    def get_weighted_vector(self, weights: np.ndarray, version: int, apply_fn) -> np.ndarray:  # type: ignore[no-untyped-def]
        if self._weighted_vector is None or self._weights_version != version:
            self._weighted_vector = apply_fn(self.vector, weights)
            self._weights_version = version
        return self._weighted_vector


class EpisodeManager:
    """エピソードのライフサイクル管理"""
    
    def __init__(self, maze_width: int, maze_height: int):
        """
        Args:
            maze_width: 迷路の幅
            maze_height: 迷路の高さ
        """
        # エピソード管理
        self.episodes: Dict[Tuple[Tuple[int, int], str], Episode] = {}
        # id -> Episode 逆引きインデックス (ベクトル近傍検索結果から Episode を高速参照)
        self.episodes_by_id: Dict[int, Episode] = {}
        self.episode_counter = 0
        self.current_step = 0

        # 訪問済み位置の追跡
        self.visited_positions: Set[Tuple[int, int]] = set()

        # ベクトル処理
        self.vector_processor = VectorProcessor(maze_width, maze_height)

        # 方向マッピング
        self.directions = ['N', 'S', 'E', 'W']
        self.direction_map = {
            'N': (0, -1),
            'S': (0, 1),
            'E': (1, 0),
            'W': (-1, 0)
        }
        # 初期visitカウントのシード方式（既訪問の隣接なら1にするか）
        try:
            self._init_visit_seed = str(__import__('os').environ.get('MAZE_INIT_VISIT_SEED','0')).strip() not in ("0","false","False","")
        except Exception:
            self._init_visit_seed = False
    
    def observe(
        self,
        current_pos: Tuple[int, int],
        maze: np.ndarray
    ) -> Dict[str, Episode]:
        """
        現在位置から4方向を観測してエピソードを生成/取得
        
        Args:
            current_pos: 現在位置 (x, y)
            maze: 迷路配列（0: 通路、1: 壁）
        
        Returns:
            方向をキーとしたエピソードの辞書
        """
        # 現在位置を訪問済みに追加
        self.visited_positions.add(current_pos)
        
        episodes = {}
        
        for direction in self.directions:
            dx, dy = self.direction_map[direction]
            next_x, next_y = current_pos[0] + dx, current_pos[1] + dy
            next_pos = (next_x, next_y)
            
            # エピソードのキー
            key = (current_pos, direction)
            
            if key in self.episodes:
                # 既存エピソードを取得 + 観測回数を加算
                ep = self.episodes[key]
                try:
                    ep.observation_count += 1
                except Exception:
                    # 念のため欠損時に初期化
                    try:
                        setattr(ep, 'observation_count', int(getattr(ep, 'observation_count', 0)) + 1)
                    except Exception:
                        pass
                episodes[direction] = ep
            else:
                # 新規エピソード作成
                is_wall = self._is_wall(next_pos, maze)
                
                # 初期訪問回数の決定
                # 既訪問位置への方向なら1にするオプション（デフォルト無効で常に0）
                if self._init_visit_seed and (next_pos in self.visited_positions):
                    initial_visits = 1
                else:
                    initial_visits = 0
                
                # ベクトル生成
                vector = self.vector_processor.create_vector(
                    current_pos,
                    direction,
                    is_wall,
                    initial_visits,
                    success_outcome=0.0,
                    goal_flag=0.0
                )
                
                # エピソード作成
                episode = Episode(
                    position=current_pos,
                    direction=direction,
                    vector=vector,
                    is_wall=is_wall,
                    visit_count=initial_visits,
                    observation_count=1,
                    episode_id=self.episode_counter,
                    timestamp=self.current_step,
                    success_outcome=0.0,
                    goal_flag=0.0
                )
                
                self.episodes[key] = episode
                self.episodes_by_id[episode.episode_id] = episode
                self.episode_counter += 1
                episodes[direction] = episode
        
        return episodes
    
    def move(
        self,
        current_pos: Tuple[int, int],
        selected_direction: str
    ) -> bool:
        """
        移動実行時に選択した方向の訪問回数を更新
        
        Args:
            current_pos: 現在位置
            selected_direction: 選択した方向
        
        Returns:
            移動が実行されたかどうか
        """
        key = (current_pos, selected_direction)
        
        if key not in self.episodes:
            return False
        
        episode = self.episodes[key]
        
        # 壁への移動は不可
        if episode.is_wall:
            return False
        
        # 訪問回数を更新
        episode.visit_count += 1
        
        # ベクトルも更新
        episode.vector = self.vector_processor.update_visit_count(
            episode.vector,
            episode.visit_count
        )
        
        # 次の位置を訪問済みに追加
        dx, dy = self.direction_map[selected_direction]
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        self.visited_positions.add(next_pos)

        # 逆方向エピソードを次の位置に作成/更新（到達を記録）
        # 例: current=(x,y) から E に移動 → next=(x+1,y) の W エピソードを visit++
        try:
            rev_map = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
            rev_dir = rev_map.get(selected_direction, '')
            if rev_dir:
                rev_key = (next_pos, rev_dir)
                if rev_key not in self.episodes:
                    # 壁ではない（いま通ってきた経路）とみなせる
                    is_wall_rev = False
                    vec_rev = self.vector_processor.create_vector(
                        next_pos,
                        rev_dir,
                        is_wall_rev,
                        0,
                        success_outcome=0.0,
                        goal_flag=0.0
                    )
                    rev_ep = Episode(
                        position=next_pos,
                        direction=rev_dir,
                        vector=vec_rev,
                        is_wall=is_wall_rev,
                        visit_count=0,
                        observation_count=0,
                        episode_id=self.episode_counter,
                        timestamp=self.current_step,
                        success_outcome=0.0,
                        goal_flag=0.0
                    )
                    self.episodes[rev_key] = rev_ep
                    self.episodes_by_id[rev_ep.episode_id] = rev_ep
                    self.episode_counter += 1
                # visit++ and update vector
                ep_rev = self.episodes[rev_key]
                ep_rev.visit_count += 1
                ep_rev.vector = self.vector_processor.update_visit_count(ep_rev.vector, ep_rev.visit_count)
        except Exception:
            # 逆方向の記録に失敗しても本流は継続
            pass

        return True
    
    def get_episode(
        self,
        position: Tuple[int, int],
        direction: str
    ) -> Optional[Episode]:
        """
        特定のエピソードを取得
        
        Args:
            position: 位置
            direction: 方向
        
        Returns:
            エピソード（存在しない場合はNone）
        """
        key = (position, direction)
        return self.episodes.get(key)

    # --- Transition observer compatibility helper ---
    def get_visit_count(self, position: Tuple[int,int]) -> int:
        """Return approximate visit count for a position by aggregating directional episodes.

        Position単位の visit_count を保持していないため、該当位置を起点とする4方向エピソードの
        visit_count の最大値を代表値として返す (一度でも通過すれば >0 になる想定)。
        エピソード未生成なら 0。
        """
        dirs = ['N','S','E','W']
        max_vis = 0
        for d in dirs:
            ep = self.episodes.get((position, d))
            if ep and ep.visit_count > max_vis:
                max_vis = ep.visit_count
        return max_vis
    
    def get_episodes_at_position(
        self,
        position: Tuple[int, int]
    ) -> Dict[str, Episode]:
        """
        特定位置の全エピソードを取得
        
        Args:
            position: 位置
        
        Returns:
            方向をキーとしたエピソードの辞書
        """
        result = {}
        for direction in self.directions:
            episode = self.get_episode(position, direction)
            if episode:
                result[direction] = episode
        return result
    
    def get_episodes_at_step(
        self,
        step: int
    ) -> List[Episode]:
        """
        特定ステップまでに作成されたエピソードを取得
        
        Args:
            step: ステップ数
        
        Returns:
            エピソードのリスト
        """
        return [
            ep for ep in self.episodes.values()
            if ep.timestamp <= step
        ]
    
    def get_unvisited_episodes(self) -> List[Episode]:
        """
        未訪問（visit_count=0）のエピソードを取得
        
        Returns:
            未訪問エピソードのリスト
        """
        return [
            ep for ep in self.episodes.values()
            if ep.visit_count == 0 and not ep.is_wall
        ]
    
    def get_least_visited_episodes(
        self,
        n: int = 5
    ) -> List[Tuple[Episode, int]]:
        """
        訪問回数が少ないエピソードを取得
        
        Args:
            n: 取得する数
        
        Returns:
            (エピソード, 訪問回数)のタプルのリスト
        """
        valid_episodes = [
            (ep, ep.visit_count)
            for ep in self.episodes.values()
            if not ep.is_wall
        ]
        
        # 訪問回数でソート
        valid_episodes.sort(key=lambda x: x[1])
        
        return valid_episodes[:n]
    
    def increment_step(self):
        """ステップカウンタを増加"""
        self.current_step += 1
    
    def get_statistics(self) -> Dict[str, any]:
        """
        統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        total_episodes = len(self.episodes)
        wall_episodes = sum(1 for ep in self.episodes.values() if ep.is_wall)
        visited_episodes = sum(1 for ep in self.episodes.values() if ep.visit_count > 0)
        
        visit_counts = [ep.visit_count for ep in self.episodes.values() if not ep.is_wall]
        
        stats = {
            'total_episodes': total_episodes,
            'wall_episodes': wall_episodes,
            'passable_episodes': total_episodes - wall_episodes,
            'visited_episodes': visited_episodes,
            'unvisited_episodes': total_episodes - wall_episodes - visited_episodes,
            'unique_positions': len(self.visited_positions),
            'current_step': self.current_step
        }
        
        if visit_counts:
            stats['max_visit_count'] = max(visit_counts)
            stats['avg_visit_count'] = np.mean(visit_counts)
            stats['total_visits'] = sum(visit_counts)
        
        return stats
    
    def reset(self):
        """状態をリセット"""
        self.episodes.clear()
        self.episode_counter = 0
        self.current_step = 0
        self.visited_positions.clear()
    
    def _is_wall(
        self,
        pos: Tuple[int, int],
        maze: np.ndarray
    ) -> bool:
        """
        指定位置が壁かどうか判定（内部メソッド）
        
        Args:
            pos: 位置 (x, y)
            maze: 迷路配列
        
        Returns:
            壁かどうか
        """
        x, y = pos
        h, w = maze.shape
        
        # 範囲外は壁扱い
        if not (0 <= x < w and 0 <= y < h):
            return True
        
        # 迷路配列で判定（1が壁）
        return maze[y, x] == 1
