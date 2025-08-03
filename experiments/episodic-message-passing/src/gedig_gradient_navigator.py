#!/usr/bin/env python3
"""真のgeDIG勾配ナビゲーター：候補行動ごとのΔGED/ΔIG予測版"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class GradientGedigEpisode:
    """勾配計算に最適化されたエピソード"""
    episode_type: str
    content: Dict
    compact_vector: np.ndarray  # 5次元
    node_id: int
    
    # geDIG関連
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    
    # メタ情報
    visit_count: int = 0
    timestamp: int = 0


class GedigGradientNavigator:
    """真のgeDIG勾配に基づくナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig, maze_size: int = 20):
        self.config = config
        self.maze_size = maze_size
        self.episodes: List[GradientGedigEpisode] = []
        self.episode_counter = 0
        self.current_step = 0
        
        # 統計
        self.position_visit_counts = defaultdict(int)
        self.action_success_rates = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))
        
        # ハイパーパラメータ（レビューを反映）
        self.ged_weight = 1.0      # 後でOptunaで調整
        self.ig_weight = 1.0       # 後でOptunaで調整
        self.look_ahead_depth = 2  # 2手先まで見る
        self.temperature = 1.0
        self.temperature_decay = 0.998
        
    def create_compact_vector(self, episode_type: str, content: Dict) -> np.ndarray:
        """5次元コンパクトベクトル（前と同じ）"""
        if episode_type == "goal_info":
            pos = content['position']
            vector = np.zeros(5)
            vector[0] = (pos[0] / self.maze_size) * 2 - 1
            vector[1] = (pos[1] / self.maze_size) * 2 - 1
            vector[2] = 0.0
            vector[3] = 1.0
            vector[4] = 0.0
            return vector
        else:
            from_pos = content['from']
            action = content['action']
            result = content['result']
            
            norm_x = (from_pos[0] / self.maze_size) * 2 - 1
            norm_y = (from_pos[1] / self.maze_size) * 2 - 1
            action_norm = -1.0 + (action * 2.0 / 3.0)
            
            if result == "成功":
                result_val = 1.0
            elif result == "壁":
                result_val = -1.0
            else:
                result_val = -0.5
                
            visit_count = self.position_visit_counts[from_pos]
            # レビューの提案：滑らかな減衰
            visit_norm = 2.0 * (1.0 / (1.0 + math.log1p(visit_count))) - 1.0
            
            return np.array([norm_x, norm_y, action_norm, result_val, visit_norm])
    
    def estimate_deltas(self, from_pos: Tuple[int, int], action: int) -> Tuple[float, float]:
        """候補行動のΔGED/ΔIGを予測（これが核心！）"""
        
        # 仮想的な次の位置
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        to_pos = (from_pos[0] + dx, from_pos[1] + dy)
        
        # 1. ΔGED推定：この移動がもたらす構造的新規性
        ged_estimate = 1.0  # デフォルト
        
        # 既存エピソードとの構造的差異を計算
        similar_transitions = 0
        for ep in self.episodes:
            if ep.episode_type == "movement" and ep.content['from'] == from_pos:
                if ep.content['action'] == action:
                    similar_transitions += 1
                    
        if similar_transitions > 0:
            # 既に試した遷移は新規性が低い
            ged_estimate = 1.0 / (1.0 + math.log1p(similar_transitions))
        
        # 到達先の新規性も考慮
        if to_pos not in self.position_visit_counts:
            ged_estimate += 0.3  # 新しい位置への到達ボーナス
            
        # 2. ΔIG推定：この移動で得られる情報量
        ig_estimate = 0.5  # デフォルト
        
        # 位置の不確実性（エントロピー）
        stats = self.action_success_rates[from_pos][action]
        if stats['total'] == 0:
            # 完全に未知
            ig_estimate = 1.0
        else:
            # 成功率の不確実性
            p = stats['success'] / stats['total']
            if 0 < p < 1:
                entropy = -p * math.log(p) - (1-p) * math.log(1-p)
                ig_estimate = entropy
                
        # 訪問回数による減衰（レビューの提案を反映）
        visit_decay = 1.0 / (1.0 + math.log1p(self.position_visit_counts[from_pos]))
        ig_estimate *= visit_decay
        
        return ged_estimate, ig_estimate
    
    def look_ahead_score(self, from_pos: Tuple[int, int], action: int, depth: int = 1) -> float:
        """Look-ahead: N手先までの期待スコア"""
        if depth == 0:
            return 0.0
            
        # 現在の行動のスコア
        ged, ig = self.estimate_deltas(from_pos, action)
        # レビューの提案：log1p圧縮でスケール整合
        current_score = math.log1p(ged) * math.log1p(ig)
        
        if depth == 1:
            return current_score
            
        # 次の位置を予測
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        next_pos = (from_pos[0] + dx, from_pos[1] + dy)
        
        # 境界チェック
        if not (0 <= next_pos[0] < self.maze_size and 0 <= next_pos[1] < self.maze_size):
            return current_score  # 壁にぶつかる
            
        # 再帰的に次の手を評価
        future_scores = []
        for next_action in range(4):
            future_score = self.look_ahead_score(next_pos, next_action, depth - 1)
            future_scores.append(future_score)
            
        # 最大値を選択（楽観的）
        if future_scores:
            max_future = max(future_scores)
            return current_score + 0.9 * max_future  # 割引率0.9
        else:
            return current_score
    
    def decide_action_gradient(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """真のgeDIG勾配に基づく行動決定"""
        
        action_scores = {}
        
        for action in possible_actions:
            # レビューの核心提案：候補行動ごとのΔGED/ΔIG計算
            if self.look_ahead_depth > 0:
                score = self.look_ahead_score(current_pos, action, self.look_ahead_depth)
            else:
                ged, ig = self.estimate_deltas(current_pos, action)
                score = math.log1p(ged) * math.log1p(ig)
                
            action_scores[action] = score
            
        # デバッグ出力（最初の数ステップのみ）
        if self.current_step < 10:
            print(f"\n位置{current_pos}での行動スコア:")
            for act, score in action_scores.items():
                ged, ig = self.estimate_deltas(current_pos, act)
                print(f"  {['↑', '→', '↓', '←'][act]}: score={score:.3f} (ΔG={ged:.3f}, ΔI={ig:.3f})")
        
        # ソフトマックス選択（温度付き）
        if action_scores:
            scores = np.array(list(action_scores.values()))
            # オーバーフロー防止
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / exp_scores.sum()
            
            return list(action_scores.keys())[np.random.choice(len(probs), p=probs)]
        else:
            return np.random.choice(possible_actions)
    
    def add_episode(self, episode_type: str, content: Dict) -> GradientGedigEpisode:
        """エピソード追加（勾配計算用）"""
        compact_vector = self.create_compact_vector(episode_type, content)
        
        episode = GradientGedigEpisode(
            episode_type=episode_type,
            content=content,
            compact_vector=compact_vector,
            node_id=self.episode_counter,
            timestamp=self.current_step
        )
        
        if episode_type == "movement":
            from_pos = content['from']
            episode.visit_count = self.position_visit_counts[from_pos]
            
            # 実際のΔGED/ΔIG（事後計算）
            episode.ged_delta, episode.ig_delta = self.estimate_deltas(from_pos, content['action'])
            episode.gedig_value = episode.ged_delta * episode.ig_delta
            
        self.episodes.append(episode)
        
        # 統計更新
        if episode_type == "movement":
            from_pos = content['from']
            action = content['action']
            result = content['result']
            
            self.position_visit_counts[from_pos] += 1
            self.action_success_rates[from_pos][action]['total'] += 1
            if result == "成功":
                self.action_success_rates[from_pos][action]['success'] += 1
                to_pos = (from_pos[0] + [(0, -1), (1, 0), (0, 1), (-1, 0)][action][0],
                         from_pos[1] + [(0, -1), (1, 0), (0, 1), (-1, 0)][action][1])
                if 0 <= to_pos[0] < self.maze_size and 0 <= to_pos[1] < self.maze_size:
                    self.position_visit_counts[to_pos] += 1
                    
        self.episode_counter += 1
        self.current_step += 1
        
        # 温度アニーリング
        self.temperature *= self.temperature_decay
        self.temperature = max(0.1, self.temperature)
        
        return episode


def run_gradient_experiment(maze_size: int = 20):
    """勾配ベース実験"""
    print("真のgeDIG勾配ナビゲーター実験")
    print("=" * 60)
    print("特徴：候補行動ごとのΔGED/ΔIG予測 + 2手先Look-ahead")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = GedigGradientNavigator(config, maze_size=maze_size)
    
    # 迷路生成
    np.random.seed(42)
    maze = SimpleMaze(size=(maze_size, maze_size), maze_type='dfs')
    
    print(f"迷路サイズ: {maze.size}")
    print(f"スタート: {maze.start_pos} → ゴール: {maze.goal_pos}")
    
    # ゴール情報追加
    navigator.add_episode("goal_info", {"position": maze.goal_pos})
    
    # メインループ
    obs = maze.reset()
    steps = 0
    max_steps = maze_size * maze_size * 5
    
    print("\n探索開始...")
    
    while steps < max_steps:
        current_pos = obs.position
        
        # 勾配ベースの行動決定
        action = navigator.decide_action_gradient(current_pos, obs.possible_moves)
        
        # 行動実行
        old_pos = current_pos
        obs, _, done, _ = maze.step(action)
        new_pos = obs.position
        steps += 1
        
        # エピソード形成
        result = "壁" if old_pos == new_pos else ("行き止まり" if obs.is_dead_end else "成功")
        
        episode = navigator.add_episode("movement", {
            "from": old_pos,
            "to": new_pos,
            "action": action,
            "result": result
        })
        
        # 進捗表示
        if steps % 200 == 0:
            unique_positions = len(navigator.position_visit_counts)
            coverage = unique_positions / (maze_size * maze_size) * 100
            max_visits = max(navigator.position_visit_counts.values()) if navigator.position_visit_counts else 0
            
            print(f"ステップ {steps}: カバレッジ {coverage:.1f}%, "
                  f"最大訪問回数 {max_visits}, 温度 {navigator.temperature:.3f}")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    else:
        print(f"\n⏱️ {max_steps}ステップで終了")
        
    # 最終統計
    unique_positions = len(navigator.position_visit_counts)
    coverage = unique_positions / (maze_size * maze_size) * 100
    max_visits = max(navigator.position_visit_counts.values()) if navigator.position_visit_counts else 0
    
    print(f"\n最終統計:")
    print(f"  カバレッジ: {coverage:.1f}%（目標60-70%）")
    print(f"  最大訪問回数: {max_visits}（目標<80）")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    
    return navigator, maze


def ablation_study():
    """アブレーション実験（レビューの提案）"""
    print("\n\n=== アブレーション実験 ===")
    
    results = {}
    
    # 1. 完全版（GED × IG）
    print("\n1. GED × IG（完全版）:")
    nav1, _ = run_gradient_experiment(maze_size=15)
    results['full'] = len(nav1.position_visit_counts) / 225 * 100
    
    # 2. GEDのみ
    print("\n2. GEDのみ:")
    # IG計算を無効化する簡易実装...
    
    # 3. IGのみ
    print("\n3. IGのみ:")
    # GED計算を無効化する簡易実装...
    
    print("\n=== 結果比較 ===")
    for variant, coverage in results.items():
        print(f"{variant}: {coverage:.1f}%")


if __name__ == "__main__":
    # メイン実験
    navigator, maze = run_gradient_experiment(maze_size=20)
    
    # アブレーション実験（オプション）
    # ablation_study()