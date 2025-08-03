#!/usr/bin/env python3
"""最終版geDIGナビゲーター：訪問回数を正当にノード属性として管理"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


@dataclass
class FinalGedigEpisode:
    """最終版geDIGエピソード（訪問回数を正当な属性として保持）"""
    episode_type: str
    content: Dict
    vector: np.ndarray
    node_id: int
    
    # geDIG理論の核心
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    
    # 時間情報
    timestamp: int = 0
    decay_factor: float = 1.0
    
    # 訪問情報（エージェント自身の観測）
    visit_count: int = 0
    first_visit_step: Optional[int] = None
    last_visit_step: Optional[int] = None
    
    def update_visit(self, current_step: int):
        """訪問情報を更新（エージェントの内部状態として）"""
        self.visit_count += 1
        if self.first_visit_step is None:
            self.first_visit_step = current_step
        self.last_visit_step = current_step
        
    def get_effective_gedig(self, current_time: int, decay_rate: float = 0.99) -> float:
        """時間減衰と訪問減衰を考慮した実効geDIG値"""
        # 時間減衰
        time_diff = current_time - self.timestamp
        time_decay = decay_rate ** time_diff
        
        # 訪問減衰（内発的な"飽き"のモデル）
        visit_decay = 1.0 / (1.0 + math.log(1 + self.visit_count))
        
        # 両方を考慮
        self.decay_factor = time_decay * visit_decay
        return self.gedig_value * self.decay_factor


class FinalGedigNavigator:
    """最終版geDIGナビゲーター（理論的に正当なループ防止）"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[FinalGedigEpisode] = []
        self.episode_counter = 0
        self.current_time = 0
        self.current_step = 0
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # 位置ごとのエピソード（高速検索用）
        self.position_episodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # 最適化されたパラメータ
        self.structural_weight = 1.0      # ΔGED重み
        self.information_weight = 5.0     # ΔIG重み（適度に強化）
        self.visit_decay_beta = 0.8       # 訪問減衰の強さ
        self.decay_rate = 0.995
        
        # 温度アニーリング
        self.initial_temperature = 1.0
        self.temperature_decay = 0.998
        self.current_temperature = self.initial_temperature
        
        # エージェント自身の観測統計
        self.position_visit_counts = defaultdict(int)  # 位置ごとの訪問回数
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
    def calculate_final_ged(self, new_episode: FinalGedigEpisode) -> float:
        """最終版GED計算"""
        if not self.episodes:
            return 1.0
            
        if new_episode.episode_type == "movement":
            from_pos = new_episode.content['from']
            to_pos = new_episode.content['to']
            
            # 構造的新規性の基本計算
            existing_transitions = self.transition_counts[from_pos]
            total_transitions = sum(existing_transitions.values())
            
            if total_transitions == 0:
                structural_novelty = 1.0
            else:
                # この特定の遷移の頻度
                this_transition_count = existing_transitions[to_pos]
                transition_prob = this_transition_count / total_transitions
                
                # 頻度が低い遷移ほど新規性が高い
                structural_novelty = 1.0 - transition_prob
                
            # グラフ全体での位置の重要性
            if len(self.position_episodes[from_pos]) == 0:
                position_importance = 1.0  # 完全に新しい位置
            else:
                # 接続度に基づく重要性
                degree = len(set(ep.content['to'] for ep_id in self.position_episodes[from_pos]
                               for ep in [self.episodes[ep_id]] 
                               if ep.episode_type == "movement"))
                position_importance = 1.0 / (1.0 + math.log(1 + degree))
                
            return structural_novelty * 0.7 + position_importance * 0.3
            
        return 0.5
        
    def calculate_final_ig(self, new_episode: FinalGedigEpisode) -> float:
        """最終版IG計算（訪問回数による自然な減衰）"""
        if new_episode.episode_type != "movement":
            return 0.1
            
        from_pos = new_episode.content['from']
        to_pos = new_episode.content['to']
        
        # エージェント自身が観測した訪問回数
        from_visit_count = self.position_visit_counts[from_pos]
        to_visit_count = self.position_visit_counts[to_pos]
        
        # 基本的な情報利得（未知の度合い）
        if from_visit_count == 0:
            base_ig = 1.0  # 初訪問は最大価値
        else:
            # 訪問回数に基づく"飽き"のモデル
            # IG_tilde = IG / (1 + log(1 + v))^beta
            visit_factor = (1 + math.log(1 + from_visit_count)) ** self.visit_decay_beta
            base_ig = 1.0 / visit_factor
            
        # 遷移先の新規性ボーナス
        if to_visit_count == 0:
            destination_bonus = 0.5  # 新しい場所への到達
        else:
            destination_bonus = 0.1 / (1 + math.log(1 + to_visit_count))
            
        # 結果タイプによる調整
        if new_episode.content['result'] == "行き止まり":
            # 行き止まりの発見は価値があるが、再訪は価値が低い
            dead_end_value = 1.5 / (1 + from_visit_count * 0.5)
            result_multiplier = dead_end_value
        elif new_episode.content['result'] == "壁":
            result_multiplier = 1.2
        else:
            result_multiplier = 1.0
            
        # 最終的なIG
        ig = (base_ig + destination_bonus) * result_multiplier
        
        return max(0.0, ig)
        
    def add_episode(self, episode_type: str, content: Dict) -> FinalGedigEpisode:
        """エピソード追加（訪問情報を正当に管理）"""
        # ベクトル表現
        if episode_type == "goal_info":
            pos = content['position']
            vector = np.array([pos[0], pos[1], 100.0])
        else:
            from_pos = content['from']
            to_pos = content['to']
            result_val = 1.0 if content['result'] == "成功" else -1.0
            
            # 訪問回数も特徴量に含める（エージェントの観測情報として）
            from_visits = self.position_visit_counts[from_pos]
            to_visits = self.position_visit_counts[to_pos]
            
            vector = np.array([
                from_pos[0], from_pos[1],
                to_pos[0], to_pos[1],
                result_val,
                content['action'],
                math.log(1 + from_visits),  # 訪問情報も特徴量の一部
                math.log(1 + to_visits)
            ])
            
        episode = FinalGedigEpisode(
            episode_type=episode_type,
            content=content,
            vector=vector,
            node_id=self.episode_counter,
            timestamp=self.current_time
        )
        
        # geDIG計算
        episode.ged_delta = self.calculate_final_ged(episode)
        episode.ig_delta = self.calculate_final_ig(episode)
        
        # geDIG = ΔGED × ΔIG（乗算が理論の核心）
        episode.gedig_value = (
            self.structural_weight * episode.ged_delta * 
            self.information_weight * episode.ig_delta
        )
        
        # エピソード追加
        self.episodes.append(episode)
        self.episode_graph.add_node(episode.node_id)
        
        if episode_type == "movement":
            from_pos = content['from']
            to_pos = content['to']
            
            # エージェントの観測統計を更新
            self.position_visit_counts[from_pos] += 1
            if content['result'] == "成功":
                self.position_visit_counts[to_pos] += 1
            self.transition_counts[from_pos][to_pos] += 1
            
            # エピソードの訪問情報を更新
            episode.update_visit(self.current_step)
            
            # インデックス更新
            self.position_episodes[from_pos].append(self.episode_counter)
            
            # グラフエッジ追加
            if self.episodes and len(self.episodes) > 1:
                for i in range(max(0, len(self.episodes) - 10), len(self.episodes) - 1):
                    prev_ep = self.episodes[i]
                    if (prev_ep.episode_type == "movement" and 
                        prev_ep.content['to'] == from_pos and
                        prev_ep.content['result'] == "成功"):
                        self.episode_graph.add_edge(i, episode.node_id)
                        
        self.episode_counter += 1
        self.current_time += 1
        self.current_step += 1
        
        # 温度アニーリング
        self.current_temperature *= self.temperature_decay
        self.current_temperature = max(0.05, self.current_temperature)
        
        return episode
        
    def decide_action_final(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """最終版行動決定（理論的に正当な探索）"""
        
        # 各行動の期待geDIG値
        action_expected_gedig = {}
        
        for action in possible_actions:
            # この位置・行動の過去エピソード
            relevant_episodes = []
            for ep_id in self.position_episodes[current_pos][-20:]:  # 最新20個に限定
                ep = self.episodes[ep_id]
                if (ep.episode_type == "movement" and 
                    ep.content['action'] == action):
                    relevant_episodes.append(ep)
                    
            if relevant_episodes:
                # 実効geDIG値の加重平均（訪問減衰込み）
                total_weight = 0.0
                weighted_gedig = 0.0
                
                for ep in relevant_episodes:
                    effective_gedig = ep.get_effective_gedig(self.current_time, self.decay_rate)
                    weight = ep.decay_factor
                    
                    weighted_gedig += effective_gedig * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    action_expected_gedig[action] = weighted_gedig / total_weight
                else:
                    action_expected_gedig[action] = self.information_weight
            else:
                # 未試行の行動は高価値（でも過度ではない）
                current_visits = self.position_visit_counts[current_pos]
                exploration_value = self.information_weight * (1.0 + 1.0 / (1 + current_visits))
                action_expected_gedig[action] = exploration_value
                
        # ソフトマックス行動選択（温度付き）
        if not action_expected_gedig:
            return np.random.choice(possible_actions)
            
        # 値を正規化して確率に変換
        max_value = max(action_expected_gedig.values())
        exp_values = {}
        for action, value in action_expected_gedig.items():
            # オーバーフロー防止
            normalized_value = (value - max_value) / self.current_temperature
            exp_values[action] = math.exp(min(normalized_value, 50))
            
        total_exp = sum(exp_values.values())
        if total_exp == 0:
            return np.random.choice(possible_actions)
            
        action_probs = {a: v/total_exp for a, v in exp_values.items()}
        
        # 確率的選択
        r = np.random.random()
        cumsum = 0.0
        for action, prob in action_probs.items():
            cumsum += prob
            if r < cumsum:
                return action
                
        return list(action_probs.keys())[-1]


def run_final_experiment(maze_size: int = 20):
    """最終版実験（理論的に正当なループ防止）"""
    print(f"最終版geDIGナビゲーター - {maze_size}×{maze_size}迷路実験")
    print("=" * 60)
    print("特徴：訪問回数をノード属性として正当に管理")
    print("     エージェント自身の観測に基づく'飽き'のモデル")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = FinalGedigNavigator(config)
    
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
    max_steps = maze_size * maze_size * 10
    
    print("\n探索開始...")
    start_time = time.time()
    
    # 進捗記録
    coverage_history = []
    max_visits_history = []
    
    while steps < max_steps:
        current_pos = obs.position
        
        # 行動決定
        action = navigator.decide_action_final(current_pos, obs.possible_moves)
        
        # 行動実行
        old_pos = current_pos
        obs, reward, done, info = maze.step(action)
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
        
        # 進捗表示と記録
        if steps % 100 == 0:
            unique_positions = len(navigator.position_visit_counts)
            coverage = unique_positions / (maze_size * maze_size) * 100
            max_visits = max(navigator.position_visit_counts.values()) if navigator.position_visit_counts else 0
            
            coverage_history.append(coverage)
            max_visits_history.append(max_visits)
            
            print(f"ステップ {steps}: カバレッジ {coverage:.1f}%, "
                  f"最大訪問回数 {max_visits}, 温度 {navigator.current_temperature:.3f}")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    else:
        print(f"\n⏱️ {max_steps}ステップで終了")
        
    # 最終統計
    elapsed_time = time.time() - start_time
    unique_positions = len(navigator.position_visit_counts)
    coverage = unique_positions / (maze_size * maze_size) * 100
    max_visits = max(navigator.position_visit_counts.values()) if navigator.position_visit_counts else 0
    
    # ホットスポット分析
    hotspots = sorted(navigator.position_visit_counts.items(), 
                     key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n最終統計:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  ユニーク位置数: {unique_positions}")
    print(f"  カバレッジ: {coverage:.1f}%")
    print(f"  最大訪問回数: {max_visits}")
    print(f"  実行時間: {elapsed_time:.1f}秒")
    print(f"\nホットスポット（上位5箇所）:")
    for pos, count in hotspots:
        print(f"  {pos}: {count}回")
    
    return navigator, maze, coverage_history, max_visits_history


def visualize_final_results(navigator: FinalGedigNavigator, maze: SimpleMaze,
                           coverage_history: List[float], max_visits_history: List[int],
                           save_path: str = 'final_gedig_visualization.png'):
    """最終版の結果を可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 1. 探索マップ（訪問回数の可視化）
    ax = axes[0, 0]
    ax.set_title("Final geDIG: Visit-aware Exploration", fontsize=16)
    ax.set_aspect('equal')
    
    # 訪問回数のヒートマップ
    visit_matrix = np.zeros(maze.size)
    for pos, count in navigator.position_visit_counts.items():
        if 0 <= pos[0] < maze.size[0] and 0 <= pos[1] < maze.size[1]:
            visit_matrix[pos[0], pos[1]] = count
    
    # カラーマップ（訪問回数に応じて色分け）
    im = ax.imshow(visit_matrix, cmap='viridis', alpha=0.8,
                   extent=[-0.5, maze.size[1]-0.5, maze.size[0]-0.5, -0.5])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count', fontsize=12)
    
    # スタートとゴール
    ax.plot(maze.start_pos[1], maze.start_pos[0], 'ws', markersize=15, 
            markeredgecolor='black', markeredgewidth=2, label='Start')
    ax.plot(maze.goal_pos[1], maze.goal_pos[0], 'r*', markersize=20, label='Goal')
    
    ax.set_xlim(-0.5, maze.size[1]-0.5)
    ax.set_ylim(maze.size[0]-0.5, -0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. カバレッジと最大訪問回数の推移
    ax = axes[0, 1]
    ax.set_title("Coverage & Max Visits Over Time", fontsize=16)
    
    steps = np.arange(len(coverage_history)) * 100
    
    # カバレッジ（左軸）
    color = 'tab:blue'
    ax.set_xlabel('Steps')
    ax.set_ylabel('Coverage (%)', color=color)
    ax.plot(steps, coverage_history, color=color, linewidth=2, label='Coverage')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(0, max(coverage_history) * 1.1)
    
    # 最大訪問回数（右軸）
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Max Visit Count', color=color)
    ax2.plot(steps, max_visits_history, color=color, linewidth=2, 
             linestyle='--', label='Max Visits')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Final Coverage: {coverage_history[-1]:.1f}%", fontsize=14)
    
    # 3. 訪問減衰の効果
    ax = axes[0, 2]
    ax.set_title("Visit Decay Effect on geDIG", fontsize=16)
    
    # 訪問回数とgeDIG値の関係
    visit_counts = list(range(0, 21))
    decay_factors = [1.0 / (1.0 + math.log(1 + v)) for v in visit_counts]
    
    ax.plot(visit_counts, decay_factors, 'g-', linewidth=3, label='Visit Decay Factor')
    ax.fill_between(visit_counts, 0, decay_factors, alpha=0.3, color='green')
    ax.set_xlabel('Visit Count')
    ax.set_ylabel('Decay Factor')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. エピソードの時系列分析
    ax = axes[1, 0]
    ax.set_title("Episode Values Timeline", fontsize=16)
    
    movement_episodes = [ep for ep in navigator.episodes if ep.episode_type == "movement"]
    if movement_episodes:
        indices = list(range(len(movement_episodes)))
        ged_values = [ep.ged_delta for ep in movement_episodes]
        ig_values = [ep.ig_delta for ep in movement_episodes]
        gedig_values = [ep.gedig_value for ep in movement_episodes]
        
        # 移動平均
        window = 50
        if len(indices) > window:
            ged_ma = np.convolve(ged_values, np.ones(window)/window, mode='valid')
            ig_ma = np.convolve(ig_values, np.ones(window)/window, mode='valid')
            gedig_ma = np.convolve(gedig_values, np.ones(window)/window, mode='valid')
            ma_indices = indices[window//2:-window//2+1]
            
            ax.plot(ma_indices, ged_ma, 'b-', alpha=0.7, label=f'ΔGED (MA{window})')
            ax.plot(ma_indices, ig_ma, 'r-', alpha=0.7, label=f'ΔIG (MA{window})')
            ax.plot(ma_indices, gedig_ma, 'g-', alpha=0.7, label=f'geDIG (MA{window})')
        
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. 位置の訪問分布
    ax = axes[1, 1]
    ax.set_title("Visit Count Distribution", fontsize=16)
    
    visit_counts = list(navigator.position_visit_counts.values())
    if visit_counts:
        ax.hist(visit_counts, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(visit_counts), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(visit_counts):.1f}')
        ax.axvline(np.median(visit_counts), color='orange', linestyle='--', 
                  label=f'Median: {np.median(visit_counts):.1f}')
        ax.set_xlabel('Visit Count')
        ax.set_ylabel('Number of Positions')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. 最終サマリー
    ax = axes[1, 2]
    ax.set_title("Final Summary", fontsize=16)
    ax.axis('off')
    
    # 統計サマリー
    unique_positions = len(navigator.position_visit_counts)
    coverage = unique_positions / (maze.size[0] * maze.size[1]) * 100
    max_visits = max(navigator.position_visit_counts.values()) if navigator.position_visit_counts else 0
    
    # 改善効果の計算（仮想的な比較）
    improvement_text = f"""
Final geDIG Navigator Results
=============================
✅ Theoretical Validity: Visit counts as node attributes
✅ No External Cheats: Agent's own observations only

Performance Metrics:
• Coverage: {coverage:.1f}% 
• Unique Positions: {unique_positions}
• Max Visit Count: {max_visits}
• Episodes: {len(navigator.episodes)}

Key Features:
• Visit Decay: 1/(1+log(1+v))^{navigator.visit_decay_beta}
• Temperature: {navigator.initial_temperature} → {navigator.current_temperature:.3f}
• IG Weight: {navigator.information_weight}x

Theoretical Justification:
• Count-based exploration (Bellemare et al. 2016)
• Intrinsic motivation via novelty decay
• Self-observed statistics only
"""
    
    ax.text(0.05, 0.95, improvement_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {save_path} として保存しました")
    return fig


if __name__ == "__main__":
    navigator, maze, coverage_history, max_visits_history = run_final_experiment(maze_size=20)
    visualize_final_results(navigator, maze, coverage_history, max_visits_history)