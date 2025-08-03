#!/usr/bin/env python3
"""バランス調整版geDIGナビゲーター：ΔIG強化＋ループ抑止"""

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
class BalancedGedigEpisode:
    """バランス調整版geDIGエピソード"""
    episode_type: str
    content: Dict
    vector: np.ndarray
    node_id: int
    
    # geDIG理論の核心
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    
    # 時間減衰
    timestamp: int = 0
    decay_factor: float = 1.0
    
    # ループ抑止用
    position_visit_count: int = 0
    
    def get_effective_gedig(self, current_time: int, decay_rate: float = 0.99) -> float:
        """時間減衰を考慮した実効geDIG値"""
        time_diff = current_time - self.timestamp
        self.decay_factor = decay_rate ** time_diff
        return self.gedig_value * self.decay_factor


class BalancedGedigNavigator:
    """バランス調整版geDIGナビゲーター"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[BalancedGedigEpisode] = []
        self.episode_counter = 0
        self.current_time = 0
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # 位置ごとのエピソード
        self.position_episodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # バランス調整されたパラメータ
        self.structural_weight = 1.0      # ΔGED重み
        self.information_weight = 10.0    # ΔIG重み（10倍に強化！）
        self.loop_penalty_factor = 5.0    # ループペナルティ係数
        self.decay_rate = 0.995
        
        # 温度アニーリング
        self.initial_temperature = 1.0
        self.temperature_decay = 0.995
        self.current_temperature = self.initial_temperature
        
        # 統計
        self.state_transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_visit_counts = defaultdict(int)
        
    def calculate_balanced_ged(self, new_episode: BalancedGedigEpisode) -> float:
        """バランス調整されたGED計算"""
        if not self.episodes:
            return 1.0
            
        if new_episode.episode_type == "movement":
            from_pos = new_episode.content['from']
            
            # 既存エッジ数
            existing_edges = len([ep for ep_id in self.position_episodes[from_pos] 
                                for ep in [self.episodes[ep_id]] 
                                if ep.episode_type == "movement"])
            
            # 構造的新規性
            if existing_edges == 0:
                structural_novelty = 1.0
            else:
                structural_novelty = 1.0 / (1.0 + math.log(existing_edges + 1))
                
            # 探索範囲ボーナス（スタートからの距離）
            start_pos = (1, 1)  # 仮定
            distance_from_start = abs(from_pos[0] - start_pos[0]) + abs(from_pos[1] - start_pos[1])
            exploration_bonus = min(0.3, distance_from_start / 50.0)
            
            return structural_novelty + exploration_bonus
            
        return 0.5
        
    def calculate_balanced_ig(self, new_episode: BalancedGedigEpisode) -> float:
        """バランス調整されたIG計算（強化版）"""
        if new_episode.episode_type != "movement":
            return 0.1
            
        from_pos = new_episode.content['from']
        to_pos = new_episode.content['to']
        visit_count = self.state_visit_counts[from_pos]
        
        # ループ抑止：訪問回数が多いほどペナルティ
        if visit_count > 5:
            loop_penalty = -self.loop_penalty_factor * math.log(visit_count)
        else:
            loop_penalty = 0
            
        # 基本的な情報利得
        total_transitions = sum(self.state_transition_counts[from_pos].values())
        
        if total_transitions == 0:
            base_ig = 1.0  # 未知は高価値
        else:
            # エントロピー計算
            prior_entropy = 0.0
            for next_state, count in self.state_transition_counts[from_pos].items():
                p = count / total_transitions
                if p > 0:
                    prior_entropy -= p * math.log(p)
                    
            # 新しい遷移後のエントロピー
            new_counts = self.state_transition_counts[from_pos].copy()
            new_counts[to_pos] = new_counts.get(to_pos, 0) + 1
            new_total = total_transitions + 1
            
            posterior_entropy = 0.0
            for next_state, count in new_counts.items():
                p = count / new_total
                if p > 0:
                    posterior_entropy -= p * math.log(p)
                    
            base_ig = prior_entropy - posterior_entropy
            
        # 結果による調整
        if new_episode.content['result'] == "行き止まり":
            result_bonus = 2.0  # 行き止まり発見は価値大
        elif new_episode.content['result'] == "壁":
            result_bonus = 1.5
        else:
            # 新規位置への到達
            if to_pos not in self.state_visit_counts or self.state_visit_counts[to_pos] == 0:
                result_bonus = 3.0  # 新規位置は超高価値！
            else:
                result_bonus = 1.0
                
        # 最終的なIG = 基本IG × 結果ボーナス + ループペナルティ
        ig = base_ig * result_bonus + loop_penalty
        
        return max(0.0, ig)
        
    def add_episode(self, episode_type: str, content: Dict) -> BalancedGedigEpisode:
        """エピソード追加（バランス調整版）"""
        # ベクトル表現
        if episode_type == "goal_info":
            pos = content['position']
            vector = np.array([pos[0], pos[1], 100.0])
        else:
            from_pos = content['from']
            to_pos = content['to']
            result_val = 1.0 if content['result'] == "成功" else -1.0
            vector = np.array([
                from_pos[0], from_pos[1],
                to_pos[0], to_pos[1],
                result_val,
                content['action']
            ])
            
        episode = BalancedGedigEpisode(
            episode_type=episode_type,
            content=content,
            vector=vector,
            node_id=self.episode_counter,
            timestamp=self.current_time,
            position_visit_count=self.state_visit_counts.get(content.get('from', (0,0)), 0)
        )
        
        # バランス調整されたgeDIG計算
        episode.ged_delta = self.calculate_balanced_ged(episode)
        episode.ig_delta = self.calculate_balanced_ig(episode)
        
        # geDIG = ΔGED × ΔIG（重み付き）
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
            self.position_episodes[from_pos].append(self.episode_counter)
            
            # 統計更新
            self.state_transition_counts[from_pos][to_pos] += 1
            self.state_visit_counts[from_pos] += 1
            
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
        
        # 温度アニーリング
        self.current_temperature *= self.temperature_decay
        self.current_temperature = max(0.1, self.current_temperature)  # 最小温度
        
        return episode
        
    def decide_action_balanced(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """バランス調整された行動決定"""
        
        # 各行動の期待geDIG値を計算
        action_expected_gedig = {}
        
        for action in possible_actions:
            # 関連エピソードを検索
            relevant_episodes = []
            for ep_id in self.position_episodes[current_pos]:
                ep = self.episodes[ep_id]
                if (ep.episode_type == "movement" and 
                    ep.content['action'] == action):
                    relevant_episodes.append(ep)
                    
            if relevant_episodes:
                # 時間減衰を考慮した加重平均
                total_weight = 0.0
                weighted_gedig = 0.0
                
                for ep in relevant_episodes:
                    effective_gedig = ep.get_effective_gedig(self.current_time, self.decay_rate)
                    weight = ep.decay_factor
                    
                    # 訪問回数によるペナルティも考慮
                    if ep.position_visit_count > 10:
                        effective_gedig *= 0.5  # 頻繁な訪問は価値半減
                        
                    weighted_gedig += effective_gedig * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    action_expected_gedig[action] = weighted_gedig / total_weight
                else:
                    action_expected_gedig[action] = self.information_weight
            else:
                # 未試行は超高価値
                action_expected_gedig[action] = self.structural_weight * self.information_weight * 2.0
                
        # アニーリングされたボルツマン分布
        exp_values = {}
        for action, gedig in action_expected_gedig.items():
            exp_values[action] = math.exp(gedig / self.current_temperature)
            
        # 正規化
        total_exp = sum(exp_values.values())
        action_probs = {a: v/total_exp for a, v in exp_values.items()}
        
        # 確率的選択
        r = np.random.random()
        cumsum = 0.0
        for action, prob in action_probs.items():
            cumsum += prob
            if r < cumsum:
                return action
                
        return np.random.choice(possible_actions)


def run_balanced_experiment(maze_size: int = 20):
    """バランス調整版実験"""
    print(f"バランス調整版geDIGナビゲーター - {maze_size}×{maze_size}迷路実験")
    print("=" * 60)
    print("改善点：ΔIG×10倍、ループペナルティ、温度アニーリング")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = BalancedGedigNavigator(config)
    
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
    
    while steps < max_steps:
        current_pos = obs.position
        
        # バランス調整された行動決定
        action = navigator.decide_action_balanced(current_pos, obs.possible_moves)
        
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
        
        # 進捗表示
        if steps % 200 == 0:
            unique_positions = set(ep.content.get('to', (0,0)) for ep in navigator.episodes 
                                 if ep.episode_type == "movement")
            coverage = len(unique_positions) / (maze_size * maze_size) * 100
            max_visits = max(navigator.state_visit_counts.values()) if navigator.state_visit_counts else 0
            print(f"ステップ {steps}: カバレッジ {coverage:.1f}%, "
                  f"最大訪問回数 {max_visits}, 温度 {navigator.current_temperature:.3f}")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    else:
        print(f"\n⏱️ {max_steps}ステップで終了")
        
    # 統計
    elapsed_time = time.time() - start_time
    unique_positions = set(ep.content.get('to', (0,0)) for ep in navigator.episodes 
                         if ep.episode_type == "movement")
    max_visits = max(navigator.state_visit_counts.values()) if navigator.state_visit_counts else 0
    
    print(f"\n統計:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  ユニーク位置数: {len(unique_positions)}")
    print(f"  カバレッジ: {len(unique_positions) / (maze_size * maze_size) * 100:.1f}%")
    print(f"  最大訪問回数: {max_visits}")
    print(f"  実行時間: {elapsed_time:.1f}秒")
    
    return navigator, maze


def visualize_balanced_results(navigator: BalancedGedigNavigator, maze: SimpleMaze, 
                              save_path: str = 'balanced_gedig_visualization.png'):
    """バランス版の結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # 1. 探索パターン比較
    ax = axes[0, 0]
    ax.set_title("Balanced geDIG Exploration (ΔIG×10 + Loop Penalty)", fontsize=16)
    ax.set_aspect('equal')
    
    # 訪問頻度のヒートマップ
    visit_matrix = np.zeros(maze.size)
    position_visits = defaultdict(int)
    
    for ep in navigator.episodes:
        if ep.episode_type == "movement":
            to_pos = ep.content['to']
            position_visits[to_pos] += 1
            if 0 <= to_pos[0] < maze.size[0] and 0 <= to_pos[1] < maze.size[1]:
                visit_matrix[to_pos[0], to_pos[1]] += 1
    
    # クリップして表示（最大10回）
    visit_matrix_clipped = np.clip(visit_matrix, 0, 10)
    im = ax.imshow(visit_matrix_clipped, cmap='viridis', alpha=0.8, 
                   extent=[-0.5, maze.size[1]-0.5, maze.size[0]-0.5, -0.5])
    plt.colorbar(im, ax=ax, label='Visit Count (capped at 10)')
    
    # スタートとゴール
    ax.plot(maze.start_pos[1], maze.start_pos[0], 'bs', markersize=15, label='Start')
    ax.plot(maze.goal_pos[1], maze.goal_pos[0], 'g*', markersize=20, label='Goal')
    
    ax.set_xlim(-0.5, maze.size[1]-0.5)
    ax.set_ylim(maze.size[0]-0.5, -0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. geDIG値の分布比較
    ax = axes[0, 1]
    ax.set_title("geDIG Components Distribution", fontsize=16)
    
    movement_episodes = [ep for ep in navigator.episodes if ep.episode_type == "movement"]
    if movement_episodes:
        # ΔGED、ΔIG、geDIGの分布を並べて表示
        ged_values = [ep.ged_delta for ep in movement_episodes]
        ig_values = [ep.ig_delta for ep in movement_episodes]
        gedig_values = [ep.gedig_value for ep in movement_episodes]
        
        ax.hist([ged_values, ig_values, gedig_values], bins=30, 
                label=['ΔGED', 'ΔIG×10', 'geDIG'], alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. 温度アニーリングの効果
    ax = axes[1, 0]
    ax.set_title("Temperature Annealing & Exploration", fontsize=16)
    
    # エピソード番号ごとの温度と新規位置発見
    temperatures = []
    new_positions = []
    unique_positions_cumulative = set()
    
    for i, ep in enumerate(navigator.episodes):
        if ep.episode_type == "movement":
            temperatures.append(navigator.initial_temperature * (navigator.temperature_decay ** i))
            to_pos = ep.content['to']
            if to_pos not in unique_positions_cumulative:
                unique_positions_cumulative.add(to_pos)
                new_positions.append(1)
            else:
                new_positions.append(0)
    
    if temperatures:
        ax2 = ax.twinx()
        ax.plot(temperatures, 'r-', alpha=0.7, label='Temperature')
        ax2.plot(np.cumsum(new_positions), 'b-', alpha=0.7, label='Cumulative Unique Positions')
        
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Temperature", color='r')
        ax2.set_ylabel("Unique Positions", color='b')
        ax.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
    
    # 4. 改善効果の要約
    ax = axes[1, 1]
    ax.set_title("Improvement Summary", fontsize=16)
    ax.axis('off')
    
    # 統計計算
    unique_positions = set(ep.content.get('to', (0,0)) for ep in navigator.episodes 
                         if ep.episode_type == "movement")
    coverage = len(unique_positions) / (maze.size[0] * maze.size[1]) * 100
    max_visits = max(navigator.state_visit_counts.values()) if navigator.state_visit_counts else 0
    
    # ホットスポット分析
    hotspots = sorted(position_visits.items(), key=lambda x: x[1], reverse=True)[:5]
    
    stats_text = f"""
Balanced geDIG Results
=====================
Total Episodes: {len(navigator.episodes)}
Unique Positions: {len(unique_positions)}
Coverage: {coverage:.1f}% ⬆️

Key Improvements:
• ΔIG weight: 1.0 → 10.0 (10x boost)
• Loop penalty: -5×log(visits) when visits > 5
• Temperature annealing: 1.0 → {navigator.current_temperature:.3f}

Hotspot Reduction:
Max visits: {max_visits} (Target: <10)
Top 5 hotspots:
{chr(10).join(f"  {pos}: {count} visits" for pos, count in hotspots[:5])}

Average ΔGED: {np.mean([ep.ged_delta for ep in movement_episodes]):.4f}
Average ΔIG: {np.mean([ep.ig_delta for ep in movement_episodes]):.4f}
Average geDIG: {np.mean([ep.gedig_value for ep in movement_episodes]):.4f}
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {save_path} として保存しました")
    return fig


if __name__ == "__main__":
    navigator, maze = run_balanced_experiment(maze_size=20)
    visualize_balanced_results(navigator, maze)