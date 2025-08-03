#!/usr/bin/env python3
"""純粋なgeDIG理論ナビゲーター：ループ検出なし、理論のみで動作"""

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
class PureGedigEpisode:
    """純粋なgeDIG理論のエピソード"""
    episode_type: str
    content: Dict
    vector: np.ndarray
    node_id: int
    
    # geDIG理論の核心
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0
    
    # 時間減衰（理論的な忘却）
    timestamp: int = 0
    decay_factor: float = 1.0
    
    def get_effective_gedig(self, current_time: int, decay_rate: float = 0.99) -> float:
        """時間減衰を考慮した実効geDIG値"""
        time_diff = current_time - self.timestamp
        self.decay_factor = decay_rate ** time_diff
        return self.gedig_value * self.decay_factor


class PureGedigNavigator:
    """純粋なgeDIG理論に基づくナビゲーター（チートなし）"""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.episodes: List[PureGedigEpisode] = []
        self.episode_counter = 0
        self.current_time = 0
        
        # グラフ構造
        self.episode_graph = nx.DiGraph()
        
        # 位置ごとのエピソード（高速検索用）
        self.position_episodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # geDIG理論パラメータ
        self.structural_weight = 1.0  # 構造的新規性の重み
        self.information_weight = 1.0  # 情報利得の重み
        self.decay_rate = 0.995  # 時間減衰率
        
        # 純粋な統計（チートではない）
        self.state_transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_visit_counts = defaultdict(int)
        
    def calculate_pure_ged(self, new_episode: PureGedigEpisode) -> float:
        """純粋なグラフ編集距離の変化"""
        if not self.episodes:
            return 1.0
            
        # 新しいエピソードがもたらす構造変化を評価
        if new_episode.episode_type == "movement":
            from_pos = new_episode.content['from']
            
            # この位置からの既存エッジ数
            existing_edges = len([ep for ep_id in self.position_episodes[from_pos] 
                                for ep in [self.episodes[ep_id]] 
                                if ep.episode_type == "movement"])
            
            # 構造的新規性：新しいエッジほど価値が高い
            if existing_edges == 0:
                return 1.0  # 完全に新しい位置からの移動
            else:
                # エッジ数が多いほど新規性は低い（対数的減衰）
                return 1.0 / (1.0 + math.log(existing_edges + 1))
                
        return 0.5  # デフォルト値
        
    def calculate_pure_ig(self, new_episode: PureGedigEpisode) -> float:
        """純粋な情報利得の計算"""
        if new_episode.episode_type != "movement":
            return 0.1
            
        from_pos = new_episode.content['from']
        to_pos = new_episode.content['to']
        
        # 状態遷移の不確実性（エントロピー）を計算
        total_transitions = sum(self.state_transition_counts[from_pos].values())
        
        if total_transitions == 0:
            # 未知の状態は最大の情報利得
            return 1.0
            
        # 事前エントロピー
        prior_entropy = 0.0
        for next_state, count in self.state_transition_counts[from_pos].items():
            p = count / total_transitions
            if p > 0:
                prior_entropy -= p * math.log(p)
                
        # この遷移を追加した後のエントロピー
        new_counts = self.state_transition_counts[from_pos].copy()
        new_counts[to_pos] = new_counts.get(to_pos, 0) + 1
        new_total = total_transitions + 1
        
        posterior_entropy = 0.0
        for next_state, count in new_counts.items():
            p = count / new_total
            if p > 0:
                posterior_entropy -= p * math.log(p)
                
        # 情報利得 = エントロピーの減少
        ig = prior_entropy - posterior_entropy
        
        # 結果の価値による調整
        if new_episode.content['result'] == "行き止まり":
            ig *= 2.0  # 行き止まりの発見は価値が高い
        elif new_episode.content['result'] == "壁":
            ig *= 1.5  # 壁の発見も重要
            
        return max(0.0, ig)
        
    def add_episode(self, episode_type: str, content: Dict) -> PureGedigEpisode:
        """エピソードを追加（純粋な理論計算）"""
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
            
        episode = PureGedigEpisode(
            episode_type=episode_type,
            content=content,
            vector=vector,
            node_id=self.episode_counter,
            timestamp=self.current_time
        )
        
        # 純粋なgeDIG計算
        episode.ged_delta = self.calculate_pure_ged(episode)
        episode.ig_delta = self.calculate_pure_ig(episode)
        
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
            self.position_episodes[from_pos].append(self.episode_counter)
            
            # 統計更新
            self.state_transition_counts[from_pos][to_pos] += 1
            self.state_visit_counts[from_pos] += 1
            
            # グラフエッジ追加
            if self.episodes and len(self.episodes) > 1:
                # 時間的に近いエピソードを接続
                for i in range(max(0, len(self.episodes) - 10), len(self.episodes) - 1):
                    prev_ep = self.episodes[i]
                    if (prev_ep.episode_type == "movement" and 
                        prev_ep.content['to'] == from_pos and
                        prev_ep.content['result'] == "成功"):
                        self.episode_graph.add_edge(i, episode.node_id)
                        
        self.episode_counter += 1
        self.current_time += 1
        
        return episode
        
    def decide_action_pure(self, current_pos: Tuple[int, int], possible_actions: List[int]) -> int:
        """純粋なgeDIG理論に基づく行動決定（チートなし）"""
        
        # 各行動の期待geDIG値を計算
        action_expected_gedig = {}
        
        for action in possible_actions:
            # この位置・行動の過去エピソードを検索
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
                    
                    weighted_gedig += effective_gedig * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    action_expected_gedig[action] = weighted_gedig / total_weight
                else:
                    action_expected_gedig[action] = 1.0  # 未知は高価値
            else:
                # 未試行の行動は最大の期待値
                action_expected_gedig[action] = self.structural_weight * self.information_weight
                
        # ボルツマン分布による確率的選択（温度パラメータで探索を制御）
        temperature = 0.5  # 低いほど最適行動を選びやすい
        
        # 期待値を確率に変換
        exp_values = {}
        for action, gedig in action_expected_gedig.items():
            # geDIG値が高いほど選ばれやすい
            exp_values[action] = math.exp(gedig / temperature)
            
        # 正規化
        total_exp = sum(exp_values.values())
        action_probs = {a: v/total_exp for a, v in exp_values.items()}
        
        # 確率的に選択
        r = np.random.random()
        cumsum = 0.0
        for action, prob in action_probs.items():
            cumsum += prob
            if r < cumsum:
                return action
                
        # フォールバック
        return np.random.choice(possible_actions)
        
    def update_gedig_values_globally(self):
        """全エピソードのgeDIG値を再計算（グローバル最適化）"""
        # PageRank的なアプローチで価値を伝播
        if len(self.episode_graph) == 0:
            return
            
        # 初期値設定
        pagerank_scores = nx.pagerank(self.episode_graph, alpha=0.85)
        
        # PageRankスコアでgeDIG値を調整
        for node_id, score in pagerank_scores.items():
            if node_id < len(self.episodes):
                episode = self.episodes[node_id]
                # 元のgeDIG値とPageRankスコアを組み合わせ
                episode.gedig_value = episode.gedig_value * (1 + score)


def run_pure_gedig_experiment(maze_size: int = 20):
    """純粋なgeDIG理論実験"""
    print(f"純粋geDIG理論ナビゲーター - {maze_size}×{maze_size}迷路実験")
    print("=" * 60)
    print("チートなし：ループ検出も外部ヒューリスティックも使用しません")
    print("-" * 60)
    
    config = MazeNavigatorConfig()
    navigator = PureGedigNavigator(config)
    
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
    max_steps = maze_size * maze_size * 20
    
    print("\n探索開始...")
    start_time = time.time()
    
    while steps < max_steps:
        current_pos = obs.position
        
        # 純粋な理論に基づく行動決定
        action = navigator.decide_action_pure(current_pos, obs.possible_moves)
        
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
        
        # 100ステップごとにグローバル最適化
        if steps % 100 == 0:
            navigator.update_gedig_values_globally()
            
        # 進捗表示
        if steps % 500 == 0:
            coverage = len(set(ep.content.get('to', (0,0)) for ep in navigator.episodes 
                             if ep.episode_type == "movement")) / (maze_size * maze_size) * 100
            print(f"ステップ {steps}: カバレッジ {coverage:.1f}%, "
                  f"平均geDIG {np.mean([ep.gedig_value for ep in navigator.episodes[-100:]]):.3f}")
            
        # ゴール到達
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ ゴール到達！ステップ数: {steps}")
            break
            
    else:
        print(f"\n⏱️ {max_steps}ステップで終了")
        
    # 統計
    elapsed_time = time.time() - start_time
    print(f"\n統計:")
    print(f"  総エピソード数: {len(navigator.episodes)}")
    print(f"  実行時間: {elapsed_time:.1f}秒")
    print(f"  平均ΔGED: {np.mean([ep.ged_delta for ep in navigator.episodes]):.3f}")
    print(f"  平均ΔIG: {np.mean([ep.ig_delta for ep in navigator.episodes]):.3f}")
    
    return navigator, maze


def visualize_pure_gedig_results(navigator: PureGedigNavigator, maze: SimpleMaze, 
                                save_path: str = 'pure_gedig_visualization.png'):
    """純粋geDIG理論の結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # 1. 探索経路と訪問頻度
    ax = axes[0, 0]
    ax.set_title("Pure geDIG Exploration (No Loop Detection)", fontsize=16)
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
    
    # 対数スケールで表示（頻繁な訪問を強調）
    visit_matrix_log = np.log1p(visit_matrix)
    im = ax.imshow(visit_matrix_log, cmap='hot', alpha=0.8, 
                   extent=[-0.5, maze.size[1]-0.5, maze.size[0]-0.5, -0.5])
    plt.colorbar(im, ax=ax, label='log(Visit Count + 1)')
    
    # スタートとゴール
    ax.plot(maze.start_pos[1], maze.start_pos[0], 'bs', markersize=15, label='Start')
    ax.plot(maze.goal_pos[1], maze.goal_pos[0], 'g*', markersize=20, label='Goal')
    
    ax.set_xlim(-0.5, maze.size[1]-0.5)
    ax.set_ylim(maze.size[0]-0.5, -0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. geDIG値の時系列変化
    ax = axes[0, 1]
    ax.set_title("geDIG Values Over Time (with Decay)", fontsize=16)
    
    # 各エピソードの実効geDIG値を計算
    episodes_data = []
    for i, ep in enumerate(navigator.episodes):
        if ep.episode_type == "movement":
            effective_gedig = ep.get_effective_gedig(navigator.current_time, navigator.decay_rate)
            episodes_data.append((i, ep.gedig_value, effective_gedig, ep.decay_factor))
    
    if episodes_data:
        indices, original_gedig, effective_gedig, decay_factors = zip(*episodes_data)
        
        # 元のgeDIG値と減衰後の値を表示
        ax.plot(indices, original_gedig, 'b-', alpha=0.3, label='Original geDIG')
        ax.plot(indices, effective_gedig, 'r-', alpha=0.7, label='Effective geDIG (with decay)')
        
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("geDIG Value")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. 位置ごとの探索パターン
    ax = axes[1, 0]
    ax.set_title("Position Visit Patterns (Hotspots)", fontsize=16)
    
    # 最も頻繁に訪問された位置トップ10
    top_positions = sorted(position_visits.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_positions:
        positions, counts = zip(*top_positions)
        y_pos = np.arange(len(positions))
        
        bars = ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{pos}" for pos in positions])
        ax.set_xlabel("Visit Count")
        ax.set_title(f"Top 10 Most Visited Positions (Total unique: {len(position_visits)})")
        
        # 色分け（訪問回数に応じて）
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.Reds(counts[i] / max(counts)))
    
    # 4. 理論的指標の分析
    ax = axes[1, 1]
    ax.set_title("Theoretical Metrics Analysis", fontsize=16)
    ax.axis('off')
    
    # 統計情報
    movement_episodes = [ep for ep in navigator.episodes if ep.episode_type == "movement"]
    
    if movement_episodes:
        # geDIG成分の分析
        ged_values = [ep.ged_delta for ep in movement_episodes]
        ig_values = [ep.ig_delta for ep in movement_episodes]
        gedig_values = [ep.gedig_value for ep in movement_episodes]
        
        # ループ検出の代替指標
        position_entropy = -sum((count/sum(position_visits.values())) * 
                               math.log(count/sum(position_visits.values())) 
                               for count in position_visits.values() if count > 0)
        
        stats_text = f"""
Pure geDIG Theory Statistics (No Cheats!)
=========================================
Total Episodes: {len(navigator.episodes)}
Movement Episodes: {len(movement_episodes)}
Unique Positions: {len(position_visits)}
Coverage: {len(position_visits) / (maze.size[0] * maze.size[1]) * 100:.1f}%

Average ΔGED: {np.mean(ged_values):.4f}
Average ΔIG: {np.mean(ig_values):.4f}
Average geDIG: {np.mean(gedig_values):.4f}

Position Entropy: {position_entropy:.3f}
(Higher = more exploration diversity)

Decay Rate: {navigator.decay_rate}
Temperature (Boltzmann): 0.5

Top 3 Hotspots (potential loops):
{chr(10).join(f"  {pos}: {count} visits" for pos, count in top_positions[:3])}
"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {save_path} として保存しました")
    return fig


if __name__ == "__main__":
    navigator, maze = run_pure_gedig_experiment(maze_size=20)
    visualize_pure_gedig_results(navigator, maze)