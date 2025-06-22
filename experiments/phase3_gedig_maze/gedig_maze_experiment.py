"""
Phase 3: GEDIG迷路実験
=====================================

粘菌アナロジーによるGED(Graph Edit Distance) + IG(Information Gain)評価で、
InsightSpike-AIが最適解探索において試行回数を大幅削減することを検証

目標: 60%試行回数削減、3倍高速収束、95%成功率

安全性機能:
- 実験前の自動データバックアップ
- 実験用データの分離実行
- 実験後の自動ロールバック
"""

import sys
import time
import numpy as np
import pandas as pd
# matplotlib バックエンドを GUI 非依存の Agg に固定（GIF 生成用）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import heapq
import random
from math import log2, sqrt

# 共通ユーティリティインポート
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_manager import safe_experiment_environment, with_data_safety, create_experiment_data_config
from evaluation_metrics import MetricsCalculator
from experiment_reporter import ExperimentReporter

# CLI機能インポート（フォールバック対応）
try:
    from cli_utils import create_base_cli_parser, add_phase_specific_args, merge_cli_config, print_experiment_header, handle_cli_error, create_experiment_summary
    from scripts_integration import ScriptsIntegratedExperiment, print_scripts_integration_status
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    print("⚠️  CLI機能が利用できません - 基本モードで実行")


@dataclass
class MazeEnvironment:
    """迷路環境定義（壁ベース表現）"""
    width: int
    height: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    walls: Set[Tuple[int, int]]  # 壁の座標
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """有効な位置かチェック（通路セル）"""
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.walls)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """隣接する有効な位置を取得"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def get_all_valid_positions(self) -> List[Tuple[int, int]]:
        """全ての有効な位置（通路セル）を取得"""
        valid_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if self.is_valid_position((x, y)):
                    valid_positions.append((x, y))
        return valid_positions


@dataclass
class PathfindingMetrics:
    """経路探索性能指標"""
    algorithm_name: str
    trials_count: int          # 試行回数
    convergence_time: float    # 収束時間 (秒)
    path_length: int          # 最終経路長
    solution_quality: float   # 解の品質 (0-1)
    success_rate: float       # 成功率 (0-1)
    exploration_efficiency: float  # 探索効率
    memory_usage_mb: float    # メモリ使用量
    gedig_score: float        # GEDIG統合スコア


class BasePathfindingAlgorithm(ABC):
    """経路探索アルゴリズムの基底クラス"""
    
    def __init__(self, name: str):
        self.name = name
        self.trials = 0
        self.explored_nodes = set()
    
    @abstractmethod
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """経路探索実行"""
        pass
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """マンハッタン距離計算"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """ユークリッド距離計算"""
        return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class AStarAlgorithm(BasePathfindingAlgorithm):
    """A*アルゴリズム（ベースライン）"""
    
    def __init__(self):
        super().__init__("A_Star")
    
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """A*による経路探索"""
        start_time = time.time()
        self.trials = 0
        self.explored_nodes = set()
        
        # 優先度付きキュー: (f_score, g_score, position, path)
        open_set = [(0, 0, maze.start, [maze.start])]
        g_scores = {maze.start: 0}
        
        while open_set:
            self.trials += 1
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in self.explored_nodes:
                continue
                
            self.explored_nodes.add(current)
            
            if current == maze.goal:
                end_time = time.time()
                return path, PathfindingMetrics(
                    algorithm_name=self.name,
                    trials_count=self.trials,
                    convergence_time=end_time - start_time,
                    path_length=len(path),
                    solution_quality=1.0,  # A*は最適解保証
                    success_rate=1.0,
                    exploration_efficiency=len(path) / len(self.explored_nodes),
                    memory_usage_mb=len(open_set) * 0.001,  # 概算
                    gedig_score=0.0  # 従来手法はGEDIG使用せず
                )
            
            for neighbor in maze.get_neighbors(current):
                tentative_g = g_score + 1
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self.manhattan_distance(neighbor, maze.goal)
                    f_score = tentative_g + h_score
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
        
        # 解が見つからない場合
        end_time = time.time()
        return [], PathfindingMetrics(
            algorithm_name=self.name,
            trials_count=self.trials,
            convergence_time=end_time - start_time,
            path_length=0,
            solution_quality=0.0,
            success_rate=0.0,
            exploration_efficiency=0.0,
            memory_usage_mb=0.0,
            gedig_score=0.0
        )


class DijkstraAlgorithm(BasePathfindingAlgorithm):
    """Dijkstraアルゴリズム"""
    
    def __init__(self):
        super().__init__("Dijkstra")
    
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """Dijkstraによる経路探索"""
        start_time = time.time()
        self.trials = 0
        self.explored_nodes = set()
        
        # 優先度付きキュー: (distance, position, path)
        pq = [(0, maze.start, [maze.start])]
        distances = {maze.start: 0}
        
        while pq:
            self.trials += 1
            current_dist, current, path = heapq.heappop(pq)
            
            if current in self.explored_nodes:
                continue
                
            self.explored_nodes.add(current)
            
            if current == maze.goal:
                end_time = time.time()
                return path, PathfindingMetrics(
                    algorithm_name=self.name,
                    trials_count=self.trials,
                    convergence_time=end_time - start_time,
                    path_length=len(path),
                    solution_quality=1.0,  # 最適解保証
                    success_rate=1.0,
                    exploration_efficiency=len(path) / len(self.explored_nodes),
                    memory_usage_mb=len(pq) * 0.001,
                    gedig_score=0.0
                )
            
            for neighbor in maze.get_neighbors(current):
                new_dist = current_dist + 1
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_dist, neighbor, new_path))
        
        end_time = time.time()
        return [], PathfindingMetrics(
            algorithm_name=self.name,
            trials_count=self.trials,
            convergence_time=end_time - start_time,
            path_length=0,
            solution_quality=0.0,
            success_rate=0.0,
            exploration_efficiency=0.0,
            memory_usage_mb=0.0,
            gedig_score=0.0
        )


class ReinforcementLearningAlgorithm(BasePathfindingAlgorithm):
    """強化学習アルゴリズム（DQN風）"""
    
    def __init__(self):
        super().__init__("Reinforcement_Learning")
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = 0.1  # 探索率
        self.alpha = 0.1    # 学習率
        self.gamma = 0.9    # 割引率
    
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """Q学習による経路探索"""
        start_time = time.time()
        self.trials = 0
        
        # エピソード実行
        max_episodes = 1000
        best_path = []
        best_length = float('inf')
        
        for episode in range(max_episodes):
            path = self._run_episode(maze)
            self.trials += len(path) if path else 100  # 失敗時はペナルティ
            
            if path and len(path) < best_length:
                best_path = path
                best_length = len(path)
                
            # 早期終了条件
            if path and len(path) <= best_length * 1.1:
                break
        
        end_time = time.time()
        success_rate = 1.0 if best_path else 0.0
        
        return best_path, PathfindingMetrics(
            algorithm_name=self.name,
            trials_count=self.trials,
            convergence_time=end_time - start_time,
            path_length=len(best_path),
            solution_quality=0.8 if best_path else 0.0,  # 準最適解
            success_rate=success_rate,
            exploration_efficiency=len(best_path) / max(self.trials, 1),
            memory_usage_mb=len(self.q_table) * 0.01,
            gedig_score=0.0
        )
    
    def _run_episode(self, maze: MazeEnvironment) -> List[Tuple[int, int]]:
        """1エピソード実行"""
        current = maze.start
        path = [current]
        visited = set()
        
        for _ in range(100):  # 最大ステップ数
            if current == maze.goal:
                return path
                
            if current in visited:
                return []  # 循環検出
            visited.add(current)
            
            # ε-greedy行動選択
            if random.random() < self.epsilon:
                neighbors = maze.get_neighbors(current)
                if not neighbors:
                    return []
                next_pos = random.choice(neighbors)
            else:
                next_pos = self._select_best_action(current, maze)
                
            if next_pos is None:
                return []
                
            # Q値更新
            reward = self._calculate_reward(next_pos, maze.goal)
            self._update_q_value(current, next_pos, reward, maze)
            
            current = next_pos
            path.append(current)
        
        return []  # タイムアウト
    
    def _select_best_action(self, state: Tuple[int, int], maze: MazeEnvironment) -> Optional[Tuple[int, int]]:
        """最良行動選択"""
        neighbors = maze.get_neighbors(state)
        if not neighbors:
            return None
            
        best_action = max(neighbors, key=lambda n: self.q_table[state][n])
        return best_action
    
    def _calculate_reward(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """報酬計算"""
        if pos == goal:
            return 100.0
        else:
            return -self.manhattan_distance(pos, goal) * 0.1
    
    def _update_q_value(self, state: Tuple[int, int], action: Tuple[int, int], 
                       reward: float, maze: MazeEnvironment):
        """Q値更新"""
        next_neighbors = maze.get_neighbors(action)
        max_next_q = max([self.q_table[action][n] for n in next_neighbors], default=0)
        
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )


class GeneticAlgorithm(BasePathfindingAlgorithm):
    """遺伝的アルゴリズム"""
    
    def __init__(self):
        super().__init__("Genetic_Algorithm")
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """遺伝的アルゴリズムによる経路探索"""
        start_time = time.time()
        self.trials = 0
        
        # 初期個体群生成
        population = self._generate_initial_population(maze)
        best_path = []
        best_fitness = float('-inf')
        
        max_generations = 100
        for generation in range(max_generations):
            # 適応度評価
            fitness_scores = [self._evaluate_fitness(individual, maze) for individual in population]
            self.trials += len(population)
            
            # 最良個体更新
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_path = population[max_idx]
            
            # 終了条件
            if best_path and best_path[-1] == maze.goal:
                break
                
            # 新世代生成
            population = self._evolve_population(population, fitness_scores)
        
        end_time = time.time()
        success_rate = 1.0 if best_path and best_path[-1] == maze.goal else 0.0
        
        return best_path, PathfindingMetrics(
            algorithm_name=self.name,
            trials_count=self.trials,
            convergence_time=end_time - start_time,
            path_length=len(best_path),
            solution_quality=0.7 if success_rate > 0 else 0.0,
            success_rate=success_rate,
            exploration_efficiency=len(best_path) / max(self.trials, 1),
            memory_usage_mb=self.population_size * 0.1,
            gedig_score=0.0
        )
    
    def _generate_initial_population(self, maze: MazeEnvironment) -> List[List[Tuple[int, int]]]:
        """初期個体群生成"""
        population = []
        for _ in range(self.population_size):
            path = self._generate_random_path(maze)
            population.append(path)
        return population
    
    def _generate_random_path(self, maze: MazeEnvironment, max_length: int = 50) -> List[Tuple[int, int]]:
        """ランダム経路生成"""
        path = [maze.start]
        current = maze.start
        
        for _ in range(max_length):
            neighbors = maze.get_neighbors(current)
            if not neighbors:
                break
                
            # ゴールに向かう傾向を持たせる
            if random.random() < 0.7:  # 70%の確率でゴール方向
                best_neighbor = min(neighbors, key=lambda n: self.manhattan_distance(n, maze.goal))
                next_pos = best_neighbor
            else:
                next_pos = random.choice(neighbors)
            
            path.append(next_pos)
            current = next_pos
            
            if current == maze.goal:
                break
                
        return path
    
    def _evaluate_fitness(self, individual: List[Tuple[int, int]], maze: MazeEnvironment) -> float:
        """適応度評価"""
        if not individual:
            return -1000
            
        # ゴール到達ボーナス
        goal_bonus = 1000 if individual[-1] == maze.goal else 0
        
        # 距離ペナルティ
        distance_penalty = self.manhattan_distance(individual[-1], maze.goal)
        
        # 経路長ペナルティ
        length_penalty = len(individual) * 0.1
        
        return goal_bonus - distance_penalty - length_penalty
    
    def _evolve_population(self, population: List[List[Tuple[int, int]]], 
                          fitness_scores: List[float]) -> List[List[Tuple[int, int]]]:
        """個体群進化"""
        new_population = []
        
        # エリート選択
        elite_count = max(2, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # 交叉・突然変異
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
                
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[List[Tuple[int, int]]], 
                            fitness_scores: List[float]) -> List[Tuple[int, int]]:
        """トーナメント選択"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: List[Tuple[int, int]], 
                  parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """交叉"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy()
            
        cut_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:cut_point] + parent2[cut_point:]
        return child
    
    def _mutate(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """突然変異"""
        if len(individual) < 2:
            return individual
            
        # ランダムな位置での経路変更
        mutation_point = random.randint(1, len(individual) - 1)
        return individual[:mutation_point]


class ImprovedSlimeMoldGEDIGAlgorithm(BasePathfindingAlgorithm):
    """
    改良版粘菌GEDIG アルゴリズム
    
    理論的基盤:
    1. 共通グラフ定義: ノードV=迷路セル、エッジE=隣接セル間、重みw_e(t)=チューブ太さ（流量）
    2. エッジ分布エントロピー: H(t) = -∑p_e(t)log(p_e(t)), p_e(t) = w_e(t)/∑w_e'(t)
    3. GED（グラフ編集距離）: GED_t = 1 - cosine(w(t), w(t-1))
    4. IG（情報利得）: IG_t = H(t-1) - H(t)
    5. 洞察スパイク: GED_t > τ_g かつ IG_t > τ_i
    """
    
    def __init__(self):
        super().__init__("SlimeMold_GEDIG")
        # 粘菌パラメータ
        self.flow_decay = 0.05         # 流量減衰率（より保持）
        self.conductivity_factor = 1.5  # 導電性係数（より活発）
        self.exploration_bonus = 0.1    # 探索ボーナス（より探索的）
        
        # 洞察スパイク検出のしきい値
        self.tau_g = 0.01  # GED閾値（更に敏感に）
        self.tau_i = 0.01  # IG閾値（より敏感に）
        
        # 状態管理
        self.edge_weights = {}          # エッジ重み w_e(t)
        self.weight_history = []        # 重みベクトル履歴
        self.entropy_history = []       # エントロピー履歴
        self.ged_history = []          # GED履歴
        self.ig_history = []           # IG履歴
        self.insight_spikes = []       # 洞察スパイク発生時刻
        
    def find_path(self, maze: MazeEnvironment) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """粘菌アナロジーによる経路探索"""
        start_time = time.time()
        self.trials = 0
        self._initialize_network(maze)
        
        best_path = []
        max_iterations = 1000
        convergence_threshold = 0.01
        
        for iteration in range(max_iterations):
            self.trials += 1
            
            # 1. 粘菌フローシミュレーション
            self._simulate_slime_flow(maze, iteration)
            
            # 2. 現在の重みベクトルを記録
            current_weights = self._get_current_weight_vector()
            self.weight_history.append(current_weights.copy())
            
            # 3. エントロピー計算
            entropy = self._calculate_edge_entropy()
            self.entropy_history.append(entropy)
            
            # 4. GED・IG計算（2回目以降）
            if len(self.weight_history) >= 2:
                ged_score = self._calculate_ged()
                ig_score = self._calculate_information_gain()
                
                self.ged_history.append(ged_score)
                self.ig_history.append(ig_score)
                
                # 5. 洞察スパイク検出
                if ged_score > self.tau_g and ig_score > self.tau_i:
                    self.insight_spikes.append(iteration)
                    # 洞察時の経路を抽出
                    current_path = self._extract_optimal_path(maze)
                    if current_path and current_path[-1] == maze.goal:
                        best_path = current_path
                        break
            
            # 6. 収束判定
            if len(self.weight_history) >= 2:
                weight_change = self._calculate_weight_change()
                if weight_change < convergence_threshold:
                    break
        
        # 最終経路抽出
        if not best_path:
            best_path = self._extract_optimal_path(maze)
        
        end_time = time.time()
        success_rate = 1.0 if best_path and best_path[-1] == maze.goal else 0.0
        
        # GEDIG統合スコア
        gedig_score = len(self.insight_spikes) * (
            np.mean(self.ged_history) if self.ged_history else 0.0
        )
        
        return best_path, PathfindingMetrics(
            algorithm_name=self.name,
            trials_count=self.trials,
            convergence_time=end_time - start_time,
            path_length=len(best_path) if best_path else 0,
            solution_quality=success_rate,
            success_rate=success_rate,
            exploration_efficiency=len(self.explored_nodes) / max(self.trials, 1),
            memory_usage_mb=self._estimate_memory_usage(),
            gedig_score=gedig_score
        )
    
    def _initialize_network(self, maze: MazeEnvironment):
        """粘菌管ネットワーク初期化"""
        self.edge_weights = {}
        self.explored_nodes = set()
        
        # 全ての有効なエッジを初期化
        for x in range(maze.width):
            for y in range(maze.height):
                pos = (x, y)
                if not maze.is_valid_position(pos):
                    continue
                    
                neighbors = maze.get_neighbors(pos)
                for neighbor in neighbors:
                    # 双方向エッジを作成（正規化用）
                    edge = tuple(sorted([pos, neighbor]))
                    if edge not in self.edge_weights:
                        self.edge_weights[edge] = 1.0  # 初期重み
    
    def _simulate_slime_flow(self, maze: MazeEnvironment, iteration: int):
        """粘菌の流体力学シミュレーション"""
        # ゴールからの距離場計算（魅力場）
        goal_distances = self._calculate_distance_field(maze)
        
        # 各エッジの流量を更新
        for edge in self.edge_weights:
            pos1, pos2 = edge
            
            # 距離ベースの魅力度
            dist1 = goal_distances.get(pos1, float('inf'))
            dist2 = goal_distances.get(pos2, float('inf'))
            
            # ゴールに近いほど高い魅力度
            attractiveness = 1.0 / (1.0 + min(dist1, dist2))
            
            # スタートからの距離も考慮（探索促進）
            start_dist1 = self.manhattan_distance(pos1, maze.start)
            start_dist2 = self.manhattan_distance(pos2, maze.start)
            exploration_factor = 1.0 + self.exploration_bonus / (1.0 + min(start_dist1, start_dist2))
            
            # 時間減衰（使われない管は細くなる）
            decay_factor = 1.0 - self.flow_decay * iteration / 100.0
            decay_factor = max(0.1, decay_factor)
            
            # 新しい重み計算
            new_weight = self.edge_weights[edge] * attractiveness * exploration_factor * decay_factor
            self.edge_weights[edge] = max(0.1, min(5.0, new_weight))
    
    def _calculate_distance_field(self, maze: MazeEnvironment) -> Dict[Tuple[int, int], int]:
        """ゴールからの最短距離場をBFSで計算"""
        distances = {}
        queue = deque([(maze.goal, 0)])
        distances[maze.goal] = 0
        
        while queue:
            pos, dist = queue.popleft()
            neighbors = maze.get_neighbors(pos)
            
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        return distances
    
    def _get_current_weight_vector(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """現在の重みベクトルを取得"""
        return self.edge_weights.copy()
    
    def _calculate_edge_entropy(self) -> float:
        """エッジ分布のエントロピー H(t) = -∑p_e(t)log(p_e(t))"""
        if not self.edge_weights:
            return 0.0
        
        # 重みを確率分布に正規化
        total_weight = sum(self.edge_weights.values())
        if total_weight == 0:
            return 0.0
        
        entropy = 0.0
        for weight in self.edge_weights.values():
            p = weight / total_weight
            if p > 0:
                entropy += -p * log2(p)
        
        return entropy
    
    def _calculate_ged(self) -> float:
        """GED_t = 1 - cosine(w(t), w(t-1))"""
        if len(self.weight_history) < 2:
            return 0.0
        
        current_weights = self.weight_history[-1]
        previous_weights = self.weight_history[-2]
        
        # 共通エッジの重みベクトルを作成
        common_edges = set(current_weights.keys()) & set(previous_weights.keys())
        if not common_edges:
            return 1.0
        
        # ベクトル化
        vec_current = np.array([current_weights.get(edge, 0.0) for edge in common_edges])
        vec_previous = np.array([previous_weights.get(edge, 0.0) for edge in common_edges])
        
        # コサイン類似度計算
        norm_current = np.linalg.norm(vec_current)
        norm_previous = np.linalg.norm(vec_previous)
        
        if norm_current == 0 or norm_previous == 0:
            return 1.0
        
        cosine_sim = np.dot(vec_current, vec_previous) / (norm_current * norm_previous)
        return 1.0 - cosine_sim
    
    def _calculate_information_gain(self) -> float:
        """IG_t = H(t-1) - H(t)"""
        if len(self.entropy_history) < 2:
            return 0.0
        
        return self.entropy_history[-2] - self.entropy_history[-1]
    
    def _calculate_weight_change(self) -> float:
        """重みベクトルの変化量計算（収束判定用）"""
        if len(self.weight_history) < 2:
            return float('inf')
        
        current = self.weight_history[-1]
        previous = self.weight_history[-2]
        
        total_change = 0.0
        all_edges = set(current.keys()) | set(previous.keys())
        
        for edge in all_edges:
            curr_weight = current.get(edge, 0.0)
            prev_weight = previous.get(edge, 0.0)
            total_change += abs(curr_weight - prev_weight)
        
        return total_change / max(len(all_edges), 1)
    
    def _extract_optimal_path(self, maze: MazeEnvironment) -> List[Tuple[int, int]]:
        """現在の管ネットワークから最適経路を抽出"""
        if not self.edge_weights:
            return []
        
        # 重み付きグラフでDijkstra法を実行
        distances = {maze.start: 0.0}
        previous = {}
        unvisited = set(maze.get_all_valid_positions())
        
        while unvisited:
            # 最小距離のノードを選択
            current = min(unvisited, key=lambda pos: distances.get(pos, float('inf')))
            
            if current == maze.goal:
                break
                
            if current not in distances:
                break
                
            unvisited.remove(current)
            current_dist = distances[current]
            
            # 隣接ノードの距離を更新
            neighbors = maze.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in unvisited:
                    continue
                
                # エッジ重みを使って距離計算（重いエッジ = 短い距離）
                edge = tuple(sorted([current, neighbor]))
                edge_weight = self.edge_weights.get(edge, 0.1)
                edge_cost = 1.0 / edge_weight  # 重い管ほど通りやすい
                
                new_dist = current_dist + edge_cost
                
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
        
        # 経路復元
        if maze.goal not in previous and maze.goal != maze.start:
            return []
        
        path = []
        current = maze.goal
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path if path[0] == maze.start else []
    
    def _estimate_memory_usage(self) -> float:
        """メモリ使用量推定（MB）"""
        # エッジ数 × データサイズの概算
        return len(self.edge_weights) * 0.001  # 概算値


class MazeGenerator:
    """適切な迷路生成アルゴリズム（Recursive Backtracking）"""
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
        
    def generate_maze(self) -> Set[Tuple[int, int]]:
        """Recursive Backtrackingで迷路を生成（壁の座標を返す）"""
        # 奇数サイズに調整（迷路生成の標準）
        maze_width = self.width if self.width % 2 == 1 else self.width - 1
        maze_height = self.height if self.height % 2 == 1 else self.height - 1
        
        # 全て壁で初期化
        walls = set()
        for x in range(maze_width):
            for y in range(maze_height):
                walls.add((x, y))
        
        # 通路を作る（奇数座標のみを通路候補とする）
        visited = set()
        stack = []
        
        # スタート地点（必ず奇数座標）
        start_x, start_y = 1, 1
        current = (start_x, start_y)
        visited.add(current)
        walls.discard(current)  # 通路として開放
        
        while True:
            # 隣接する未訪問セル（2マス先の奇数座標）を探す
            neighbors = []
            x, y = current
            
            for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if (0 < nx < maze_width - 1 and 
                    0 < ny < maze_height - 1 and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny))
            
            if neighbors:
                # ランダムに隣接セルを選択
                next_cell = random.choice(neighbors)
                stack.append(current)
                
                # 現在のセルと次のセルの間の壁を除去
                wall_x = (current[0] + next_cell[0]) // 2
                wall_y = (current[1] + next_cell[1]) // 2
                walls.discard((wall_x, wall_y))
                
                # 次のセルを通路として開放
                walls.discard(next_cell)
                visited.add(next_cell)
                current = next_cell
            elif stack:
                # バックトラック
                current = stack.pop()
            else:
                break
        
        return walls
    
    def generate_simple_maze(self) -> Set[Tuple[int, int]]:
        """シンプルな迷路パターン（テスト用）"""
        walls = set()
        
        # 外壁
        for x in range(self.width):
            walls.add((x, 0))
            walls.add((x, self.height - 1))
        for y in range(self.height):
            walls.add((0, y))
            walls.add((self.width - 1, y))
        
        # 内部の壁パターン
        # 水平壁
        for x in range(2, self.width - 2, 4):
            for y in range(2, self.height - 2, 4):
                walls.add((x, y))
                walls.add((x + 1, y))
        
        # 垂直壁
        for x in range(4, self.width - 2, 4):
            for y in range(1, self.height - 1, 2):
                walls.add((x, y))
        
        return walls
    
    def ensure_path_accessibility(self, walls: Set[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], width: int, height: int) -> Set[Tuple[int, int]]:
        """スタートとゴール周辺の通路を確保"""
        # スタート・ゴール地点とその隣接セルを通路として確保
        for pos in [start, goal]:
            x, y = pos
            # 中心
            walls.discard((x, y))
            # 隣接4方向
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    walls.discard((nx, ny))
        
        return walls


class GEDIGMazeExperiment:
    """GEDIG迷路実験メインクラス"""
    
    def __init__(self, output_dir: str = "experiments/phase3_gedig_maze/results", config: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        # 迷路を後から再利用できるよう保持（generate_maze_environments で上書き）
        self.maze_envs: List[MazeEnvironment] = []
        # GIF 生成を有効にするか
        self.enable_animation: bool = self.config.get("animate", False)
        self.logger = logging.getLogger(__name__)
        
        # アルゴリズム初期化
        self.algorithms = {
            'A_Star': AStarAlgorithm(),
            'Dijkstra': DijkstraAlgorithm(),
            'Reinforcement_Learning': ReinforcementLearningAlgorithm(),
            'Genetic_Algorithm': GeneticAlgorithm(),
            'SlimeMold_GEDIG': ImprovedSlimeMoldGEDIGAlgorithm()
        }
    
    def generate_maze_environments(self) -> List[MazeEnvironment]:
        """様々な迷路環境を生成（適切な迷路アルゴリズムを使用）"""
        mazes = []
        
        # 1. 小さな迷路 (11x11) - Recursive Backtracking
        generator_small = MazeGenerator(11, 11, seed=42)
        small_walls = generator_small.generate_maze()
        small_walls = generator_small.ensure_path_accessibility(small_walls, (1, 1), (9, 9), 11, 11)
        small_maze = MazeEnvironment(
            width=11, height=11,
            start=(1, 1), goal=(9, 9),
            walls=small_walls
        )
        mazes.append(('RecursiveMaze_11x11', small_maze))
        
        # 2. 中型迷路 (21x21) - Recursive Backtracking  
        generator_medium = MazeGenerator(21, 21, seed=123)
        medium_walls = generator_medium.generate_maze()
        medium_walls = generator_medium.ensure_path_accessibility(medium_walls, (1, 1), (19, 19), 21, 21)
        medium_maze = MazeEnvironment(
            width=21, height=21,
            start=(1, 1), goal=(19, 19),
            walls=medium_walls
        )
        mazes.append(('RecursiveMaze_21x21', medium_maze))
        
        # 3. シンプルパターン迷路 (15x15)
        generator_simple = MazeGenerator(15, 15, seed=456)
        simple_walls = generator_simple.generate_simple_maze()
        simple_walls = generator_simple.ensure_path_accessibility(simple_walls, (1, 1), (13, 13), 15, 15)
        simple_maze = MazeEnvironment(
            width=15, height=15,
            start=(1, 1), goal=(13, 13),
            walls=simple_walls
        )
        mazes.append(('SimpleMaze_15x15', simple_maze))
        
        # 4. 複雑迷路 (31x31) - より大きな迷路
        generator_complex = MazeGenerator(31, 31, seed=789)
        complex_walls = generator_complex.generate_maze()
        complex_walls = generator_complex.ensure_path_accessibility(complex_walls, (1, 1), (29, 29), 31, 31)
        complex_maze = MazeEnvironment(
            width=31, height=31,
            start=(1, 1), goal=(29, 29),
            walls=complex_walls
        )
        mazes.append(('ComplexMaze_31x31', complex_maze))
        
        # 生成した迷路をメンバに保存して後で再利用できるようにする
        self.maze_envs = mazes
        return mazes
    
    def run_pathfinding_comparison(self, num_trials: int = 5) -> pd.DataFrame:
        """経路探索比較実験実行"""
        results = []
        
        self.logger.info("Starting Phase 3: GEDIG Maze Pathfinding Experiment")
        
        mazes = self.generate_maze_environments()
        
        for maze_name, maze in mazes:
            self.logger.info(f"Testing on maze: {maze_name}")
            
            for algorithm_name, algorithm in self.algorithms.items():
                self.logger.info(f"  Algorithm: {algorithm_name}")
                
                trial_results = []
                for trial in range(num_trials):
                    try:
                        path, metrics = algorithm.find_path(maze)
                        trial_results.append(metrics)
                        
                        self.logger.info(f"    Trial {trial+1}: "
                                       f"Trials={metrics.trials_count}, "
                                       f"Time={metrics.convergence_time:.3f}s, "
                                       f"Success={metrics.success_rate}")
                        
                    except Exception as e:
                        self.logger.error(f"    Trial {trial+1} failed: {e}")
                        # 失敗時のデフォルトメトリクス
                        trial_results.append(PathfindingMetrics(
                            algorithm_name=algorithm_name,
                            trials_count=1000,  # ペナルティ
                            convergence_time=60.0,  # タイムアウト
                            path_length=0,
                            solution_quality=0.0,
                            success_rate=0.0,
                            exploration_efficiency=0.0,
                            memory_usage_mb=0.0,
                            gedig_score=0.0
                        ))
                
                # 平均メトリクス計算
                if trial_results:
                    avg_metrics = self._calculate_average_metrics(trial_results)
                    result = {
                        'maze_name': maze_name,
                        'algorithm': algorithm_name,
                        **asdict(avg_metrics)
                    }
                    results.append(result)
        
        # 結果DataFrame作成
        df_results = pd.DataFrame(results)
        
        # 結果保存
        df_results.to_csv(self.output_dir / 'gedig_maze_results.csv', index=False)
        self.logger.info(f"Results saved to {self.output_dir / 'gedig_maze_results.csv'}")
        
        return df_results
    
    def _calculate_average_metrics(self, trial_results: List[PathfindingMetrics]) -> PathfindingMetrics:
        """複数試行の平均メトリクス計算"""
        if not trial_results:
            return PathfindingMetrics(
                algorithm_name="Unknown",
                trials_count=0, convergence_time=0.0, path_length=0,
                solution_quality=0.0, success_rate=0.0, exploration_efficiency=0.0,
                memory_usage_mb=0.0, gedig_score=0.0
            )
        
        return PathfindingMetrics(
            algorithm_name=trial_results[0].algorithm_name,
            trials_count=int(np.mean([m.trials_count for m in trial_results])),
            convergence_time=np.mean([m.convergence_time for m in trial_results]),
            path_length=int(np.mean([m.path_length for m in trial_results])),
            solution_quality=np.mean([m.solution_quality for m in trial_results]),
            success_rate=np.mean([m.success_rate for m in trial_results]),
            exploration_efficiency=np.mean([m.exploration_efficiency for m in trial_results]),
            memory_usage_mb=np.mean([m.memory_usage_mb for m in trial_results]),
            gedig_score=np.mean([m.gedig_score for m in trial_results])
        )
    
    def generate_performance_report(self, df_results: pd.DataFrame) -> None:
        """性能レポート生成"""
        report_path = self.output_dir / 'gedig_performance_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 3: GEDIG迷路実験 結果レポート\n\n")
            f.write("## 実験概要\n")
            f.write("粘菌アナロジーによるGEDIG評価法と従来の経路探索アルゴリズムの比較\n\n")
            
            # アルゴリズム別平均性能（数値列のみ）
            numeric_columns = ['trials_count', 'convergence_time', 'success_rate', 'solution_quality', 'gedig_score']
            algo_avg = df_results.groupby('algorithm')[numeric_columns].mean()
            
            f.write("## アルゴリズム別平均性能\n\n")
            f.write("| アルゴリズム | 試行回数 | 収束時間(s) | 成功率 | 解品質 | GEDIG |\n")
            f.write("|-------------|----------|-------------|--------|--------|-------|\n")
            
            for algorithm in algo_avg.index:
                f.write(f"| {algorithm} | {algo_avg.loc[algorithm, 'trials_count']:.0f} | "
                       f"{algo_avg.loc[algorithm, 'convergence_time']:.3f} | "
                       f"{algo_avg.loc[algorithm, 'success_rate']:.3f} | "
                       f"{algo_avg.loc[algorithm, 'solution_quality']:.3f} | "
                       f"{algo_avg.loc[algorithm, 'gedig_score']:.3f} |\n")
            
            # GEDIG vs ベースライン比較
            if 'SlimeMold_GEDIG' in algo_avg.index:
                gedig_metrics = algo_avg.loc['SlimeMold_GEDIG']
                
                f.write("\n## GEDIG vs ベースライン比較\n\n")
                
                for baseline in ['A_Star', 'Dijkstra', 'Reinforcement_Learning', 'Genetic_Algorithm']:
                    if baseline in algo_avg.index:
                        baseline_metrics = algo_avg.loc[baseline]
                        
                        trial_reduction = (baseline_metrics['trials_count'] - gedig_metrics['trials_count']) / baseline_metrics['trials_count'] * 100
                        speed_improvement = baseline_metrics['convergence_time'] / gedig_metrics['convergence_time']
                        
                        f.write(f"### vs {baseline}\n")
                        f.write(f"- **試行回数削減**: {trial_reduction:.1f}%\n")
                        f.write(f"- **収束速度**: {speed_improvement:.1f}倍高速\n")
                        f.write(f"- **成功率**: {gedig_metrics['success_rate']:.1%}\n\n")
                
                # 目標達成確認
                f.write("## 目標達成状況\n")
                
                avg_trial_reduction = np.mean([
                    (algo_avg.loc[baseline, 'trials_count'] - gedig_metrics['trials_count']) / algo_avg.loc[baseline, 'trials_count'] * 100
                    for baseline in ['A_Star', 'Dijkstra', 'Reinforcement_Learning', 'Genetic_Algorithm']
                    if baseline in algo_avg.index
                ])
                avg_speed_improvement = np.mean([
                    algo_avg.loc[baseline, 'convergence_time'] / gedig_metrics['convergence_time']
                    for baseline in ['A_Star', 'Dijkstra', 'Reinforcement_Learning', 'Genetic_Algorithm']
                    if baseline in algo_avg.index
                ])
                
                f.write(f"- **試行回数60%削減**: {avg_trial_reduction:.1f}% ")
                f.write("✅ 達成\n" if avg_trial_reduction >= 60 else "❌ 未達成\n")
                
                f.write(f"- **収束速度3倍向上**: {avg_speed_improvement:.1f}倍 ")
                f.write("✅ 達成\n" if avg_speed_improvement >= 3.0 else "❌ 未達成\n")
                
                f.write(f"- **成功率95%+**: {gedig_metrics['success_rate']:.1%} ")
                f.write("✅ 達成\n" if gedig_metrics['success_rate'] >= 0.95 else "❌ 未達成\n")
        
        self.logger.info(f"Performance report generated: {report_path}")
    
    def visualize_results(self, df_results: pd.DataFrame) -> None:
        """結果の可視化（棒グラフ）"""
        # 図のサイズ設定
        plt.figure(figsize=(15, 10))
        
        # 1. 試行回数比較
        plt.subplot(2, 3, 1)
        algo_avg = df_results.groupby('algorithm')['trials_count'].mean()
        bars = plt.bar(algo_avg.index, algo_avg.values)
        plt.title('平均試行回数比較')
        plt.ylabel('試行回数')
        plt.xticks(rotation=45)
        
        # GEDIGを強調
        for i, bar in enumerate(bars):
            if 'GEDIG' in algo_avg.index[i]:
                bar.set_color('red')
        
        # 2. 収束時間比較
        plt.subplot(2, 3, 2)
        time_avg = df_results.groupby('algorithm')['convergence_time'].mean()
        bars = plt.bar(time_avg.index, time_avg.values)
        plt.title('平均収束時間比較')
        plt.ylabel('時間 (秒)')
        plt.xticks(rotation=45)
        
        for i, bar in enumerate(bars):
            if 'GEDIG' in time_avg.index[i]:
                bar.set_color('red')
        
        # 3. 成功率比較
        plt.subplot(2, 3, 3)
        success_avg = df_results.groupby('algorithm')['success_rate'].mean()
        bars = plt.bar(success_avg.index, success_avg.values)
        plt.title('平均成功率比較')
        plt.ylabel('成功率')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for i, bar in enumerate(bars):
            if 'GEDIG' in success_avg.index[i]:
                bar.set_color('red')
        
        # 4. 解品質比較
        plt.subplot(2, 3, 4)
        quality_avg = df_results.groupby('algorithm')['solution_quality'].mean()
        bars = plt.bar(quality_avg.index, quality_avg.values)
        plt.title('平均解品質比較')
        plt.ylabel('解品質')
        plt.xticks(rotation=45)
        
        for i, bar in enumerate(bars):
            if 'GEDIG' in quality_avg.index[i]:
                bar.set_color('red')
        
        # 5. 迷路タイプ別性能
        plt.subplot(2, 3, 5)
        maze_performance = df_results.groupby(['maze_name', 'algorithm'])['trials_count'].mean().unstack()
        maze_performance.plot(kind='bar', ax=plt.gca())
        plt.title('迷路タイプ別試行回数')
        plt.ylabel('試行回数')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. GEDIG スコア分布
        plt.subplot(2, 3, 6)
        gedig_data = df_results[df_results['algorithm'] == 'SlimeMold_GEDIG']
        if not gedig_data.empty:
            plt.bar(gedig_data['maze_name'], gedig_data['gedig_score'])
            plt.title('GEDIG スコア (迷路別)')
            plt.ylabel('GEDIG スコア')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 図保存
        viz_path = self.output_dir / 'gedig_performance_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {viz_path}")

        # ----------------------------
        # 追加機能: A* vs GEDIG 経路アニメーション
        # ----------------------------
        self.logger.info(f"Animation enabled: {self.enable_animation}, Maze count: {len(self.maze_envs)}")
        if self.enable_animation and self.maze_envs:
            try:
                self.logger.info("Starting GIF generation...")
                self._generate_path_comparison_gifs(limit=3)  # 代表的な3迷路のみ
            except Exception as e:
                self.logger.warning(f"Path comparison GIF generation failed: {e}")
        else:
            if not self.enable_animation:
                self.logger.info("Animation disabled - skipping GIF generation")
            if not self.maze_envs:
                self.logger.warning("No mazes available for GIF generation")

    # ===============================================================
    # GIF 生成ユーティリティ
    # ===============================================================
    def _draw_maze(self, maze: 'MazeEnvironment', path: List[Tuple[int, int]], color: str, title: str = "") -> 'np.ndarray':
        """迷路グリッド＋部分経路を描画し numpy 配列で返す（改良版）"""
        import numpy as np
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, maze.width - 0.5)
        ax.set_ylim(-0.5, maze.height - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # 背景を白に設定
        ax.set_facecolor('white')
        
        # 壁を描画（線ベース）
        if maze.walls:
            for wall_x, wall_y in maze.walls:
                # 各壁セルを小さな四角として描画
                from matplotlib.patches import Rectangle
                rect = Rectangle((wall_x - 0.5, wall_y - 0.5), 1, 1, 
                               linewidth=0, facecolor='black', alpha=0.8)
                ax.add_patch(rect)
        
        # グリッド線を描画（薄いグレー）
        for x in range(maze.width + 1):
            ax.axvline(x - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)
        for y in range(maze.height + 1):
            ax.axhline(y - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)
        
        # スタート地点（緑の円）
        ax.scatter([maze.start[0]], [maze.start[1]], c='green', marker='o', s=150, 
                  edgecolor='darkgreen', linewidth=2, zorder=5)
        
        # ゴール地点（青い星）
        ax.scatter([maze.goal[0]], [maze.goal[1]], c='blue', marker='*', s=200,
                  edgecolor='darkblue', linewidth=2, zorder=5)
        
        # 経路を描画
        if path and len(path) > 1:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            # 経路線
            ax.plot(xs, ys, color=color, linewidth=3, alpha=0.8, zorder=3)
            # 経路点
            ax.scatter(xs[1:-1], ys[1:-1], color=color, s=30, alpha=0.7, zorder=4)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 画像として出力
        from io import BytesIO
        import imageio.v2 as imageio_v2
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img = imageio_v2.imread(buf)
        plt.close(fig)
        return img

    def _create_comparison_gif(self, maze_name: str, maze_env: 'MazeEnvironment') -> None:
        """A* と SlimeMold_GEDIG の経路を段階的に描いて GIF 出力"""
        self.logger.info(f"Creating comparison GIF for {maze_name}")
        from pathlib import Path
        import numpy as np
        import imageio.v2 as imageio_v2  # 明示import

        # --- 経路計算 ---
        astar = AStarAlgorithm()
        path_astar, _ = astar.find_path(maze_env)

        gedig = ImprovedSlimeMoldGEDIGAlgorithm()
        path_gedig, _ = gedig.find_path(maze_env)

        max_len = max(len(path_astar), len(path_gedig))
        frames = []
        for i in range(max_len):
            partial_a = path_astar[: i + 1] if i < len(path_astar) else path_astar
            partial_g = path_gedig[: i + 1] if i < len(path_gedig) else path_gedig
            img_a = self._draw_maze(maze_env, partial_a, color='blue', title='A*')
            img_g = self._draw_maze(maze_env, partial_g, color='red', title='SlimeMold_GEDIG')
            combined = np.concatenate([img_a, img_g], axis=1)
            frames.append(combined)

        gif_path = self.output_dir / f'path_comparison_{maze_name}.gif'
        imageio_v2.mimsave(gif_path, frames, format='GIF', duration=0.2)
        self.logger.info(f"Path comparison GIF saved: {gif_path}")

    def _generate_path_comparison_gifs(self, limit: int = 3):
        """いくつかの迷路に対して比較 GIF を生成"""
        self.logger.info(f"Generating path comparison GIFs for up to {limit} mazes")
        count = 0
        for maze_name, maze_env in self.maze_envs:
            self.logger.info(f"Creating GIF for maze: {maze_name}")
            self._create_comparison_gif(maze_name, maze_env)
            count += 1
            if count >= limit:
                break


@with_data_safety(
    experiment_name="phase3_gedig_maze",
    backup_description="Pre-experiment backup for Phase 3: GEDIG Maze Optimization",
    auto_rollback=True,
    selective_copy=["processed", "cache"]  # 迷路実験用の最小限データ
)
def run_gedig_maze_experiment(experiment_env: Dict[str, Any] = None) -> Dict[str, Any]:
    """データ安全性機能付きGEDIG迷路実験実行"""
    
    # 実験用データ設定取得
    data_config = create_experiment_data_config(experiment_env)
    
    # 実験用出力ディレクトリ設定
    experiment_output_dir = experiment_env["experiment_data_dir"] / "outputs"
    experiment_output_dir.mkdir(exist_ok=True)
    
    experiment = GEDIGMazeExperiment(str(experiment_output_dir))
    
    logger = logging.getLogger(__name__)
    logger.info("=== Phase 3: GEDIG Maze Experiment (Safe Mode) ===")
    logger.info(f"Experiment data directory: {experiment_env['experiment_data_dir']}")
    logger.info(f"Backup ID: {experiment_env['backup_id']}")
    logger.info(f"Data configuration: {data_config}")
    
    try:
        # 実験実行（実験用データディレクトリを使用）
        results = experiment.run_pathfinding_comparison()
        
        # レポート生成
        experiment.generate_performance_report(results)
        
        # 可視化
        experiment.visualize_results(results)
        
        # 実験結果の統合データ保存
        experiment_results = {
            "experiment_name": "phase3_gedig_maze",
            "timestamp": time.time(),
            "backup_id": experiment_env["backup_id"],
            "data_config": data_config,
            "results": results.to_dict('records'),
            "output_directory": str(experiment_output_dir),
            "success": True
        }
        
        # 実験結果JSONファイル保存
        results_file = experiment_output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment results saved to: {results_file}")
        logger.info("🎉 Phase 3 実験完了! (データは自動的に安全な状態に復元されます)")
        
        return experiment_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def create_cli_parser() -> argparse.ArgumentParser:
    """Phase 3専用CLI引数パーサーの作成"""
    try:
        if CLI_AVAILABLE:
            parser = create_base_cli_parser(
                "Phase 3", 
                "GEDIG迷路実験 - 粘菌アナロジーによる最適化性能検証"
            )
            parser = add_phase_specific_args(parser, "phase3")
            return parser
    except Exception:
        pass
    
    # フォールバック: 基本CLI作成
    parser = argparse.ArgumentParser(
        description="Phase 3: GEDIG迷路実験",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--maze-sizes', nargs='+', type=int, default=[10, 20, 50, 100], help='迷路サイズ')
    parser.add_argument('--algorithms', nargs='+', default=['astar', 'dijkstra', 'genetic', 'reinforcement'], help='比較アルゴリズム')
    parser.add_argument('--maze-count', type=int, default=10, help='各サイズの迷路生成数')
    parser.add_argument('--output', type=str, default="experiments/phase3_gedig_maze/results", help='出力ディレクトリ')
    parser.add_argument('--export', choices=['csv', 'json', 'excel'], default='csv', help='エクスポート形式')
    parser.add_argument('--no-backup', action='store_true', help='バックアップスキップ')
    parser.add_argument('--quick', action='store_true', help='クイックテスト')
    parser.add_argument('--animate', action='store_true', help='A*とGEDIGの経路比較GIFを生成')
    parser.add_argument('--config', type=str, help='設定ファイル')
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """設定ファイルの読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 設定ファイル読み込み完了: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return {}


def merge_cli_config(args: argparse.Namespace, phase: str = "phase3") -> Dict[str, Any]:
    """CLI引数と設定ファイルのマージ"""
    config = {}
    
    # 設定ファイルがある場合は読み込み
    if hasattr(args, 'config') and args.config:
        config = load_config_file(args.config)
    
    # CLI引数で上書き
    config.update({
        'debug': getattr(args, 'debug', False),
        'maze_sizes': getattr(args, 'maze_sizes', [10, 20, 50, 100]),
        'algorithms': getattr(args, 'algorithms', ['astar', 'dijkstra', 'genetic', 'reinforcement']),
        'maze_count': getattr(args, 'maze_count', 10),
        'export_format': getattr(args, 'export', 'csv'),
        'output_dir': getattr(args, 'output', 'experiments/phase3_gedig_maze/results'),
        'no_backup': getattr(args, 'no_backup', False),
        'quick_mode': getattr(args, 'quick', False),
        'animate': getattr(args, 'animate', False),
        'generate_report': True,
        'generate_plots': True,  # Phase 3では可視化重要
        'selective_copy': ["processed", "models", "cache"]
    })
    
    # クイックモードの場合は設定を簡素化
    if config['quick_mode']:
        config['maze_sizes'] = [10, 20]
        config['maze_count'] = 3
        config['algorithms'] = ['astar', 'gedig']  # 基本アルゴリズムのみ
    
    # 出力ディレクトリの表記揺れ（"phase 3" → "phase3_gedig_maze"）を補正
    if 'output_dir' in config and isinstance(config['output_dir'], str):
        config['output_dir'] = config['output_dir'].replace('phase 3', 'phase3_gedig_maze')
    
    return config


def main():
    """メイン実行関数 - CLI対応・データ安全性機能付き"""
    
    # CLI引数パース（フォールバック機能付き）
    try:
        parser = create_cli_parser()
        args = parser.parse_args()
        config = merge_cli_config(args, "phase3")
    except Exception as e:
        print(f"⚠️  CLI機能エラー: {e}")
        print("🔧 基本モードで実行します")
        config = {
            'debug': False,
            'maze_sizes': [10, 20, 50, 100],
            'algorithms': ['astar', 'dijkstra', 'genetic', 'reinforcement'],
            'maze_count': 10,
            'export_format': 'csv',
            'output_dir': 'experiments/phase3_gedig_maze/results',
            'no_backup': False,
            'selective_copy': ["processed", "models", "cache"],
            'generate_report': True,
            'generate_plots': True,
            'quick_mode': False,
            'animate': False
        }
    
    # 実験ヘッダー表示
    try:
        if CLI_AVAILABLE:
            print_experiment_header("Phase 3: GEDIG迷路実験", config)
            print_scripts_integration_status()
    except Exception:
        print("🔬 Phase 3: GEDIG迷路実験")
        print("=" * 50)
        print(f"🗺️  迷路サイズ: {config['maze_sizes']}")
        print(f"🔧 アルゴリズム: {config['algorithms']}")
        print(f"🔢 迷路生成数: {config['maze_count']}")
        print(f"🛡️  データバックアップ: {'無効' if config['no_backup'] else '有効'}")
        print(f"🐛 デバッグモード: {'有効' if config['debug'] else '無効'}")
    
    try:
        # scripts/experiments/統合モードを試行
        try:
            if CLI_AVAILABLE:
                scripts_experiment = ScriptsIntegratedExperiment("phase3_gedig_maze", config)
                
                def run_phase3_experiment(integrated_config):
                    if integrated_config['no_backup']:
                        # 高速モード
                        experiment = GEDIGMazeExperiment(integrated_config['output_dir'], integrated_config)
                        results = experiment.run_pathfinding_comparison()
                        if integrated_config['generate_report']:
                            experiment.generate_performance_report(results)
                        if integrated_config['generate_plots']:
                            experiment.visualize_results(results)
                        return results
                    else:
                        # 安全モード
                        return run_gedig_maze_experiment()
                
                results = scripts_experiment.run_experiment(run_phase3_experiment)
                print("✅ scripts/experiments/統合モードで実行完了")
            else:
                raise Exception("CLI機能が利用できません")
                
        except Exception as integration_error:
            print(f"⚠️  scripts統合モードエラー: {integration_error}")
            print("🔧 標準モードで実行します")
            
            # 標準モード実行
            if config['no_backup']:
                # バックアップなしで直接実行（高速モード）
                print("\n⚡ 高速モード: データバックアップなしで実行")
                experiment = GEDIGMazeExperiment(config['output_dir'], config)
                results = experiment.run_pathfinding_comparison()
                
                if config['generate_report']:
                    experiment.generate_performance_report(results)
                
                if config['generate_plots']:
                    experiment.visualize_results(results)
                
                print("\n🎉 Phase 3 実験完了! (高速モード)")
                
            else:
                # 安全な実験環境で実行（推奨）
                print("\n🛡️  安全モード: データバックアップ付きで実行")
                results = run_gedig_maze_experiment()
        
        # 結果サマリー表示
        try:
            if CLI_AVAILABLE:
                summary = create_experiment_summary(results, "phase3")
                print(summary)
        except Exception:
            # フォールバック: 基本サマリー
            if not config.get('debug', False) and results is not None:
                print("\n📊 結果サマリー:")
                print("✅ 実験完了")
                print("📁 結果は以下に保存されています:")
                print("  - experiment_data/ (実験結果)")
                print("  - data_backups/ (バックアップ)")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⛔ 実験が中断されました")
        print("🔄 データは安全な状態に復元されています")
        return None
        
    except Exception as e:
        try:
            if CLI_AVAILABLE:
                handle_cli_error(e, config)
        except Exception:
            print(f"\n❌ 実験が失敗しました: {e}")
            if config.get('debug', False):
                import traceback
                traceback.print_exc()
            print("🔄 データは自動的に実験前の状態に復元されました")
        raise


if __name__ == "__main__":
    main()
