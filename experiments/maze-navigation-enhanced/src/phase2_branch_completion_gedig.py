#!/usr/bin/env python3
"""
未探索分岐探索完了時のgeDIG値調査
==================================

ゴールに繋がらない分岐を探索し終わった時の
グラフ短絡とgeDIG値の変化を調査
"""

import numpy as np
import sys
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# geDIG計算を直接実装
class SimpleGeDIG:
    """シンプルなgeDIG計算"""
    
    def calculate_multihop(self, g1: nx.Graph, g2: nx.Graph, max_hop: int = 10) -> Dict[int, float]:
        """マルチホップgeDIG計算"""
        results = {}
        
        # グラフが空の場合
        if g1.number_of_nodes() == 0:
            # 初期グラフ作成時
            for hop in range(1, min(max_hop + 1, 11)):
                results[hop] = 0.5 * (0.7 ** hop)  # 正の値（新情報）
            return results
        
        for hop in range(1, min(max_hop + 1, 11)):
            # グラフの変化を計算
            n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
            e1, e2 = g1.number_of_edges(), g2.number_of_edges()
            
            # ノードとエッジの変化
            node_added = max(0, n2 - n1)
            edge_added = max(0, e2 - e1)
            
            # 構造の変化（GED的な要素）
            if n1 + n2 > 0:
                structural_change = (node_added + edge_added) / (n1 + n2 + 1)
            else:
                structural_change = 0
            
            # 情報利得（新規ノードが多いほど高い）
            if node_added > 0:
                information_gain = np.log1p(node_added) / np.log1p(n2)
            else:
                information_gain = 0
            
            # グラフの密度変化（短絡の検出）
            density1 = 2 * e1 / (n1 * (n1 - 1)) if n1 > 1 else 0
            density2 = 2 * e2 / (n2 * (n2 - 1)) if n2 > 1 else 0
            density_change = density2 - density1
            
            # geDIG計算
            # - 正の値: 新しい情報の追加
            # - 負の値: グラフの短絡や統合
            if density_change > 0.1:  # 密度が大幅に増加 = 短絡
                gedig = -abs(density_change) * 2  # 負の値（短絡検出）
            elif node_added == 0 and edge_added > 2:  # 新規ノードなしで多数のエッジ追加
                gedig = -abs(edge_added / (e1 + 1))  # 負の値（既存ノード間の接続）
            else:
                # 通常の探索
                gedig = structural_change - 0.3 * information_gain
            
            # ホップ数による減衰
            gedig *= (0.7 ** (hop - 1))
            
            results[hop] = gedig
        
        return results

@dataclass
class Episode:
    position: Tuple[int, int]
    direction: str
    vector: np.ndarray
    is_wall: bool
    visit_count: int = 0
    timestamp: int = 0
    episode_id: int = 0

@dataclass
class BranchCompletionEvent:
    """分岐探索完了イベント"""
    step: int
    branch_entry: Tuple[int, int]  # 分岐の入口
    branch_positions: List[Tuple[int, int]]  # 探索した位置
    return_position: Tuple[int, int]  # 戻ってきた位置
    gedig_value: float
    gedig_by_hop: Dict[int, float]
    graph_diameter_before: int
    graph_diameter_after: int
    new_edges_created: int

class BranchCompletionAnalyzer:
    """分岐探索完了分析"""
    
    def __init__(self):
        # T字型迷路を作成（明確な分岐を持つ）
        self.maze = self.create_t_junction_maze()
        self.h, self.w = self.maze.shape
        self.pos = (5, 9)  # 下部中央からスタート
        self.goal = (9, 1)  # 右上にゴール
        
        # エピソード管理
        self.episodes = {}
        self.all_episodes = []
        self.episode_counter = 0
        self.episode_graph = nx.Graph()
        
        # 探索状態
        self.step = 0
        self.path = [self.pos]
        self.unique_positions = set([self.pos])
        
        # 分岐検出
        self.branch_events = []
        self.current_branch_entry = None
        self.branch_positions = set()
        self.pre_branch_graph = None
        
        # geDIG
        self.gedig = SimpleGeDIG()
        self.graph_history = []
        
        # 重み
        self.weights = np.array([
            1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0
        ])
        self.temperature = 0.1
        
        print("="*80)
        print("BRANCH COMPLETION geDIG ANALYSIS")
        print("="*80)
        print(f"Maze: {self.w}x{self.h} T-junction")
        print(f"Start: {self.pos}, Goal: {self.goal}")
        print()
        self.print_maze()
    
    def create_t_junction_maze(self):
        """T字型迷路作成（左の分岐は行き止まり）"""
        maze = np.ones((11, 11), dtype=int)
        
        # 縦の通路（中央）
        for y in range(1, 10):
            maze[y, 5] = 0
        
        # 横の通路（上部）
        for x in range(1, 10):
            maze[1, x] = 0
        
        # 左の行き止まり分岐
        for x in range(1, 5):
            maze[3, x] = 0
        maze[3, 5] = 0  # 接続部
        
        # 右への小さな分岐
        maze[7, 6] = 0
        maze[7, 7] = 0
        
        return maze
    
    def print_maze(self):
        """迷路を表示"""
        print("Maze structure:")
        for y in range(self.h):
            row = ""
            for x in range(self.w):
                if (x, y) == self.pos:
                    row += "S "
                elif (x, y) == self.goal:
                    row += "G "
                elif self.maze[y, x] == 1:
                    row += "█ "
                else:
                    row += "· "
            print(row)
        print()
    
    def create_vector(self, pos, dir, is_wall, visits=0):
        direction_map = {'N': (0,-1), 'S': (0,1), 'E': (1,0), 'W': (-1,0)}
        dx, dy = direction_map.get(dir, (0,0))
        
        return np.array([
            pos[0]/self.w, pos[1]/self.h,
            dx, dy,
            -1.0 if is_wall else 1.0,
            np.log1p(visits),
            0.0, 0.0
        ])
    
    def observe(self):
        """4方向観測"""
        episodes = []
        
        for d, (dx, dy) in [('N', (0,-1)), ('S', (0,1)), ('E', (1,0)), ('W', (-1,0))]:
            nx, ny = self.pos[0]+dx, self.pos[1]+dy
            key = (self.pos, d)
            
            if key in self.episodes:
                ep = self.episodes[key]
            else:
                is_wall = True
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    is_wall = (self.maze[ny, nx] == 1)
                
                ep = Episode(
                    self.pos, d,
                    self.create_vector(self.pos, d, is_wall),
                    is_wall,
                    timestamp=self.step,
                    episode_id=self.episode_counter
                )
                
                self.episode_counter += 1
                self.episodes[key] = ep
                self.all_episodes.append(ep)
                
                # グラフにノード追加
                self.episode_graph.add_node(
                    ep.episode_id,
                    position=ep.position,
                    direction=ep.direction
                )
                
                # 直前エピソードと接続
                if len(self.all_episodes) > 1:
                    prev_ep = self.all_episodes[-2]
                    self.episode_graph.add_edge(ep.episode_id, prev_ep.episode_id)
            
            episodes.append(ep)
        
        return episodes
    
    def detect_branch_entry(self):
        """分岐への進入を検出"""
        # 現在位置で利用可能な方向数
        available_dirs = sum(1 for _, (dx, dy) in [('N', (0,-1)), ('S', (0,1)), ('E', (1,0)), ('W', (-1,0))]
                           if 0 <= self.pos[0]+dx < self.w and 0 <= self.pos[1]+dy < self.h
                           and self.maze[self.pos[1]+dy, self.pos[0]+dx] == 0)
        
        # 3方向以上なら分岐点
        if available_dirs >= 3 and not self.current_branch_entry:
            self.current_branch_entry = self.pos
            self.branch_positions = {self.pos}
            self.pre_branch_graph = self.episode_graph.copy()
            print(f"\n🔍 Branch entry detected at {self.pos} (Step {self.step})")
            return True
        
        return False
    
    def detect_branch_completion(self):
        """分岐探索の完了を検出"""
        if not self.current_branch_entry:
            return False
        
        # 分岐入口に戻ってきたか
        if self.pos == self.current_branch_entry and len(self.branch_positions) > 3:
            print(f"\n✅ Branch exploration completed at Step {self.step}")
            print(f"   Explored {len(self.branch_positions)} positions in branch")
            return True
        
        # 分岐内を探索中
        if self.current_branch_entry:
            self.branch_positions.add(self.pos)
        
        return False
    
    def calculate_gedig_multihop(self) -> Dict[int, float]:
        """マルチホップgeDIG計算"""
        # 分岐完了時は、分岐前のグラフと比較
        if self.pre_branch_graph:
            current_graph = self.episode_graph.copy()
            prev_graph = self.pre_branch_graph
        elif len(self.graph_history) > 0:
            current_graph = self.episode_graph.copy()
            prev_graph = self.graph_history[-1]
        else:
            return {}
        
        # SimpleGeDIGを使用
        results = self.gedig.calculate_multihop(prev_graph, current_graph)
        
        return results
    
    def analyze_branch_completion(self):
        """分岐完了時の分析"""
        # デバッグ: グラフサイズ確認
        print(f"\n  Debug: Current graph has {self.episode_graph.number_of_nodes()} nodes, {self.episode_graph.number_of_edges()} edges")
        print(f"  Debug: Previous graph has {self.pre_branch_graph.number_of_nodes()} nodes, {self.pre_branch_graph.number_of_edges()} edges")
        
        # geDIG計算
        hop_results = self.calculate_gedig_multihop()
        gedig_value = hop_results.get(1, 0.0)  # 1-hop geDIG
        
        # デバッグ: 計算結果
        print(f"  Debug: hop_results = {hop_results}")
        
        # グラフ直径の変化
        diameter_before = nx.diameter(self.pre_branch_graph) if nx.is_connected(self.pre_branch_graph) else -1
        diameter_after = nx.diameter(self.episode_graph) if nx.is_connected(self.episode_graph) else -1
        
        # 新規エッジ数
        edges_before = self.pre_branch_graph.number_of_edges()
        edges_after = self.episode_graph.number_of_edges()
        new_edges = edges_after - edges_before
        
        # イベント記録
        event = BranchCompletionEvent(
            step=self.step,
            branch_entry=self.current_branch_entry,
            branch_positions=list(self.branch_positions),
            return_position=self.pos,
            gedig_value=gedig_value,
            gedig_by_hop=hop_results,
            graph_diameter_before=diameter_before,
            graph_diameter_after=diameter_after,
            new_edges_created=new_edges
        )
        
        self.branch_events.append(event)
        
        # 詳細出力
        print(f"\n📊 Branch Completion Analysis:")
        print(f"  geDIG value: {gedig_value:.6f}")
        print(f"  Graph diameter: {diameter_before} → {diameter_after}")
        print(f"  New edges created: {new_edges}")
        print(f"\n  geDIG by hop:")
        for hop in sorted(hop_results.keys())[:10]:
            print(f"    {hop:2d}-hop: {hop_results[hop]:8.6f}")
        
        # グラフ短絡の検出
        if diameter_after < diameter_before and diameter_before > 0:
            print(f"\n  🔗 Graph shortcut detected! Diameter reduced by {diameter_before - diameter_after}")
        
        # リセット
        self.current_branch_entry = None
        self.branch_positions = set()
    
    def decide(self, episodes):
        """意思決定"""
        query = self.create_vector(self.pos, '', False, 0)
        query[4] = 1.0
        query_weighted = query * self.weights
        
        valid_episodes = [ep for ep in episodes if not ep.is_wall]
        if not valid_episodes:
            return None
        
        distances = []
        for ep in valid_episodes:
            ep_weighted = ep.vector * self.weights
            dist = np.linalg.norm(query_weighted - ep_weighted)
            distances.append(dist)
        
        distances = np.array(distances)
        scores = np.exp(-distances / self.temperature)
        probabilities = scores / np.sum(scores)
        
        best_idx = np.argmax(probabilities)
        return valid_episodes[best_idx].direction
    
    def move(self, d):
        """移動実行"""
        if not d:
            return False
        
        direction_map = {'N': (0,-1), 'S': (0,1), 'E': (1,0), 'W': (-1,0)}
        dx, dy = direction_map[d]
        nx, ny = self.pos[0]+dx, self.pos[1]+dy
        
        key = (self.pos, d)
        if key in self.episodes:
            ep = self.episodes[key]
            ep.visit_count += 1
            ep.vector[5] = np.log1p(ep.visit_count)
            
            if 0 <= nx < self.w and 0 <= ny < self.h and self.maze[ny, nx] == 0:
                self.pos = (nx, ny)
                self.path.append(self.pos)
                self.unique_positions.add(self.pos)
                
                if self.pos == self.goal:
                    return True
        
        return False
    
    def run(self, max_steps=200):
        """メインループ"""
        print("Starting exploration...")
        print("-" * 60)
        
        for i in range(max_steps):
            self.step = i + 1
            
            # 観測
            episodes = self.observe()
            
            # 分岐検出
            self.detect_branch_entry()
            
            # 分岐完了検出
            if self.detect_branch_completion():
                self.analyze_branch_completion()
            
            # グラフ履歴更新（毎ステップ）
            self.graph_history.append(self.episode_graph.copy())
            
            # 意思決定と移動
            d = self.decide(episodes)
            if d and self.move(d):
                print(f"\n🎯 GOAL REACHED in {self.step} steps!")
                break
            
            # 進捗
            if self.step % 50 == 0:
                print(f"\nStep {self.step}: Position {self.pos}")
                print(f"  Unique positions: {len(self.unique_positions)}")
                print(f"  Graph: {self.episode_graph.number_of_nodes()} nodes, "
                      f"{self.episode_graph.number_of_edges()} edges")
        
        self.final_analysis()
    
    def final_analysis(self):
        """最終分析"""
        print("\n" + "="*80)
        print("FINAL ANALYSIS - Branch Completion Events")
        print("="*80)
        
        if not self.branch_events:
            print("No branch completion events detected")
            return
        
        print(f"\nTotal branch completion events: {len(self.branch_events)}")
        
        for i, event in enumerate(self.branch_events):
            print(f"\n📍 Event {i+1} (Step {event.step}):")
            print(f"  Branch entry: {event.branch_entry}")
            print(f"  Positions explored: {len(event.branch_positions)}")
            print(f"  geDIG value: {event.gedig_value:.6f}")
            print(f"  Graph diameter change: {event.graph_diameter_before} → {event.graph_diameter_after}")
            
            if event.gedig_by_hop:
                print(f"  geDIG by hop:")
                for hop in sorted(event.gedig_by_hop.keys())[:10]:
                    gedig = event.gedig_by_hop[hop]
                    print(f"    {hop:2d}-hop: {gedig:8.6f}")
        
        # 統計
        if self.branch_events:
            gedig_values = [e.gedig_value for e in self.branch_events]
            print(f"\n📊 geDIG Statistics at Branch Completion:")
            print(f"  Average: {np.mean(gedig_values):.6f}")
            print(f"  Min: {min(gedig_values):.6f}")
            print(f"  Max: {max(gedig_values):.6f}")
            
            print(f"\n💡 Backtrack Trigger Recommendation:")
            threshold = np.mean(gedig_values) - np.std(gedig_values)
            print(f"  Suggested geDIG threshold: {threshold:.6f}")
            print(f"  (When geDIG < {threshold:.3f}, consider backtracking)")

def main():
    analyzer = BranchCompletionAnalyzer()
    analyzer.run(max_steps=200)

if __name__ == "__main__":
    main()