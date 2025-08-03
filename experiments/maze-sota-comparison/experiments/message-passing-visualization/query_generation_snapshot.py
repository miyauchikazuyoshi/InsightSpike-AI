#!/usr/bin/env python3
"""行き止まりから分岐点に戻った時のクエリ生成スナップショット"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


def create_query_generation_snapshot():
    """分岐点でのクエリ生成の様子を1枚の画像で表現"""
    
    # 迷路の設定（シンプルなT字路）
    fig = plt.figure(figsize=(20, 10))
    
    # 左側：迷路の状況
    ax1 = plt.subplot(121)
    ax1.set_title('Maze Situation: Returning to Junction after Dead End', fontsize=16)
    
    # 簡単な迷路を手動で作成（T字路）
    maze_size = (7, 7)
    ax1.set_xlim(-0.5, maze_size[1] - 0.5)
    ax1.set_ylim(-0.5, maze_size[0] - 0.5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # 迷路の構造（T字路）
    walls = [
        # 外壁
        *[(0, j) for j in range(7)],
        *[(6, j) for j in range(7)],
        *[(i, 0) for i in range(7)],
        *[(i, 6) for i in range(7)],
        # 内部の壁（T字路を作る）
        (1, 2), (1, 4),
        (2, 2), (2, 4),
        (4, 2), (4, 4),
        (5, 2), (5, 4),
    ]
    
    # 壁を描画
    for i, j in walls:
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor='black')
        ax1.add_patch(rect)
        
    # 特別な位置
    start_pos = (5, 3)
    junction_pos = (3, 3)
    dead_end_left = (3, 1)
    dead_end_right = (3, 5)
    goal_pos = (1, 3)
    
    # スタート地点
    rect = patches.Rectangle((start_pos[1]-0.5, start_pos[0]-0.5), 1, 1, 
                           facecolor='lightgreen', alpha=0.5)
    ax1.add_patch(rect)
    ax1.text(start_pos[1], start_pos[0], 'S', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # ゴール
    rect = patches.Rectangle((goal_pos[1]-0.5, goal_pos[0]-0.5), 1, 1, 
                           facecolor='yellow', alpha=0.5)
    ax1.add_patch(rect)
    ax1.text(goal_pos[1], goal_pos[0], 'G', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # 分岐点（青）
    circle = Circle((junction_pos[1], junction_pos[0]), 0.3, 
                   facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(junction_pos[1], junction_pos[0], 'J', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # 行き止まり（赤×）
    ax1.plot(dead_end_left[1], dead_end_left[0], 'rx', markersize=20, markeredgewidth=3)
    ax1.plot(dead_end_right[1], dead_end_right[0], 'rx', markersize=20, markeredgewidth=3)
    
    # エージェントの現在位置（分岐点に戻ってきた）
    agent_circle = Circle((junction_pos[1], junction_pos[0]), 0.2, 
                         facecolor='darkblue', edgecolor='white', linewidth=2)
    ax1.add_patch(agent_circle)
    
    # 経路の軌跡
    # 左の行き止まりへの経路（失敗）
    ax1.annotate('', xy=(dead_end_left[1], dead_end_left[0]), 
                xytext=(junction_pos[1], junction_pos[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))
    ax1.text(dead_end_left[1]-0.3, dead_end_left[0]+0.3, 'Dead End!', 
            fontsize=8, color='red')
    
    # 右の行き止まりへの経路（失敗）
    ax1.annotate('', xy=(dead_end_right[1], dead_end_right[0]), 
                xytext=(junction_pos[1], junction_pos[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))
    ax1.text(dead_end_right[1]+0.3, dead_end_right[0]+0.3, 'Dead End!', 
            fontsize=8, color='red')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)
    
    # 右側：エピソード記憶グラフ
    ax2 = plt.subplot(122)
    ax2.set_title('Episode Memory Graph with Query Generation', fontsize=16)
    
    # グラフ構築
    G = nx.DiGraph()
    
    # ノード定義
    nodes = {
        'Q1': {'label': 'Query:\n"How to reach goal?"', 'color': 'yellow', 'size': 1200},
        'E1': {'label': 'Episode 1:\nFrom S to J\n(Success)', 'color': 'lightblue', 'size': 800},
        'E2': {'label': 'Episode 2:\nFrom J, go Left\n(Wall collision)', 'color': 'red', 'size': 800},
        'E3': {'label': 'Episode 3:\nFrom J, go Right\n(Wall collision)', 'color': 'red', 'size': 800},
        'AG1': {'label': 'Action Guidance:\nAt J, avoid Left', 'color': 'white', 'size': 700},
        'AG2': {'label': 'Action Guidance:\nAt J, avoid Right', 'color': 'white', 'size': 700},
        'Q2': {'label': 'NEW Query:\n"At J, which way?\nNot Left, Not Right"', 'color': 'gold', 'size': 1500},
    }
    
    # ノードを追加
    for node_id, attrs in nodes.items():
        G.add_node(node_id)
        
    # エッジを追加（エピソードの流れ）
    edges = [
        ('Q1', 'E1', 'solid'),
        ('E1', 'E2', 'solid'),
        ('E1', 'E3', 'solid'),
        ('E2', 'AG1', 'dashed'),  # メッセージパッシング
        ('E3', 'AG2', 'dashed'),  # メッセージパッシング
        ('AG1', 'Q2', 'dotted'),  # クエリ生成
        ('AG2', 'Q2', 'dotted'),  # クエリ生成
    ]
    
    for u, v, style in edges:
        G.add_edge(u, v, style=style)
        
    # レイアウト
    pos = {
        'Q1': (0, 2),
        'E1': (0, 1),
        'E2': (-1.5, 0),
        'E3': (1.5, 0),
        'AG1': (-1.5, -1),
        'AG2': (1.5, -1),
        'Q2': (0, -2),
    }
    
    # ノード描画
    for node_id, (x, y) in pos.items():
        attrs = nodes[node_id]
        if attrs['color'] == 'white':
            # 白いノードは黒い縁を付ける
            circle = Circle((x, y), 0.5, facecolor='white', 
                          edgecolor='black', linewidth=2)
        else:
            circle = Circle((x, y), 0.5, facecolor=attrs['color'], 
                          edgecolor='darkgray', linewidth=1)
        ax2.add_patch(circle)
        
    # エッジ描画
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        style = data['style']
        if style == 'solid':
            ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        elif style == 'dashed':
            ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='orange', 
                                      lw=2, linestyle='dashed'))
        else:  # dotted
            ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='green', 
                                      lw=3, linestyle='dotted'))
            
    # ラベル描画
    for node_id, (x, y) in pos.items():
        attrs = nodes[node_id]
        # 複数行のテキストを描画
        lines = attrs['label'].split('\n')
        for i, line in enumerate(lines):
            y_offset = y + 0.1 - i * 0.15
            fontsize = 10 if i == 0 else 8
            fontweight = 'bold' if i == 0 else 'normal'
            ax2.text(x, y_offset, line, ha='center', va='center',
                    fontsize=fontsize, fontweight=fontweight)
            
    # 凡例
    legend_elements = [
        patches.Patch(color='yellow', label='Goal Query'),
        patches.Patch(color='red', label='Wall Collision Episode'),
        patches.Patch(color='white', label='Action Guidance (Message Passing)'),
        patches.Patch(color='lightblue', label='Successful Move Episode'),
        patches.Patch(color='gold', label='Generated Query at Junction'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 説明テキスト
    ax2.text(0, -3, 'Message Passing creates new query:\n"Which direction at junction?"\nwith constraints from dead ends',
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3.5, 3)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # 保存
    plt.savefig('query_generation_at_junction.png', dpi=200, bbox_inches='tight')
    print("✅ query_generation_at_junction.png として保存しました")
    print("\n説明:")
    print("- 左側: T字路の迷路で、エージェントが両方の行き止まりを経験後、分岐点Jに戻った状況")
    print("- 右側: エピソード記憶グラフ")
    print("  - 赤: 壁衝突エピソード（行き止まり）")
    print("  - 黄: 元のゴールクエリ")
    print("  - 白: メッセージパッシングによる行動指針")
    print("  - 金: 分岐点で生成された新しいクエリ")
    print("  - 青: 成功した移動エピソード")
    
    plt.show()


if __name__ == "__main__":
    create_query_generation_snapshot()