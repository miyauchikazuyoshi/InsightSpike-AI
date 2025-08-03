#!/usr/bin/env python3
"""動的記憶アルゴリズムにとっての迷路問題の難易度分析。"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

print("動的記憶アルゴリズムにとっての迷路問題の本質的難易度")
print("=" * 60)

print("\n1. なぜ迷路が「簡単」になってしまうのか")
print("-" * 40)
print("• 一度通った道は記憶される → 同じ失敗を繰り返さない")
print("• 行き止まりは記憶される → 次は回避")
print("• 分岐点は記憶される → 体系的な探索が可能")
print("• 壁の位置は記憶される → 地図が徐々に完成")

print("\n2. 従来の迷路問題の前提")
print("-" * 40)
print("従来のRL研究では：")
print("• エージェントは「記憶喪失」が前提")
print("• 毎回リセットされて何も覚えていない")
print("• だから同じ失敗を何度も繰り返す")
print("• これを克服するために「学習」が必要だった")

print("\n3. 動的記憶で変わるもの")
print("-" * 40)
comparison = """
| 課題 | 従来RL | 動的記憶 | 影響 |
|------|--------|----------|------|
| 同じ道を何度も通る | よくある | 避けられる | 効率↑ |
| 行き止まりに何度も入る | よくある | 一度だけ | 効率↑ |
| 最適経路の発見 | 多数の試行が必要 | 自然に収束 | 効率↑ |
| 部分観測性 | 大きな問題 | 徐々に解決 | 難易度↓ |
"""
print(comparison)

print("\n4. それでも残る挑戦")
print("-" * 40)
print("動的記憶があっても難しい要素：")
print("• 初回探索の効率性（どう探索するか）")
print("• 大規模迷路でのメモリ管理")
print("• 動的に変化する迷路")
print("• 複数の目標がある場合の優先順位")
print("• 部分的に観測可能な環境での推論")

print("\n5. より難しいタスクへの発展")
print("-" * 40)
print("迷路が簡単になったなら、次の挑戦：")
print("")
print("【レベル1: 静的迷路】← 今ここ")
print("• 固定された壁")
print("• 単一のゴール")
print("• 完全に観測可能（隣接）")
print("")
print("【レベル2: 動的環境】")
print("• 移動する障害物")
print("• 時間制限")
print("• 複数の目標")
print("")
print("【レベル3: 不確実性のある環境】") 
print("• 霧で視界が制限")
print("• ノイズのあるセンサー")
print("• 確率的な壁（ドアが開いたり閉じたり）")
print("")
print("【レベル4: マルチエージェント】")
print("• 他のエージェントとの協調/競争")
print("• 通信制限")
print("• 役割分担")
print("")
print("【レベル5: 実世界タスク】")
print("• 連続空間")
print("• 物理シミュレーション")
print("• 実ロボット制御")

# 難易度の定量的分析
print("\n\n6. 難易度の定量的分析")
print("-" * 40)

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig

config = MazeNavigatorConfig(
    ged_weight=1.0,
    ig_weight=2.0,
    temperature=1.0,
    exploration_epsilon=0.0
)

# 異なるサイズの迷路でテスト
sizes = [(5, 5), (10, 10), (20, 20), (30, 30)]
results = []

print("\n迷路サイズと必要ステップ数の関係：")
print("サイズ | 理論最小 | 実際 | 効率")
print("-------|----------|------|------")

for size in sizes:
    np.random.seed(42)
    maze = SimpleMaze(size=size, maze_type='dfs')
    navigator = ExperienceMemoryNavigator(config)
    
    obs = maze.reset()
    steps = 0
    
    for _ in range(size[0] * size[1] * 10):  # 十分なステップ数
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        steps += 1
        if done and maze.agent_pos == maze.goal_pos:
            break
    
    # マンハッタン距離（理論最小）
    min_steps = abs(maze.start_pos[0] - maze.goal_pos[0]) + abs(maze.start_pos[1] - maze.goal_pos[1])
    efficiency = min_steps / steps if steps > 0 else 0
    
    print(f"{size[0]}x{size[1]} | {min_steps:8} | {steps:4} | {efficiency:.2%}")
    results.append((size[0] * size[1], steps, min_steps))

# グラフ化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

maze_areas = [r[0] for r in results]
actual_steps = [r[1] for r in results]
min_steps = [r[2] for r in results]

ax1.plot(maze_areas, actual_steps, 'b-o', label='実際のステップ数')
ax1.plot(maze_areas, min_steps, 'r--o', label='理論最小ステップ数')
ax1.set_xlabel('迷路の面積（マス数）')
ax1.set_ylabel('ステップ数')
ax1.set_title('迷路サイズと探索効率')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 効率性の分析
efficiencies = [m/a if a > 0 else 0 for m, a in zip(min_steps, actual_steps)]
ax2.plot(maze_areas, efficiencies, 'g-o')
ax2.axhline(y=0.5, color='r', linestyle='--', label='50%効率')
ax2.set_xlabel('迷路の面積（マス数）')
ax2.set_ylabel('探索効率（理論最小/実際）')
ax2.set_title('迷路サイズと探索効率の関係')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('maze_difficulty_scaling.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n→ グラフを maze_difficulty_scaling.png に保存しました")

print("\n\n【結論】")
print("=" * 60)
print("はい、動的記憶により単純な迷路問題は「簡単」になります。")
print("")
print("しかし、これは弱点ではなく強みです：")
print("• 人間にとっても迷路は「記憶があれば簡単」")
print("• 実世界の問題も「経験を活かせば簡単」になるべき")
print("• より複雑な問題に挑戦する準備ができた")
print("")
print("次のステップ：")
print("→ 動的環境、マルチエージェント、実世界タスクへ！")
print("=" * 60)