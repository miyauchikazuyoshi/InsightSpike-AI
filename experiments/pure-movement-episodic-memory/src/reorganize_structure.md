# ディレクトリ構造の再編成計画

## 現在の構造（混在している）
```
src/
├── pure_memory_agent.py
├── pure_memory_agent_adaptive.py
├── pure_memory_agent_goal_oriented.py
├── pure_memory_agent_with_goal_beacon.py
├── gedig_aware_integrated_index.py
├── run_experiment.py
├── run_quick_test.py
├── run_15x15_experiment.py
├── test_adaptive_quick.py
├── test_11x11_adaptive.py
├── test_11x11_quick.py
├── test_goal_oriented.py
├── test_goal_oriented_7x7.py
├── test_goal_beacon.py
├── analyze_multihop.py
└── analyze_goal_effect.py
```

## 提案する新構造
```
src/
├── agents/                          # エージェント実装
│   ├── __init__.py
│   ├── base_agent.py               # 基底クラス
│   ├── pure_memory_agent.py        # 基本実装
│   ├── adaptive_agent.py           # geDIG適応的深度選択
│   ├── goal_oriented_agent.py      # ゴール指向クエリ
│   └── beacon_agent.py             # ゴールビーコン
│
├── core/                            # コア機能
│   ├── __init__.py
│   ├── gedig_index.py              # GeDIG統合インデックス
│   ├── episode_memory.py           # エピソード管理
│   └── message_passing.py          # メッセージパッシング
│
├── experiments/                     # 実験実行
│   ├── __init__.py
│   ├── quick_test.py               # クイックテスト
│   ├── maze_sizes_comparison.py    # サイズ別比較
│   ├── strategy_comparison.py      # 戦略比較
│   └── beacon_effectiveness.py     # ビーコン効果測定
│
├── analysis/                        # 分析ツール
│   ├── __init__.py
│   ├── multihop_analyzer.py        # マルチホップ分析
│   ├── goal_effect_analyzer.py     # ゴール効果分析
│   └── performance_metrics.py      # 性能指標計算
│
├── visualization/                   # 可視化
│   ├── __init__.py
│   ├── maze_visualizer.py         # 迷路と経路の可視化
│   ├── episode_graph_viz.py       # エピソードグラフ
│   └── metrics_plotter.py         # 指標のプロット
│
└── utils/                          # ユーティリティ
    ├── __init__.py
    ├── maze_generator.py           # 迷路生成
    └── config_manager.py           # 設定管理
```

## 移行計画

### Phase 1: エージェントの整理
- 共通インターフェース（BaseAgent）を定義
- 各エージェントを継承ベースに整理
- 重複コードを削除

### Phase 2: 実験スクリプトの統合
- 共通の実験フレームワークを作成
- パラメータ化された比較実験
- 結果の自動保存

### Phase 3: 可視化の追加
- 迷路と経路のリアルタイム表示
- エピソードグラフの可視化
- 学習曲線のプロット

## メリット
1. **再利用性向上**：エージェントを簡単に切り替え可能
2. **実験の系統化**：統一的な実験フレームワーク
3. **拡張性**：新しいエージェントや分析を追加しやすい
4. **可読性**：役割が明確で理解しやすい

## 実行例
```python
# 新しい構造での実験実行
from agents import AdaptiveAgent, BeaconAgent
from experiments import strategy_comparison
from visualization import maze_visualizer

# エージェントの比較実験
results = strategy_comparison.run(
    agents=[AdaptiveAgent, BeaconAgent],
    maze_sizes=[(7,7), (11,11), (15,15)],
    trials=3
)

# 結果の可視化
maze_visualizer.plot_comparison(results)
```