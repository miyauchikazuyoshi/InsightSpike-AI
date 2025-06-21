# Phase 3: GEDIG迷路実験

## 📋 実験設計

### 🎯 目標
- **試行回数削減**: 60%削減
- **収束速度**: 3倍高速収束
- **成功率**: 95%達成

### 🔬 実験設計

#### 比較対象アルゴリズム
1. **A* (A-Star)**
   - 最適解保証アルゴリズム
   - ヒューリスティック: マンハッタン距離
   - 完全性と最適性を持つ

2. **Dijkstra**
   - 最短経路アルゴリズム
   - 重み付きグラフ対応
   - 最適解保証

3. **強化学習 (Q-Learning)**
   - ε-greedy探索
   - 学習率: 0.1, 割引率: 0.9
   - エピソード最大: 1000回

4. **遺伝的アルゴリズム**
   - 個体数: 50
   - 突然変異率: 0.1
   - 交叉率: 0.8

5. **SlimeMold_GEDIG（提案手法）**
   - 粘菌インスパイア仮想管ネットワーク
   - GEDIG = α×GED + β×IG (α=0.6, β=0.4)
   - 適応的強化と減衰メカニズム

#### テスト迷路環境
1. **Simple_10x10**: 基本的な10×10迷路
2. **Complex_20x20**: 複雑な20×20迷路  
3. **Dynamic_15x15**: 動的障害物15×15迷路
4. **MultiGoal_12x12**: 複数ゴール12×12迷路

#### 評価指標
- **試行回数**: アルゴリズムの探索ステップ数
- **収束時間（秒）**: 解発見までの実行時間
- **経路長**: 最終解の経路長
- **解の品質（0-1）**: 最適解からの乖離度
- **成功率（0-1）**: 解発見率
- **探索効率**: 経路長/探索ノード数
- **GEDIGスコア**: 提案手法の内部評価値

### 🧠 GEDIG手法の詳細

#### Graph Edit Distance (GED)
前回状態との管強度変化をコストとして計算
```python
def _calculate_graph_edit_distance(self) -> float:
    total_change = 0.0
    for edge, strength in self.virtual_tubes.items():
        prev_strength = self._prev_tubes.get(edge, 1.0)
        change = abs(strength - prev_strength)
        total_change += change
    return total_change / max(len(self.virtual_tubes), 1)
```

#### Information Gain (IG)
エントロピーベースの不確実性削減量
```python
def _calculate_information_gain(self, maze) -> float:
    # 各位置の不確実性を計算
    for pos in valid_positions:
        incoming_strengths = [管強度 for 流入管]
        probabilities = [正規化された確率分布]
        entropy = -sum(p * log2(p + 1e-10) for p in probabilities)
    
    # 情報獲得 = 最大エントロピー - 現在のエントロピー
    return max_entropy - normalized_entropy
```

#### 粘菌インスパイア機構
```python
def _update_virtual_tubes(self, maze):
    # ゴールからの距離計算（BFS）
    goal_distances = self._calculate_goal_distances(maze)
    
    # 流量シミュレーション
    self._simulate_flow(maze, goal_distances)
    
    # 管強度の適応的更新
    for edge in self.virtual_tubes:
        importance = 1.0 / (1.0 + min_distance_to_goal)
        flow_factor = 1.0 + flow * 0.1
        new_strength = strength * importance * flow_factor
```

### 🛡️ 安全性機能
- 実験前の自動データバックアップ
- 無限ループ防止機構
- メモリ使用量監視

### ⚖️ バイアス検証
実験設計には以下のバイアスが含まれることを認識：
1. **試行回数定義の不統一**
2. **事前知識の利用**（ゴール距離の事前計算）
3. **早期収束判定**（緩い収束条件）

## 2025-06-21 修正メモ（Fairness Update）

1. **GED (ΔGED) の再定義**  
   * これまでは *直前ステップとの差分のみ* を計算していたため、収束時に過小評価される問題があった。  
   * 今後は **初期状態との累積差分 + Edge Insert/Delete コスト** を含めた正式 GED を実装予定。
2. **IG (ΔIG) の正規化**  
   * `max_entropy = log2(|E| + 1)` では迷路サイズ依存のバイアスが生じる。  
   * **ノード数ベース** かつ **KL-divergence** 版へ置き換え、迷路スケールに不変な指標とする。
3. **ベースライン比較の対称化**  
   * A*/Dijkstra など従来法にも GED/IG をポスト計算し、スコア列を埋めて相対比較を行う。  
   * レポートでは GEDIG は "内部報酬" 指標である旨を明示し、外部性能（試行回数・収束時間等）とは分離して評価する。

これにより「GEDIG が一方的に高い」という誤解を防ぎ、フェアな実験比較を保証する。
