# Episodic Memory Navigation with geDIG - Experiment Design

## 実験概要
純粋なエピソード記憶とマルチホップ評価による迷路ナビゲーションの実証実験

## 1. 迷路設計

### 1.1 複雑な50×50迷路の生成
```python
- Recursive backtracking algorithm
- 複数のループパスを追加（探索の多様性確保）
- 開始地点: (1, 1)
- ゴール地点: (48, 48)
```

### 1.2 迷路の可視化
- 壁: 黒
- 通路: 白
- エージェント位置: 青
- ゴール: 金色
- 訪問済み: 薄い青

## 2. エピソード記憶の定義

### 2.1 純粋なエピソード記憶（訪問回数なし）
```python
Episode = {
    'id': int,                    # エピソード番号
    'position': (x, y),           # 位置
    'action': str,                # 行動 (up/right/down/left)
    'result': str,                # 結果 (success/wall/visited)
    'local_topology': float,      # 局所的な壁の数 [-1, 1]
    'is_junction': bool,          # 分岐点かどうか
    'reached_goal': bool,         # ゴール到達
    'embedding': np.array([...])  # 正規化された埋め込みベクトル
}
```

### 2.2 埋め込みベクトル（6次元）
```python
embedding = [
    x / width,                    # 位置X [0, 1]
    y / height,                   # 位置Y [0, 1]
    action_encoding,              # 行動 [-1, 1]
    result_encoding,              # 結果 {success: 1, wall: -1, visited: 0}
    local_topology,               # 局所トポロジー [-1, 1]
    goal_signal                   # ゴール信号 {reached: 10, not: 0}
]
```

## 3. geDIG計算による記憶選定

### 3.1 クエリベクトルの定義
```python
query = [
    current_x / width,
    current_y / height,
    proposed_action,
    0.0,  # unknown result
    current_topology,
    10.0  # seeking goal
]
```

### 3.2 類似度計算
- コサイン類似度ベース
- 空間的距離によるペナルティ

### 3.3 マルチホップ評価
- 1-hop: 直接隣接するエピソード
- 2-hop: 2ステップ以内で到達可能
- 3-hop: 3ステップ以内で到達可能

## 4. 深層メッセージパッシング

### 4.1 エピソードグラフの構築
- 空間的に近い（3マンハッタン距離以内）エピソードを接続

### 4.2 メッセージパッシング
```python
for round in range(depth):
    for episode in episodes:
        neighbors = get_spatial_neighbors(episode)
        if neighbors:
            # ゴール次元は強く伝播
            new_embedding = weighted_average(episode, neighbors)
```

### 4.3 洞察ベクトルの生成
- メッセージパッシング後の埋め込みを統合
- 行動価値の計算

## 5. 実験計画

### 5.1 実験条件
1. **ベースライン**: ランダムウォーク
2. **1-hop geDIG**: 局所的な記憶のみ
3. **3-hop geDIG**: マルチホップ評価
4. **3-hop + 深層メッセージパッシング**: フル実装

### 5.2 評価指標
- ゴール到達までのステップ数
- 壁への衝突回数
- カバレッジ（探索した領域の割合）
- 計算時間

### 5.3 各条件5回実行
- 統計的信頼性の確保
- 平均と標準偏差を報告

## 6. 期待される結果

### 6.1 仮説
- 訪問回数なしでも、マルチホップ評価により効率的な探索が可能
- 深層メッセージパッシングによりさらに性能向上
- 50×50迷路でも1000ステップ以内で解決

### 6.2 生物学的妥当性
- 訪問回数の明示的な記憶は不要
- 空間的な関連性とゴール信号の伝播が重要
- 純粋な経験の蓄積と想起で知的行動が創発

## 7. 実装のポイント

### 7.1 効率化
- エピソード数の上限設定（メモリ管理）
- 空間インデックスによる高速検索
- バッチメッセージパッシング

### 7.2 可視化
- 探索過程のアニメーション
- ホップ選択の分布
- メッセージパッシングの効果

## 8. 論文へのアピールポイント

1. **訪問回数なしの純粋なエピソード記憶**
2. **学習なし・報酬なしで迷路を解決**
3. **生物学的に妥当な記憶メカニズム**
4. **AGIへの新しいアプローチ**