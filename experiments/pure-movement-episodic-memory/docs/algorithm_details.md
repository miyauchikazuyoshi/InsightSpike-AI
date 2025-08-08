# アルゴリズム詳細仕様

## 1. 全体アーキテクチャ

```
┌─────────────────────────────────────────────┐
│                Navigator                     │
├─────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Episode  │  │  GeDIG   │  │ Message  │ │
│  │  Memory  │→ │  Index   │→ │ Passing  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│       ↓             ↓              ↓        │
│  ┌──────────────────────────────────────┐  │
│  │        Insight Generation            │  │
│  └──────────────────────────────────────┘  │
│       ↓                                     │
│  ┌──────────┐                              │
│  │  Action  │                              │
│  │ Selection│                              │
│  └──────────┘                              │
└─────────────────────────────────────────────┘
```

## 2. メッセージパッシングアルゴリズム

### 2.1 初期化フェーズ

```python
def initialize_messages(indices, scores):
    messages = {}
    for i, idx in enumerate(indices):
        # 初期重みは検索スコアに基づく
        messages[idx] = scores[i]
    return messages
```

### 2.2 伝播フェーズ（各ホップ）

```python
def propagate_hop(messages, graph, depth):
    new_messages = {}
    decay = 0.8 ** depth  # 指数減衰
    
    for node, value in messages.items():
        # セルフループ（減衰あり）
        new_messages[node] = value * 0.7 * decay
        
        # 隣接ノードへの伝播
        for neighbor in graph.neighbors(node):
            edge_data = graph[node][neighbor]
            
            # エッジ品質の計算
            weight = edge_data['weight']      # コサイン類似度
            gedig = edge_data['gedig']        # GeDIG値
            quality = weight * (1 - gedig * 0.3)
            
            # メッセージ伝播
            propagation = value * quality * decay
            
            # 最大値更新
            if neighbor not in new_messages:
                new_messages[neighbor] = propagation
            else:
                new_messages[neighbor] = max(
                    new_messages[neighbor], 
                    propagation
                )
    
    return new_messages
```

### 2.3 集約フェーズ

```python
def aggregate_messages(messages, episodes):
    direction_vector = np.zeros(7)
    total_weight = 0
    
    for idx, value in messages.items():
        episode = episodes[idx]
        vec = episode['vec']
        
        # エピソードタイプによる重み付け
        if episode['type'] == 'movement':
            if episode['success']:
                weight = value * 1.5  # 成功移動を重視
            else:
                weight = value * 0.5  # 失敗移動は参考程度
        else:  # visual
            weight = value * 1.0  # 視覚情報は中立
        
        direction_vector += vec * weight
        total_weight += weight
    
    # 正規化
    if total_weight > 0:
        direction_vector /= total_weight
    
    return direction_vector
```

## 3. 洞察生成プロセス

### 3.1 多段階検索と統合

```python
def generate_insight(query, k=20, max_depth=5):
    insights = []
    qualities = []
    
    for depth in range(1, max_depth + 1):
        # 深度ごとに検索
        indices, scores = search_episodes(query, k=k+depth*5)
        
        # メッセージパッシング
        messages = initialize_messages(indices, scores)
        for d in range(depth):
            messages = propagate_hop(messages, graph, d)
        
        # 集約
        insight = aggregate_messages(messages, episodes)
        quality = evaluate_quality(messages)
        
        insights.append(insight)
        qualities.append(quality)
    
    # 品質に基づく重み付き平均
    final_insight = weighted_average(insights, qualities)
    return final_insight
```

### 3.2 品質評価

```python
def evaluate_quality(messages):
    # メッセージの分散（多様性）
    values = list(messages.values())
    diversity = np.std(values)
    
    # メッセージの強度（確信度）
    strength = np.mean(values)
    
    # 総合品質
    quality = strength * (1 + diversity)
    return quality
```

## 4. 方向抽出と確率変換

### 4.1 洞察から方向成分を抽出

```python
def extract_direction(insight_vector):
    # 次元2が方向情報
    direction_value = insight_vector[2]
    
    # 成功度による信頼度
    confidence = insight_vector[3]
    
    # 4方向の確率分布を生成
    probs = np.ones(4) * 0.1  # ベース確率
    
    # メイン方向への確率割り当て
    main_dir = int(round(direction_value * 3))
    if 0 <= main_dir < 4:
        probs[main_dir] += 0.6 * confidence
    
    # 正規化
    probs = probs / probs.sum()
    
    return probs
```

### 4.2 ソフトな確率割り当て

```python
def soft_assignment(direction_value, confidence):
    probs = np.zeros(4)
    
    # 連続値を離散確率に変換
    for i in range(4):
        # ガウシアンカーネルによるソフト割り当て
        center = i / 3.0
        distance = abs(direction_value - center)
        probs[i] = np.exp(-distance**2 / 0.1) * confidence
    
    # 正規化
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(4) / 4  # 一様分布
    
    return probs
```

## 5. GeDIG統合インデックス

### 5.1 エピソード追加時の処理

```python
def add_episode(episode_vector, metadata):
    # 既存エピソードとの類似度計算
    similarities = compute_similarities(episode_vector, all_episodes)
    
    # 閾値以上の類似エピソードとエッジ作成
    for idx, sim in enumerate(similarities):
        if sim > similarity_threshold:
            # GeDIG値の計算
            gedig = compute_gedig(episode_vector, all_episodes[idx])
            
            # エッジ追加
            if gedig < gedig_threshold:
                add_edge(new_idx, idx, weight=sim, gedig=gedig)
```

### 5.2 GeDIG計算

```python
def compute_gedig(vec1, vec2):
    # 情報理論的差異の計算
    # (簡略化版、実際はより複雑)
    
    # エントロピー計算
    h1 = entropy(vec1)
    h2 = entropy(vec2)
    
    # 相互情報量
    mi = mutual_information(vec1, vec2)
    
    # GeDIG値
    gedig = 1 - (2 * mi) / (h1 + h2)
    
    return gedig
```

## 6. 行動選択

### 6.1 確率的行動選択

```python
def select_action(probabilities):
    # 純粋な確率分布からサンプリング
    actions = ['up', 'right', 'down', 'left']
    return np.random.choice(actions, p=probabilities)
```

### 6.2 探索と活用のバランス（禁止）

以下のような人工的な調整は**使用しない**：

```python
# ❌ 禁止例：ε-greedy
if random.random() < epsilon:
    return random_action()
else:
    return best_action()

# ❌ 禁止例：温度パラメータ
probs = softmax(scores / temperature)
```

## 7. 性能最適化

### 7.1 インデックスの効率化

- **FAISS統合**: 大規模な類似検索の高速化
- **グラフプルーニング**: エッジ数の制限
- **バッチ処理**: 複数クエリの同時処理

### 7.2 メモリ管理

```python
class EpisodeMemory:
    def __init__(self, max_episodes=10000):
        self.max_episodes = max_episodes
        self.episodes = []
    
    def add(self, episode):
        if len(self.episodes) >= self.max_episodes:
            # 古い非成功エピソードを優先的に削除
            self.prune_old_failures()
        self.episodes.append(episode)
    
    def prune_old_failures(self):
        # 成功エピソードは保持
        failures = [e for e in self.episodes if not e['success']]
        if failures:
            # 最も古い失敗を削除
            oldest = min(failures, key=lambda e: e['timestamp'])
            self.episodes.remove(oldest)
```

## 8. デバッグとモニタリング

### 8.1 メトリクス収集

```python
metrics = {
    'hop_usage': {},      # 各深度の使用頻度
    'quality_scores': [], # 品質スコアの推移
    'wall_hits': 0,       # 壁衝突回数
    'unique_positions': set(),  # 訪問位置
    'episode_types': {},  # エピソードタイプ別統計
}
```

### 8.2 可視化

- **パス可視化**: 移動経路の表示
- **ヒートマップ**: 訪問頻度の可視化
- **グラフ構造**: エピソード間の関係
- **深度分析**: 各ホップの寄与度

## 9. 制約と制限

### 9.1 計算量

- 検索: O(n log n) where n = エピソード数
- メッセージパッシング: O(d × e) where d = 深度, e = エッジ数
- 全体: O(n log n + d × e)

### 9.2 メモリ使用量

- エピソード: 7 × 4 bytes × max_episodes
- グラフ: O(e) where e ≤ max_edges_per_node × n
- 総計: ~400KB for 10,000 episodes

### 9.3 収束性

- 保証なし（純粋な記憶ベースのため）
- ローカルミニマムの可能性あり
- 長期的には経験の蓄積により改善