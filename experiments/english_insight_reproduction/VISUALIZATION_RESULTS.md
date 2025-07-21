# InsightSpike ビジュアライゼーション結果

## 概要
知識グラフの構築と洞察スパイク検出プロセスを可視化しました。

## グラフ構造
- **ノード数**: 10（知識項目）
- **エッジ数**: 24（意味的類似度 > 0.5 の接続）

## ビジュアライゼーション結果

### 1. 知識グラフ全体図
- **ファイル**: `insight_detection_combined.png`
- **内容**: 
  - 左上：完全な知識グラフ（10ノード、24エッジ）
  - 右上〜左下：各質問に対する洞察検出の可視化

### 2. 質問1: "How are energy and information fundamentally related?"
- **結果**: ✨ **スパイク検出！**（スコア: 0.408）
- **ファイル**: `insight_detection_q1.png`, `insight_detection_animation_q1.gif`
- **検出された概念の結びつき**:
  - 🔴 Node 2: "Information and entropy have a deep mathematical relationship"
  - 🔴 Node 8: "Energy, information, and consciousness are different aspects..."
  - 🔴 Node 4: "Energy, information, and entropy form the fundamental trinity..."
- **重要な接続**: これら3つのノードが相互に強く接続されており、高いスパイクスコアを生成

### 3. 質問2: "Can consciousness be understood through information theory?"
- **結果**: ✨ **スパイク検出！**（スコア: 0.410）
- **ファイル**: `insight_detection_q2.png`
- **検出された概念の結びつき**:
  - 🔴 Node 6: "Consciousness might be quantified by Integrated Information Theory"
  - 🔴 Node 8: "Energy, information, and consciousness are different aspects..."
  - 🔴 Node 2: "Information and entropy have a deep mathematical relationship"
- **重要な接続**: 意識と情報理論の直接的な関連が複数のパスで確認

### 4. 質問3: "How does life organize information against entropy?"
- **結果**: 📝 **スパイクなし**（スコア: 0.249、閾値0.3未満）
- **ファイル**: `insight_detection_q3.png`
- **関連ノード**: 
  - Node 2: "Information and entropy..."
  - Node 8: "Energy, information, and consciousness..."
  - Node 5: "Life is a dissipative structure..."
- **問題**: 関連ノード間の接続性が不十分（閾値を下回る）

## ビジュアライゼーションの見方

### 色の意味
- 🔴 **赤色ノード**: 質問に最も関連する上位3ノード
- 🟡 **黄色ノード**: 関連はあるが重要度が低いノード
- 🟢 **緑色ノード**: 関連性の低いノード
- **赤い太線**: スパイクを形成する重要な接続

### アニメーション（`insight_detection_animation_q1.gif`）
1. **Phase 1 (0-30フレーム)**: 完全な知識グラフを表示
2. **Phase 2 (30-60フレーム)**: 関連ノードをハイライト
3. **Phase 3 (60-90フレーム)**: スパイク検出と接続の脈動表示

## 技術的詳細

### グラフ構築アルゴリズム
```python
# 1. 各知識項目を埋め込みベクトル化
embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(text)

# 2. コサイン類似度 > 0.5 でエッジを作成
similarity = cosine_similarity(embedding1, embedding2)
if similarity > 0.5:
    G.add_edge(node1, node2, weight=similarity)

# 3. スパイクスコア計算
connectivity_ratio = connections / max_possible_connections
avg_weight = total_weight / num_connections
spike_score = connectivity_ratio * avg_weight
```

### 洞察検出の妥当性
- **質問1と2**: 複数の関連概念が密に接続され、明確なスパイクを形成
- **質問3**: 関連概念は存在するが、相互接続が弱く、スパイク形成に至らず

## 結論

ビジュアライゼーションにより、InsightSpikeが：
1. **知識項目間の意味的関連を正確に捉えている**
2. **質問に対して適切な関連ノードを特定している**
3. **ノード間の接続パターンから洞察の存在を検出している**

ことが視覚的に確認できました。