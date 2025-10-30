---
status: active
category: insight
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# InsightSpike 技術実装の詳細

## 目次

1. [プロンプト生成メカニズム](#プロンプト生成メカニズム)
2. [メッセージパッシングの詳細](#メッセージパッシングの詳細)
3. [類似度計算の仕組み](#類似度計算の仕組み)
4. [エピソード統合プロセス](#エピソード統合プロセス)
5. [現在の実装の制約と改善案](#現在の実装の制約と改善案)

---

## プロンプト生成メカニズム

### InsightSpike vs Simple RAG のプロンプト比較

#### Simple RAG のプロンプト（約300文字）
```
Context: ['doc1 content...', 'doc2 content...']
Query: What is machine learning?
Answer:
```

#### InsightSpike のプロンプト（約800文字）
```
Context Enhanced with Graph Analysis:
- Episode 1: "機械学習は..." (relevance: 0.92, C-value: 0.875)
- Episode 2: "教師あり学習..." (relevance: 0.88, C-value: 0.734)
[Graph Structure Insights]
- Connected Components: 1
- Graph Density: 0.67
- Recent Changes: New connection between ML and Deep Learning detected
[Insight Detection]
- ΔGED: -0.45 (構造が単純化)
- ΔIG: 0.32 (情報利得あり)
- Spike Detected: True (Understanding pattern recognized)
Query: What is machine learning?
Answer:
```

### 4層エンリッチメントプロセス

```
Layer 1: Error Monitor (50 → 50 chars)
├─ エラーチェックのみ
└─ コンテンツ変更なし

Layer 2: Memory Manager (50 → 200 chars)
├─ 関連エピソード取得
├─ C値による重み付け
└─ 基本的なコンテキスト追加

Layer 3: Graph Reasoner (200 → 600 chars)
├─ グラフ構造分析
├─ GED/IG計算
├─ 洞察スパイク検出
└─ グラフメトリクス追加

Layer 4: LLM Interface (600 → 800 chars)
├─ 最終フォーマット整形
├─ プロンプトテンプレート適用
└─ LLM用に最適化
```

---

## メッセージパッシングの詳細

### 現在の実装フロー

1. **エピソード取得**
   ```python
   # FAISSによる類似度検索
   distances, indices = faiss_index.search(query_embedding, k=5)
   similarities = 1 / (1 + distances)  # L2距離から類似度へ
   ```

2. **グラフ構築**
   ```python
   for i, ep1 in enumerate(episodes):
       for j, ep2 in enumerate(episodes[i+1:]):
           similarity = calculate_similarity(ep1, ep2)
           if similarity > 0.7:  # 固定閾値
               graph.add_edge(i, j, weight=similarity)
   ```

3. **メッセージ集約**
   ```python
   # 各ノードのメッセージを近傍に伝播
   for node in graph.nodes():
       neighbors = graph.neighbors(node)
       message = aggregate_messages(neighbors)
       node.update_state(message)
   ```

### エピソード統合の仕組み

**重要：複数エピソードは1つのグラフに統合される**

```
Episode A ─┐
Episode B ─┼─→ 統合グラフ → メッセージパッシング → 強化されたコンテキスト
Episode C ─┘
```

並列パスではなく、全エピソードが1つのグラフ構造を形成し、
その上でメッセージパッシングが実行される。

---

## 類似度計算の仕組み

### 使用モデルと埋め込み

- **モデル**: sentence-transformers/all-MiniLM-L6-v2
- **埋め込み次元**: 384次元
- **言語**: 多言語対応（日本語含む）

### 計算プロセス

1. **テキストの埋め込み化**
   ```python
   # SentenceTransformerによる埋め込み
   embeddings = model.encode(texts)  # shape: (n_texts, 384)
   ```

2. **L2距離の計算**
   ```python
   # FAISSを使用した高速計算
   L2_distance = ||embedding1 - embedding2||²
   ```

3. **類似度への変換**
   ```python
   similarity = 1 / (1 + L2_distance)
   # L2距離が0 → 類似度1.0（完全一致）
   # L2距離が大きい → 類似度0に近づく
   ```

### エッジ形成の判定

```python
EDGE_THRESHOLD = 0.7

if similarity > EDGE_THRESHOLD:
    # エッジを形成
    graph.add_edge(node1, node2, weight=similarity)
else:
    # エッジなし（孤立ノード）
    pass
```

**閾値の影響：**
- similarity = 0.69 → エッジなし → 洞察検出されない
- similarity = 0.71 → エッジ形成 → 洞察検出の可能性

わずか0.02の差が、最終的な出力を劇的に変える！

---

## エピソード統合プロセス

### 現在の統合メカニズム

1. **全エピソードの収集**
   ```python
   all_episodes = []
   for query_part in query_parts:
       episodes = retrieve_similar_episodes(query_part)
       all_episodes.extend(episodes)
   ```

2. **重複除去と統合**
   ```python
   unique_episodes = deduplicate_by_embedding_similarity(all_episodes)
   ```

3. **統一グラフの構築**
   ```python
   graph = build_unified_graph(unique_episodes)
   # 全エピソードが1つのグラフに統合される
   ```

4. **グラフ上でのメッセージパッシング**
   ```python
   enhanced_episodes = run_message_passing(graph)
   # グラフ全体で情報が伝播・強化される
   ```

### C値報酬システムの影響

```python
# 洞察スパイク検出時
if insight_spike_detected:
    # C値が急上昇（例：0.5 → 0.985）
    for episode in contributing_episodes:
        episode.c_value *= 1.97  # 大幅な報酬
        
# 通常時
else:
    # C値は微減（例：0.5 → 0.495）
    for episode in all_episodes:
        episode.c_value *= 0.99  # わずかな減衰
```

---

## 現在の実装の制約と改善案

### 制約事項

1. **固定閾値問題**
   - 0.7という固定値
   - 文脈を考慮しない
   - 創造的な関連性を見逃す

2. **即時確定問題**
   - エッジは一度作ると変更されない
   - 最適化の余地なし
   - 後から良い構造が見つかっても修正不可

3. **線形的な処理**
   - 類似度計算 → エッジ形成 → 評価
   - フィードバックループなし
   - 探索的な最適化なし

### 改善提案のまとめ

1. **動的閾値**
   ```python
   threshold = adaptive_threshold(context, domain, task_type)
   # 文脈に応じて0.5〜0.8の間で調整
   ```

2. **仮結線システム**
   ```python
   tentative_edges = propose_edges(similarity_matrix, threshold=0.5)
   evaluated_edges = evaluate_with_ged_ig(tentative_edges)
   final_edges = select_best_edges(evaluated_edges)
   ```

3. **探索的最適化**
   ```python
   for iteration in range(max_iterations):
       candidate_graphs = generate_variations(current_graph)
       best_graph = select_by_ged_ig_score(candidate_graphs)
       if converged(best_graph, current_graph):
           break
       current_graph = best_graph
   ```

---

## 実装の優先順位

### 短期的改善（影響大・実装容易）

1. **閾値の動的調整**
   - タスクタイプに応じた閾値設定
   - ドメイン別の最適値学習

2. **エッジタイプの追加**
   - 直接エッジ
   - ブリッジエッジ
   - 仮説エッジ

### 中期的改善（構造的変更）

1. **仮結線メカニズム**
   - 評価前の仮エッジ
   - バッチ評価
   - 最適選択

2. **メッセージパッシング拡張**
   - 空白地帯検出
   - 仮説生成
   - 創発的概念

### 長期的ビジョン

1. **完全な二重システム**
   - 理解モード（現在のGED/IG）
   - 閃きモード（新メッセージパッシング）
   - 適応的モード選択

---

## パフォーマンス考慮事項

### 現在のボトルネック

1. **埋め込み計算**: O(n) - 各エピソードごと
2. **類似度計算**: O(n²) - 全ペア比較
3. **グラフ構築**: O(n²) - エッジ評価
4. **メッセージパッシング**: O(n·m) - nノード、m反復

### 最適化戦略

1. **バッチ処理**
   ```python
   # 埋め込みをバッチで計算
   embeddings = model.encode(texts, batch_size=32)
   ```

2. **近似最近傍探索**
   ```python
   # FAISSのIVFインデックス使用
   index = faiss.IndexIVFFlat(quantizer, d, nlist)
   ```

3. **グラフのスパース化**
   ```python
   # 重要なエッジのみ保持
   pruned_graph = prune_low_weight_edges(graph, min_weight=0.5)
   ```

---

## まとめ

InsightSpikeの現在の実装は、固定閾値による即時的なグラフ構築と、GED/IGによる「理解」の検出に焦点を当てています。提案される改善により、より柔軟で創造的な「閃き」の検出が可能になり、人間の知的プロセスにより近いシステムが実現されます。

---

*このドキュメントは、InsightSpikeの技術実装の詳細をまとめたものです。*
*Last Updated: 2024-01-19*