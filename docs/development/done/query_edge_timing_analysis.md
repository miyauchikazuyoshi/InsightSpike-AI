---
status: active
category: edges
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# クエリエッジのタイミング設計分析

## 問題の概要

クエリエッジの追加タイミングによって、エピソード分岐の判定に影響を与える可能性がある。特に洞察に至らなかったクエリのエッジが多数追加されると、エピソードの文脈が分散し、意図しない分岐が発生する可能性がある。

## シナリオ分析

### シナリオ1: 即時エッジ追加の問題

```
1. エピソード「apple」が存在（果物と企業の両方の文脈）
2. クエリ「What is Apple's stock price?」→ 洞察なし
3. 即座にquery_retrievalエッジを追加
4. 次のクエリ「Is apple healthy?」の処理時
   → 「apple」の隣接ノードに企業関連クエリが増える
   → 文脈の分散を検出して分岐が発生
   → 本来不要な分岐が生じる
```

### シナリオ2: バッチ処理の問題

```
1. クエリエッジをバッファに蓄積
2. 定期的にバッチで追加
3. 問題：リアルタイムな学習ができない
4. 問題：メモリにバッファを保持する必要
```

## 設計オプション

### オプション1: 遅延評価型エッジ追加

```python
class QueryEdgeManager:
    def __init__(self):
        self.pending_edges = []  # 保留中のエッジ
        self.evaluation_threshold = 10  # 評価までのクエリ数
        
    def add_query_edge(self, edge):
        """エッジを保留リストに追加"""
        self.pending_edges.append({
            **edge,
            "status": "pending",
            "added_at": time.time()
        })
        
        # 閾値に達したら評価
        if len(self.pending_edges) >= self.evaluation_threshold:
            self.evaluate_pending_edges()
    
    def evaluate_pending_edges(self):
        """保留中のエッジを評価して確定"""
        for edge in self.pending_edges:
            # 同じエピソードへの他のクエリを確認
            related_queries = self.get_related_queries(edge["target_id"])
            
            # 洞察生成率を計算
            spike_rate = sum(1 for q in related_queries if q["led_to_spike"]) / len(related_queries)
            
            if spike_rate > 0.3:  # 30%以上が洞察に繋がっている
                edge["status"] = "confirmed"
                edge["weight_modifier"] = spike_rate
            else:
                edge["status"] = "low_value"
                edge["weight_modifier"] = 0.5
```

### オプション2: 仮想グラフでの分岐評価

```python
def should_branch_with_query_edges(episode_id, graph, pending_query_edges):
    """クエリエッジを考慮した分岐判定"""
    
    # 仮想的にクエリエッジを追加したグラフを作成
    virtual_graph = graph.copy()
    
    # クエリノードを仮想的に追加
    query_nodes = []
    for edge in pending_query_edges:
        if edge["target_id"] == episode_id:
            query_node = f"query_{edge['source_id']}"
            virtual_graph.add_node(
                query_node,
                type="query",
                led_to_spike=edge["metadata"]["led_to_spike"]
            )
            virtual_graph.add_edge(
                query_node,
                episode_id,
                relation=edge["edge_type"]
            )
            query_nodes.append(query_node)
    
    # 文脈クラスタリングを実行
    clusters = detect_context_clusters(episode_id, virtual_graph)
    
    # クエリノードが異なるクラスタに分散している場合は分岐を推奨
    query_clusters = defaultdict(list)
    for cluster in clusters:
        for node_id in cluster["neighbor_ids"]:
            if node_id in query_nodes:
                query_clusters[cluster["type"]].append(node_id)
    
    # 分岐判定
    if len(query_clusters) > 1:
        # クエリが複数の文脈に分散
        return True, clusters
    
    return False, []
```

### オプション3: エッジタイプによる重み付け戦略

```python
EDGE_BRANCHING_WEIGHTS = {
    "query_spike": 1.0,      # 洞察生成クエリは分岐に強く影響
    "query_retrieval": 0.3,  # 通常の検索は弱い影響
    "query_bypass": 0.1,     # バイパスは最小限の影響
    "branch": 0.8,           # 既存の分岐関係
    "semantic": 0.6          # 意味的関係
}

def calculate_branching_score(episode_id, graph):
    """エピソードの分岐スコアを計算"""
    neighbors = graph.neighbors(episode_id)
    
    # エッジタイプごとの重み付き集計
    context_scores = defaultdict(float)
    
    for neighbor in neighbors:
        edge_data = graph.edges[episode_id, neighbor]
        edge_type = edge_data.get("relation", "semantic")
        weight = EDGE_BRANCHING_WEIGHTS.get(edge_type, 0.5)
        
        # 隣接ノードの文脈を取得
        context = determine_context(neighbor, graph)
        context_scores[context] += weight
    
    # 文脈の分散度を計算
    total_score = sum(context_scores.values())
    if total_score == 0:
        return 0
    
    # エントロピー計算
    entropy = 0
    for score in context_scores.values():
        p = score / total_score
        if p > 0:
            entropy -= p * math.log(p)
    
    return entropy
```

## 推奨設計

### 1. ハイブリッドアプローチ

```python
class QueryEdgeTimingManager:
    def __init__(self):
        self.immediate_types = {"query_spike"}  # 即座に追加
        self.delayed_types = {"query_retrieval", "query_bypass"}  # 遅延追加
        self.buffer = defaultdict(list)
        
    def process_query_result(self, query_id, cycle_result):
        """クエリ結果に基づいてエッジを処理"""
        
        edges = self.extract_edges(query_id, cycle_result)
        
        for edge in edges:
            if edge["edge_type"] in self.immediate_types:
                # 洞察エッジは即座に追加
                self.add_edge_to_graph(edge)
            else:
                # その他は評価待ち
                self.buffer[edge["target_id"]].append(edge)
        
        # バッファの評価
        self.evaluate_buffer()
    
    def evaluate_buffer(self):
        """バッファ内のエッジを評価"""
        for episode_id, edges in list(self.buffer.items()):
            if len(edges) >= 5:  # 5つ以上のクエリが蓄積
                # 統計的評価
                spike_count = sum(1 for e in edges if e["metadata"].get("led_to_spike"))
                spike_rate = spike_count / len(edges)
                
                if spike_rate > 0.2:  # 20%以上が洞察に繋がる
                    # このエピソードは価値がある
                    for edge in edges:
                        edge["metadata"]["episode_value"] = "high"
                        self.add_edge_to_graph(edge)
                else:
                    # 低価値として記録（グラフには追加しない）
                    self.save_to_datastore_only(edges)
                
                del self.buffer[episode_id]
```

### 2. 分岐評価の独立性

```python
class EpisodeBranchingEvaluator:
    def __init__(self):
        self.branching_threshold = 0.7  # エントロピー閾値
        
    def should_branch(self, episode_id, graph, include_query_edges=False):
        """エピソード分岐の判定"""
        
        if include_query_edges:
            # クエリエッジを含めて評価（オプション）
            entropy = self.calculate_entropy_with_queries(episode_id, graph)
        else:
            # 通常のエッジのみで評価（デフォルト）
            entropy = self.calculate_entropy_without_queries(episode_id, graph)
        
        return entropy > self.branching_threshold
    
    def calculate_entropy_without_queries(self, episode_id, graph):
        """クエリエッジを除外してエントロピー計算"""
        neighbors = []
        for neighbor in graph.neighbors(episode_id):
            edge_data = graph.edges[episode_id, neighbor]
            if not edge_data.get("relation", "").startswith("query_"):
                neighbors.append(neighbor)
        
        return self._calculate_entropy(neighbors, graph)
```

## 実装優先順位

1. **Phase 1**: クエリエッジをDataStoreのみに保存（グラフに追加しない）
2. **Phase 2**: 洞察エッジのみグラフに追加
3. **Phase 3**: 統計的評価に基づく選択的追加
4. **Phase 4**: 分岐判定の高度化

## まとめ

- **即時追加は避ける**: 特に洞察に至らなかったクエリ
- **統計的評価**: 複数クエリの結果を見てから判断
- **分岐判定の独立性**: クエリエッジを分岐判定から分離可能に
- **段階的実装**: まずDataStoreのみ、次に選択的グラフ追加