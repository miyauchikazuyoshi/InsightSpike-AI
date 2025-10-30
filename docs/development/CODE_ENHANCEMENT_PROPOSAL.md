---
status: proposal
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
decision_due: 2025-09-15
---

# コード改善提案：クエリ変成アーキテクチャ

## 🎯 理想：クエリがグラフを通じて洞察に変成

### 現在の実装の制限
- クエリは検索キーとしてのみ使用
- グラフは静的（検索後に構築）
- GNNは文書特徴量の処理のみ

### 提案する新アーキテクチャ

## 1. Query-as-Node アプローチ

```python
class QueryTransformationGraphReasoner:
    def process_query_as_node(self, query: str):
        # Step 1: クエリをグラフに仮配置
        query_node = self.create_query_node(query)
        self.graph.add_node("QUERY", features=query_node.features)
        
        # Step 2: 最適な配置を探索（geDIG評価）
        best_position = self.find_optimal_placement(query_node)
        
        # Step 3: メッセージパッシング（クエリが変化）
        for cycle in range(max_cycles):
            # クエリノードも含めてGNN処理
            node_features = self.gnn_forward(self.graph)
            query_features = node_features["QUERY"]
            
            # クエリの特徴量が変化 = 洞察を獲得
            if self.detect_insight_spike(query_features):
                break
        
        # Step 4: 変成したクエリから回答生成
        answer = self.decode_transformed_query(query_features)
        return answer
```

## 2. Dynamic Graph Construction

```python
class DynamicGraphManager:
    def incremental_graph_update(self, query_context):
        # クエリに応じて動的にノード/エッジを追加
        potential_nodes = self.identify_latent_concepts(query_context)
        
        for concept in potential_nodes:
            if self.evaluate_emergence_criteria(concept):
                self.graph.add_node(concept)
                self.connect_emergent_node(concept)
```

## 3. Query Transformation Pipeline

```python
class QueryTransformationPipeline:
    def __init__(self):
        self.stages = [
            EmbeddingStage(),      # Query → Vector
            PlacementStage(),      # Vector → Graph Position
            PropagationStage(),    # Position → Message Passing
            TransformationStage(), # Messages → Insight
            DecodingStage()        # Insight → Answer
        ]
    
    def process(self, query):
        state = {"query": query, "features": None, "graph_state": None}
        
        for stage in self.stages:
            state = stage.transform(state)
            
            # 各段階でクエリの「色」が変わる
            self.visualize_transformation(state)
        
        return state["answer"]
```

## 4. 実装の具体的な変更点

### graph_reasoner.py の拡張
```python
def process_with_query_node(self, query_embedding, documents):
    # 既存のグラフにクエリノードを追加
    graph = self.construct_graph_from_documents(documents)
    
    # クエリをノードとして追加
    query_idx = len(graph.x)
    graph.x = torch.cat([graph.x, query_embedding.unsqueeze(0)])
    
    # クエリと関連ノードをエッジで接続
    query_edges = self.connect_query_to_graph(query_idx, graph)
    graph.edge_index = torch.cat([graph.edge_index, query_edges], dim=1)
    
    # GNNでクエリも含めて処理
    if self.use_gnn:
        node_features = self.gnn(graph.x, graph.edge_index)
        transformed_query = node_features[query_idx]
        
        # 変成したクエリから洞察を抽出
        insight = self.extract_insight_from_transformation(
            original_query=query_embedding,
            transformed_query=transformed_query
        )
    
    return insight
```

### main_agent.py の改善
```python
def process_question_with_transformation(self, question: str):
    """クエリ変成を可視化しながら処理"""
    
    # 初期状態
    query_state = {
        "text": question,
        "embedding": None,
        "graph_position": None,
        "transformation_history": []
    }
    
    for cycle in range(self.max_cycles):
        # L2: メモリ検索（クエリの文脈を豊かに）
        relevant_memories = self.memory_manager.search(query_state["text"])
        query_state["context"] = relevant_memories
        
        # L3: グラフ配置と変成
        transformation = self.graph_reasoner.transform_query(
            query_state, 
            relevant_memories
        )
        query_state["transformation_history"].append(transformation)
        
        # 洞察検出
        if self.detect_insight_emergence(transformation):
            break
    
    # L4: 変成したクエリから回答生成
    answer = self.language_interface.generate_from_transformation(
        query_state["transformation_history"]
    )
    
    return answer, query_state["transformation_history"]
```

## 5. 視覚化の改善

```python
class QueryTransformationVisualizer:
    def animate_transformation(self, transformation_history):
        """クエリの変成過程をリアルタイムで可視化"""
        
        for step in transformation_history:
            # クエリの「色」（特徴量）の変化を表示
            self.show_query_state(step.query_features)
            
            # グラフ上での位置と接続の変化
            self.show_graph_state(step.graph_state)
            
            # 獲得した洞察の可視化
            if step.insights:
                self.highlight_insights(step.insights)
```

## 実装優先順位

1. **Phase 1**: QueryをGraphに配置する機能
2. **Phase 2**: GNNでクエリノードも処理
3. **Phase 3**: 動的なノード生成
4. **Phase 4**: 変成過程の可視化

## 実装状況（2025-09-08）

- Phase 1（QueryをGraphに配置）: 概ね実装済
  - `features/query_transformation/query_transformer.py` に `place_query_on_graph` 実装済。
  - L3 Reasoner が最新グラフを属性 `current_graph` で外部公開（`implementations/layers/layer3_graph_reasoner.py`）。

- Phase 2（GNNでクエリノードも処理）: 進行中 → 実装済に近い
  - Query 変成用 GNN は `QueryGraphGNN` 実装済。
  - `ConfigurableAgent` が `QueryTransformer(use_gnn=...)` を L3 の `graph.use_gnn` 設定に同期するよう調整（挙動一致）。

- Phase 3（動的なノード生成）: 最小連携は動作
  - 既存の `graph/construction.py` と L2/L3 経路により、クエリノード追加と関連エッジ付与が動作。
  - `implementations/layers/cached_memory_manager.py` にクエリ記録とグラフ接続の実装あり（分析/追跡用途）。

- Phase 4（変成過程の可視化）: スタブ投入済
  - `visualization/query_transform_viz.py` を追加。`animate_transformation`/`snapshot` で軽量な可視化・要約を提供（matplotlib が無い環境でもフォールバック）。

補足（入口とガード）
- geDIG の入口は `algorithms/gedig/selector.py` に統一。CI ガードで非 selector 呼び出しを検知し、`STRICT_GEDIG_SELECTOR=1` で Fail 可能（段階導入）。

## 期待される効果

- クエリが本当にグラフを「旅する」
- 新しい概念が動的に生成される
- 変成過程が追跡・可視化できる
- より直感的で説明可能なAI
