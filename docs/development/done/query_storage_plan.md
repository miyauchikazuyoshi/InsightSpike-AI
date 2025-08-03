# クエリ保存機能実装計画

## 概要

InsightSpikeパイプラインにおいて、ユーザーからのクエリ（質問）を保存・管理する機能を実装する。これにより、システムの使用パターン分析、洞察生成率の追跡、ユーザー行動の理解が可能になる。

## 背景と動機

現在のシステムでは：
- ユーザーのクエリは処理後に破棄される
- 洞察（インサイト）のみが保存される
- クエリと洞察の関連性が追跡できない
- システムのパフォーマンス分析が困難

## 設計方針

### 1. クエリレコードの構造

```python
{
    "id": "query_<timestamp>_<uuid>",
    "text": "元のクエリテキスト",
    "vec": np.ndarray,  # クエリのエンベディング
    "timestamp": float,
    "has_spike": bool,  # 洞察が生成されたか
    "spike_episode_id": Optional[str],  # 生成された洞察のID
    "response": str,  # LLMからの応答
    "metadata": {
        "cycle_number": int,
        "reasoning_quality": float,
        "graph_metrics": dict,
        "retrieved_docs_count": int,
        "processing_time": float,
        "llm_provider": str,
        "error_state": Optional[dict]
    }
}
```

### 2. 保存タイミング

- **process_question()完了時**: 処理結果に関わらず全てのクエリを保存
- **洞察生成の有無で分類**: has_spikeフラグで区別
- **非同期保存**: パフォーマンスへの影響を最小化

### 3. 名前空間設計

```
queries/
├── all/          # 全てのクエリ
├── with_spike/   # 洞察を生成したクエリ
└── no_spike/     # 洞察を生成しなかったクエリ
```

### 4. エッジ（関連性）の保存戦略

クエリと既存エピソード間の関連性を保存する設計：

#### 4.1 エッジタイプの定義

現在のシステムでは`relation`属性は一部で使用されているが（例：'branch'）、体系的ではない。後方互換性を保ちつつ、クエリ関連のエッジを追加する方針：

```python
# クエリ関連のエッジタイプ（relation属性の値）
QUERY_EDGE_TYPES = {
    "query_spike": "query_spike",              # クエリ→生成された洞察
    "query_retrieval": "query_retrieval",      # クエリ→検索されたエピソード
    "query_context": "query_context",          # クエリ→文脈として使用されたエピソード
    "query_bypass": "query_bypass",            # クエリ→L1バイパスで使用されたエピソード
}
```

**設計方針**:
- 既存のエッジ（relation='branch'など）には影響を与えない
- クエリ関連のエッジは'query_'プレフィックスで区別
- 既存のグラフ処理はrelation属性を必須としていないため、追加しても問題ない
- 将来的な統一は別途検討（破壊的変更を避ける）

#### 4.2 クエリノードとエッジの保存方針

**設計方針**: クエリもエピソードと同様にグラフにノードとして追加する。統一的な扱いにより：
- システムの一貫性を保つ
- クエリとエピソードの関係性を直接グラフで表現
- 既存のグラフアルゴリズムをクエリ分析にも活用可能

**クエリノードの構造**:
```python
# グラフに追加するクエリノード
graph.add_node(
    query_id,  # "query_<timestamp>_<uuid>"
    text=query_text,
    vec=query_embedding,
    node_type="query",  # エピソードと区別
    has_spike=False,    # 洞察生成の有無
    metadata={
        "timestamp": time.time(),
        "llm_provider": "anthropic",
        "processing_time": 1.23,
        # その他のメタデータ
    }
)
```

**エッジの追加方法**:
```python
# クエリ→エピソードのエッジ
graph.add_edge(
    query_id,
    episode_id,
    weight=similarity_score,  # 既存の慣習に従う
    relation="query_retrieval"  # エッジタイプ
)

```

**注意事項**: 
- クエリノードもエピソードと同様にグラフに追加
- node_type="query"で区別することで、必要に応じてフィルタリング可能
- 既存のweight属性を活用（類似度スコアなど）
- relation属性でエッジタイプを明示

#### 4.3 ケース別のエッジ保存戦略

**保存の基本方針**：
- 検索された全エピソードではなく、実際に処理に使用されたエピソードのみエッジを作成
- パフォーマンスとストレージ効率を考慮
- 分析に必要な最小限の情報を保存

**1. 洞察が生成された場合**
- クエリ → 生成された洞察エピソード（relation='query_spike'）
- クエリ → LLMに渡されたトップNエピソード（relation='query_retrieval'）
  - 通常はトップ5-10件程度
  - CycleResultのretrieved_documentsに含まれるもの
  - metadata.led_to_spike = true

**2. 洞察が生成されなかった場合**
- クエリ → LLMに渡されたトップNエピソード（relation='query_retrieval'）
  - metadata.led_to_spike = false
  - 検索精度改善の分析に活用

**3. L1バイパスが選択された場合**
- クエリ → known_elementsのエピソード（relation='query_bypass'）
- 最大3-5件程度（処理効率のため）
- metadata.bypass_confidence = error_state.confidence

**4. Adaptive Loopの場合**
- 最終的にLLMに渡されたエピソードのみ保存
- 探索履歴はクエリのmetadataに記録（エッジは作らない）
- エッジ数の爆発を防ぐ

**5. エッジ数の制限**
```python
MAX_EDGES_PER_QUERY = {
    'query_spike': 1,      # 生成された洞察は1つ
    'query_retrieval': 10, # 最大10件の検索結果
    'query_bypass': 5,     # 最大5件のbypass要素
    'query_context': 5     # グラフ文脈は最大5件
}
```

#### 4.4 実装上の考慮点

```python
# CachedMemoryManagerに追加するメソッド
def save_query_edges(
    self,
    query_id: str,
    edges: List[Dict[str, Any]]
) -> bool:
    """クエリと関連エピソードのエッジを保存"""
    pass

def get_query_edges(
    self,
    query_id: str,
    edge_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """クエリに関連するエッジを取得"""
    pass
```

#### 4.5 分析への活用

保存されたエッジ情報から：
- **検索効率分析**: どのエピソードが実際に使用されているか
- **洞察生成パターン**: 高weightで検索されたが洞察に至らないエピソード群の特定
- **L1バイパス効率**: バイパスの頻度と精度
- **エピソード価値評価**: 使用頻度とweight分布から各エピソードの実用性を評価

#### 4.6 実装例

```python
# MainAgent内でのクエリとエッジの保存
def _save_query_and_edges(self, query_id: str, query_text: str, query_vec: np.ndarray, cycle_result: CycleResult):
    # 1. クエリノードをグラフに追加
    self.graph.add_node(
        query_id,
        text=query_text,
        vec=query_vec,
        node_type="query",
        has_spike=cycle_result.spike_detected,
        metadata={
            "timestamp": time.time(),
            "cycle_number": cycle_result.cycle_number,
            "reasoning_quality": cycle_result.reasoning_quality,
            "llm_provider": self.config.get("llm_provider", "unknown")
        }
    )
    
    # 2. 洞察が生成された場合
    if cycle_result.spike_detected and cycle_result.spike_episode_id:
        self.graph.add_edge(
            query_id,
            cycle_result.spike_episode_id,
            weight=1.0,  # 洞察生成は最高の重み
            relation="query_spike"
        )
    
    # 3. 検索結果のエッジ（LLMに渡されたもののみ）
    for idx, doc in enumerate(cycle_result.retrieved_documents[:10]):
        episode_id = doc.get("id") or f"episode_{doc.get('index')}"
        weight = doc.get("similarity", 0.5)
        
        # 洞察に至らなかった場合は重みを下げる
        if not cycle_result.spike_detected:
            weight *= 0.7
            
        self.graph.add_edge(
            query_id,
            episode_id,
            weight=weight,
            relation="query_retrieval",
            rank=idx + 1,
            led_to_spike=cycle_result.spike_detected
        )
    
    # 4. L1バイパスの場合
    if cycle_result.error_state and cycle_result.error_state.get("known_elements"):
        for idx, element in enumerate(cycle_result.error_state["known_elements"][:5]):
            self.graph.add_edge(
                query_id,
                element.get("id"),
                weight=0.9,  # バイパスは高信頼度
                relation="query_bypass",
                rank=idx + 1
            )
    
    # 5. DataStoreにも保存（永続化）
    self.memory_manager.save_query_to_datastore(query_id, query_text, query_vec, cycle_result)
```

## 影響範囲分析

### 1. 変更が必要なコンポーネント

#### DataStore抽象クラス
- `save_queries()`: クエリ保存メソッド
- `load_queries()`: クエリ取得メソッド（フィルタリング対応）
- `search_queries()`: クエリ検索メソッド（将来的な拡張）

#### FileSystemDataStore実装
- クエリ用のディレクトリ構造管理
- JSONファイルへの保存・読み込み実装

#### CachedMemoryManager
- `save_query()`: クエリ保存のラッパーメソッド
- `get_query_history()`: 履歴取得メソッド
- `get_query_stats()`: 統計情報取得

#### MainAgent
- `process_question()`内でのクエリ保存処理追加
- CycleResultにクエリIDを追加
- エラーハンドリングの考慮

### 2. 後方互換性

- 既存のAPIは変更なし
- クエリ保存は新規機能として追加
- 設定で無効化可能（デフォルトは有効）

### 3. パフォーマンスへの影響

- 非同期保存により処理速度への影響は最小
- ディスク使用量の増加（クエリ1件あたり約2-5KB）
- キャッシュ戦略の検討が必要

### 4. Adaptive Loop対応

adaptive_loopが有効な場合の特別な考慮事項：

#### 4.1 探索履歴の保存
```python
"metadata": {
    "adaptive_loop": {
        "enabled": true,
        "strategy": "narrowing",  # or "expanding", "alternating"
        "exploration_attempts": 3,
        "exploration_path": [
            {
                "attempt": 1,
                "params": {"topK": 10, "radius": 0.8},
                "spike_detected": false,
                "metrics": {"delta_ged": -0.2, "delta_ig": 0.1}
            },
            {
                "attempt": 2,
                "params": {"topK": 15, "radius": 0.64},
                "spike_detected": true,
                "metrics": {"delta_ged": -0.6, "delta_ig": 0.3}
            }
        ],
        "total_api_calls": 1,  # LLMは最後に1回のみ
        "total_exploration_time": 2.34
    }
}
```

#### 4.2 実装上の考慮点
- AdaptiveProcessorのprocess()メソッドからも同様にクエリを保存
- 探索パスの全履歴を保存（パラメータ調整の学習に活用）
- スパイク検出前の試行も含めて記録
- Pattern Learnerへのフィードバックに活用可能

## 実装フェーズ

### Phase 1: 基盤整備（2日）
1. DataStoreインターフェースにクエリ関連メソッド追加
   - `save_queries()`, `load_queries()`
   - ※エッジ専用メソッドは不要（グラフの一部として保存）
2. FileSystemDataStoreへの実装
   - queries.jsonファイルの管理
3. 基本的なユニットテスト作成

### Phase 2: グラフ統合（3日）
1. CachedMemoryManagerの拡張
   - クエリノードをグラフに追加する処理
   - クエリ→エピソードのエッジ作成
2. MainAgentでの実装
   - `_save_query_and_edges()`メソッド追加
   - `process_question()`の最後でクエリ保存
   - CycleResultにquery_id追加
3. エッジ評価タイミングの実装
   - 即時追加: query_spikeエッジ
   - 遅延評価: query_retrievalエッジ

### Phase 3: Adaptive Loop対応（1日）
1. AdaptiveProcessorでのクエリ保存
2. 探索履歴のメタデータ記録
3. 統合テストの実装

### Phase 4: 分析機能（2日・オプション）
1. クエリ履歴の取得API
2. グラフ分析を活用した洞察
   - よく検索されるエピソード
   - クエリ間の関連性分析

## リスクと対策

### 1. ディスク容量
- **リスク**: 長期運用でのディスク使用量増加
- **対策**: 定期的なアーカイブ、古いクエリの圧縮

### 2. プライバシー
- **リスク**: ユーザークエリに含まれる機密情報
- **対策**: 暗号化オプション、自動削除ポリシー

### 3. パフォーマンス
- **リスク**: 大量クエリでの検索性能低下
- **対策**: インデックス化、ページネーション

### 4. 後方互換性
- **リスク**: 既存のグラフ処理への影響
- **対策**: 
  - node_type="query"で既存エピソードと区別
  - relation属性で新規エッジタイプを明示
  - 既存のweight属性の慣習に従う（類似度など）
  - 必要に応じてnode_typeでフィルタリング可能
  - グラフアルゴリズムは変更不要（ノードが増えるだけ）

## テスト計画

### 1. ユニットテスト

#### 1.1 DataStore層
```python
# test_datastore_query_methods.py
class TestDataStoreQueryMethods:
    def test_save_single_query(self):
        """単一クエリの保存と取得"""
        query = {
            "id": "query_123",
            "text": "What is insight?",
            "vec": np.random.rand(384),
            "has_spike": False,
            "metadata": {}
        }
        assert datastore.save_queries([query])
        loaded = datastore.load_queries()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "query_123"
    
    def test_load_queries_with_filter(self):
        """has_spikeフィルタでのクエリ取得"""
        # 洞察ありと無しのクエリを保存
        queries = [
            {"id": "q1", "has_spike": True, ...},
            {"id": "q2", "has_spike": False, ...}
        ]
        datastore.save_queries(queries)
        
        # フィルタリングテスト
        spike_queries = datastore.load_queries(has_spike=True)
        assert len(spike_queries) == 1
        assert spike_queries[0]["id"] == "q1"
    
    def test_query_edge_save_load(self):
        """クエリエッジの保存と取得"""
        edges = [
            {
                "source_id": "query_123",
                "target_id": "episode_456",
                "edge_type": "query_retrieval",
                "metadata": {"similarity_score": 0.85}
            }
        ]
        assert datastore.save_query_edges(edges)
        loaded = datastore.load_query_edges("query_123")
        assert len(loaded) == 1
```

#### 1.2 CachedMemoryManager層
```python
# test_cached_memory_query_methods.py
class TestCachedMemoryQueryMethods:
    def test_save_query_record(self):
        """クエリレコードの保存"""
        manager = CachedMemoryManager(datastore)
        query_id = manager.save_query(
            text="What is apple?",
            embedding=np.random.rand(384),
            has_spike=False,
            metadata={"source": "test"}
        )
        assert query_id is not None
        assert query_id.startswith("query_")
    
    def test_save_query_edges_integration(self):
        """クエリエッジの保存統合テスト"""
        manager = CachedMemoryManager(datastore)
        edges = [
            {
                "source_id": "query_123",
                "target_id": "episode_456",
                "edge_type": "query_retrieval",
                "metadata": {"similarity_score": 0.9}
            }
        ]
        assert manager.save_query_edges("query_123", edges)
    
    def test_get_query_history(self):
        """クエリ履歴の取得"""
        # 複数クエリを保存
        for i in range(5):
            manager.save_query(f"Query {i}", ...)
        
        history = manager.get_query_history(limit=3)
        assert len(history) == 3
        # 最新順であることを確認
```

### 2. レイヤー別テスト

#### 2.1 L1 StreamProcessor
```python
# test_l1_query_handling.py
class TestL1QueryHandling:
    def test_l1_bypass_query_recording(self):
        """L1バイパス時のクエリ記録"""
        processor = L1StreamProcessor()
        error_state = {
            "uncertainty": 0.1,
            "known_elements": [
                {"id": "ep_123", "text": "known fact"}
            ]
        }
        
        # バイパス判定とクエリ記録
        should_bypass = processor.should_bypass(error_state)
        if should_bypass:
            # クエリとknown_elementsの関連を記録する処理を確認
            pass
```

#### 2.2 L2 MemoryManager
```python
# test_l2_query_tracking.py
class TestL2QueryTracking:
    def test_search_with_query_tracking(self):
        """検索時のクエリ追跡"""
        memory = CachedMemoryManager(datastore)
        query = "What is consciousness?"
        
        # 検索実行
        results = memory.search_episodes(query, top_k=10)
        
        # クエリが記録されているか確認
        # （実際の実装では検索メソッド内で記録）
```

#### 2.3 L3 GraphReasoner
```python
# test_l3_query_context.py
class TestL3QueryContext:
    def test_graph_analysis_query_tracking(self):
        """グラフ分析時のクエリコンテキスト追跡"""
        reasoner = L3GraphReasoner()
        
        # グラフ分析でどのエピソードが文脈として使われたか記録
        graph_analysis = reasoner.analyze_documents(docs, context)
        
        # 使用されたエピソードのIDを確認
        assert "used_episode_ids" in graph_analysis
```

### 3. パイプライン統合テスト

#### 3.1 MainAgent統合
```python
# test_mainagent_query_integration.py
class TestMainAgentQueryIntegration:
    def test_full_pipeline_query_save(self):
        """フルパイプラインでのクエリ保存"""
        agent = MainAgent(config)
        query = "What causes aha moments?"
        
        # 処理実行
        result = agent.process_question(query)
        
        # クエリがグラフに追加されているか確認
        assert result.query_id in agent.current_graph
        node_data = agent.current_graph.nodes[result.query_id]
        assert node_data["text"] == query
        assert node_data["node_type"] == "query"
        
        # エッジが作成されているか確認
        edges = list(agent.current_graph.edges(result.query_id))
        assert len(edges) > 0
        
        # DataStoreにも保存されているか確認
        saved_queries = agent.memory_manager.datastore.load_queries(limit=1)
        assert len(saved_queries) == 1
        assert saved_queries[0]["text"] == query
    
    def test_spike_vs_no_spike_recording(self):
        """洞察生成有無での記録の違い"""
        agent = MainAgent(config)
        
        # 洞察が生成される可能性の高いクエリ
        spike_query = "What is the connection between sleep and creativity?"
        result1 = agent.process_question(spike_query)
        
        # 洞察が生成されにくいクエリ
        simple_query = "What is 2+2?"
        result2 = agent.process_question(simple_query)
        
        # 両方のクエリが適切に分類されて保存されているか
        spike_queries = agent.memory_manager.datastore.load_queries(has_spike=True)
        no_spike_queries = agent.memory_manager.datastore.load_queries(has_spike=False)
        
        # エッジのメタデータを確認
        if result1.spike_detected:
            edges1 = agent.memory_manager.datastore.load_query_edges(result1.query_id)
            assert any(e["metadata"]["led_to_spike"] for e in edges1)
```

#### 3.2 AdaptiveProcessor統合
```python
# test_adaptive_query_integration.py
class TestAdaptiveQueryIntegration:
    def test_adaptive_loop_query_recording(self):
        """Adaptive Loop使用時のクエリ記録"""
        processor = AdaptiveProcessor(...)
        query = "Complex philosophical question"
        
        result = processor.process(query)
        
        # クエリのメタデータに探索履歴が含まれているか
        saved_query = get_latest_saved_query()
        assert "adaptive_loop" in saved_query["metadata"]
        assert saved_query["metadata"]["adaptive_loop"]["exploration_attempts"] > 0
        
        # 最終的に使用されたエピソードのみエッジが作られているか
        edges = load_query_edges(saved_query["id"])
        assert len(edges) <= 10  # 制限内
```

### 4. パフォーマンステスト

#### 4.1 負荷テスト
```python
# test_query_storage_performance.py
class TestQueryStoragePerformance:
    def test_bulk_query_save_performance(self):
        """大量クエリ保存のパフォーマンス"""
        queries = []
        for i in range(1000):
            queries.append({
                "id": f"query_{i}",
                "text": f"Test query {i}",
                "vec": np.random.rand(384),
                "has_spike": i % 10 == 0,
                "metadata": {}
            })
        
        start_time = time.time()
        datastore.save_queries(queries)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0  # 5秒以内
        
    def test_query_search_performance(self):
        """クエリ検索のパフォーマンス"""
        # 10000件のクエリがある状態で
        start_time = time.time()
        recent = datastore.load_queries(limit=100)
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1  # 100ms以内
```

#### 4.2 メモリ使用量テスト
```python
# test_memory_usage.py
class TestMemoryUsage:
    def test_memory_with_query_storage(self):
        """クエリ保存時のメモリ使用量"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 1000クエリを処理
        agent = MainAgent(config)
        for i in range(1000):
            agent.process_question(f"Question {i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # 100MB以内の増加
```

### 5. エラーハンドリングテスト

```python
# test_error_handling.py
class TestErrorHandling:
    def test_datastore_failure_handling(self):
        """DataStore障害時の処理継続性"""
        # DataStoreをモックして例外を発生させる
        with patch.object(datastore, 'save_queries', side_effect=Exception):
            agent = MainAgent(config)
            result = agent.process_question("Test query")
            
            # クエリ保存に失敗してもメイン処理は継続
            assert result is not None
            assert hasattr(result, 'response')
    
    def test_corrupted_query_handling(self):
        """破損したクエリデータの処理"""
        # 不正なデータを保存
        datastore.save_queries([{"id": "bad", "text": None}])
        
        # 読み込み時にエラーにならない
        queries = datastore.load_queries()
        # 不正なデータはスキップされる
```

### 6. 後方互換性テスト

```python
# test_backward_compatibility.py
class TestBackwardCompatibility:
    def test_existing_graph_compatibility(self):
        """既存のグラフ処理との互換性確認"""
        # 既存のグラフを構築
        G = nx.Graph()
        G.add_node("ep1", text="Episode 1", vec=np.random.rand(384))
        G.add_node("ep2", text="Episode 2", vec=np.random.rand(384))
        G.add_edge("ep1", "ep2", weight=0.8, relation="semantic")
        
        # クエリノードとエッジを追加
        G.add_node("query_123", text="What is?", vec=np.random.rand(384), node_type="query")
        G.add_edge("query_123", "ep1", weight=0.7, relation="query_retrieval")
        
        # 既存のアルゴリズムが動作することを確認
        # 例：エピソードのみをフィルタリング
        episodes = [n for n, d in G.nodes(data=True) if d.get("node_type") != "query"]
        assert len(episodes) == 2
        
        # グラフアルゴリズムが正常動作
        assert nx.is_connected(G)
```

## 設定例

```yaml
query_storage:
  enabled: true
  namespace: "queries"
  retention_days: 90
  compression: true
  encryption: false
  async_save: true
  batch_size: 100
```

## 成功指標

1. 全クエリの95%以上が正常に保存される
2. クエリ保存によるレスポンス時間の増加が5%未満
3. 洞察生成率の正確な追跡が可能
4. クエリ履歴から使用パターンが分析可能

## タイムライン

- Phase 1: 2日
- Phase 2: 3日
- Phase 3: 2日（オプション）
- テスト・修正: 3日

合計: 8-10日

## エラー処理とフォールバック

### 1. DataStore障害時のフォールバック
```python
class QueryStorageWithFallback:
    def __init__(self, primary_store, fallback_store=None):
        self.primary = primary_store
        self.fallback = fallback_store or InMemoryStore()
        self.failed_queries = []
        
    def save_query(self, query):
        try:
            return self.primary.save_queries([query])
        except Exception as e:
            logger.warning(f"Primary store failed: {e}")
            # フォールバックストアに保存
            self.fallback.save_queries([query])
            self.failed_queries.append(query["id"])
            return True
            
    def retry_failed_queries(self):
        """定期的に失敗したクエリを再試行"""
        for query_id in self.failed_queries[:]:
            query = self.fallback.get_query(query_id)
            if self.primary.save_queries([query]):
                self.failed_queries.remove(query_id)
                self.fallback.delete_query(query_id)
```

### 2. 部分的失敗の処理
- クエリ本体は保存できたがエッジ保存に失敗した場合
- メタデータの一部が欠損している場合
- エンベディング生成に失敗した場合

## 今後の拡張可能性

1. **クエリ推薦システム**: 過去のクエリから関連質問を提案
2. **洞察予測**: クエリから洞察生成確率を予測
3. **グラフ分析**: クエリ-エピソード間のエッジからナレッジグラフを構築
4. **検索改善**: よく検索されるが使われないエピソードの識別と改善

## 二重管理問題への対応

### 現状の課題
システムは現在、データを二重管理している：
- エピソード: episodes.json + グラフノード
- ベクトル: FAISS index + ノードのvec属性
- グラフ: graph_pyg.pt（永続化） + メモリ上のNetworkX

クエリを追加する際もこの問題を考慮する必要がある。

### 対応方針
1. **短期的対応**: 既存の二重管理パターンに従う
   - クエリもepisodes.json相当のファイルに保存
   - 同時にグラフノードとしても追加（メモリ上）
   - グラフ全体はDataStore経由で永続化
   - 整合性は既存の仕組みに依存

### グラフの管理方法
- **実行時**: メモリ上でNetworkX/PyG形式で保持
- **永続化**: DataStore（FileSystem/SQLite等）に保存
- **起動時**: DataStoreから読み込んでメモリに展開
- **更新時**: メモリ上で更新後、定期的または終了時に永続化

## データフロー図の改訂計画

現在のDATA_FLOW_ARCHITECTURE.mermaidを確認した結果、クエリ保存機能に関する以下の改訂が必要：

### 1. 現状の問題点
- クエリデータの保存場所が明示されていない
- クエリエッジの処理フローが含まれていない
- P3（Episode Creation）とP4（Graph Update）の間でクエリ処理が欠落
- Conflict Detection（P6）でクエリエッジの影響が考慮されていない

### 2. 改訂内容

#### 2.1 新規コンポーネントの追加
```mermaid
subgraph "Query Management"
    QStore[data/queries/<br/>Query History]
    QEdges[data/query_edges/<br/>Query-Episode Relations]
    QBuffer[Query Edge Buffer<br/>Evaluation Queue]
end
```

#### 2.2 処理パイプラインの拡張
```mermaid
P3.5[Query Recording<br/>save_query]
P3.6[Edge Evaluation<br/>evaluate_edges]
P3 --> P3.5 --> P3.6 --> P4
```

#### 2.3 データアクセスパターンの追加
```mermaid
Read --> ReadQuery[load_queries<br/>Query History]
Write --> WriteQuery[save_query<br/>Record Query]
Write --> WriteEdges[save_query_edges<br/>Store Relations]
```

### 3. タイミングの修正提案

#### 現在の設計の問題
私の当初の設計では、P4（Graph Update）の後にクエリエッジを追加することになっていたが、これは適切ではない：
- P6（Conflict Detection）でクエリエッジの影響を考慮できない
- 分岐判定が不正確になる可能性

#### 改善案
```
P3（Episode Creation）
  ↓
P3.5（Query Recording）- クエリをDataStoreに保存
  ↓
P3.6（Edge Evaluation）- バッファで評価、価値あるもののみ選択
  ↓
P4（Graph Update）- 選択されたエッジを含めて更新
  ↓
P6（Conflict Detection）- クエリエッジも考慮した分岐判定
```

### 4. 具体的な改訂手順

1. **Phase 1**: Query Managementサブグラフを追加
2. **Phase 2**: 処理パイプラインにP3.5, P3.6を挿入
3. **Phase 3**: データアクセスパターンを拡張
4. **Phase 4**: パフォーマンス注釈を追加（バッファサイズ、評価頻度）

### 5. 改訂後の利点
- クエリ処理の流れが明確化
- エッジ評価のタイミングが適切に
- 既存の分岐判定メカニズムとの統合が改善

## 参考資料

- 既存のDataStore実装: `/src/insightspike/core/base/datastore.py`
- FileSystemDataStore: `/src/insightspike/storage/filesystem_datastore.py`
- CachedMemoryManager: `/src/insightspike/implementations/layers/cached_memory_manager.py`
- MainAgent: `/src/insightspike/implementations/agents/main_agent.py`
- AdaptiveProcessor: `/src/insightspike/adaptive/core/adaptive_processor.py`
- Adaptive Loop実装: `/docs/development/done/adaptive_loop_implementation_summary.md`
- データフロー図: `/docs/diagrams/DATA_FLOW_ARCHITECTURE.mermaid`
- エッジタイミング分析: `/docs/development/query_edge_timing_analysis.md`