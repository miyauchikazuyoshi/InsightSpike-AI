# InsightSpike 理想的な修正戦略

## 設計原則

1. **単一責任の原則**: 各レイヤーは明確な責任を持つ
2. **インターフェース一貫性**: 全てのレイヤー間で一貫したデータ形式
3. **後方互換性**: 既存の実験スクリプトが動作し続ける
4. **テスタビリティ**: 各レイヤーを独立してテスト可能

## 1. Embedding形状の統一

### 現状の問題
```python
# 現在：各所で異なる形状処理
embedder.get_embedding(text)  # → (1, 384)
episode.vec                   # → (1, 384) または (384,)
search_episodes(query)        # → (384,) を期待
build_graph(docs)            # → (384,) を期待
```

### 理想的な修正
```python
# 1. Embedderレベルで統一
class StandardizedEmbedder:
    def __init__(self, base_embedder):
        self.base_embedder = base_embedder
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Always returns (384,) shape"""
        emb = self.base_embedder.get_embedding(text)
        if emb.ndim > 1:
            emb = emb.squeeze()  # (1, 384) → (384,)
        return emb

# 2. Episode作成時の検証
class Episode:
    def __init__(self, text: str, vec: np.ndarray, ...):
        assert vec.ndim == 1, f"Expected 1D embedding, got shape {vec.shape}"
        self.vec = vec
        ...
```

## 2. グラフ型の統一

### 現状の問題
```python
# 混在する型
ScalableGraphBuilder → NetworkX Graph
L3GraphReasoner → PyTorch Geometric Data
GED/IG calculators → NetworkX を期待
```

### 理想的な修正

#### Option A: NetworkXで統一（推奨）
```python
# 全てNetworkXで統一
class UnifiedGraphBuilder:
    def build_graph(self, documents) -> nx.Graph:
        """Always returns NetworkX graph"""
        ...

class L3GraphReasoner:
    def analyze_documents(self, docs) -> Dict:
        graph = self.graph_builder.build_graph(docs)  # nx.Graph
        
        # PyG必要な場合のみ内部で変換
        if self.use_gnn:
            pyg_data = nx_to_pyg(graph)
            gnn_features = self.gnn(pyg_data)
            # 結果をNetworkXに戻す
            graph = update_nx_with_features(graph, gnn_features)
        
        return {
            'graph': graph,  # Always nx.Graph
            'metrics': self.calculate_metrics(graph)
        }
```

#### Option B: 明示的な型変換レイヤー
```python
class GraphTypeAdapter:
    """明示的な型変換を行うアダプター"""
    
    @staticmethod
    def to_networkx(graph: Union[nx.Graph, Data]) -> nx.Graph:
        if isinstance(graph, nx.Graph):
            return graph
        return pyg_to_nx(graph)
    
    @staticmethod
    def to_pyg(graph: Union[nx.Graph, Data]) -> Data:
        if isinstance(graph, Data):
            return graph
        return nx_to_pyg(graph)
```

## 3. Config処理の統一

### 現状の問題
```python
# dict/Pydantic混在
MainAgent(config=dict)  # または InsightSpikeConfig
L4LLMInterface(config=dict)  # LLMConfig期待
LLMProviderRegistry.get_instance(dict)  # .provider属性アクセス
```

### 理想的な修正
```python
class ConfigNormalizer:
    """全ての設定を正規化"""
    
    @staticmethod
    def normalize(config: Union[dict, BaseModel]) -> InsightSpikeConfig:
        if isinstance(config, dict):
            return InsightSpikeConfig.from_dict(config)
        return config
    
    @staticmethod
    def get_llm_config(config: Union[dict, InsightSpikeConfig]) -> LLMConfig:
        if isinstance(config, dict):
            llm_dict = config.get('llm', {})
            return LLMConfig(
                provider=llm_dict.get('provider', 'mock'),
                api_key=llm_dict.get('api_key', ''),
                model=llm_dict.get('model', ''),
                # 追加属性は別途保存
                _extra={
                    'prompt_style': llm_dict.get('prompt_style'),
                    'use_simple_prompt': llm_dict.get('use_simple_prompt')
                }
            )
        return config.llm
```

## 4. 実装優先順位

### Phase 1: 即座に修正（影響最小）
```python
# 1. Embedder出力の修正
class HuggingFaceEmbedder:
    def get_embedding(self, text: str) -> np.ndarray:
        emb = self.model.encode([text])
        return emb.squeeze()  # 常に1D

# 2. LLMProviderRegistryのdict対応
class LLMProviderRegistry:
    def get_instance(self, config):
        if isinstance(config, dict):
            provider = config.get('provider', 'mock')
        else:
            provider = config.provider
        ...
```

### Phase 2: 中期的改善（後方互換維持）
```python
# 1. グラフ型アダプター導入
# 2. Config正規化レイヤー
# 3. 各レイヤーのインターフェース明確化
```

### Phase 3: 長期的リファクタリング
```python
# 1. 全レイヤーの責任範囲再定義
# 2. 依存性注入パターンの導入
# 3. 統合テストスイートの充実
```

## 5. テスト戦略

### ユニットテスト
```python
def test_embedder_shape():
    embedder = StandardizedEmbedder(...)
    emb = embedder.get_embedding("test")
    assert emb.shape == (384,)
    assert emb.ndim == 1

def test_graph_type_consistency():
    builder = UnifiedGraphBuilder()
    graph = builder.build_graph(docs)
    assert isinstance(graph, nx.Graph)
```

### 統合テスト
```python
def test_end_to_end_pipeline():
    agent = MainAgent(config)
    agent.add_knowledge("test")
    
    # 各ステップで型と形状を検証
    assert agent.l2_memory.episodes[0].vec.shape == (384,)
    
    result = agent.process_question("test?")
    assert 'metrics' in result.graph_analysis
    assert result.graph_analysis['metrics']['delta_ged'] != 0
```

## 6. 移行計画

1. **Week 1**: Embedding形状修正（最小影響）
2. **Week 2**: Config処理統一
3. **Week 3**: グラフ型アダプター導入
4. **Week 4**: 統合テストと既存実験の動作確認