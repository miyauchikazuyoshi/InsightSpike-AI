# geDIG-based Layer1 Development Roadmap

## UPDATE (2025-07-30): Layer1's Vision Realized Through Donut Search

The original Layer1 filtering concept has been elegantly realized through the **Donut Search** mechanism. What Layer1 aimed to achieve through sequential filters (relevance → novelty → value) is now accomplished with a single geometric operation. See [Donut Search and Known Information Ignorance](/docs/research/donut_search_and_known_ignorance.md) for details.

### Key Transformation:
- **Original**: Complex multi-stage filtering with unclear parameters
- **Donut Search**: Query-centric sphere with inner radius (known) and outer radius (irrelevant)
- **Benefits**: Intuitive parameters, single-pass computation, theoretical foundation

---

## 概要（Original）
Layer1の検索メカニズムを単純なコサイン類似度から、グラフ編集距離と情報利得（geDIG）を活用した構造的類似度検索へと進化させる開発ロードマップ。

## 背景と動機

### 現状の課題
- **表面的な類似性**: 現在のコサイン類似度は単語の出現頻度に基づく表層的なマッチング
- **文脈の欠如**: 「銀行（金融）」と「銀行（川岸）」のような文脈依存の区別が困難
- **構造的関係の無視**: 概念間の関係性や階層構造を考慮できない

### geDIG導入の利点
- **構造的類似性**: グラフ構造として概念間の関係を捉える
- **意味的深度**: 表面的でない深い意味の類似性を検出
- **人間的な連想**: 人間の思考に近い連想的な検索が可能

## フェーズ1: 基礎実装（2-3週間）

### 1.1 概念グラフビルダーの実装
```python
class ConceptGraphBuilder:
    """テキストから概念グラフを構築"""
    def build_from_text(self, text: str) -> nx.Graph
    def extract_concepts(self, text: str) -> List[Concept]
    def identify_relations(self, concepts: List[Concept]) -> List[Relation]
```

### 1.2 高速geDIG計算器の開発
```python
class FastGeDIGCalculator:
    """高速なグラフ編集距離計算"""
    def calculate_ged(self, g1: nx.Graph, g2: nx.Graph) -> float
    def calculate_ig(self, g1: nx.Graph, g2: nx.Graph) -> float
    def compute_similarity(self, ged: float, ig: float) -> float
```

### 1.3 プロトタイプ評価
- ベンチマークデータセットの作成
- コサイン類似度との性能比較
- 計算速度の最適化

## フェーズ2: 高度な機能実装（3-4週間）

### 2.1 階層的検索システム
```python
class HierarchicalGeDIGRetriever:
    """多段階の精度で効率的に検索"""
    def coarse_search(self, query_graph: nx.Graph) -> List[Candidate]
    def fine_search(self, query_graph: nx.Graph, candidates: List[Candidate]) -> List[Episode]
    def rerank_results(self, results: List[Episode]) -> List[Episode]
```

### 2.2 動的グラフキャッシング
- 頻繁にアクセスされるエピソードのグラフ表現をキャッシュ
- インクリメンタルなグラフ更新メカニズム
- メモリ効率的な圧縮表現

### 2.3 文脈依存の検索
```python
class ContextAwareGeDIGRetriever:
    """文脈を考慮した構造的検索"""
    def incorporate_context(self, query_graph: nx.Graph, context: Dict) -> nx.Graph
    def disambiguate_concepts(self, ambiguous_nodes: List[Node]) -> List[Node]
```

## フェーズ3: 統合と最適化（2-3週間）

### 3.1 既存システムとの統合
- Layer1インターフェースの拡張
- 後方互換性の維持
- 段階的な移行戦略

### 3.2 パフォーマンス最適化
- GPU活用によるグラフ計算の高速化
- バッチ処理の実装
- 並列化戦略

### 3.3 メトリクスと監視
```python
class GeDIGMetrics:
    """geDIG検索の性能監視"""
    def track_retrieval_quality(self) -> Dict[str, float]
    def measure_structural_similarity(self) -> float
    def analyze_concept_coverage(self) -> Dict[str, Any]
```

## フェーズ4: 高度な応用（4-6週間）

### 4.1 学習可能なgeDIG
```python
class LearnableGeDIG:
    """フィードバックから最適な重みを学習"""
    def learn_edge_weights(self, feedback: List[Feedback]) -> Dict[str, float]
    def adapt_similarity_function(self, examples: List[Example]) -> Callable
```

### 4.2 マルチモーダル拡張
- テキスト以外のモダリティ（画像、音声）への対応
- クロスモーダル検索の実現

### 4.3 分散グラフ検索
- 大規模グラフの分散処理
- エッジコンピューティング対応

## 評価指標

### 定量的指標
- **検索精度**: Precision@K, Recall@K, F1スコア
- **構造的類似度**: グラフ編集距離の分布
- **計算効率**: クエリあたりの処理時間
- **メモリ使用量**: グラフ表現のメモリフットプリント

### 定性的指標
- **意味的妥当性**: 人間による関連性評価
- **発見的価値**: 予期しない有用な関連の発見
- **説明可能性**: なぜその結果が返されたかの理解

## 実装例

### 基本的なgeDIG検索
```python
class GeDIGLayer1(Layer1Interface):
    def __init__(self, config: Config):
        super().__init__(config)
        self.graph_builder = ConceptGraphBuilder()
        self.gediq_calculator = FastGeDIGCalculator()
        self.episode_graphs = {}  # エピソードIDとグラフのマッピング
        
    def process(self, input_data: LayerInput) -> LayerOutput:
        # クエリをグラフ化
        query_graph = self.graph_builder.build_from_text(input_data.data)
        
        # 構造的類似度で検索
        results = []
        for episode_id, episode_graph in self.episode_graphs.items():
            similarity = self.gediq_calculator.compute_similarity(
                query_graph, episode_graph
            )
            results.append((episode_id, similarity))
        
        # 上位K件を返す
        top_k = sorted(results, key=lambda x: x[1], reverse=True)[:self.k]
        
        return LayerOutput(
            data=[self.episodes[eid] for eid, _ in top_k],
            confidence=top_k[0][1] if top_k else 0.0,
            metadata={
                "method": "geDIG",
                "query_graph_size": len(query_graph.nodes()),
                "structural_similarities": dict(top_k)
            }
        )
```

## リスクと対策

### 技術的リスク
- **計算コスト**: グラフ編集距離の計算は高コスト
  - 対策: 近似アルゴリズム、キャッシング、GPU活用

- **スケーラビリティ**: 大規模データセットでの性能
  - 対策: 階層的インデックス、分散処理

### 運用リスク
- **移行の複雑性**: 既存システムからの移行
  - 対策: 段階的ロールアウト、A/Bテスト

## 期待される成果

1. **検索品質の向上**: より深い意味的関連性の発見
2. **新しい洞察の創出**: 構造的パターンからの予期しない発見
3. **人間的な推論**: より自然な連想的検索
4. **統一的アーキテクチャ**: 全レイヤーでグラフベースの処理

## タイムライン

```
2024年Q1: フェーズ1完了（基礎実装）
2024年Q2: フェーズ2完了（高度な機能）
2024年Q3: フェーズ3完了（統合と最適化）
2024年Q4: フェーズ4完了（高度な応用）
```

## 次のステップ

1. プロトタイプの実装開始
2. ベンチマークデータセットの準備
3. 性能評価フレームワークの構築
4. コミュニティからのフィードバック収集

---

このロードマップは、InsightSpikeの検索層を根本的に進化させ、より人間的で洞察に富んだAIシステムの実現を目指すものです。