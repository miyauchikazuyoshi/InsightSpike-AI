# 多次元エッジ実装計画

## 1. 概要

FAISS除去後、エッジを単一の数値から多次元的な関係性表現へと拡張します。これにより、知識グラフがより豊かで解釈可能なものになります。

## 2. エッジの次元設計

### 2.1 基本的な次元（Phase 1）
```python
@dataclass
class EdgeDimensions:
    """エッジの多次元表現"""
    # 基本的な類似度
    semantic_similarity: float      # 意味的類似度（ベクトル空間）
    structural_similarity: float    # 構造的類似度（文章構造）
    lexical_similarity: float      # 語彙的類似度（単語の重複）
    
    # 時間的関係
    temporal_distance: float       # 時間的距離（正規化）
    temporal_order: int           # 時間順序（-1: 前, 0: 同時, 1: 後）
    
    # 論理的関係
    contradiction_score: float     # 矛盾度（0: 一致, 1: 完全矛盾）
    entailment_score: float       # 含意関係の強さ
    
    # メタデータ
    edge_type: str               # "similarity", "temporal", "causal", etc.
    confidence: float            # エッジの信頼度
    last_updated: float         # 最終更新時刻
```

### 2.2 高度な次元（Phase 2）
```python
@dataclass
class AdvancedEdgeDimensions(EdgeDimensions):
    """高度なエッジ表現"""
    # 概念的関係
    abstraction_difference: float   # 抽象度の差（-1: より具体的, +1: より抽象的）
    conceptual_distance: float     # 概念階層での距離
    
    # 因果関係
    causal_strength: float        # 因果関係の強さ
    causal_direction: int         # 因果の方向（-1, 0, 1）
    
    # 文脈依存
    context_similarity: float      # 文脈の類似度
    domain_similarity: float      # ドメインの類似度
    
    # 学習された特徴
    learned_importance: float     # GNNで学習された重要度
    task_specific_scores: Dict[str, float]  # タスク特化スコア
```

## 3. 実装計画

### Phase 1: 基本構造の実装（2週間）

#### 3.1.1 エッジクラスの定義
```python
class MultiDimensionalEdge:
    """多次元エッジの実装"""
    
    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id
        self.dimensions = EdgeDimensions(
            semantic_similarity=0.0,
            structural_similarity=0.0,
            lexical_similarity=0.0,
            temporal_distance=0.0,
            temporal_order=0,
            contradiction_score=0.0,
            entailment_score=0.0,
            edge_type="unknown",
            confidence=0.0,
            last_updated=time.time()
        )
        
    def to_vector(self) -> np.ndarray:
        """エッジをベクトル表現に変換"""
        return np.array([
            self.dimensions.semantic_similarity,
            self.dimensions.structural_similarity,
            self.dimensions.lexical_similarity,
            self.dimensions.temporal_distance,
            self.dimensions.contradiction_score,
            self.dimensions.entailment_score,
            self.dimensions.confidence
        ])
        
    def from_dict(self, data: Dict[str, Any]):
        """辞書からエッジを復元"""
        for key, value in data.items():
            if hasattr(self.dimensions, key):
                setattr(self.dimensions, key, value)
```

#### 3.1.2 エッジ計算器の実装
```python
class EdgeDimensionCalculator:
    """エッジの各次元を計算"""
    
    def __init__(self, embedder: EmbeddingManager):
        self.embedder = embedder
        
    def calculate_all_dimensions(
        self, 
        episode1: Episode, 
        episode2: Episode
    ) -> EdgeDimensions:
        """全次元を計算"""
        return EdgeDimensions(
            semantic_similarity=self._semantic_similarity(episode1, episode2),
            structural_similarity=self._structural_similarity(episode1, episode2),
            lexical_similarity=self._lexical_similarity(episode1, episode2),
            temporal_distance=self._temporal_distance(episode1, episode2),
            temporal_order=self._temporal_order(episode1, episode2),
            contradiction_score=self._contradiction_score(episode1, episode2),
            entailment_score=self._entailment_score(episode1, episode2),
            edge_type=self._determine_edge_type(episode1, episode2),
            confidence=self._calculate_confidence(episode1, episode2),
            last_updated=time.time()
        )
    
    def _semantic_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """意味的類似度（コサイン類似度）"""
        return float(np.dot(ep1.vec, ep2.vec) / 
                    (np.linalg.norm(ep1.vec) * np.linalg.norm(ep2.vec)))
    
    def _structural_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """構造的類似度"""
        # 文の数、段落構造、平均文長などを比較
        struct1 = self._extract_structure(ep1.text)
        struct2 = self._extract_structure(ep2.text)
        
        similarities = []
        # 文の数の類似度
        similarities.append(1 - abs(struct1['sentence_count'] - struct2['sentence_count']) / 
                          max(struct1['sentence_count'], struct2['sentence_count'], 1))
        
        # 平均文長の類似度
        similarities.append(1 - abs(struct1['avg_sentence_length'] - struct2['avg_sentence_length']) / 
                          max(struct1['avg_sentence_length'], struct2['avg_sentence_length'], 1))
        
        return float(np.mean(similarities))
    
    def _lexical_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """語彙的類似度（Jaccard係数）"""
        words1 = set(ep1.text.split())
        words2 = set(ep2.text.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _temporal_distance(self, ep1: Episode, ep2: Episode) -> float:
        """時間的距離（符号付き、ChatGPT提案の統合版）"""
        # タイムスタンプの差を日単位で計算
        time_diff = ep1.timestamp - ep2.timestamp  # 符号を保持
        days_diff = time_diff / 86400
        
        # 対数スケールで正規化（符号を保持）
        sign = np.sign(days_diff)
        normalized = sign * np.log1p(abs(days_diff)) / np.log1p(365)
        
        return np.clip(normalized, -1.0, 1.0)
        
    def _causal_strength(self, ep1: Episode, ep2: Episode) -> float:
        """因果関係の強さ（ChatGPT提案）"""
        # 簡易版：時間順序と意味的関連を組み合わせ
        if not hasattr(ep1, 'timestamp') or not hasattr(ep2, 'timestamp'):
            return 0.0
            
        # ep1 → ep2 の因果性をチェック
        if ep1.timestamp >= ep2.timestamp:
            return 0.0  # 時間的に後のものは原因になれない
            
        # 時間的近接性と意味的関連性から推定
        temporal_proximity = np.exp(-abs(self._temporal_distance(ep1, ep2)))
        semantic_relation = self._semantic_similarity(ep1, ep2)
        
        # 簡易的な因果スコア
        return temporal_proximity * semantic_relation * 0.5
    
    def _contradiction_score(self, ep1: Episode, ep2: Episode) -> float:
        """矛盾度の計算（NLIモデル使用を推奨）"""
        # 簡易版：意味的類似度が中程度で語彙的差異が大きい場合
        semantic = self._semantic_similarity(ep1, ep2)
        lexical = self._lexical_similarity(ep1, ep2)
        
        # 意味は似ているが表現が大きく異なる = 潜在的矛盾
        if 0.3 < semantic < 0.7 and lexical < 0.2:
            return 0.6
        
        # TODO: 実際のNLIモデル（RoBERTa-large-MNLI等）を使用
        return 0.0
        
    def _entailment_score(self, ep1: Episode, ep2: Episode) -> float:
        """含意関係の強さ"""
        # 簡易版：高い意味的類似度と一方向性
        semantic = self._semantic_similarity(ep1, ep2)
        
        # 高い類似度 = 潜在的含意関係
        if semantic > 0.8:
            return semantic * 0.7
            
        # TODO: 実際のNLIモデルを使用
        return 0.0
```

### Phase 2: グラフビルダーの更新（1週間）

#### 3.2.1 新しいグラフビルダー
```python
class MultiDimensionalGraphBuilder:
    """多次元エッジを構築するグラフビルダー"""
    
    def __init__(self, edge_calculator: EdgeDimensionCalculator):
        self.edge_calculator = edge_calculator
        
    def build_graph(
        self, 
        episodes: List[Episode],
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> nx.Graph:
        """多次元エッジを持つグラフを構築"""
        if dimension_weights is None:
            dimension_weights = {
                'semantic': 0.5,
                'structural': 0.2,
                'lexical': 0.1,
                'temporal': 0.2
            }
            
        graph = nx.Graph()
        
        # ノードを追加
        for episode in episodes:
            graph.add_node(
                episode.id,
                episode=episode,
                vec=episode.vec,
                text=episode.text,
                c_value=episode.c
            )
        
        # エッジを計算して追加
        for i, ep1 in enumerate(episodes):
            for ep2 in episodes[i+1:]:
                # 全次元を計算
                edge_dims = self.edge_calculator.calculate_all_dimensions(ep1, ep2)
                
                # 総合スコアを計算
                total_score = (
                    dimension_weights['semantic'] * edge_dims.semantic_similarity +
                    dimension_weights['structural'] * edge_dims.structural_similarity +
                    dimension_weights['lexical'] * edge_dims.lexical_similarity +
                    dimension_weights['temporal'] * (1 - edge_dims.temporal_distance)
                )
                
                # 閾値を超えたらエッジを追加
                if total_score > 0.3:  # 設定可能な閾値
                    graph.add_edge(
                        ep1.id,
                        ep2.id,
                        weight=total_score,  # 後方互換性
                        dimensions=edge_dims,  # 新しい多次元データ
                        **edge_dims.__dict__  # 個別アクセス用
                    )
                    
        return graph
```

#### 3.2.2 エッジフィルタリング
```python
class EdgeFilterer:
    """条件に基づいてエッジをフィルタリング"""
    
    def filter_by_dimension(
        self,
        graph: nx.Graph,
        dimension: str,
        min_value: float,
        max_value: float = 1.0
    ) -> nx.Graph:
        """特定の次元でフィルタリング"""
        filtered_graph = graph.copy()
        
        edges_to_remove = []
        for u, v, data in graph.edges(data=True):
            if dimension in data:
                value = data[dimension]
                if value < min_value or value > max_value:
                    edges_to_remove.append((u, v))
                    
        filtered_graph.remove_edges_from(edges_to_remove)
        return filtered_graph
    
    def filter_by_type(
        self,
        graph: nx.Graph,
        edge_types: List[str]
    ) -> nx.Graph:
        """エッジタイプでフィルタリング"""
        filtered_graph = graph.copy()
        
        edges_to_remove = []
        for u, v, data in graph.edges(data=True):
            if data.get('edge_type') not in edge_types:
                edges_to_remove.append((u, v))
                
        filtered_graph.remove_edges_from(edges_to_remove)
        return filtered_graph
```

### Phase 3: 検索と推論の拡張（2週間）

#### 3.3.1 多次元検索
```python
class MultiDimensionalSearch:
    """多次元エッジを活用した検索"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        
    def search_by_relationship(
        self,
        query_episode: Episode,
        relationship_type: str,
        dimension_weights: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        特定の関係性で検索
        
        Args:
            query_episode: クエリエピソード
            relationship_type: "similar", "contradictory", "causal", etc.
            dimension_weights: 各次元の重み
        """
        results = []
        
        # 関係タイプに応じた重み設定
        if relationship_type == "similar":
            weights = {
                'semantic_similarity': 0.7,
                'structural_similarity': 0.3,
                'contradiction_score': -0.5  # 矛盾は避ける
            }
        elif relationship_type == "contradictory":
            weights = {
                'semantic_similarity': 0.3,
                'contradiction_score': 0.7
            }
        elif relationship_type == "analogical":
            weights = {
                'structural_similarity': 0.8,
                'semantic_similarity': -0.2  # 意味は異なる
            }
        else:
            weights = dimension_weights
            
        # グラフを探索
        for node_id in self.graph.nodes():
            if node_id == query_episode.id:
                continue
                
            # エッジが存在する場合
            if self.graph.has_edge(query_episode.id, node_id):
                edge_data = self.graph.edges[query_episode.id, node_id]
                
                # 重み付きスコアを計算
                score = 0.0
                for dim, weight in weights.items():
                    if dim in edge_data:
                        score += weight * edge_data[dim]
                        
                results.append((node_id, score))
                
        # スコアでソート
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:10]  # Top-10
```

#### 3.3.2 パス探索の拡張
```python
class MultiDimensionalPathFinder:
    """多次元エッジを考慮したパス探索"""
    
    def find_semantic_path(
        self,
        graph: nx.Graph,
        start: str,
        end: str,
        dimension: str = "semantic_similarity"
    ) -> List[str]:
        """特定の次元を優先したパス探索"""
        
        def edge_weight(u, v, data):
            # 指定された次元の逆数を重みとする（高い類似度 = 短い距離）
            dim_value = data.get(dimension, 0.0)
            return 1.0 / (dim_value + 0.1)  # ゼロ除算回避
            
        try:
            path = nx.shortest_path(
                graph, 
                start, 
                end, 
                weight=edge_weight
            )
            return path
        except nx.NetworkXNoPath:
            return []
    
    def find_reasoning_chain(
        self,
        graph: nx.Graph,
        start: str,
        end: str
    ) -> List[Tuple[str, str, str]]:
        """推論チェーンを構築"""
        path = self.find_semantic_path(graph, start, end)
        
        if len(path) < 2:
            return []
            
        chain = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = graph.edges[u, v]
            
            # エッジタイプに基づいた関係性の説明
            relationship = self._explain_relationship(edge_data)
            chain.append((u, relationship, v))
            
        return chain
    
    def _explain_relationship(self, edge_data: Dict) -> str:
        """エッジデータから関係性を説明"""
        edge_type = edge_data.get('edge_type', 'related')
        
        if edge_type == 'similarity':
            return f"類似（{edge_data.get('semantic_similarity', 0):.2f}）"
        elif edge_type == 'contradiction':
            return f"矛盾（{edge_data.get('contradiction_score', 0):.2f}）"
        elif edge_type == 'temporal':
            order = edge_data.get('temporal_order', 0)
            if order < 0:
                return "時間的に前"
            elif order > 0:
                return "時間的に後"
            else:
                return "同時期"
        else:
            return edge_type
```

### Phase 4: GNN統合（2週間）

#### 3.4.1 多次元エッジGNN
```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MultiDimensionalEdgeGNN(MessagePassing):
    """多次元エッジを扱うGNN"""
    
    def __init__(
        self, 
        node_dim: int, 
        edge_dim: int = 7,  # エッジの次元数
        hidden_dim: int = 256
    ):
        super().__init__(aggr='mean')
        
        # エッジ特徴の処理
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # メッセージ生成
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # ノード更新
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: ノード特徴 [num_nodes, node_dim]
            edge_index: エッジインデックス [2, num_edges]
            edge_attr: エッジ特徴 [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """メッセージ生成"""
        # エッジ特徴をエンコード
        edge_features = self.edge_encoder(edge_attr)
        
        # ノード特徴とエッジ特徴を結合
        combined = torch.cat([x_i, x_j, edge_features], dim=-1)
        
        # メッセージを生成
        return self.message_mlp(combined)
    
    def update(self, aggr_out, x):
        """ノード更新"""
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)
```

## 4. データ構造の更新

### 4.1 グラフの永続化
```python
def save_multidim_graph(graph: nx.Graph, filepath: str):
    """多次元グラフを保存"""
    data = {
        'nodes': [],
        'edges': []
    }
    
    # ノードデータ
    for node_id, node_data in graph.nodes(data=True):
        node_info = {
            'id': node_id,
            'text': node_data.get('text', ''),
            'c_value': node_data.get('c_value', 0.5),
            'vec': node_data.get('vec', []).tolist() if hasattr(node_data.get('vec', []), 'tolist') else []
        }
        data['nodes'].append(node_info)
    
    # エッジデータ
    for u, v, edge_data in graph.edges(data=True):
        edge_info = {
            'source': u,
            'target': v,
            'dimensions': {}
        }
        
        # EdgeDimensionsの全フィールドを保存
        if 'dimensions' in edge_data:
            dims = edge_data['dimensions']
            edge_info['dimensions'] = {
                k: v for k, v in dims.__dict__.items()
                if not k.startswith('_')
            }
        else:
            # 後方互換性
            for key in ['semantic_similarity', 'structural_similarity', 
                       'temporal_distance', 'contradiction_score']:
                if key in edge_data:
                    edge_info['dimensions'][key] = edge_data[key]
                    
        data['edges'].append(edge_info)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

### 4.2 SQLiteでの保存
```sql
-- エッジテーブルの拡張
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    semantic_similarity REAL,
    structural_similarity REAL,
    lexical_similarity REAL,
    temporal_distance REAL,
    temporal_order INTEGER,
    contradiction_score REAL,
    entailment_score REAL,
    edge_type TEXT,
    confidence REAL,
    last_updated REAL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES episodes(id),
    FOREIGN KEY (target_id) REFERENCES episodes(id)
);

-- インデックス
CREATE INDEX idx_edge_semantic ON edges(semantic_similarity);
CREATE INDEX idx_edge_type ON edges(edge_type);
CREATE INDEX idx_edge_temporal ON edges(temporal_order);
```

## 5. 移行計画

### 5.1 後方互換性の維持
```python
class EdgeCompatibilityLayer:
    """既存のコードとの互換性を保つレイヤー"""
    
    @staticmethod
    def get_weight(edge_data: Dict) -> float:
        """従来のweight属性を返す"""
        if 'weight' in edge_data:
            return edge_data['weight']
        elif 'dimensions' in edge_data:
            # 主要な次元の重み付き平均
            dims = edge_data['dimensions']
            return (
                0.6 * dims.semantic_similarity +
                0.2 * dims.structural_similarity +
                0.2 * (1 - dims.temporal_distance)
            )
        else:
            return edge_data.get('semantic_similarity', 0.5)
```

### 5.2 段階的移行
1. **Week 1-2**: 基本実装とテスト
2. **Week 3**: 既存コードの更新
3. **Week 4-5**: 高度な機能の実装
4. **Week 6**: 統合テストとドキュメント

## 6. 期待される効果

1. **解釈可能性の向上**
   - なぜ関連しているかが明確に
   - 複数の観点から関係性を分析可能

2. **検索精度の向上**
   - ユーザーの意図に応じた検索
   - 多様な関係性の発見

3. **推論能力の向上**
   - アナロジー推論
   - 矛盾の検出
   - 時系列的な分析

4. **拡張性**
   - 新しい次元の追加が容易
   - タスク特化の最適化が可能

## 9. 実装の重要ポイント（まとめ）

### 9.1 速度維持の鍵
1. **遅延評価**: semantic_similarityのみ事前計算、他は必要時のみ
2. **使用頻度統計**: よく使う特徴を自動で事前計算に切り替え
3. **バッチ更新**: GPU-CPU転送を最小化

### 9.2 メモリ効率の鍵  
1. **FP16圧縮**: 32byte → 16byteで50%削減
2. **Edge Embedding Table**: 128次元ベクトルに圧縮
3. **正規化**: 各軸を適切にスケーリング

### 9.3 先行研究の活用
1. **BR-GCN**: 二段attention（Node + Relation level）
2. **TGN**: 時系列メモリセルで動的更新
3. **HGNN**: 異種エッジタイプの標準的扱い方

### 9.4 段階的実装で低リスク
- Phase 2.1a: 3軸のみ（現行速度維持）
- Phase 2.1b: 正規化と圧縮
- Phase 2.2: BR-GCN統合
- Phase 2.3: 高度な特徴追加

これにより、**現在の速度を維持しながら**、**表現力を大幅に向上**させることが可能。