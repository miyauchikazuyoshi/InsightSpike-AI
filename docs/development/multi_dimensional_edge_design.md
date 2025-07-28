# 多次元エッジ設計：FAISSなしで実現する豊かなグラフ表現

## 現状の制約

FAISSは高速ですが、**単一の類似度指標**しか扱えません：
- コサイン類似度のみ
- エッジ = 1つの数値
- 関係性の種類を区別できない

## 多次元エッジの可能性

FAISSを外すことで、エッジに複数の類似度を持たせられます：

```python
class MultiDimensionalEdge:
    """多次元的な関係性を表現するエッジ"""
    
    def __init__(self):
        self.semantic_similarity: float      # 意味的類似度
        self.structural_similarity: float    # 構造的類似度
        self.temporal_proximity: float       # 時間的近接性
        self.causal_strength: float         # 因果関係の強さ
        self.contradiction_level: float     # 矛盾度
        self.abstraction_diff: float        # 抽象度の差
```

## 具体的な実装案

### 1. 意味的類似度と構造的類似度の分離

```python
class EnhancedGraphBuilder:
    """多次元エッジを構築するグラフビルダー"""
    
    def build_edge(self, node1: Episode, node2: Episode) -> Dict[str, float]:
        edge_features = {}
        
        # 1. 意味的類似度（ベクトル空間）
        edge_features['semantic'] = cosine_similarity(node1.vec, node2.vec)
        
        # 2. 構造的類似度（テキスト構造）
        edge_features['structural'] = self._calculate_structural_similarity(
            node1.text, node2.text
        )
        
        # 3. 概念的距離（抽象度）
        edge_features['conceptual'] = self._calculate_conceptual_distance(
            node1, node2
        )
        
        # 4. 時系列的関係
        edge_features['temporal'] = self._calculate_temporal_relation(
            node1.timestamp, node2.timestamp
        )
        
        return edge_features
    
    def _calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """文章構造の類似性を計算"""
        # 段落数、文の長さ分布、句読点パターンなど
        struct1 = self._extract_structure(text1)
        struct2 = self._extract_structure(text2)
        
        return self._compare_structures(struct1, struct2)
    
    def _calculate_conceptual_distance(self, ep1: Episode, ep2: Episode) -> float:
        """概念階層での距離を計算"""
        # 具体的 ↔ 抽象的
        abstraction1 = self._measure_abstraction_level(ep1.text)
        abstraction2 = self._measure_abstraction_level(ep2.text)
        
        return 1.0 - abs(abstraction1 - abstraction2)
```

### 2. GNNでの活用

```python
class MultiEdgeGNN(torch.nn.Module):
    """多次元エッジを活用するGNN"""
    
    def __init__(self, edge_dim: int = 6):
        super().__init__()
        
        # エッジタイプごとの重み学習
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=edge_dim,
            num_heads=edge_dim  # 各次元に1つのヘッド
        )
        
        # メッセージパッシング
        self.message_mlp = nn.Sequential(
            nn.Linear(edge_dim * 2 + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x, edge_index, edge_features):
        # エッジの種類に応じた重み付けメッセージパッシング
        for src, dst in edge_index:
            # 多次元エッジ特徴を考慮
            edge_vec = edge_features[src, dst]
            
            # 注意機構でエッジタイプの重要度を動的に決定
            edge_weights = self.edge_attention(edge_vec)
            
            # 重み付きメッセージ
            message = self.compute_message(x[src], x[dst], edge_weights)
            x[dst] = self.update_node(x[dst], message)
        
        return x
```

### 3. 洞察検出への応用

```python
def detect_multi_dimensional_insight(graph: nx.Graph) -> List[Insight]:
    """多次元エッジを活用した洞察検出"""
    
    insights = []
    
    for node in graph.nodes():
        neighbors = graph.neighbors(node)
        
        for neighbor in neighbors:
            edge_data = graph.edges[node, neighbor]
            
            # 1. 意味は似ているが構造が異なる → 言い換えの発見
            if (edge_data['semantic'] > 0.8 and 
                edge_data['structural'] < 0.3):
                insights.append(Paraphrase(node, neighbor))
            
            # 2. 構造は似ているが意味が異なる → アナロジーの発見
            elif (edge_data['structural'] > 0.8 and 
                  edge_data['semantic'] < 0.5):
                insights.append(Analogy(node, neighbor))
            
            # 3. 時間的に近いが意味が矛盾 → 認識の変化
            elif (edge_data['temporal'] > 0.9 and 
                  edge_data['contradiction'] > 0.7):
                insights.append(ConceptEvolution(node, neighbor))
            
            # 4. 抽象度が大きく異なる → 一般化/具体化
            elif abs(edge_data['abstraction_diff']) > 0.7:
                if edge_data['abstraction_diff'] > 0:
                    insights.append(Generalization(node, neighbor))
                else:
                    insights.append(Instantiation(node, neighbor))
    
    return insights
```

### 4. 検索の高度化

```python
class MultiDimensionalSearch:
    """多次元類似度を使った高度な検索"""
    
    def search(
        self, 
        query: str,
        weights: Dict[str, float] = None
    ) -> List[Episode]:
        """
        重み付き多次元検索
        
        Args:
            query: 検索クエリ
            weights: 各類似度の重み
                - semantic: 意味的類似度の重み
                - structural: 構造的類似度の重み
                - conceptual: 概念的距離の重み
        """
        if weights is None:
            weights = {
                'semantic': 0.6,
                'structural': 0.2,
                'conceptual': 0.2
            }
        
        results = []
        query_features = self._extract_features(query)
        
        for episode in self.episodes:
            # 多次元スコアの計算
            score = 0.0
            
            # 意味的類似度
            semantic_sim = cosine_similarity(
                query_features['vec'], 
                episode.vec
            )
            score += weights['semantic'] * semantic_sim
            
            # 構造的類似度
            structural_sim = self._structural_similarity(
                query_features['structure'],
                episode.structure
            )
            score += weights['structural'] * structural_sim
            
            # 概念的距離
            conceptual_sim = self._conceptual_similarity(
                query_features['concepts'],
                episode.concepts
            )
            score += weights['conceptual'] * conceptual_sim
            
            results.append((episode, score))
        
        # スコアでソート
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [ep for ep, _ in results[:10]]
```

## メリット

### 1. より豊かな関係性の表現
- 「なぜ」関連しているかが明確に
- 複数の観点からの関連性を同時に扱える
- 矛盾や対立関係も表現可能

### 2. 高度な推論
- アナロジー推論
- 抽象化/具体化
- 時系列的な概念進化の追跡

### 3. 柔軟な検索
- ユーザーの意図に応じた重み調整
- 「構造が似た文章」「意味が反対の文章」など多様な検索

### 4. 学習可能性
- GNNでエッジタイプの重要度を学習
- タスクに応じた最適な重み付け

## 実装の段階的アプローチ

### Phase 1: 基本的な多次元エッジ
```python
edge_data = {
    'weight': 0.8,  # 後方互換性
    'semantic': 0.8,
    'structural': 0.6
}
```

### Phase 2: 高度な類似度追加
```python
edge_data.update({
    'temporal': 0.3,
    'causal': 0.7,
    'contradiction': 0.1
})
```

### Phase 3: 学習ベースの最適化
- GNNでエッジ重みを学習
- タスク特化の類似度定義

## まとめ

FAISSを外すことで：
- **エッジが単なる数値から、豊かな関係性の表現へ**
- **意味と構造を分離して扱える**
- **より高度な推論と検索が可能に**

これは単なる性能トレードオフではなく、**知識表現の質的な向上**をもたらします。