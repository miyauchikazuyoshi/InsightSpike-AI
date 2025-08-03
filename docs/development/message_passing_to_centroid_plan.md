# メッセージパッシングから重心計算への移行実装計画

## 概要
現在の複雑なメッセージパッシングアルゴリズムを、シンプルで効率的な重心計算に置き換える実装計画。

## 理論的背景

### 現在のアプローチ（メッセージパッシング）
```
1. クエリから近傍ノードを選択
2. ノード間でメッセージを伝播
3. 収束するまで反復
4. 最終的な表現を取得
```

### 新しいアプローチ（重心計算）
```
1. クエリから近傍ノードを選択
2. 重み付き重心を計算
3. 重心位置が洞察の位置
```

## 実装詳細

### Phase 1: 重心計算器の実装

#### 1.1 基本的な重心計算クラス
```python
# src/insightspike/graph/centroid_calculator.py

class CentroidCalculator:
    """Calculate weighted centroid of selected nodes."""
    
    def calculate_centroid(
        self,
        query_vec: np.ndarray,
        neighbor_vecs: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Calculate weighted centroid including query.
        
        Args:
            query_vec: Query vector
            neighbor_vecs: List of neighbor vectors
            weights: Optional weights for each vector (query + neighbors)
        
        Returns:
            Centroid vector (normalized)
        """
        # Combine query and neighbors
        all_vecs = [query_vec] + neighbor_vecs
        
        # Default weights: higher for query
        if weights is None:
            weights = [2.0] + [1.0] * len(neighbor_vecs)
        
        # Calculate weighted centroid
        weighted_sum = sum(w * v for w, v in zip(weights, all_vecs))
        centroid = weighted_sum / sum(weights)
        
        # Normalize
        return centroid / (np.linalg.norm(centroid) + 1e-8)
```

#### 1.2 重み戦略
```python
class WeightingStrategy:
    """Different strategies for weighting nodes."""
    
    @staticmethod
    def distance_based(distances: List[float]) -> List[float]:
        """Weight inversely proportional to distance."""
        return [1.0 / (d + 0.1) for d in distances]
    
    @staticmethod
    def rank_based(n_items: int) -> List[float]:
        """Weight based on rank (closer = higher weight)."""
        return [1.0 / (i + 1) for i in range(n_items)]
    
    @staticmethod
    def uniform(n_items: int) -> List[float]:
        """Equal weights for all."""
        return [1.0] * n_items
```

### Phase 2: 既存のメッセージパッシングとの統合

#### 2.1 切り替え可能な実装
```python
# src/insightspike/graph/insight_generator.py

class InsightGenerator:
    """Generate insights using either message passing or centroid."""
    
    def __init__(self, method: str = "centroid"):
        self.method = method
        self.centroid_calc = CentroidCalculator()
        self.message_passer = MessagePassing()  # 既存実装
    
    def generate_insight(
        self,
        query_vec: np.ndarray,
        neighbor_data: List[Tuple[np.ndarray, float]],
        **kwargs
    ) -> np.ndarray:
        """Generate insight vector."""
        if self.method == "centroid":
            vecs = [v for v, _ in neighbor_data]
            distances = [d for _, d in neighbor_data]
            weights = WeightingStrategy.distance_based(distances)
            return self.centroid_calc.calculate_centroid(
                query_vec, vecs, weights
            )
        else:
            # Fallback to message passing
            return self.message_passer.process(query_vec, neighbor_data)
```

### Phase 3: 設定システムへの統合

#### 3.1 設定モデルの更新
```python
# src/insightspike/config/models.py に追加

class InsightGenerationConfig(BaseModel):
    """Configuration for insight generation."""
    method: Literal["centroid", "message_passing"] = "centroid"
    
    # Centroid specific
    weighting_strategy: Literal["distance", "rank", "uniform"] = "distance"
    query_weight_multiplier: float = 2.0
    
    # Message passing specific (既存)
    n_iterations: int = 3
    damping_factor: float = 0.85
```

### Phase 4: Layer3への統合

#### 4.1 GraphReasonerの更新
```python
# src/insightspike/layer3_graph_reasoner.py の更新

class GraphReasoner:
    def __init__(self, config: dict):
        # ... existing init ...
        self.insight_config = InsightGenerationConfig(
            **config.get("insight_generation", {})
        )
        self.insight_generator = InsightGenerator(
            method=self.insight_config.method
        )
    
    def reason(self, query: str) -> Dict:
        # ... existing neighbor search ...
        
        # Generate insight using configured method
        insight_vec = self.insight_generator.generate_insight(
            query_vec, neighbors
        )
        
        # Decode insight to text (existing decoder)
        insight_text = self.decoder.decode(insight_vec)
        
        return {
            "insight": insight_text,
            "method": self.insight_config.method,
            "position": insight_vec
        }
```

## 実装順序

1. **Week 1**: 基本実装
   - [ ] CentroidCalculator クラス
   - [ ] WeightingStrategy クラス
   - [ ] 単体テスト

2. **Week 2**: 統合
   - [ ] InsightGenerator クラス
   - [ ] 設定システムへの追加
   - [ ] 既存テストの更新

3. **Week 3**: 実験と評価
   - [ ] A/Bテスト環境の構築
   - [ ] パフォーマンス比較
   - [ ] 精度比較

## 期待される効果

### パフォーマンス向上
- メッセージパッシング: O(n² × iterations)
- 重心計算: O(n)

### コードの簡潔性
- 実装行数: ~80% 削減
- 理解しやすさ: 大幅向上

### 精度
- 理論的には等価（空間的な解釈）
- 実験で検証予定

## 実験計画

### 比較実験
```python
# experiments/pre-experiment/test_centroid_vs_message_passing.py

def compare_methods():
    # 1. 同じクエリと近傍での結果比較
    # 2. 処理時間の測定
    # 3. 生成される洞察の質の評価
    pass
```

### 評価指標
1. 処理時間
2. メモリ使用量
3. 洞察の意味的類似度
4. 人間による評価

## リスクと対策

### リスク
1. メッセージパッシングの方が優れている可能性
2. 既存の動作との互換性

### 対策
1. 設定で切り替え可能にする
2. 段階的な移行
3. 十分な比較実験

## 次のステップ

1. このプランのレビュー
2. CentroidCalculatorの実装開始
3. 小規模な実験での検証