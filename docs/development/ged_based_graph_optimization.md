# GED-Based Graph Optimization Plan

## 概要

現在のInsightSpikeは類似度ベースで即座にエッジを確定していますが、より洗練されたGED（Graph Edit Distance）ベースの最適化アプローチを実装することで、グラフ構造の品質を向上させることができます。

## 現在の実装

### フロー
1. **類似度検索**: FAISSを使用してtop-k近傍を検索
2. **エッジ作成**: 類似度 > 0.7 なら即座にエッジを作成（永続的）
3. **事後評価**: ΔGED、ΔIGを計算して洞察検出とC値報酬に使用

### 課題
- エッジは一度作成されると変更されない
- 類似度閾値（0.7）は固定
- ノイズの多いエッジが含まれる可能性
- グラフ構造の最適化機会を逃している

## 提案: GEDベースの最適化

### 新しいフロー
```
1. 仮エッジ作成
   ├─ 類似度ベースで幅広くエッジ候補を生成
   └─ 閾値を緩める（例: 0.5以上）

2. GED最適化
   ├─ 各エッジ候補についてGEDへの影響を評価
   ├─ GEDが改善（負の方向）するエッジを優先
   └─ 組み合わせ最適化でベストな構造を選択


3. エッジ確定
   ├─ 最適化されたエッジセットを確定
   └─ IG計算で情報利得を評価
```

## 実装設計

### Phase 1: 基本実装

```python
class OptimizedGraphBuilder(ScalableGraphBuilder):
    def build_graph_with_optimization(self, documents, embeddings):
        # Step 1: 仮エッジ候補の生成
        candidate_edges = self._generate_candidate_edges(
            embeddings, 
            similarity_threshold=0.5  # 緩い閾値
        )
        
        # Step 2: GED最適化
        optimized_edges = self._optimize_edges_by_ged(
            candidate_edges,
            previous_graph=self.previous_graph,
            max_iterations=10
        )
        
        # Step 3: グラフ構築
        return self._build_graph_from_edges(documents, optimized_edges)
    
    def _optimize_edges_by_ged(self, candidates, previous_graph, max_iterations):
        """GEDを最小化するエッジセットを選択"""
        current_edges = candidates.copy()
        best_ged = float('inf')
        
        for _ in range(max_iterations):
            # 各エッジの追加/削除によるGED変化を計算
            edge_impacts = self._calculate_edge_impacts(
                current_edges, 
                previous_graph
            )
            
            # GEDが改善するエッジを選択
            improved_edges = self._select_improving_edges(
                edge_impacts,
                improvement_threshold=-0.1
            )
            
            # 収束判定
            new_ged = self._calculate_ged(improved_edges, previous_graph)
            if new_ged >= best_ged:
                break
                
            current_edges = improved_edges
            best_ged = new_ged
            
        return current_edges
```

### Phase 2: 高度な最適化

#### 1. 適応的閾値
```python
def _adaptive_threshold(self, graph_density, target_density=0.3):
    """グラフ密度に基づいて閾値を動的調整"""
    if graph_density > target_density:
        return 0.8  # 密なグラフは閾値を上げる
    else:
        return 0.6  # 疎なグラフは閾値を下げる
```

#### 2. エッジ重み最適化
```python
def _optimize_edge_weights(self, edges, ged_gradient):
    """GED勾配に基づいてエッジ重みを調整"""
    for edge in edges:
        edge.weight *= (1 + learning_rate * ged_gradient[edge])
    return edges
```

#### 3. 多目的最適化
```python
def _multi_objective_optimization(self, candidates):
    """GED最小化とIG最大化を同時に最適化"""
    pareto_front = []
    
    for edge_subset in self._generate_edge_combinations(candidates):
        ged = self._calculate_ged(edge_subset)
        ig = self._calculate_ig(edge_subset)
        
        # パレート最適解を更新
        if self._is_pareto_optimal(ged, ig, pareto_front):
            pareto_front.append((edge_subset, ged, ig))
    
    # 重み付けで最終選択
    return self._select_best_from_pareto(pareto_front)
```

## 評価指標

### 1. グラフ品質メトリクス
- **構造的一貫性**: グラフの連結成分数、クラスタリング係数
- **情報密度**: エッジあたりの情報利得
- **ノイズ比率**: 低重要度エッジの割合

### 2. パフォーマンスメトリクス
- **計算時間**: 最適化にかかる時間
- **メモリ使用量**: 候補エッジの保持コスト
- **収束速度**: 最適解への到達イテレーション数

### 3. 洞察品質メトリクス
- **スパイク検出率**: 洞察の発見頻度
- **誤検出率**: ノイズによる誤った洞察
- **応答品質**: 生成される回答の関連性

## 実装計画

### フェーズ1: プロトタイプ（2週間）
- [ ] OptimizedGraphBuilderクラスの基本実装
- [ ] 単純なGED最適化アルゴリズム
- [ ] ユニットテストの作成

### フェーズ2: 最適化（3週間）
- [ ] 高速化（近似アルゴリズム、キャッシング）
- [ ] 適応的閾値の実装
- [ ] パフォーマンスベンチマーク

### フェーズ3: 統合（2週間）
- [ ] Layer3GraphReasonerとの統合
- [ ] 既存のテストスイートでの検証
- [ ] A/Bテストによる品質評価

### フェーズ4: 高度な機能（4週間）
- [ ] エッジ重み最適化
- [ ] 多目的最適化
- [ ] 実験と調整

## リスクと対策

### 1. 計算コストの増大
**リスク**: 組み合わせ最適化により計算時間が爆発的に増加
**対策**: 
- ヒューリスティックによる枝刈り
- 並列処理の活用
- 段階的な最適化（粗い→細かい）

### 2. 過学習
**リスク**: 特定のグラフ構造に過剰適応
**対策**:
- 正則化項の追加
- クロスバリデーション
- 多様性を保つ制約

### 3. 後方互換性
**リスク**: 既存のシステムとの互換性問題
**対策**:
- フィーチャーフラグによる段階的導入
- 旧アルゴリズムとの並行稼働
- 徹底的な回帰テスト

## 期待される効果

1. **ノイズ削減**: 不要なエッジを事前に除去
2. **構造最適化**: より意味のあるグラフ構造
3. **洞察品質向上**: 関連性の高い情報の強調
4. **計算効率改善**: 疎で効率的なグラフ

## 参考文献

- [Graph Edit Distance Computation](https://arxiv.org/abs/1709.08652)
- [Scalable Graph Construction Algorithms](https://dl.acm.org/doi/10.1145/3318464.3389706)
- [Multi-objective Graph Optimization](https://ieeexplore.ieee.org/document/8851875)

## 付録: 実験用コード

```python
# experiments/ged_optimization/test_optimization.py
def compare_graph_construction_methods():
    """現在の手法とGED最適化手法を比較"""
    
    # データセット準備
    documents = load_test_documents()
    
    # 現在の手法
    current_graph = ScalableGraphBuilder().build_graph(documents)
    
    # GED最適化手法
    optimized_graph = OptimizedGraphBuilder().build_graph_with_optimization(documents)
    
    # メトリクス比較
    metrics = {
        "current": calculate_metrics(current_graph),
        "optimized": calculate_metrics(optimized_graph)
    }
    
    return metrics
```

---

*Last Updated: 2024-01-19*
*Author: Claude & User*
*Status: Planning Phase*