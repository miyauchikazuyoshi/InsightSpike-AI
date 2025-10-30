---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# グラフエッジ生成問題の解決策

## 問題の概要

グラフ構築時にエッジが生成されない問題が頻発しています。主な原因は：

1. **類似度閾値が高すぎる** (デフォルト: 0.3)
2. **正規化の問題**
3. **データの多様性**
4. **バッチ処理の制限**

## 推奨される解決策

### 1. 類似度閾値の動的調整

```python
def calculate_adaptive_threshold(embeddings: np.ndarray, percentile: float = 10) -> float:
    """
    データセットに基づいて適応的な閾値を計算
    
    Args:
        embeddings: 正規化済みの埋め込みベクトル
        percentile: 使用するパーセンタイル (デフォルト: 10%)
    
    Returns:
        適応的な類似度閾値
    """
    # サンプルペアの類似度を計算
    n_samples = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # ペアワイズ類似度を計算
    similarities = np.dot(sample_embeddings, sample_embeddings.T)
    
    # 対角要素を除外（自己類似度）
    mask = ~np.eye(n_samples, dtype=bool)
    similarities = similarities[mask]
    
    # パーセンタイルベースの閾値を計算
    threshold = np.percentile(similarities, 100 - percentile)
    
    # 最小値を保証
    return max(threshold, 0.1)
```

### 2. 最小エッジ保証

```python
def ensure_minimum_edges(
    distances: np.ndarray,
    neighbors: np.ndarray,
    min_edges_per_node: int = 1
) -> List[List[int]]:
    """
    各ノードが最低限のエッジを持つことを保証
    
    Args:
        distances: FAISSからの距離/類似度行列
        neighbors: FAISSからの近傍インデックス
        min_edges_per_node: ノードあたりの最小エッジ数
    
    Returns:
        エッジリスト
    """
    edge_list = []
    
    for i, (dists, neighs) in enumerate(zip(distances, neighbors)):
        # 有効な近傍を取得（-1を除外）
        valid_indices = neighs != -1
        valid_neighs = neighs[valid_indices]
        valid_dists = dists[valid_indices]
        
        # 自己接続を除外
        non_self_mask = valid_neighs != i
        valid_neighs = valid_neighs[non_self_mask]
        valid_dists = valid_dists[non_self_mask]
        
        if len(valid_neighs) > 0:
            # 最低限のエッジを追加（最も類似度の高いものから）
            n_edges = min(min_edges_per_node, len(valid_neighs))
            top_indices = np.argsort(valid_dists)[-n_edges:]
            
            for idx in top_indices:
                edge_list.append([i, valid_neighs[idx]])
    
    return edge_list
```

### 3. 改善されたScalableGraphBuilder

```python
class ImprovedScalableGraphBuilder(ScalableGraphBuilder):
    def __init__(
        self,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.2,  # より低い閾値
        top_k: int = 50,
        batch_size: int = 1000,
        use_adaptive_threshold: bool = True,
        min_edges_per_node: int = 1,
        embedder: Optional[Any] = None,
        monitor: Optional[Any] = None
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            batch_size=batch_size,
            embedder=embedder,
            monitor=monitor
        )
        self.use_adaptive_threshold = use_adaptive_threshold
        self.min_edges_per_node = min_edges_per_node
    
    def _build_from_scratch(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """改善されたグラフ構築"""
        # 埋め込みを正規化
        embeddings = self._normalize_embeddings(embeddings)
        
        # 適応的閾値を計算
        if self.use_adaptive_threshold and len(embeddings) > 10:
            self.similarity_threshold = calculate_adaptive_threshold(embeddings)
            print(f"Adaptive threshold: {self.similarity_threshold:.3f}")
        
        # FAISSインデックスを構築
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        
        # エッジリストを作成
        edge_list = []
        
        # バッチ処理
        for start_idx in range(0, len(embeddings), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(embeddings))
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # k近傍を検索
            k = min(self.top_k + 1, len(embeddings))
            distances, neighbors = self.index.search(batch_embeddings, k)
            
            # 最小エッジ保証付きでエッジを作成
            batch_edges = ensure_minimum_edges(
                distances, 
                neighbors,
                self.min_edges_per_node
            )
            
            # バッチオフセットを追加
            for src, dst in batch_edges:
                edge_list.append([start_idx + src, dst])
        
        # エッジ統計をログ
        if len(edge_list) == 0:
            print(f"Warning: No edges created! Consider lowering similarity threshold.")
            print(f"Current threshold: {self.similarity_threshold}")
            # 緊急措置：最近傍を強制的に接続
            edge_list = self._create_emergency_edges(embeddings)
        
        return embeddings, edge_list
    
    def _create_emergency_edges(self, embeddings: np.ndarray) -> List[List[int]]:
        """エッジが全く作成されない場合の緊急措置"""
        edge_list = []
        k = min(3, len(embeddings))
        
        distances, neighbors = self.index.search(embeddings, k)
        
        for i, neighs in enumerate(neighbors):
            # 最も近い非自己ノードに接続
            for n in neighs:
                if n != -1 and n != i:
                    edge_list.append([i, n])
                    break
        
        print(f"Emergency edges created: {len(edge_list)}")
        return edge_list
```

### 4. 設定ファイルの更新

`config.py`を更新して、より柔軟な設定を可能にする：

```python
@dataclass
class GraphConfig:
    """Graph construction configuration"""
    
    # 基本設定
    similarity_threshold: float = 0.2  # 0.3から0.2に下げる
    top_k: int = 50
    batch_size: int = 1000
    
    # 適応的設定
    use_adaptive_threshold: bool = True
    adaptive_percentile: float = 10.0  # 上位10%の類似度ペア
    min_edges_per_node: int = 1  # 各ノードの最小エッジ数
    
    # 緊急措置
    force_minimum_connectivity: bool = True  # 最小接続性を強制
```

### 5. デバッグとモニタリング

```python
def log_graph_statistics(edge_list: List[List[int]], num_nodes: int):
    """グラフ統計をログ出力"""
    if not edge_list:
        print("WARNING: Empty edge list!")
        return
    
    # ノードごとの次数を計算
    degrees = np.zeros(num_nodes)
    for src, dst in edge_list:
        degrees[src] += 1
        degrees[dst] += 1
    
    print(f"Graph Statistics:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Edges: {len(edge_list)}")
    print(f"  - Avg degree: {np.mean(degrees):.2f}")
    print(f"  - Min degree: {np.min(degrees)}")
    print(f"  - Max degree: {np.max(degrees)}")
    print(f"  - Isolated nodes: {np.sum(degrees == 0)}")
```

## 実装の優先順位

1. **即座に実装すべき**:
   - 類似度閾値を0.3から0.2に下げる
   - 最小エッジ保証を実装

2. **次に実装すべき**:
   - 適応的閾値計算
   - グラフ統計のロギング

3. **将来的に検討**:
   - より高度なグラフ構築アルゴリズム
   - 階層的クラスタリングの改善

## テスト方法

```python
def test_edge_generation():
    """エッジ生成のテスト"""
    # 多様なデータセットでテスト
    embeddings = np.random.randn(100, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    builder = ImprovedScalableGraphBuilder(
        similarity_threshold=0.2,
        use_adaptive_threshold=True,
        min_edges_per_node=2
    )
    
    graph = builder.build_graph(
        documents=[{"text": f"doc_{i}"} for i in range(100)],
        embeddings=embeddings
    )
    
    print(f"Edges created: {graph.edge_index.shape[1]}")
    assert graph.edge_index.shape[1] > 0, "No edges created!"
```

この改善により、グラフ構築の信頼性が大幅に向上するはずです。