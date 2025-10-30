---
status: active
category: infra
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# FAISS除去実装計画

## 1. 概要

FAISSへの依存を段階的に除去し、純粋なPython実装に移行する計画です。これにより、インストールの簡素化、デバッグの容易さ、プラットフォーム独立性を実現します。

## 2. 現状分析

### 2.1 FAISSを使用しているモジュール
- `ScalableGraphBuilder` - グラフ構築時の近傍検索
- `SQLiteStore` - ベクトル検索の高速化
- `FileSystemStore` - ベクトル検索の高速化
- `ScalableGraphManager` - グラフ管理

### 2.2 現在のフォールバック実装
```python
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False
    faiss = MockFaiss()  # 最小限のモック
```

## 3. 実装フェーズ

### Phase 1: NumPyベースの近傍検索実装（2週間）

#### 3.1.1 基本実装
```python
class NumpyNearestNeighborIndex:
    """NumPyベースの近傍検索インデックス"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None
        self.ids = None
        self.is_trained = True  # FAISSとの互換性
        
    def add(self, vectors: np.ndarray):
        """ベクトルを追加"""
        if self.vectors is None:
            self.vectors = vectors.copy()
            self.ids = np.arange(len(vectors))
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            new_ids = np.arange(len(self.vectors) - len(vectors), len(self.vectors))
            self.ids = np.concatenate([self.ids, new_ids])
            
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """k近傍検索"""
        distances = []
        indices = []
        
        for query in queries:
            # コサイン類似度計算
            similarities = np.dot(self.vectors, query)
            similarities = similarities / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query))
            
            # Top-k取得
            if k >= len(similarities):
                top_k_idx = np.argsort(similarities)[::-1]
            else:
                top_k_idx = np.argpartition(similarities, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
            
            distances.append(similarities[top_k_idx])
            indices.append(self.ids[top_k_idx])
            
        return np.array(distances), np.array(indices)
```

#### 3.1.2 最適化版（バッチ処理）
```python
class OptimizedNumpyIndex:
    """最適化されたNumPy検索"""
    
    def search_batch(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """バッチ検索で高速化"""
        # 正規化済みベクトルを事前計算
        if not hasattr(self, '_normalized_vectors'):
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self._normalized_vectors = self.vectors / (norms + 1e-8)
            
        # バッチでコサイン類似度計算
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        normalized_queries = queries / (query_norms + 1e-8)
        
        # 行列積で一括計算
        similarities = np.dot(normalized_queries, self._normalized_vectors.T)
        
        # 各クエリのTop-k
        distances = np.zeros((len(queries), k))
        indices = np.zeros((len(queries), k), dtype=int)
        
        for i, sim in enumerate(similarities):
            if k >= len(sim):
                top_k_idx = np.argsort(sim)[::-1][:k]
            else:
                top_k_idx = np.argpartition(sim, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(sim[top_k_idx])[::-1]]
                
            distances[i] = sim[top_k_idx]
            indices[i] = self.ids[top_k_idx]
            
        return distances, indices
```

### Phase 2: インターフェース統一（1週間）

#### 3.2.1 ファクトリーパターンで切り替え
```python
class VectorIndexFactory:
    """ベクトルインデックスのファクトリー"""
    
    @staticmethod
    def create_index(
        dimension: int,
        index_type: str = "auto",
        **kwargs
    ) -> VectorIndexInterface:
        """
        インデックスを作成
        
        Args:
            dimension: ベクトル次元
            index_type: "faiss", "numpy", "auto"
        """
        if index_type == "auto":
            # FAISSが利用可能ならFAISS、なければNumPy
            if FAISS_AVAILABLE:
                index_type = "faiss"
            else:
                index_type = "numpy"
                
        if index_type == "faiss" and FAISS_AVAILABLE:
            return FaissIndexWrapper(dimension, **kwargs)
        else:
            return NumpyNearestNeighborIndex(dimension, **kwargs)
```

#### 3.2.2 設定での切り替え
```yaml
# config.yaml
vector_search:
  backend: "numpy"  # "faiss", "numpy", "auto"
  numpy_options:
    batch_size: 1000
    use_optimization: true
```

### Phase 3: 移行とテスト（1週間）

#### 3.3.1 パフォーマンステスト
```python
def benchmark_vector_search():
    """検索性能のベンチマーク"""
    sizes = [1000, 10000, 100000]
    dimensions = [384, 768]
    
    results = {}
    
    for size in sizes:
        for dim in dimensions:
            # テストデータ生成
            vectors = np.random.randn(size, dim).astype(np.float32)
            queries = np.random.randn(100, dim).astype(np.float32)
            
            # NumPy版
            numpy_index = NumpyNearestNeighborIndex(dim)
            numpy_index.add(vectors)
            
            start = time.time()
            numpy_index.search(queries, k=10)
            numpy_time = time.time() - start
            
            # FAISS版（利用可能な場合）
            if FAISS_AVAILABLE:
                faiss_index = faiss.IndexFlatIP(dim)
                faiss_index.add(vectors)
                
                start = time.time()
                faiss_index.search(queries, k=10)
                faiss_time = time.time() - start
            else:
                faiss_time = None
                
            results[f"{size}x{dim}"] = {
                "numpy": numpy_time,
                "faiss": faiss_time,
                "ratio": numpy_time / faiss_time if faiss_time else None
            }
            
    return results
```

#### 3.3.2 精度検証
```python
def verify_search_accuracy():
    """検索精度の検証"""
    # 同じデータで両方のインデックスをテスト
    vectors = np.random.randn(1000, 384)
    queries = np.random.randn(10, 384)
    
    numpy_idx = NumpyNearestNeighborIndex(384)
    numpy_idx.add(vectors)
    numpy_dist, numpy_ids = numpy_idx.search(queries, k=10)
    
    if FAISS_AVAILABLE:
        faiss_idx = faiss.IndexFlatIP(384)
        faiss_idx.add(vectors)
        faiss_dist, faiss_ids = faiss_idx.search(queries, k=10)
        
        # 結果の一致を確認
        for i in range(len(queries)):
            # Top-10の順序が同じか確認
            assert np.array_equal(numpy_ids[i], faiss_ids[i]), \
                f"Query {i}: Results differ"
```

### Phase 4: FAISS依存の除去（1週間）

#### 3.4.1 条件付きインポートの削除
```python
# Before
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

# After
# FAISSインポートを完全に削除
FAISS_AVAILABLE = False  # 後方互換性のため一時的に残す
```

#### 3.4.2 pyproject.tomlの更新
```toml
[tool.poetry.dependencies]
# faiss-cpu = { version = "^1.7.4", optional = true }  # 削除

[tool.poetry.extras]
# faiss = ["faiss-cpu"]  # 削除
```

## 4. 移行スケジュール

### Week 1-2: Phase 1
- NumPyベース実装の作成
- 単体テストの作成
- パフォーマンス最適化

### Week 3: Phase 2
- インターフェース統一
- 設定システムの更新
- ドキュメント更新

### Week 4: Phase 3
- 統合テスト
- ベンチマーク実施
- 移行ガイド作成

### Week 5: Phase 4
- FAISS依存の除去
- 最終テスト
- リリース準備

## 5. リスク管理

### 5.1 パフォーマンス低下
- **対策**: 10万件を超える場合は外部ベクトルDBを推奨
- **緩和策**: 階層的インデックスの実装を検討

### 5.2 後方互換性
- **対策**: 移行期間中は両方サポート
- **緩和策**: 明確な移行ガイドとツール提供

### 5.3 既存ユーザーへの影響
- **対策**: 設定で選択可能に
- **緩和策**: パフォーマンス比較データの公開

## 6. 成功基準

1. **機能面**
   - 全ての既存テストがパス
   - 検索精度がFAISSと同等

2. **性能面**
   - 1万件以下：2倍以内の速度低下
   - 10万件以下：10倍以内の速度低下

3. **ユーザビリティ**
   - インストール成功率100%
   - プラットフォーム依存なし

## 7. 今後の拡張

### 7.1 階層的インデックス
- K-meansクラスタリング
- 2段階検索で高速化

### 7.2 近似アルゴリズム
- LSH（Locality Sensitive Hashing）
- Random Projection

### 7.3 外部統合
- PostgreSQL pgvector
- Elasticsearch vector search
- 専用ベクトルDB（Pinecone等）