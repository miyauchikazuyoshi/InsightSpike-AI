---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# エピソード重複削除とリファクタリング設計

## 目的

コードリファクタリングの原則を知識グラフに適用：
- **DRY原則** (Don't Repeat Yourself) - 重複知識の削除
- **単一責任原則** - 各エピソードは1つの明確な知識を持つ
- **関心の分離** - 異なる概念は別エピソードに

## 重複パターンの種類と検出方法

### 1. 完全重複（Exact Duplicates）

```python
def detect_exact_duplicates(self) -> List[Tuple[int, int]]:
    """完全に同一のエピソードを検出"""
    duplicates = []
    
    # ベクトル空間で完全一致を検索
    for i in range(len(self.episodes)):
        # FAISSで距離0の近傍を検索
        distances, indices = self.index.search(
            self.episodes[i].vector.reshape(1, -1), 
            k=10
        )
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != i and dist < 1e-6:  # ほぼ0
                if (idx, i) not in duplicates:  # 順序を考慮
                    duplicates.append((i, idx))
    
    return duplicates
```

### 2. 意味的重複（Semantic Duplicates）

```python
def detect_semantic_duplicates(self, threshold: float = 0.95) -> List[Dict]:
    """意味的に重複するエピソードを検出"""
    semantic_groups = []
    
    # 高類似度クラスタを形成
    from sklearn.cluster import DBSCAN
    
    # コサイン距離でクラスタリング
    clustering = DBSCAN(eps=1-threshold, metric='cosine', min_samples=2)
    labels = clustering.fit_predict(self.vectors)
    
    # 各クラスタ内で重複を評価
    for label in set(labels):
        if label == -1:  # ノイズ
            continue
            
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        if len(cluster_indices) > 1:
            # クラスタ内の情報量を比較
            info_contents = [
                self.calculate_information_content(i) 
                for i in cluster_indices
            ]
            
            semantic_groups.append({
                'indices': cluster_indices,
                'info_contents': info_contents,
                'redundancy': self.calculate_redundancy(cluster_indices)
            })
    
    return semantic_groups
```

### 3. 部分重複（Partial Redundancy）

```python
def detect_partial_redundancy(self) -> List[Dict]:
    """部分的に重複する知識を検出"""
    partial_overlaps = []
    
    for i, ep1 in enumerate(self.episodes):
        for j, ep2 in enumerate(self.episodes[i+1:], i+1):
            # テキストの重複部分を検出
            overlap = self.find_text_overlap(ep1.text, ep2.text)
            
            if overlap['ratio'] > 0.5:  # 50%以上重複
                partial_overlaps.append({
                    'indices': (i, j),
                    'overlap': overlap,
                    'unique_1': overlap['unique_to_1'],
                    'unique_2': overlap['unique_to_2']
                })
    
    return partial_overlaps
```

## リファクタリング戦略

### 1. Extract Method パターン

```python
def extract_common_knowledge(self, episode_group: List[int]) -> Episode:
    """共通知識を抽出して新エピソードを作成"""
    episodes = [self.episodes[i] for i in episode_group]
    
    # 共通部分を抽出
    common_tokens = set(episodes[0].tokens)
    for ep in episodes[1:]:
        common_tokens &= set(ep.tokens)
    
    if len(common_tokens) > 3:  # 意味のある共通部分
        # 共通知識エピソードを作成
        common_text = self.reconstruct_text(common_tokens)
        common_vector = self.average_vectors([ep.vector for ep in episodes])
        
        common_episode = Episode(
            text=f"[COMMON] {common_text}",
            vector=common_vector,
            c_value=max(ep.c_value for ep in episodes),
            derived_from=episode_group
        )
        
        return common_episode
```

### 2. Replace with Reference パターン

```python
def replace_with_reference(self, duplicate_indices: List[int], 
                          canonical_index: int):
    """重複を参照で置き換え"""
    canonical = self.episodes[canonical_index]
    
    for idx in duplicate_indices:
        if idx == canonical_index:
            continue
            
        # 軽量な参照エピソードに置き換え
        ref_episode = ReferenceEpisode(
            reference_to=canonical_index,
            additional_context=self.extract_unique_context(idx, canonical_index),
            c_value=self.episodes[idx].c_value
        )
        
        # メモリ使用量大幅削減
        self.episodes[idx] = ref_episode
```

### 3. Consolidate パターン

```python
def consolidate_knowledge(self, redundant_group: Dict) -> Episode:
    """冗長な知識を統合"""
    indices = redundant_group['indices']
    episodes = [self.episodes[i] for i in indices]
    
    # 最も情報量の多いエピソードをベースに
    base_idx = np.argmax(redundant_group['info_contents'])
    base_episode = episodes[base_idx]
    
    # 他のエピソードから固有情報を抽出して追加
    consolidated_text = base_episode.text
    for i, ep in enumerate(episodes):
        if i != base_idx:
            unique_info = self.extract_unique_info(ep, base_episode)
            if unique_info:
                consolidated_text += f" [+{unique_info}]"
    
    # 統合エピソードを作成
    consolidated = Episode(
        text=consolidated_text,
        vector=self.weighted_average_vectors(
            [ep.vector for ep in episodes],
            weights=redundant_group['info_contents']
        ),
        c_value=self.weighted_average(
            [ep.c_value for ep in episodes],
            weights=[ep.access_count for ep in episodes]
        )
    )
    
    return consolidated
```

## 評価指標

### 1. 冗長性スコア

```python
def calculate_redundancy_score(self) -> float:
    """グラフ全体の冗長性を評価"""
    total_info = 0
    unique_info = 0
    
    # 各エピソードの情報量を計算
    for i, episode in enumerate(self.episodes):
        info = self.calculate_information_content(i)
        total_info += info
        
        # 他のエピソードと重複しない情報量
        unique = self.calculate_unique_information(i)
        unique_info += unique
    
    redundancy = 1 - (unique_info / total_info)
    return redundancy
```

### 2. 圧縮可能性

```python
def estimate_compression_potential(self) -> Dict:
    """リファクタリングによる圧縮可能性を推定"""
    
    # 各種重複を検出
    exact_dups = self.detect_exact_duplicates()
    semantic_dups = self.detect_semantic_duplicates()
    partial_dups = self.detect_partial_redundancy()
    
    # 削減可能なメモリを推定
    reducible_memory = 0
    reducible_memory += len(exact_dups) * self.episode_size  # 完全削除
    reducible_memory += sum(
        len(g['indices']) - 1 for g in semantic_dups
    ) * self.episode_size * 0.8  # 80%削減
    reducible_memory += len(partial_dups) * self.episode_size * 0.3  # 30%削減
    
    current_memory = len(self.episodes) * self.episode_size
    
    return {
        'current_memory': current_memory,
        'reducible_memory': reducible_memory,
        'compression_ratio': reducible_memory / current_memory,
        'exact_duplicates': len(exact_dups),
        'semantic_groups': len(semantic_dups),
        'partial_overlaps': len(partial_dups)
    }
```

### 3. 知識の完全性チェック

```python
def verify_knowledge_preservation(self, before: List[Episode], 
                                after: List[Episode]) -> bool:
    """リファクタリング後も知識が保持されているか検証"""
    
    # 検証用クエリセット
    test_queries = self.generate_test_queries(before)
    
    # リファクタリング前後で検索結果を比較
    for query in test_queries:
        results_before = self.search(query, episodes=before)
        results_after = self.search(query, episodes=after)
        
        # 意味的に同等の結果が得られるか
        if not self.results_semantically_equivalent(
            results_before, results_after
        ):
            return False
    
    return True
```

## 実装例

```python
def auto_refactor(self, aggressive: bool = False):
    """自動リファクタリング実行"""
    
    # 1. 現状分析
    compression_potential = self.estimate_compression_potential()
    print(f"Compression potential: {compression_potential['compression_ratio']:.1%}")
    
    if compression_potential['compression_ratio'] < 0.1 and not aggressive:
        print("Already optimized. Skipping refactoring.")
        return
    
    # 2. バックアップ
    backup = self.create_backup()
    
    try:
        # 3. 完全重複を削除
        self.remove_exact_duplicates()
        
        # 4. 意味的重複を統合
        self.consolidate_semantic_duplicates()
        
        # 5. 部分重複をリファクタ
        self.refactor_partial_overlaps()
        
        # 6. 知識の完全性を検証
        if not self.verify_knowledge_preservation(backup, self.episodes):
            raise Exception("Knowledge loss detected!")
        
        # 7. グラフを再構築
        self.rebuild_graph()
        
        # 8. メトリクスを報告
        self.report_refactoring_results(backup)
        
    except Exception as e:
        print(f"Refactoring failed: {e}. Rolling back.")
        self.restore_from_backup(backup)
```

## まとめ

エピソードの重複削除は、コードリファクタリングの原則を適用：
1. **検出**: 完全・意味的・部分的重複を特定
2. **統合**: Extract/Replace/Consolidateパターンで整理
3. **検証**: 知識の完全性を保証
4. **最適化**: メモリ効率と検索性能の向上

これにより、知識グラフがクリーンで効率的に保たれる。