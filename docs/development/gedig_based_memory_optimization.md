# geDIGベースのメモリ最適化設計

## 核心的アイデア

GED（グラフ編集距離）を使って知識グラフの「構造的冗長性」を検出し、共通部分を抽出しながら個別の「核」は保持する。

## 現在の問題

```
リンゴは赤い → [CONCEPT] 赤色は成熟のシグナル
```
↑ リンゴの存在が消えてしまう

## geDIGベースの解決策

### 1. 差分グラフ構造

```python
class DifferentialEpisode:
    """差分表現によるエピソード"""
    
    def __init__(self, core_content: str, parent_pattern: Optional[int] = None):
        self.core = core_content  # 固有の核（例：「リンゴ」）
        self.parent = parent_pattern  # 共通パターンへの参照
        self.delta = {}  # 親パターンからの差分
        
    def get_full_content(self) -> str:
        """完全な内容を再構築"""
        if self.parent:
            parent_content = self.memory.get_pattern(self.parent)
            return self.apply_delta(parent_content, self.core, self.delta)
        return self.core
```

### 2. geDIGによる共通構造検出

```python
def detect_common_structures(self, episode_cluster: List[Episode]):
    """GEDを使って共通構造を検出"""
    
    # 各エピソードをミニグラフとして表現
    graphs = [self.episode_to_graph(ep) for ep in episode_cluster]
    
    # ペアワイズGED計算
    ged_matrix = np.zeros((len(graphs), len(graphs)))
    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            if i < j:
                ged_matrix[i][j] = self.ged_calculator.compute(g1, g2)
                ged_matrix[j][i] = ged_matrix[i][j]
    
    # 低GEDのペアから共通構造を抽出
    common_structure = self.extract_maximum_common_subgraph(graphs, ged_matrix)
    
    return common_structure
```

### 3. 核保持型の知識圧縮

```python
def compress_with_core_preservation(self, episodes: List[Episode]):
    """核を保持しながら圧縮"""
    
    # Step 1: 共通パターンを検出
    common = detect_common_structures(episodes)
    
    # Step 2: 各エピソードを「核＋差分」に分解
    compressed_episodes = []
    for ep in episodes:
        # 核の抽出（固有名詞、特定概念）
        core = self.extract_core_concept(ep)  # "リンゴ", "イチゴ", etc.
        
        # 共通部分からの差分を計算
        delta = self.compute_delta(ep, common)
        
        # 差分エピソードとして保存
        diff_ep = DifferentialEpisode(
            core_content=core,
            parent_pattern=common.id,
            delta=delta
        )
        compressed_episodes.append(diff_ep)
    
    # Step 3: 共通パターンを上位概念として保存
    pattern_episode = Episode(
        text=f"[PATTERN] {common.description}",
        is_pattern=True,
        child_cores=[ep.core for ep in compressed_episodes]
    )
    
    return pattern_episode, compressed_episodes
```

### 4. 具体例での動作

#### 元のエピソード群
```
1. "リンゴは赤い果物である"
2. "イチゴは赤い果物である"  
3. "トマトは赤い野菜である"
4. "血液は赤い液体である"
```

#### geDIG分析後の構造

```python
# 共通パターン
pattern = {
    "id": "P001",
    "structure": "[SUBJECT]は赤い[CATEGORY]である",
    "common_property": "赤色"
}

# 差分エピソード（核を保持）
episodes = [
    {"core": "リンゴ", "category": "果物", "parent": "P001"},
    {"core": "イチゴ", "category": "果物", "parent": "P001"},
    {"core": "トマト", "category": "野菜", "parent": "P001"},
    {"core": "血液", "category": "液体", "parent": "P001"}
]
```

### 5. メモリ効率の計算

```python
def calculate_memory_efficiency(self):
    """GED最小化によるメモリ効率を計算"""
    
    # 圧縮前：各エピソードの完全グラフ
    original_size = sum(ep.graph.size() for ep in self.episodes)
    
    # 圧縮後：パターン＋差分
    pattern_size = self.pattern.graph.size()
    delta_sizes = sum(ep.delta_size() for ep in self.differential_episodes)
    compressed_size = pattern_size + delta_sizes
    
    # GED的観点：編集操作数の削減
    edit_reduction = 1 - (compressed_size / original_size)
    
    return {
        "memory_saved": edit_reduction,
        "cores_preserved": len(self.differential_episodes),
        "patterns_extracted": len(self.patterns)
    }
```

### 6. 動的な再構築とアクセス

```python
def access_episode(self, query: str):
    """クエリに応じて必要な知識を再構築"""
    
    # 関連する核を検索
    relevant_cores = self.search_cores(query)
    
    # 必要なパターンを特定
    required_patterns = set()
    for core in relevant_cores:
        if core.parent:
            required_patterns.add(core.parent)
    
    # 最小限の再構築
    reconstructed = []
    for pattern_id in required_patterns:
        pattern = self.get_pattern(pattern_id)
        # 関連する核だけを使って部分的に再構築
        partial_episodes = [
            core.reconstruct_with_pattern(pattern)
            for core in relevant_cores
            if core.parent == pattern_id
        ]
        reconstructed.extend(partial_episodes)
    
    return reconstructed
```

### 7. スパイク検出の活用

```python
def optimize_on_spike(self, spike_info: Dict):
    """スパイク検出時にメモリ構造を最適化"""
    
    if spike_info['delta_ged'] < -0.5:  # 構造的簡略化
        # グラフが簡略化された = より良い抽象化が見つかった
        affected_episodes = spike_info['affected_episodes']
        
        # 新しい共通構造でre-compress
        new_pattern = self.extract_pattern_from_spike(spike_info)
        self.recompress_episodes(affected_episodes, new_pattern)
        
        logger.info(f"Spike-driven optimization: {len(affected_episodes)} episodes recompressed")
```

## 実装の利点

1. **核の保持**: 「リンゴ」「イチゴ」などの固有概念は失われない
2. **構造的効率**: GEDに基づく最適な共通構造抽出
3. **動的最適化**: スパイク検出時に構造を改善
4. **最小再構築**: 必要な部分だけを動的に復元
5. **スケーラブル**: 階層的パターンで大規模化に対応

## メモリ使用量の例

- 元: 4エピソード × 50bytes = 200bytes
- 圧縮後: 
  - パターン: 30bytes
  - 核×4: 10bytes × 4 = 40bytes
  - 差分: 5bytes × 4 = 20bytes
  - 合計: 90bytes（55%削減）

## 結論

geDIGのグラフ編集距離を使うことで：
- 構造的な共通性を数学的に評価
- 最小限の編集で知識を表現
- 核となる概念は保持しながら冗長性を削減
- スパイク検出が最適化の機会を提供

これにより、「リンゴ」は忘れずに、「赤い」という共通性を効率的に管理できる。