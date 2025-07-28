# Episode Branching Architecture (エピソード分岐アーキテクチャ)

## 概要

2025年7月26日の議論で、エピソード分裂の新しいアーキテクチャが提案された。従来の「1→2（元を削除）」から「1→1+N（元を保持）」への転換。

## 背景：なぜ分岐が必要か

### 問題例：「apple」の多義性

```
「apple」に関連するエピソード：
- 「appleは赤い」
- 「appleは甘酸っぱい」  
- 「appleはスティーブ・ジョブスが作った」
- 「appleはコンピューター企業だ」
```

現在の実装では、テキストを物理的に分割しようとするが、これは無意味。「apple」という単語を分割できない。

## 新アーキテクチャ：概念の分岐

### 基本原理

1. **元のエピソードは保持**
   - 曖昧な状態も知識の一部
   - 文脈によって適切な解釈を選択可能

2. **文脈ごとに分岐エピソードを生成**
   - 同じテキスト、異なるベクトル
   - 中間ノードとして機能

3. **グラフの矛盾を解消**
   ```
   従来：親が矛盾するエッジを持つ
   [apple] ──── [赤い]
      └──── [IT企業]  → 矛盾！
   
   新案：中間ノードが文脈を分離  
   [apple(親)]
      ├─[apple(果物)] ──── [赤い]
      └─[apple(IT)] ──── [IT企業]  → 矛盾なし！
   ```

## 実装設計

### 1. 分岐検出

```python
def detect_context_divergence(episode: Episode, graph: nx.Graph) -> List[ContextCluster]:
    """
    隣接ノードの文脈クラスタを検出
    """
    neighbors = list(graph.neighbors(episode.id))
    
    # 隣接ノードを文脈でクラスタリング
    clusters = []
    
    # 例：
    # cluster1: ["赤い", "甘酸っぱい"] → 果物文脈
    # cluster2: ["ジョブス", "IT企業"] → テクノロジー文脈
    
    return clusters
```

### 2. 分岐エピソード生成

```python
def create_branch_episodes(
    parent: Episode,
    context_clusters: List[ContextCluster],
    graph: nx.Graph
) -> List[Episode]:
    """
    文脈ごとに分岐エピソードを生成（親は保持）
    """
    branches = [parent]  # 親を含める
    
    for cluster in context_clusters:
        # メッセージパッシングで文脈特化ベクトル生成
        messages = [
            (parent.vec, 0.4),  # 親から基本的意味
        ]
        
        # クラスタ内の隣接ノードから文脈情報
        for neighbor_id in cluster.neighbor_ids:
            neighbor_vec = graph.nodes[neighbor_id]['vec']
            weight = 0.6 / len(cluster.neighbor_ids)
            messages.append((neighbor_vec, weight))
        
        # ベクトル集約
        branch_vec = aggregate_messages(messages)
        
        # 分岐エピソード作成
        branch = Episode(
            text=parent.text,  # テキストは同じ！
            vec=branch_vec,    # 文脈特化ベクトル
            c=parent.c * 0.8,  # やや減衰
            metadata={
                'parent_id': parent.id,
                'context_type': cluster.type,
                'context_neighbors': cluster.neighbor_ids,
                'is_branch': True
            }
        )
        branches.append(branch)
    
    return branches
```

### 3. グラフ更新

```python
def update_graph_with_branches(
    graph: nx.Graph,
    parent: Episode,
    branches: List[Episode]
) -> None:
    """
    分岐エピソードをグラフに統合
    """
    for branch in branches[1:]:  # 親以外
        # 分岐ノードを追加
        graph.add_node(
            branch.id,
            text=branch.text,
            vec=branch.vec,
            metadata=branch.metadata
        )
        
        # 親から分岐へのエッジ
        graph.add_edge(
            parent.id,
            branch.id,
            weight=0.9,
            relation='branch'
        )
        
        # 文脈に応じて既存エッジを移動
        context_neighbors = branch.metadata['context_neighbors']
        for neighbor in context_neighbors:
            if graph.has_edge(parent.id, neighbor):
                # 親からのエッジを分岐に移動
                edge_data = graph.edges[parent.id, neighbor]
                graph.remove_edge(parent.id, neighbor)
                graph.add_edge(branch.id, neighbor, **edge_data)
```

## 利点

### 1. 多義性の保持
- 「apple」は果物でもIT企業でもある
- 文脈によって適切な解釈を選択

### 2. 検索の階層性
```python
# 汎用検索：親エピソードがヒット
results = search("apple")  # すべてのapple

# 文脈特化検索：分岐エピソードがヒット  
results = search("apple", context="technology")  # IT関連のみ
```

### 3. 知識の成長過程
```
t1: [apple] → 初期の曖昧な理解
t2: [apple] → [apple(果物)] → 文脈が分化
t3: [apple] → [apple(果物)]
           └→ [apple(IT)] → さらに分化
```

### 4. 人間の概念形成との一致
- 最初は曖昧な理解から始まる
- 経験により文脈ごとに特殊化
- 元の曖昧な概念も保持

## 実装優先順位

1. **Phase 1**: 基本的な分岐メカニズム
   - 文脈クラスタリング
   - メッセージパッシングによるベクトル生成
   - グラフ構造の更新

2. **Phase 2**: 検索システムの対応
   - 親・分岐の階層を考慮した検索
   - 文脈ヒントによる絞り込み

3. **Phase 3**: 洞察検出との統合
   - 分岐時のスパイク検出
   - 新しい文脈の発見を洞察として記録

## 関連ドキュメント

- `/docs/research/multi_dimensional_edge_research_2025_01.md` - 多次元エッジの研究
- `/src/insightspike/episodic/hybrid_episode_splitter.py` - 現在の分割実装
- `/src/insightspike/implementations/layers/cached_memory_manager.py` - メモリ管理層

## 実装状況（2025-07-26）

### 完了した実装

1. **CachedMemoryManager.branch_episode()**
   - `split_episode()` を `branch_episode()` にリダイレクト
   - 1→1+N パターンの実装完了
   - メッセージパッシングによる文脈特化ベクトル生成

2. **文脈クラスタリング**
   - コサイン類似度ベースのクラスタリング実装
   - 閾値0.5で隣接ノードをグループ化
   - 果物/技術などの文脈タイプ自動判定

3. **テスト実装**
   - `/test_episode_branching.py` で動作確認
   - "apple"の多義性を正しく分岐

### 実行結果例

```
Original: apple (C=0.8)
├─ apple_branch_fruit_0 (C=0.64) - 果物文脈
├─ apple_branch_context_1_1 (C=0.64) - 一般文脈  
└─ apple_branch_technology_2 (C=0.64) - 技術文脈

Vector similarities:
- apple vs fruit branch: 0.689
- apple vs tech branch: 0.770
- fruit vs tech branch: 0.568 (低い = 異なる文脈)
```

## 次のステップ

1. ~~このアーキテクチャに基づいて実装~~ ✅ 完了
2. ~~実装完了後、アーキテクチャドキュメントを更新~~ ✅ 完了
3. 数学概念実験で効果を検証（次のタスク）
4. クラスタリングアルゴリズムの改善（類似文脈の統合）