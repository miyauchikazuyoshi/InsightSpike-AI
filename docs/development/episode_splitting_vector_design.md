---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# エピソード分割時のベクトル表現設計

## 現状分析

### 問題点
現在のCachedMemoryManagerには分割ロジックが実装されていません。L2MemoryManagerの文字数ベースの分割から、グラフベースの分割への移行が必要です。

### 分割エピソードのベクトル生成方法案

## 1. 単純な再エンベディング方式
```python
def split_episode_simple(episode: Episode, split_points: List[int]) -> List[Episode]:
    """分割されたテキストごとに新しいベクトルを生成"""
    segments = split_text_at_points(episode.text, split_points)
    new_episodes = []
    
    for segment in segments:
        # 新しいテキストセグメントのベクトルを生成
        new_vec = embedder.encode(segment)
        new_episodes.append(Episode(
            text=segment,
            vec=new_vec,
            c=episode.c * 0.8  # 分割によるC値の減衰
        ))
    
    return new_episodes
```

## 2. メッセージパッシング方式（提案）
```python
def split_episode_with_message_passing(
    episode: Episode, 
    graph: nx.Graph,
    split_points: List[int]
) -> List[Episode]:
    """グラフ上のメッセージパッシングで分割後のベクトルを調整"""
    
    # 1. 初期分割
    segments = split_text_at_points(episode.text, split_points)
    temp_vecs = [embedder.encode(seg) for seg in segments]
    
    # 2. 元のエピソードの近傍ノードを取得
    neighbors = graph.neighbors(episode.id)
    neighbor_vecs = [graph.nodes[n]['vec'] for n in neighbors]
    
    # 3. メッセージパッシングで各セグメントのベクトルを調整
    refined_vecs = []
    for i, temp_vec in enumerate(temp_vecs):
        # セグメントの位置に応じた重み付け
        position_weight = i / len(temp_vecs)
        
        # 近傍からのメッセージ集約
        messages = []
        for n_vec in neighbor_vecs:
            similarity = cosine_similarity(temp_vec, n_vec)
            if similarity > 0.5:  # 関連性のある近傍のみ
                messages.append(n_vec * similarity)
        
        # メッセージの集約
        if messages:
            aggregated_message = np.mean(messages, axis=0)
            # 元のベクトルとメッセージの重み付き結合
            refined_vec = (0.7 * temp_vec + 0.3 * aggregated_message)
            refined_vec = refined_vec / np.linalg.norm(refined_vec)
        else:
            refined_vec = temp_vec
            
        refined_vecs.append(refined_vec)
    
    # 4. 新しいエピソードを生成
    new_episodes = []
    for seg, vec in zip(segments, refined_vecs):
        new_episodes.append(Episode(
            text=seg,
            vec=vec,
            c=episode.c * (0.8 + 0.2 * len(messages)/len(neighbors))
        ))
    
    return new_episodes
```

## 3. グラフ構造保存方式
```python
def split_episode_preserving_structure(
    episode: Episode,
    graph: nx.Graph,
    split_info: Dict[str, Any]
) -> Tuple[List[Episode], List[Edge]]:
    """分割後もグラフ構造を保存"""
    
    # 1. セマンティックな境界で分割
    boundaries = find_semantic_boundaries(episode.text)
    segments = split_at_boundaries(episode.text, boundaries)
    
    # 2. 各セグメントの重要度を計算
    segment_importances = []
    for seg in segments:
        # TF-IDFやキーワード密度などで重要度計算
        importance = calculate_importance(seg, graph.nodes)
        segment_importances.append(importance)
    
    # 3. ベクトル生成と調整
    new_episodes = []
    for seg, imp in zip(segments, segment_importances):
        base_vec = embedder.encode(seg)
        
        # 重要度に基づいて元のベクトルからの継承度を調整
        inheritance_ratio = imp / sum(segment_importances)
        adjusted_vec = (
            inheritance_ratio * episode.vec + 
            (1 - inheritance_ratio) * base_vec
        )
        adjusted_vec = adjusted_vec / np.linalg.norm(adjusted_vec)
        
        new_episodes.append(Episode(
            text=seg,
            vec=adjusted_vec,
            c=episode.c * imp
        ))
    
    # 4. エッジの再配置
    new_edges = []
    old_edges = list(graph.edges(episode.id, data=True))
    
    for new_ep in new_episodes:
        for _, target, data in old_edges:
            # セグメントと隣接ノードの関連性を再評価
            relevance = calculate_relevance(new_ep.text, graph.nodes[target]['text'])
            if relevance > 0.3:
                new_edges.append({
                    'source': new_ep.id,
                    'target': target,
                    'weight': data['weight'] * relevance
                })
    
    return new_episodes, new_edges
```

## 推奨実装: ハイブリッドアプローチ

```python
class GraphAwareEpisodeSplitter:
    """グラフ構造を考慮したエピソード分割"""
    
    def split_episode(
        self,
        episode: Episode,
        graph: Optional[nx.Graph] = None,
        method: str = 'hybrid'
    ) -> List[Episode]:
        """
        エピソードを分割し、適切なベクトル表現を生成
        
        Args:
            episode: 分割対象のエピソード
            graph: 現在のナレッジグラフ
            method: 'simple', 'message_passing', 'structure_preserving', 'hybrid'
        """
        
        # 1. セマンティックな分割点を検出
        split_points = self._detect_split_points(episode.text)
        
        if not split_points:
            return [episode]  # 分割不要
        
        # 2. 分割方法に応じた処理
        if method == 'simple' or graph is None:
            return self._split_simple(episode, split_points)
        
        elif method == 'message_passing':
            return self._split_with_message_passing(episode, graph, split_points)
        
        elif method == 'structure_preserving':
            return self._split_preserving_structure(episode, graph, split_points)
        
        else:  # hybrid
            # グラフの密度に応じて方法を選択
            if self._is_dense_region(episode, graph):
                # 密な領域ではメッセージパッシング
                return self._split_with_message_passing(episode, graph, split_points)
            else:
                # 疎な領域では構造保存
                return self._split_preserving_structure(episode, graph, split_points)
    
    def _detect_split_points(self, text: str) -> List[int]:
        """セマンティックな分割点を検出"""
        # 実装: 段落、文章構造、トピック変化などを検出
        pass
    
    def _is_dense_region(self, episode: Episode, graph: nx.Graph) -> bool:
        """エピソード周辺のグラフ密度を評価"""
        neighbors = list(graph.neighbors(episode.id))
        if len(neighbors) < 3:
            return False
        
        # 近傍のエッジ密度を計算
        subgraph = graph.subgraph(neighbors + [episode.id])
        density = nx.density(subgraph)
        return density > 0.5
```

## まとめ

エピソード分割時のベクトル生成は以下の要因を考慮すべき：

1. **セマンティックな一貫性**: 分割後も意味的な関連性を保持
2. **グラフ構造の保存**: 既存の関係性を適切に継承
3. **情報の局所性**: 各セグメントの独自性も反映
4. **計算効率**: リアルタイムでの分割にも対応

メッセージパッシングアプローチは、グラフ構造を活用して分割後のベクトルを最適化し、知識の関係性を保持しながら効果的な分割を実現します。