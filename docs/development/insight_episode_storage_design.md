# 仮説洞察エピソードの保存設計

## 現状の問題点

1. **InsightFactRegistry**は洞察を登録するが、エピソードとして保存されない
2. 洞察はSQLiteに保存されるが、ベクトル化されていない
3. 洞察と元エピソードの関係が追跡されていない

## 提案する実装

### 1. 洞察エピソードの作成タイミング

スパイク検出時に以下の処理を行う：

```python
def _process_spike_insight(self, response: str, graph_analysis: Dict, temp_episode_idx: int):
    """スパイク検出時に洞察をエピソードとして保存"""
    
    # 1. 洞察を抽出
    from ...detection.insight_registry import InsightFactRegistry
    registry = InsightFactRegistry()
    insights = registry.extract_and_evaluate_insights(
        response, 
        graph_analysis,
        self.query_analysis  # クエリとの関連付け
    )
    
    # 2. 高品質な洞察をエピソードとして追加
    for insight in insights:
        if insight.quality_score > 0.7:
            # 洞察エピソードのテキスト作成
            insight_text = f"[INSIGHT] {insight.text}"
            
            # ベクトル化（SentenceTransformerで埋め込み生成）
            insight_vector = self.l1_transform.embedder.embed_text(insight_text)
            
            # 洞察エピソードを追加（低いC値で開始 - 仮説段階）
            insight_episode_idx = self.l2_memory.add_episode(
                text=insight_text,
                vector=insight_vector,  # FAISSインデックスに追加される
                c_value=0.3,  # 仮説として低い信頼度から開始
                metadata={
                    "type": "insight",
                    "source_episode": temp_episode_idx,
                    "insight_id": insight.id,
                    "query": self.current_query,
                    "initial_quality": insight.quality_score
                }
            )
            
            # 3. エッジを作成
            if hasattr(self.l2_memory, 'graph_builder'):
                # 入力クエリとの強い相関
                self.l2_memory.graph_builder.add_edge(
                    temp_episode_idx, 
                    insight_episode_idx,
                    weight=0.9,  # 強い関連
                    edge_type="generates_insight"
                )
                
                # 元となったエピソードとの関連
                for source_idx in graph_analysis.get('key_episodes', []):
                    self.l2_memory.graph_builder.add_edge(
                        source_idx,
                        insight_episode_idx, 
                        weight=0.7,
                        edge_type="contributes_to_insight"
                    )
```

### 2. C値の段階的向上メカニズム

洞察エピソードのC値は以下の条件で上昇：

```python
def update_insight_confidence(self, insight_episode_idx: int):
    """洞察の信頼度を段階的に向上"""
    episode = self.l2_memory.episodes[insight_episode_idx]
    
    # 1. 使用頻度による向上
    if episode.retrieval_count > 5:
        episode.c_value = min(episode.c_value + 0.1, 0.8)
    
    # 2. 高信頼エピソードとの関連
    connected_episodes = self.graph_builder.get_connected_episodes(insight_episode_idx)
    high_confidence_connections = [
        e for e in connected_episodes 
        if self.l2_memory.episodes[e].c_value > 0.7
    ]
    if len(high_confidence_connections) > 3:
        episode.c_value = min(episode.c_value + 0.15, 0.9)
    
    # 3. 外部検証による向上（実験結果との一致など）
    if episode.metadata.get('externally_validated', False):
        episode.c_value = min(episode.c_value + 0.2, 0.95)
```

### 3. 洞察エピソードの並列扱い

洞察エピソードは通常のエピソードと同等に扱われる：
- 同じFAISSインデックスに格納
- 同じグラフ構造に参加
- C値による自然な優先順位付け
- 特別なクラスや処理は不要

```python
def retrieve_episodes(self, query: str, k: int = 5):
    """通常の検索 - 洞察も通常エピソードも平等に扱う"""
    
    # FAISSでベクトル検索
    query_vector = self.embedder.embed_text(query)
    distances, indices = self.index.search(query_vector, k)
    
    # C値でスコアを調整（洞察も通常エピソードも同じ）
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        episode = self.episodes[idx]
        # C値が高いほどスコアが高くなる
        adjusted_score = (1.0 - dist) * episode.c_value
        results.append({
            'index': idx,
            'score': adjusted_score,
            'text': episode.text,
            'c_value': episode.c_value
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)
```

### 4. 実装ステップ

1. **Phase 1**: スパイク検出時に洞察エピソード作成（低C値で開始）
2. **Phase 2**: SentenceTransformerでベクトル化してFAISSに追加
3. **Phase 3**: C値の段階的向上ロジック実装
4. **Phase 4**: 使用実績に基づくC値自動調整

## 期待される効果

1. **知識の再利用性向上**
   - 一度発見した洞察が将来のクエリで活用される
   - 洞察の連鎖による深い理解

2. **推論の効率化**
   - 類似クエリに対して即座に洞察を提供
   - 計算コストの削減

3. **知識グラフの質向上**
   - 洞察がハブノードとなり、知識を統合
   - より意味的に豊かなグラフ構造

4. **仮説から確信への進化**
   - 低C値から始まり、検証を経て高信頼度へ
   - 科学的方法論に沿った知識の成熟