# エピソード分割のベクトル生成：2つのアプローチ

## アプローチ1: メッセージパッシングによるベクトル形成

### 基本アイデア
分裂前のベクトルを「親」として重視しつつ、グラフ上の近傍ノードからの影響を類似度に基づいて取り込む。

```python
def split_with_message_passing(
    episode: Episode,
    graph: nx.Graph,
    split_texts: List[str]
) -> List[Episode]:
    """
    メッセージパッシングで分割後のベクトルを生成
    
    親ベクトル（分裂前）の影響を最も強く受けつつ、
    グラフ上の近傍ノードからも情報を集約
    """
    new_episodes = []
    parent_vec = episode.vec
    
    # 近傍ノードとその類似度を取得
    neighbors = []
    for node_id in graph.neighbors(episode.id):
        neighbor_vec = graph.nodes[node_id]['vec']
        similarity = cosine_similarity(parent_vec, neighbor_vec)
        neighbors.append((node_id, neighbor_vec, similarity))
    
    # 各分割テキストに対してベクトル生成
    for i, text in enumerate(split_texts):
        # 1. ベースとなる新規エンベディング
        base_vec = embedder.encode(text)
        
        # 2. 親ベクトルの影響（最大重み）
        parent_weight = 0.5  # 親の影響を50%確保
        
        # 3. 近傍からのメッセージ集約
        neighbor_messages = []
        remaining_weight = 1.0 - parent_weight
        
        # 類似度でソートして上位k個を使用
        top_neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)[:5]
        
        # 類似度を正規化して重みとする
        total_similarity = sum(sim for _, _, sim in top_neighbors)
        
        for node_id, neighbor_vec, similarity in top_neighbors:
            # テキストの関連性も考慮
            text_relevance = calculate_text_relevance(
                text, 
                graph.nodes[node_id].get('text', '')
            )
            
            # 総合的な重み = グラフ類似度 × テキスト関連性
            weight = (similarity / total_similarity) * text_relevance * remaining_weight
            neighbor_messages.append(neighbor_vec * weight)
        
        # 4. 最終的なベクトル = 親 + ベース + 近傍メッセージ
        final_vec = (
            parent_vec * parent_weight +
            base_vec * 0.3 +
            sum(neighbor_messages)
        )
        
        # 正規化
        final_vec = final_vec / np.linalg.norm(final_vec)
        
        # 5. C値の計算（親からの継承 + 独自性）
        position_factor = 1.0 - abs(i - len(split_texts)/2) / len(split_texts)
        new_c = episode.c * (0.7 + 0.3 * position_factor)
        
        new_episodes.append(Episode(
            text=text,
            vec=final_vec,
            c=new_c,
            metadata={
                'parent_id': episode.id,
                'split_index': i,
                'split_method': 'message_passing'
            }
        ))
    
    return new_episodes
```

### 利点
- グラフ構造を活用した文脈保持
- 親子関係が明確
- 近傍の知識を自然に取り込める

### 欠点
- 計算コストがやや高い
- パラメータ調整が必要

## アプローチ2: LLMによる意味的分割

### 基本アイデア
LLMに分割後の各セグメントの「意味的な要約ベクトル」を生成してもらう。

```python
def split_with_llm_guidance(
    episode: Episode,
    split_texts: List[str],
    llm_provider: LLMProvider
) -> List[Episode]:
    """
    LLMに各分割セグメントの意味的特徴を抽出してもらい、
    それを基にベクトルを生成
    """
    new_episodes = []
    
    # LLMへのプロンプト構築
    prompt = f"""
    以下のテキストを{len(split_texts)}個のセグメントに分割しました。
    各セグメントの主要な概念と、他のセグメントとの関係性を分析してください。
    
    元のテキスト: {episode.text[:500]}...
    
    分割されたセグメント:
    """
    
    for i, text in enumerate(split_texts):
        prompt += f"\nセグメント{i+1}: {text[:200]}..."
    
    prompt += """
    
    各セグメントについて以下を出力してください：
    1. 主要な概念（3-5個のキーワード）
    2. 他のセグメントとの関係性（0-1のスコア）
    3. 独立性の度合い（0-1のスコア）
    4. 元のテキスト全体における重要度（0-1のスコア）
    
    JSON形式で出力してください。
    """
    
    # LLMの応答を解析
    llm_response = llm_provider.generate(prompt)
    segment_analysis = parse_json_response(llm_response)
    
    # 各セグメントのベクトル生成
    for i, (text, analysis) in enumerate(zip(split_texts, segment_analysis)):
        # 1. 基本エンベディング
        base_vec = embedder.encode(text)
        
        # 2. キーワードベースの強調ベクトル
        keywords = analysis['keywords']
        keyword_vecs = [embedder.encode(kw) for kw in keywords]
        keyword_vec = np.mean(keyword_vecs, axis=0) if keyword_vecs else base_vec
        
        # 3. 関係性に基づく親ベクトルの影響
        parent_influence = 1.0 - analysis['independence_score']
        
        # 4. 最終ベクトル計算
        final_vec = (
            base_vec * 0.5 +
            keyword_vec * 0.3 +
            episode.vec * parent_influence * 0.2
        )
        final_vec = final_vec / np.linalg.norm(final_vec)
        
        # 5. LLMが判断した重要度をC値に反映
        new_c = episode.c * analysis['importance_score']
        
        new_episodes.append(Episode(
            text=text,
            vec=final_vec,
            c=new_c,
            metadata={
                'parent_id': episode.id,
                'split_index': i,
                'split_method': 'llm_guided',
                'keywords': keywords,
                'relationships': analysis['relationships']
            }
        ))
    
    return new_episodes
```

### 利点
- 意味的に正確な分割
- 文脈理解が深い
- 重要度判定が高精度

### 欠点
- LLMコールのコスト
- レイテンシが高い
- 決定論的でない

## ハイブリッドアプローチ（推奨）

```python
class HybridEpisodeSplitter:
    """状況に応じて最適な分割方法を選択"""
    
    def split_episode(
        self,
        episode: Episode,
        graph: nx.Graph,
        context: Dict[str, Any]
    ) -> List[Episode]:
        
        # 分割判断
        should_split, split_points = self._analyze_split_necessity(episode, graph)
        
        if not should_split:
            return [episode]
        
        # テキストを分割
        split_texts = self._split_at_boundaries(episode.text, split_points)
        
        # 分割方法の選択
        if len(split_texts) > 5 or episode.c > 0.8:
            # 重要なエピソードや多数分割の場合はLLM使用
            return self._split_with_llm(episode, split_texts, graph)
        
        elif self._has_dense_connections(episode, graph):
            # グラフが密な場合はメッセージパッシング
            return self._split_with_message_passing(episode, split_texts, graph)
        
        else:
            # それ以外は高速な簡易分割
            return self._split_simple(episode, split_texts)
    
    def _analyze_split_necessity(
        self, 
        episode: Episode, 
        graph: nx.Graph
    ) -> Tuple[bool, List[int]]:
        """分割の必要性と分割点を判定"""
        
        # 1. グラフ構造の変化（ΔGED）をチェック
        local_complexity = self._calculate_local_complexity(episode, graph)
        
        # 2. テキストの意味的境界を検出
        semantic_boundaries = self._detect_semantic_boundaries(episode.text)
        
        # 3. 情報密度の変化点を検出
        density_changes = self._detect_information_density_changes(episode.text)
        
        # 総合的に判断
        if local_complexity > 0.7 and len(semantic_boundaries) > 0:
            return True, semantic_boundaries
        
        return False, []
```

## 実装の優先順位

1. **フェーズ1**: メッセージパッシング実装
   - 既存のグラフ構造を最大限活用
   - 計算効率が良い
   - 決定論的

2. **フェーズ2**: LLM統合
   - 高精度が必要な場合のオプション
   - コスト/レイテンシを考慮した選択的使用

3. **フェーズ3**: ハイブリッド最適化
   - 両方の利点を活かす
   - 文脈に応じた自動選択