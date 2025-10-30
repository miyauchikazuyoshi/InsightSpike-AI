---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# 洞察エピソードのメッセージパッシング実装計画

## 1. 概要

洞察エピソードの動的な変化と分裂を、多次元エッジを活用したメッセージパッシングで実現します。これにより、知識グラフが自律的に進化し、より深い洞察を生成できるようになります。

## 2. 洞察エピソードの特性

### 2.1 洞察エピソードの定義
```python
@dataclass
class InsightEpisode(Episode):
    """洞察を表すエピソード"""
    # 基本属性（Episodeから継承）
    # text: str
    # vec: np.ndarray
    # c: float
    
    # 洞察特有の属性
    insight_type: str  # "analogy", "contradiction", "synthesis", "abstraction"
    source_episodes: List[str]  # 洞察の元となったエピソードID
    confidence: float  # 洞察の確信度
    evolution_history: List[Dict[str, Any]]  # 変化の履歴
    
    # メッセージパッシング用
    accumulated_messages: np.ndarray  # 蓄積されたメッセージ
    message_count: int  # 受信したメッセージ数
    stability_score: float  # 安定性スコア（変化の少なさ）
    
    def is_ready_to_split(self) -> bool:
        """分裂の準備ができているか"""
        return (
            self.stability_score < 0.3 and  # 不安定
            self.message_count > 10 and      # 十分なメッセージ
            len(self.text) > 200            # 十分な内容
        )
```

### 2.2 洞察の種類と特性
```python
class InsightTypes:
    """洞察の種類と処理方法"""
    
    ANALOGY = "analogy"  # アナロジー：構造的類似性
    CONTRADICTION = "contradiction"  # 矛盾：意味的対立
    SYNTHESIS = "synthesis"  # 統合：複数概念の融合
    ABSTRACTION = "abstraction"  # 抽象化：一般化
    INSTANTIATION = "instantiation"  # 具体化：特殊化
    
    @staticmethod
    def get_processing_weights(insight_type: str) -> Dict[str, float]:
        """洞察タイプに応じたメッセージ処理の重み"""
        weights = {
            InsightTypes.ANALOGY: {
                "structural_similarity": 0.8,
                "semantic_similarity": 0.2,
                "temporal_distance": 0.0
            },
            InsightTypes.CONTRADICTION: {
                "contradiction_score": 0.7,
                "semantic_similarity": 0.3,
                "temporal_order": 0.0
            },
            InsightTypes.SYNTHESIS: {
                "semantic_similarity": 0.5,
                "entailment_score": 0.3,
                "temporal_distance": 0.2
            },
            InsightTypes.ABSTRACTION: {
                "abstraction_difference": 0.6,
                "semantic_similarity": 0.4,
                "structural_similarity": 0.0
            }
        }
        return weights.get(insight_type, {})
```

## 3. メッセージパッシングによる洞察の進化

### 3.1 基本的なメッセージパッシング
```python
class InsightMessagePassing:
    """洞察エピソードのメッセージパッシング"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.message_buffer = defaultdict(list)
        
    def propagate_messages(self, num_iterations: int = 5):
        """メッセージを伝播"""
        for iteration in range(num_iterations):
            # 各ノードからメッセージを送信
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                
                if isinstance(node_data.get('episode'), InsightEpisode):
                    self._send_messages_from_insight(node_id, node_data['episode'])
                else:
                    self._send_messages_from_regular(node_id, node_data['episode'])
            
            # メッセージを処理
            self._process_messages()
            
            # 洞察の状態を更新
            self._update_insights()
    
    def _send_messages_from_insight(self, node_id: str, insight: InsightEpisode):
        """洞察エピソードからメッセージを送信"""
        neighbors = list(self.graph.neighbors(node_id))
        
        # 洞察タイプに応じた重みを取得
        weights = InsightTypes.get_processing_weights(insight.insight_type)
        
        for neighbor_id in neighbors:
            edge_data = self.graph.edges[node_id, neighbor_id]
            
            # エッジの多次元情報を使用
            message_strength = 0.0
            for dim, weight in weights.items():
                if dim in edge_data:
                    message_strength += weight * edge_data[dim]
            
            if message_strength > 0.1:  # 閾値
                message = {
                    'from': node_id,
                    'to': neighbor_id,
                    'vector': insight.vec * message_strength,
                    'type': insight.insight_type,
                    'strength': message_strength,
                    'edge_data': edge_data
                }
                self.message_buffer[neighbor_id].append(message)
    
    def _process_messages(self):
        """受信したメッセージを処理"""
        for node_id, messages in self.message_buffer.items():
            if not messages:
                continue
                
            node_data = self.graph.nodes[node_id]
            episode = node_data['episode']
            
            if isinstance(episode, InsightEpisode):
                self._process_insight_messages(episode, messages)
            else:
                # 通常のエピソードが洞察に変化する可能性
                if self._should_become_insight(episode, messages):
                    new_insight = self._create_insight_from_episode(episode, messages)
                    node_data['episode'] = new_insight
        
        # バッファをクリア
        self.message_buffer.clear()
    
    def _process_insight_messages(self, insight: InsightEpisode, messages: List[Dict]):
        """洞察エピソードのメッセージ処理"""
        # メッセージを集約
        aggregated_vector = np.zeros_like(insight.vec)
        total_strength = 0.0
        
        for msg in messages:
            aggregated_vector += msg['vector']
            total_strength += msg['strength']
            
        if total_strength > 0:
            aggregated_vector /= total_strength
            
            # 洞察のベクトルを更新（部分的に）
            update_rate = 0.1  # 学習率
            old_vec = insight.vec.copy()
            insight.vec = (1 - update_rate) * insight.vec + update_rate * aggregated_vector
            insight.vec /= np.linalg.norm(insight.vec)
            
            # 安定性スコアを更新
            vec_change = np.linalg.norm(insight.vec - old_vec)
            insight.stability_score = 1.0 - vec_change
            
            # メッセージカウントを更新
            insight.message_count += len(messages)
            
            # 履歴に記録
            insight.evolution_history.append({
                'timestamp': time.time(),
                'vec_change': float(vec_change),
                'message_count': len(messages),
                'stability': insight.stability_score
            })
```

### 3.2 洞察の分裂メカニズム
```python
class InsightSplitter:
    """洞察エピソードの分裂処理"""
    
    def __init__(self, embedder: EmbeddingManager):
        self.embedder = embedder
        
    def check_and_split_insights(self, graph: nx.Graph) -> List[str]:
        """分裂が必要な洞察をチェックして分裂"""
        split_nodes = []
        
        for node_id, node_data in list(graph.nodes(data=True)):
            episode = node_data.get('episode')
            
            if isinstance(episode, InsightEpisode) and episode.is_ready_to_split():
                new_nodes = self._split_insight(node_id, episode, graph)
                split_nodes.extend(new_nodes)
                
        return split_nodes
    
    def _split_insight(
        self, 
        node_id: str, 
        insight: InsightEpisode, 
        graph: nx.Graph
    ) -> List[str]:
        """洞察を分裂"""
        # 分裂の理由を分析
        split_reason = self._analyze_split_reason(insight, graph)
        
        if split_reason['type'] == 'conceptual_divergence':
            # 概念的な分岐による分裂
            return self._split_by_concepts(node_id, insight, graph, split_reason)
        elif split_reason['type'] == 'contradiction':
            # 内部矛盾による分裂
            return self._split_by_contradiction(node_id, insight, graph, split_reason)
        elif split_reason['type'] == 'complexity':
            # 複雑性による分裂
            return self._split_by_complexity(node_id, insight, graph, split_reason)
        else:
            return []
    
    def _split_by_concepts(
        self,
        node_id: str,
        insight: InsightEpisode,
        graph: nx.Graph,
        split_reason: Dict
    ) -> List[str]:
        """概念的分岐による分裂"""
        # メッセージ履歴から主要な概念を抽出
        concepts = split_reason['concepts']  # 例: ['量子', '古典']
        
        new_insights = []
        neighbors = list(graph.neighbors(node_id))
        
        for i, concept in enumerate(concepts):
            # 概念に関連する部分を抽出
            concept_text = self._extract_concept_text(insight.text, concept)
            
            # 新しいベクトルを生成（メッセージパッシング的に）
            concept_vec = self._generate_concept_vector(
                insight, concept, neighbors, graph
            )
            
            # 新しい洞察エピソードを作成
            new_insight = InsightEpisode(
                text=concept_text,
                vec=concept_vec,
                c=insight.c * 0.8,  # C値は少し減衰
                insight_type=insight.insight_type,
                source_episodes=insight.source_episodes + [node_id],
                confidence=insight.confidence * 0.9,
                evolution_history=[{
                    'event': 'split_from_parent',
                    'parent_id': node_id,
                    'reason': 'conceptual_divergence',
                    'concept': concept,
                    'timestamp': time.time()
                }],
                accumulated_messages=np.zeros_like(concept_vec),
                message_count=0,
                stability_score=0.5
            )
            
            new_insights.append(new_insight)
        
        # グラフを更新
        return self._update_graph_after_split(
            graph, node_id, new_insights, neighbors
        )
    
    def _generate_concept_vector(
        self,
        parent_insight: InsightEpisode,
        concept: str,
        neighbors: List[str],
        graph: nx.Graph
    ) -> np.ndarray:
        """概念特化のベクトルを生成（メッセージパッシング風）"""
        # 基本ベクトル（概念のエンベディング）
        concept_vec = self.embedder.encode(concept)
        
        # 親の影響（50%）
        parent_influence = parent_insight.vec * 0.5
        
        # 近傍からの影響（30%）
        neighbor_influence = np.zeros_like(concept_vec)
        total_weight = 0.0
        
        for neighbor_id in neighbors:
            neighbor_data = graph.nodes[neighbor_id]
            edge_data = graph.edges[parent_insight.id if hasattr(parent_insight, 'id') else '', neighbor_id]
            
            # エッジの多次元情報を考慮
            relevance = self._calculate_concept_relevance(
                concept, 
                neighbor_data.get('text', ''),
                edge_data
            )
            
            if relevance > 0.1:
                neighbor_vec = neighbor_data.get('vec', np.zeros_like(concept_vec))
                neighbor_influence += neighbor_vec * relevance
                total_weight += relevance
        
        if total_weight > 0:
            neighbor_influence /= total_weight
            neighbor_influence *= 0.3
        
        # 概念ベクトル（20%）
        concept_influence = concept_vec * 0.2
        
        # 統合
        final_vec = parent_influence + neighbor_influence + concept_influence
        final_vec /= np.linalg.norm(final_vec)
        
        return final_vec
```

### 3.3 分裂エピソードのグラフ統合
```python
def _update_graph_after_split(
    self,
    graph: nx.Graph,
    parent_id: str,
    new_insights: List[InsightEpisode],
    original_neighbors: List[str]
) -> List[str]:
    """分裂後のグラフ更新"""
    new_node_ids = []
    
    # 親ノードのデータを保存
    parent_data = graph.nodes[parent_id].copy()
    parent_edges = list(graph.edges(parent_id, data=True))
    
    # 親ノードを削除
    graph.remove_node(parent_id)
    
    # 新しいノードを追加
    for i, new_insight in enumerate(new_insights):
        new_id = f"{parent_id}_split_{i}"
        new_insight.id = new_id
        
        graph.add_node(
            new_id,
            episode=new_insight,
            vec=new_insight.vec,
            text=new_insight.text,
            c_value=new_insight.c
        )
        new_node_ids.append(new_id)
    
    # エッジの再配置（多次元情報を考慮）
    edge_calculator = EdgeDimensionCalculator(self.embedder)
    
    for new_id, new_insight in zip(new_node_ids, new_insights):
        # 元の近傍との新しいエッジを計算
        for _, neighbor_id, old_edge_data in parent_edges:
            if neighbor_id not in graph:
                continue
                
            neighbor_episode = graph.nodes[neighbor_id].get('episode')
            if neighbor_episode:
                # 新しいエッジの次元を計算
                new_edge_dims = edge_calculator.calculate_all_dimensions(
                    new_insight, neighbor_episode
                )
                
                # 親エッジの情報も部分的に継承
                inherited_dims = self._inherit_edge_dimensions(
                    old_edge_data, new_edge_dims, new_insight
                )
                
                # 総合スコアが閾値を超えたら追加
                if self._should_create_edge(inherited_dims):
                    graph.add_edge(
                        new_id,
                        neighbor_id,
                        **inherited_dims.__dict__
                    )
    
    # 分裂した洞察同士の関係も追加
    for i, id1 in enumerate(new_node_ids):
        for id2 in new_node_ids[i+1:]:
            sibling_dims = EdgeDimensions(
                semantic_similarity=0.6,  # 同じ親から生まれたので中程度の類似
                structural_similarity=0.7,  # 構造は似ている
                temporal_distance=0.0,  # 同時に生成
                temporal_order=0,
                edge_type="sibling",
                confidence=0.8,
                last_updated=time.time()
            )
            
            graph.add_edge(id1, id2, **sibling_dims.__dict__)
    
    return new_node_ids
```

## 4. 実装スケジュール

### Phase 1: 基礎実装（2週間）

#### Week 1: 洞察エピソードクラス
- InsightEpisodeクラスの実装
- 基本的なメッセージパッシング機構
- 安定性スコアの計算

#### Week 2: 分裂判定ロジック
- 分裂条件の実装
- 概念抽出アルゴリズム
- ベクトル生成（メッセージパッシング風）

### Phase 2: 高度な分裂メカニズム（2週間）

#### Week 3: 分裂タイプの実装
- 概念的分岐による分裂
- 矛盾による分裂
- 複雑性による分裂

#### Week 4: グラフ統合
- エッジの再配置
- 多次元情報の継承
- 兄弟関係の管理

### Phase 3: 最適化と学習（1週間）

#### Week 5: パフォーマンス最適化
- バッチ処理
- 非同期メッセージパッシング
- メモリ効率化

### Phase 4: 統合とテスト（1週間）

#### Week 6: システム統合
- 既存システムとの統合
- エンドツーエンドテスト
- ドキュメント作成

## 5. 期待される効果

### 5.1 自律的な知識の進化
- 洞察が自然に分化・特化
- 矛盾の自動解消
- 概念の精緻化

### 5.2 より深い推論
- 多段階の洞察生成
- 複雑な関係性の発見
- 創発的な知識の出現

### 5.3 適応的な学習
- 使用パターンに応じた進化
- ドメイン特化の自動化
- 知識の自己組織化

## 6. 実装例

### 6.1 使用例
```python
# 初期設定
graph = nx.Graph()
embedder = EmbeddingManager()
message_passer = InsightMessagePassing(graph)
splitter = InsightSplitter(embedder)

# 洞察エピソードの追加
insight = InsightEpisode(
    text="量子コンピュータは古典コンピュータとは根本的に異なる計算原理を持つ",
    vec=embedder.encode("量子計算と古典計算の違い"),
    c=0.9,
    insight_type=InsightTypes.CONTRADICTION,
    source_episodes=["ep1", "ep2"],
    confidence=0.8,
    evolution_history=[],
    accumulated_messages=np.zeros(384),
    message_count=0,
    stability_score=1.0
)

graph.add_node("insight_001", episode=insight)

# メッセージパッシングで進化
for iteration in range(10):
    message_passer.propagate_messages(num_iterations=1)
    
    # 分裂チェック
    split_nodes = splitter.check_and_split_insights(graph)
    
    if split_nodes:
        print(f"Iteration {iteration}: Split into {len(split_nodes)} new insights")
        
# 結果の確認
for node_id, data in graph.nodes(data=True):
    if isinstance(data.get('episode'), InsightEpisode):
        insight = data['episode']
        print(f"{node_id}: {insight.text[:50]}... (stability: {insight.stability_score:.2f})")
```

### 6.2 可視化
```python
def visualize_insight_evolution(graph: nx.Graph):
    """洞察の進化を可視化"""
    import matplotlib.pyplot as plt
    
    # 洞察ノードを特定
    insight_nodes = [
        node for node, data in graph.nodes(data=True)
        if isinstance(data.get('episode'), InsightEpisode)
    ]
    
    # 色分け
    node_colors = []
    for node in graph.nodes():
        if node in insight_nodes:
            insight = graph.nodes[node]['episode']
            if insight.insight_type == InsightTypes.ANALOGY:
                node_colors.append('lightblue')
            elif insight.insight_type == InsightTypes.CONTRADICTION:
                node_colors.append('lightcoral')
            elif insight.insight_type == InsightTypes.SYNTHESIS:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightyellow')
        else:
            node_colors.append('lightgray')
    
    # 描画
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=node_colors, with_labels=True)
    plt.title("Insight Evolution Network")
    plt.show()
```

## 7. まとめ

この実装により、洞察エピソードが：
1. **メッセージパッシングで動的に進化**
2. **適切なタイミングで自律的に分裂**
3. **多次元エッジ情報を活用して最適な構造を形成**

することが可能になり、知識グラフが生きた有機体のように成長・進化するシステムが実現されます。