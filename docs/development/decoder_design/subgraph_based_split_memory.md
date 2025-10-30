---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Subgraph-Based Split Memory Management

## なぜサブグラフ管理が優れているか

### タグベース管理の限界

1. **フラットな構造**：概念間の関係性が失われる
2. **スケーラビリティ**：タグが増えると管理が困難
3. **推論の制限**：グラフ構造を活用した推論ができない

### サブグラフ管理の利点

1. **構造保持**：分裂した概念も元の関係性を維持
2. **自然な階層**：基礎→高度の進化を構造的に表現
3. **グラフ推論**：GNNやメッセージパッシングが可能
4. **動的再構成**：クエリに応じてサブグラフを組み換え

## 提案アーキテクチャ

### 1. 概念サブグラフ構造

```python
class ConceptSubgraph:
    """単一概念の進化を表すサブグラフ"""
    def __init__(self, concept_id: str):
        self.concept_id = concept_id
        self.evolution_graph = nx.DiGraph()
        self.root_node = None
        
    def add_evolution(self, from_level: str, to_level: str, trigger: str):
        """概念の進化（分裂）を記録"""
        self.evolution_graph.add_edge(
            from_level, 
            to_level, 
            trigger=trigger,
            timestamp=time.now()
        )
```

### 2. 階層的サブグラフ管理

```python
class HierarchicalMemoryGraph:
    """全体のメモリをサブグラフの集合として管理"""
    def __init__(self):
        self.master_graph = nx.DiGraph()
        self.concept_subgraphs = {}  # concept_id -> ConceptSubgraph
        self.cross_concept_edges = []  # サブグラフ間の接続
        
    def register_split(self, episode, split_info):
        """分裂をサブグラフとして登録"""
        concept_id = split_info['concept_id']
        
        if concept_id not in self.concept_subgraphs:
            self.concept_subgraphs[concept_id] = ConceptSubgraph(concept_id)
            
        subgraph = self.concept_subgraphs[concept_id]
        subgraph.add_evolution(
            split_info['from_level'],
            split_info['to_level'],
            split_info['trigger']
        )
```

### 3. クエリ対応サブグラフ抽出

```python
class SubgraphSelector:
    """クエリに最適なサブグラフを選択・構築"""
    
    def select_subgraph(self, query: str, memory_graph: HierarchicalMemoryGraph):
        # 1. 関連概念の特定
        relevant_concepts = self.identify_concepts(query)
        
        # 2. 各概念から適切なレベルを選択
        selected_nodes = {}
        for concept in relevant_concepts:
            subgraph = memory_graph.concept_subgraphs[concept]
            level = self.select_level(query, subgraph)
            selected_nodes[concept] = level
            
        # 3. 選択したノードを含む最小部分グラフを構築
        return self.build_minimal_subgraph(selected_nodes, memory_graph)
    
    def select_level(self, query: str, concept_subgraph: ConceptSubgraph):
        """概念の進化グラフから適切なレベルを選択"""
        # クエリの複雑度分析
        complexity = self.analyze_query_complexity(query)
        
        # 進化パスを辿って最適なレベルを決定
        if complexity < 0.3:
            return self.find_earliest_node(concept_subgraph)
        elif complexity > 0.7:
            return self.find_latest_node(concept_subgraph)
        else:
            return self.find_intermediate_node(concept_subgraph, complexity)
```

### 4. 動的サブグラフ融合

```python
class DynamicSubgraphFusion:
    """複数のサブグラフを動的に融合"""
    
    def fuse_for_query(self, query: str, subgraphs: List[ConceptSubgraph]):
        # 1. クエリが要求する概念間の関係を分析
        required_relations = self.analyze_required_relations(query)
        
        # 2. 各サブグラフから必要な部分を抽出
        partial_graphs = []
        for subgraph in subgraphs:
            partial = self.extract_relevant_part(subgraph, required_relations)
            partial_graphs.append(partial)
            
        # 3. 部分グラフを統合
        fused_graph = self.merge_partial_graphs(partial_graphs)
        
        # 4. クロスコンセプトエッジを追加
        self.add_cross_concept_edges(fused_graph, required_relations)
        
        return fused_graph
```

## 実装例：数学概念の場合

```python
# 掛け算の進化サブグラフ
multiplication_subgraph = ConceptSubgraph("multiplication")
multiplication_subgraph.add_node("basic", {
    "definition": "同じ数を何回も足すこと",
    "examples": ["3×4 = 3+3+3+3"],
    "phase": 1
})
multiplication_subgraph.add_node("advanced", {
    "definition": "スケーリング操作",
    "examples": ["3×0.5 = 1.5"],
    "phase": 3
})
multiplication_subgraph.add_evolution("basic", "advanced", 
                                    trigger="decimal_multiplication")

# クエリ: "3×0.5を小学生に説明して"
# → 基礎ノードと高度ノードの橋渡しサブグラフを生成
bridging_subgraph = generator.create_bridge(
    from_node=multiplication_subgraph.get_node("basic"),
    to_node=multiplication_subgraph.get_node("advanced"),
    context="elementary_education"
)
```

## サブグラフベースの利点

### 1. 表現力
- 概念の進化を**有向グラフ**として自然に表現
- 分岐、合流、循環などの複雑な進化パターンに対応

### 2. 推論能力
- グラフニューラルネットワーク（GNN）の適用が可能
- メッセージパッシングで概念間の影響を計算

### 3. 柔軟性
- クエリごとに最適なサブグラフを動的構築
- 新しい分裂パターンへの適応が容易

### 4. 説明可能性
- 選択理由をグラフパスとして可視化
- デバッグとチューニングが容易

## 課題と対策

### 課題
1. **計算コスト**：サブグラフ操作は高コスト
2. **メモリ使用量**：グラフ構造の保持
3. **最適化の難しさ**：組み合わせ最適化問題

### 対策
1. **キャッシング**：頻出サブグラフをキャッシュ
2. **プルーニング**：低頻度パスの削除
3. **ヒューリスティクス**：貪欲法による近似解

## 実装ロードマップ

### Phase 1（1ヶ月）
- [ ] 基本的なConceptSubgraphクラスの実装
- [ ] 分裂の記録と可視化

### Phase 2（2-3ヶ月）
- [ ] HierarchicalMemoryGraphの実装
- [ ] 基本的なサブグラフ選択アルゴリズム

### Phase 3（4-6ヶ月）
- [ ] 動的サブグラフ融合
- [ ] GNNベースの推論機構

### Phase 4（6ヶ月以降）
- [ ] 大規模評価と最適化
- [ ] プロダクション対応

## 結論

サブグラフ管理は、タグベースよりも**本質的に優れたアプローチ**である。概念の進化と関係性を自然に表現し、高度な推論を可能にする。実装は複雑だが、長期的な拡張性と性能を考慮すると、投資する価値がある。