# 実装計画書：制御された創発メカニズム

## Phase 1: 基盤機能実装 (2-3日)

### 1.1 Split機能の追加

#### UpdateType拡張
```python
# delta_gedig.py に追加
class UpdateType(Enum):
    ADD = "add"
    PRUNE = "prune" 
    MERGE = "merge"
    SUMMARIZE = "summarize"
    SPLIT = "split"  # ← 新規追加
```

#### Split判定ロジック
```python
# dynamic_update.py に追加
def should_split_node(self, 
                     graph: EpisodicMemoryGraph,
                     node_id: str,
                     contradiction_context: Dict) -> UpdateDecision:
    """矛盾を含むノードの分裂判定"""
    
    if not self.mode.enable_split:
        return UpdateDecision(should_update=False, ...)
    
    # 矛盾度評価
    contradiction_score = self._calculate_contradiction_score(graph, node_id)
    
    # 複雑度評価（分裂に値するか）
    complexity_score = self._calculate_node_complexity(graph, node_id)
    
    # geDIG評価による分裂メリット判定
    split_candidates = self._generate_split_candidates(graph, node_id)
    
    for candidate in split_candidates:
        gedig_result = graph.evaluate_update(candidate)
        if gedig_result.delta_ig > self.thresholds.split_ig_threshold:
            return UpdateDecision(should_update=True, update=candidate, ...)
    
    return UpdateDecision(should_update=False, ...)
```

#### Split実行機構
```python
def _generate_split_candidates(self, graph: EpisodicMemoryGraph, node_id: str) -> List[GraphUpdate]:
    """分裂候補の生成"""
    original_node = graph.nodes[node_id]
    
    # LLMによる内容分析・分割
    split_texts = self._analyze_and_split_content(original_node.text)
    
    # 各分割に対する新ノード作成
    split_updates = []
    for i, split_text in enumerate(split_texts):
        split_node_id = f"{node_id}_split_{i}"
        
        update = GraphUpdate(
            update_type=UpdateType.SPLIT,
            target_nodes=[node_id],
            new_node_data={
                'id': split_node_id,
                'text': split_text,
                'embedding': self._generate_embedding(split_text),
                'node_type': 'split_fragment',
                'parent_node': node_id,
                'split_reason': 'contradiction_resolution'
            },
            remove_nodes=[node_id] if i == len(split_texts) - 1 else []
        )
        split_updates.append(update)
    
    return split_updates
```

### 1.2 実際のNLI統合

#### NLI計算クラス
```python
# nli_calculator.py (新規作成)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class NLICalculator:
    """Natural Language Inference計算クラス"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.nli_model = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name
        )
    
    def calculate_entailment_contradiction(self, text_a: str, text_b: str) -> Tuple[float, float]:
        """含意・矛盾スコア計算"""
        
        # Forward direction: A entails B?
        result_ab = self.nli_model(f"{text_a} [SEP] {text_b}")
        
        # Backward direction: B entails A?  
        result_ba = self.nli_model(f"{text_b} [SEP] {text_a}")
        
        # スコア抽出・正規化
        entailment_score = self._extract_entailment_score(result_ab, result_ba)
        contradiction_score = self._extract_contradiction_score(result_ab, result_ba)
        
        return entailment_score, contradiction_score
    
    def _extract_entailment_score(self, result_ab: Dict, result_ba: Dict) -> float:
        """含意スコア抽出"""
        # ラベル("ENTAILMENT")の信頼度を取得
        # 双方向の最大値を採用
        pass
    
    def _extract_contradiction_score(self, result_ab: Dict, result_ba: Dict) -> float:
        """矛盾スコア抽出"""
        # ラベル("CONTRADICTION")の信頼度を取得
        pass
```

#### EpisodeEdge更新
```python
# graph_manager.py 修正
def add_edge(self, source: str, target: str, **kwargs):
    """エッジ追加時にNLI計算を実行"""
    
    source_text = self.nodes[source].text
    target_text = self.nodes[target].text
    
    # NLI計算
    if hasattr(self, 'nli_calculator'):
        entailment_score, contradiction_score = self.nli_calculator.calculate_entailment_contradiction(
            source_text, target_text
        )
    else:
        entailment_score, contradiction_score = 0.0, 0.0
    
    # EpisodeEdgeに反映
    edge = EpisodeEdge(
        source=source,
        target=target,
        relation=kwargs.get('relation', 'semantic'),
        weight=kwargs.get('weight', 1.0),
        semantic_similarity=kwargs.get('semantic_similarity', 0.0),
        lexical_overlap=kwargs.get('lexical_overlap', 0.0),
        temporal_distance=kwargs.get('temporal_distance', 0.0),
        entailment_score=entailment_score,      # ← 実計算値
        contradiction_score=contradiction_score # ← 実計算値
    )
```

### 1.3 中間ノード生成機構

#### 中間概念生成クラス
```python
# intermediate_concept_generator.py (新規作成)
class IntermediateConceptGenerator:
    """矛盾解決のための中間概念生成"""
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    
    def generate_intermediate_concept(self, 
                                    contradictory_nodes: List[str],
                                    graph: EpisodicMemoryGraph) -> Optional[EpisodeNode]:
        """矛盾ノード群から中間概念を生成"""
        
        # 矛盾ノードのテキスト取得
        node_texts = [graph.nodes[nid].text for nid in contradictory_nodes]
        
        # LLMによる中間概念生成プロンプト
        prompt = self._build_intermediate_concept_prompt(node_texts)
        
        # LLM呼び出し
        response = self.llm_provider.generate(prompt)
        
        # 概念妥当性検証
        if self._validate_intermediate_concept(response, node_texts):
            return self._create_intermediate_node(response, contradictory_nodes)
        
        return None
    
    def _build_intermediate_concept_prompt(self, node_texts: List[str]) -> str:
        """中間概念生成プロンプト構築"""
        return f"""
以下の矛盾する概念群から、これらを統合・橋渡しする中間概念を生成してください：

矛盾する概念：
{chr(10).join(f"- {text}" for text in node_texts)}

要求：
1. 両方の概念の部分的真実を認識する
2. より高次の抽象化レベルで統合する  
3. 新たな洞察や視点を提供する
4. 論理的整合性を保つ

中間概念：
"""
    
    def _validate_intermediate_concept(self, concept: str, original_texts: List[str]) -> bool:
        """生成された中間概念の妥当性検証"""
        # 1. 元概念との関連性チェック
        # 2. 論理的整合性チェック
        # 3. 新規性チェック
        pass
    
    def _create_intermediate_node(self, concept_text: str, parent_nodes: List[str]) -> EpisodeNode:
        """中間概念ノード作成"""
        return EpisodeNode(
            node_id=f"intermediate_{int(time.time())}",
            text=concept_text,
            node_type='intermediate_concept',
            confidence=0.8,  # 生成概念の初期信頼度
            timestamp=time.time(),
            metadata={
                'generation_method': 'contradiction_resolution',
                'parent_nodes': parent_nodes,
                'concept_type': 'bridge_concept'
            }
        )
```

## Phase 2: 制御メカニズム実装 (2-3日)

### 2.1 創発品質評価システム

#### 品質評価クラス
```python
# emergence_quality_evaluator.py (新規作成)
class EmergenceQualityEvaluator:
    """創発的知識の品質評価"""
    
    def evaluate_emergence_quality(self, new_node: EpisodeNode, graph: EpisodicMemoryGraph) -> EmergenceQuality:
        """総合品質評価"""
        
        novelty_score = self._calculate_novelty_score(new_node, graph)
        utility_score = self._calculate_utility_score(new_node, graph)  
        consistency_score = self._calculate_consistency_score(new_node, graph)
        
        return EmergenceQuality(
            overall_score=(novelty_score + utility_score + consistency_score) / 3.0,
            novelty_score=novelty_score,
            utility_score=utility_score,
            consistency_score=consistency_score,
            confidence=min(novelty_score, utility_score, consistency_score)
        )
    
    def _calculate_novelty_score(self, new_node: EpisodeNode, graph: EpisodicMemoryGraph) -> float:
        """新規性スコア算出"""
        # 既存知識との類似度逆数
        similar_nodes = graph.find_similar_nodes(new_node.embedding, k=10)
        
        if not similar_nodes:
            return 1.0
        
        max_similarity = max(sim for _, sim in similar_nodes)
        return 1.0 - max_similarity
    
    def _calculate_utility_score(self, new_node: EpisodeNode, graph: EpisodicMemoryGraph) -> float:
        """有用性スコア算出"""
        # 潜在的な接続可能性を評価
        connection_potential = self._estimate_connection_potential(new_node, graph)
        
        # 推論タスクへの貢献度推定
        reasoning_contribution = self._estimate_reasoning_contribution(new_node, graph)
        
        return (connection_potential + reasoning_contribution) / 2.0
    
    def _calculate_consistency_score(self, new_node: EpisodeNode, graph: EpisodicMemoryGraph) -> float:
        """一貫性スコア算出"""
        # 既存知識体系との整合性評価
        consistency_violations = self._detect_consistency_violations(new_node, graph)
        
        return 1.0 - min(1.0, len(consistency_violations) / 10.0)
```

### 2.2 適応的閾値調整システム

#### 閾値学習クラス
```python
# adaptive_threshold_manager.py (新規作成)
class AdaptiveThresholdManager:
    """創発プロセスの適応的制御"""
    
    def __init__(self, initial_thresholds: Dict[str, float]):
        self.thresholds = initial_thresholds.copy()
        self.success_history = []
        self.failure_history = []
    
    def update_thresholds(self, emergence_results: List[EmergenceResult]):
        """成功・失敗パターンから閾値調整"""
        
        # 成功パターン分析
        successful_results = [r for r in emergence_results if r.was_successful]
        failed_results = [r for r in emergence_results if not r.was_successful]
        
        # 各閾値の最適化
        for threshold_name in self.thresholds:
            optimal_value = self._optimize_threshold(
                threshold_name, successful_results, failed_results
            )
            
            # 漸進的調整（急激な変更を避ける）
            current_value = self.thresholds[threshold_name]
            self.thresholds[threshold_name] = current_value * 0.9 + optimal_value * 0.1
    
    def _optimize_threshold(self, 
                          threshold_name: str, 
                          successful_results: List[EmergenceResult],
                          failed_results: List[EmergenceResult]) -> float:
        """個別閾値の最適化"""
        
        if not successful_results:
            return self.thresholds[threshold_name]  # 変更なし
        
        # 成功時の閾値範囲を分析
        successful_values = [r.threshold_values[threshold_name] for r in successful_results]
        failed_values = [r.threshold_values[threshold_name] for r in failed_results]
        
        # 成功確率の高い閾値を算出
        return self._find_optimal_threshold_value(successful_values, failed_values)
```

### 2.3 創発履歴追跡システム

#### 系譜管理クラス
```python
# emergence_lineage_tracker.py (新規作成)
class EmergenceLineageTracker:
    """創発プロセスの系譜追跡"""
    
    def __init__(self):
        self.lineage_graph = nx.DiGraph()  # 系譜グラフ
        self.emergence_events = {}         # 創発イベント詳細
    
    def record_emergence_event(self, 
                              event_id: str,
                              event_type: str,  # split, merge, intermediate_generation
                              source_nodes: List[str],
                              target_nodes: List[str],
                              quality_metrics: Dict):
        """創発イベント記録"""
        
        # イベント詳細保存
        self.emergence_events[event_id] = EmergenceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            quality_metrics=quality_metrics
        )
        
        # 系譜グラフ更新
        for source in source_nodes:
            for target in target_nodes:
                self.lineage_graph.add_edge(
                    source, target, 
                    event_id=event_id,
                    event_type=event_type
                )
    
    def trace_node_lineage(self, node_id: str) -> List[EmergenceEvent]:
        """ノードの系譜追跡"""
        
        # 祖先ノードを辿る
        ancestors = nx.ancestors(self.lineage_graph, node_id)
        
        # 関連する創発イベントを時系列順に取得
        related_events = []
        for event_id, event in self.emergence_events.items():
            if (node_id in event.target_nodes or 
                any(anc in event.source_nodes for anc in ancestors)):
                related_events.append(event)
        
        return sorted(related_events, key=lambda e: e.timestamp)
    
    def analyze_emergence_patterns(self) -> Dict[str, Any]:
        """創発パターン分析"""
        
        return {
            'most_productive_lineages': self._find_most_productive_lineages(),
            'common_emergence_patterns': self._identify_common_patterns(),
            'quality_evolution_trends': self._analyze_quality_trends()
        }
```

## Phase 3: 評価実験システム (3-4日)

### 3.1 評価データセット管理

#### データセット構築
```python
# evaluation_datasets.py (新規作成)
class ContradictoryKnowledgeDataset:
    """矛盾を含む知識データセット"""
    
    def __init__(self):
        self.datasets = {
            'scientific_evolution': self._load_scientific_evolution_data(),
            'historical_perspectives': self._load_historical_perspective_data(),
            'theoretical_frameworks': self._load_theoretical_framework_data()
        }
    
    def _load_scientific_evolution_data(self) -> List[ContradictionExample]:
        """科学理論の変遷データ"""
        return [
            ContradictionExample(
                topic="太陽系の構造",
                perspective_a="天動説：地球が宇宙の中心で、太陽が地球の周りを回る",
                perspective_b="地動説：太陽が中心で、地球が太陽の周りを回る", 
                ground_truth_resolution="地動説が正しく、さらに太陽も銀河系内を移動している",
                domain="astronomy"
            ),
            # 他の例...
        ]
```

### 3.2 評価指標システム

#### メトリクス計算クラス
```python
# evaluation_metrics.py (新規作成)
class ControlledEmergenceMetrics:
    """制御された創発の評価指標"""
    
    def calculate_comprehensive_metrics(self, 
                                      experiment_results: ExperimentResults) -> MetricsReport:
        """総合評価指標計算"""
        
        return MetricsReport(
            knowledge_graph_quality=self._calculate_kg_quality(experiment_results),
            contradiction_resolution_rate=self._calculate_resolution_rate(experiment_results),
            intermediate_node_utility=self._calculate_intermediate_utility(experiment_results),
            reasoning_accuracy_improvement=self._calculate_reasoning_improvement(experiment_results),
            memory_efficiency=self._calculate_memory_efficiency(experiment_results),
            emergence_quality_distribution=self._analyze_emergence_quality(experiment_results)
        )
    
    def _calculate_kg_quality(self, results: ExperimentResults) -> KGQualityMetrics:
        """知識グラフ品質指標"""
        return KGQualityMetrics(
            node_coherence_score=self._measure_node_coherence(results.final_graph),
            edge_validity_score=self._measure_edge_validity(results.final_graph),
            graph_connectivity_score=self._measure_connectivity(results.final_graph),
            information_density_score=self._measure_information_density(results.final_graph)
        )
    
    def _calculate_intermediate_utility(self, results: ExperimentResults) -> float:
        """中間ノード有用性指標"""
        
        intermediate_nodes = [n for n in results.final_graph.nodes.values() 
                            if n.node_type == 'intermediate_concept']
        
        if not intermediate_nodes:
            return 0.0
        
        total_utility = 0.0
        for node in intermediate_nodes:
            # 下流タスクでの活用度
            task_utility = self._measure_task_utility(node, results.task_results)
            
            # 他ノードとの接続性
            connectivity_utility = self._measure_connectivity_utility(node, results.final_graph)
            
            # 概念的新規性
            novelty_utility = self._measure_concept_novelty(node, results.baseline_graph)
            
            total_utility += (task_utility + connectivity_utility + novelty_utility) / 3.0
        
        return total_utility / len(intermediate_nodes)
```

## 実装スケジュール

### Week 1: 基盤実装
- **Day 1-2**: Split機能・NLI統合
- **Day 3-4**: 中間ノード生成機構
- **Day 5**: 統合テスト・バグ修正

### Week 2: 制御システム実装  
- **Day 6-7**: 品質評価システム
- **Day 8-9**: 適応的閾値調整
- **Day 10**: 履歴追跡システム

### Week 3: 評価実験
- **Day 11-12**: データセット準備・評価指標実装
- **Day 13-14**: 本実験実施
- **Day 15**: 結果分析・レポート作成

## 成功基準・検証項目

### 機能的成功基準
- [ ] Split機能が矛盾検知時に適切に動作する
- [ ] 中間ノードが意味的に有用な概念を生成する  
- [ ] geDIG評価による品質制御が効果的に機能する
- [ ] 適応的閾値調整が性能向上をもたらす

### 性能的成功基準
- [ ] ベースライン比で推論精度が5%以上向上
- [ ] 矛盾解決率が80%以上
- [ ] 生成された中間概念の妥当性が専門家評価で70%以上
- [ ] メモリ効率がベースライン以上を維持

### 品質的成功基準
- [ ] 創発的知識の新規性スコア平均0.7以上
- [ ] システムの安定性（異常終了なし）
- [ ] 実行時間が実用的範囲内（クエリ当たり<10秒）
- [ ] 再現性確保（同条件で一貫した結果）