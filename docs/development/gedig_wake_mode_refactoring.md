---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# geDIG計算ロジックのWake/Sleepモード対応リファクタリング

## 概要

現在のgeDIG計算は単一モードで動作していますが、実際の認知システムでは異なる処理モードが必要です：

- **Wake Mode（覚醒モード）**: クエリ駆動で効率的に問題解決（geDIG最小化）
- **Sleep Mode（睡眠モード）**: エッジ密度に基づく自律的な記憶整理（geDIG評価による剪定）

## 実装優先順位

**Phase 1: Wake Modeのみ実装**（まず着手）
- Sleep Modeは後回しにして、Wake Modeの実装に集中
- クエリ応答時のgeDIG最小化を実現

## 現状の問題点

### 1. 単一モードの制約

```python
# 現在のgedig_core.pyでの計算
def calculate_gedig(self, graph, focal_nodes):
    ged = self.calculate_normalized_ged(graph, focal_nodes)
    ig = self.calculate_entropy_variance_ig(graph, focal_nodes)
    
    # 常に最大化を前提とした計算
    gedig = ged * ig
    return gedig
```

この実装では、エージェントは常に「新しいことを発見しよう」とする探索モードで動作します。

### 2. 効率性の欠如

既知の問題に対しても毎回新しい解法を探索しようとするため、学習した知識を活用できません。

## 提案する解決策

### 1. モード対応のgeDIG計算

```python
@dataclass
class GeDIGMode(Enum):
    WAKE = "wake"    # クエリ駆動モード（geDIG最小化）
    SLEEP = "sleep"  # 自律整理モード（インタラプト処理）

class GeDIGCalculator:
    def calculate_gedig(
        self, 
        graph: nx.Graph, 
        focal_nodes: Set[str],
        mode: GeDIGMode = GeDIGMode.SLEEP
    ) -> GeDIGResult:
        """
        モードに応じたgeDIG計算
        
        Args:
            graph: 知識グラフ
            focal_nodes: 焦点ノード
            mode: 計算モード（SLEEP/WAKE）
        """
        
        if mode == GeDIGMode.WAKE:
            # Wake Mode: クエリに対する効率的な応答
            ged = self._calculate_minimal_ged(graph, focal_nodes)
            ig = self._calculate_convergent_ig(graph, focal_nodes)
            
            # 最小化スコア（既知パターンへの収束）
            gedig_value = 1.0 / (1.0 + ged * ig)
            
        else:  # SLEEP mode
            # Sleep Mode: 自律的な記憶整理（クエリなし）
            # インタラプト処理で記憶を再構成
            return self._process_sleep_mode_consolidation(graph)
        
        return GeDIGResult(
            gedig_value=gedig_value,
            ged_value=ged,
            ig_value=ig,
            mode=mode.value
        )
```

### 2. Sleep Modeの自律処理

```python
def _process_sleep_mode_consolidation(self, graph: nx.Graph) -> GeDIGResult:
    """
    Sleep Mode: クエリなしで自律的に記憶を整理
    """
    # 1. 弱い接続の刈り込み
    weak_edges = self._identify_weak_connections(graph)
    
    # 2. 類似パターンの統合
    patterns = self._find_similar_patterns(graph)
    consolidated = self._consolidate_patterns(patterns)
    
    # 3. 新しい接続の発見（夢のような処理）
    novel_connections = self._discover_latent_connections(graph)
    
    # Sleep modeではgeDIG値は整理の進捗を表す
    consolidation_score = len(consolidated) / len(patterns)
    
    return GeDIGResult(
        gedig_value=consolidation_score,
        mode="sleep",
        consolidated_patterns=consolidated,
        novel_connections=novel_connections
    )
```

### 3. Wake Mode専用のメトリクス

```python
def _calculate_minimal_ged(self, graph: nx.Graph, focal_nodes: Set[str]) -> float:
    """
    Wake Mode用のGED計算
    既存構造への最小限の変更を評価
    """
    # 既存の接続パターンとの一致度
    pattern_match_score = self._evaluate_pattern_matching(graph, focal_nodes)
    
    # 構造的安定性（変更の少なさ）
    stability_score = 1.0 / (1.0 + self._count_structural_changes(graph, focal_nodes))
    
    return pattern_match_score * stability_score

def _calculate_convergent_ig(self, graph: nx.Graph, focal_nodes: Set[str]) -> float:
    """
    Wake Mode用のIG計算
    予測可能性と確実性を重視
    """
    # エントロピーの低さ（確実性）
    certainty = 1.0 / (1.0 + self._calculate_entropy(graph, focal_nodes))
    
    # 既知パターンへの収束度
    convergence = self._evaluate_convergence_to_known_patterns(graph, focal_nodes)
    
    return certainty * convergence
```

### 4. Sleep Modeのトリガー条件（Phase 2で実装）

```python
class EdgeDensityMonitor:
    """エッジ密度に基づくSleep Modeトリガー"""
    
    def __init__(self):
        self.edge_threshold = 10  # ノードあたりのエッジ数閾値
        self.density_trigger = 0.3  # 30%のノードが閾値超えたらSleep
        
    def should_sleep(self, graph: nx.Graph) -> bool:
        """Sleep Modeに入るべきか判定"""
        high_degree_nodes = 0
        total_nodes = len(graph.nodes())
        
        for node in graph.nodes():
            if graph.degree(node) > self.edge_threshold:
                high_degree_nodes += 1
                
        density_ratio = high_degree_nodes / total_nodes
        return density_ratio > self.density_trigger
    
    def prune_edges_with_gedig(self, graph: nx.Graph) -> nx.Graph:
        """geDIG評価による最適な剪定"""
        # 全エッジの組み合わせを評価
        edge_combinations = self._generate_pruning_candidates(graph)
        
        best_combination = None
        min_entropy_loss = float('inf')
        
        for combination in edge_combinations:
            # 剪定後のグラフを仮想的に作成
            pruned_graph = self._apply_pruning(graph, combination)
            
            # 情報エントロピーの損失を計算
            entropy_loss = self._calculate_entropy_loss(
                graph, pruned_graph
            )
            
            # geDIG評価（最小の損失を探す）
            if entropy_loss < min_entropy_loss:
                min_entropy_loss = entropy_loss
                best_combination = combination
                
        return self._apply_pruning(graph, best_combination)
```

### 5. 自動モード切り替え

```python
class AdaptiveGeDIGCalculator(GeDIGCalculator):
    def __init__(self):
        super().__init__()
        self.confidence_threshold = 0.8
        self.exploration_threshold = 0.3
        
    def calculate_adaptive_gedig(
        self, 
        graph: nx.Graph, 
        focal_nodes: Set[str],
        task_context: Dict[str, Any]
    ) -> GeDIGResult:
        """
        タスクコンテキストに基づいて自動的にモードを選択
        """
        # モード判定
        mode = self._determine_mode(graph, focal_nodes, task_context)
        
        # モードに応じた計算
        result = self.calculate_gedig(graph, focal_nodes, mode)
        
        # モード切り替えの記録
        result.auto_mode_selected = True
        result.mode_confidence = self._calculate_mode_confidence(task_context)
        
        return result
    
    def _determine_mode(
        self, 
        graph: nx.Graph, 
        focal_nodes: Set[str],
        task_context: Dict[str, Any]
    ) -> GeDIGMode:
        """
        現在の状況からモードを自動判定
        """
        # 既知のパターンとの類似度
        familiarity = task_context.get('familiarity_score', 0.0)
        
        # タスクの進捗
        progress = task_context.get('task_progress', 0.0)
        
        # 最近のパフォーマンス
        recent_performance = task_context.get('recent_success_rate', 0.5)
        
        if familiarity > self.confidence_threshold and recent_performance > 0.7:
            return GeDIGMode.WAKE
        elif progress < self.exploration_threshold or recent_performance < 0.3:
            return GeDIGMode.SLEEP
        else:
            # 境界的な場合は確率的に選択
            wake_probability = familiarity * recent_performance
            return GeDIGMode.WAKE if np.random.random() < wake_probability else GeDIGMode.SLEEP
```

## 実装計画

### Phase 1: Wake Mode実装（最優先 - 1週間）

1. **Wake Mode専用のgeDIG計算**
   - `calculate_wake_mode_gedig`メソッドの実装
   - 既存パターンへの収束を評価
   - 最小限の構造変更で目標達成

2. **迷路探索での検証**
   - 既知経路の効率的な活用
   - geDIG最小化による高速化測定

### Phase 2: Sleep Mode設計（後日実装 - 2週間）

1. **エッジ密度モニタリング**
   - ノードごとのエッジ数カウント
   - 閾値超過ノードの割合計算
   - 自動トリガー機構

2. **geDIG評価による剪定**
   - エントロピー損失最小化
   - 重要な接続の保持
   - 冗長な接続の削除

3. **剪定アルゴリズム**
   ```python
   # 擬似コード
   for edge in candidate_edges:
       # 剪定前後の情報損失を計算
       loss = calculate_entropy_loss(graph, edge)
       # geDIG評価で最適な組み合わせを探索
       if loss < threshold:
           prune_candidates.add(edge)
   ```

## 期待される効果

1. **Wake Mode**:
   - クエリに対する高速応答
   - 既知パターンの効率的活用
   - 計算リソースの節約

2. **Sleep Mode**:
   - 記憶の自律的な整理・統合
   - 弱い接続の刈り込み
   - 潜在的なパターンの発見
   - 「ひらめき」の生成

## 実際の動作イメージ

### Phase 1: Wake Modeのみ

```python
# Wake Modeの実装（最初はこれだけ）
class WakeModeGedigCalculator:
    def calculate_gedig(self, graph, focal_nodes, query_context):
        """クエリに対して最小限の変更で応答"""
        # 既存パターンとの距離
        pattern_distance = self._find_nearest_pattern(graph, query_context)
        
        # 構造変更の最小化
        minimal_ged = 1.0 / (1.0 + self._required_changes(graph, focal_nodes))
        
        # 収束的な情報利得
        convergent_ig = self._calculate_convergence_score(graph, focal_nodes)
        
        # Wake Mode: 小さい値が良い
        return minimal_ged * convergent_ig * pattern_distance
```

### Phase 2: Sleep Mode追加（将来実装）

```python
# エッジ密度ベースの自動トリガー
monitor = EdgeDensityMonitor()
if monitor.should_sleep(agent.knowledge_graph):
    # Sleep Mode: geDIG評価による剪定
    pruned_graph = monitor.prune_edges_with_gedig(
        agent.knowledge_graph
    )
    agent.update_knowledge_graph(pruned_graph)
```

## 後方互換性

既存のAPIとの互換性を保つため、デフォルトはWAKEモードとします：

```python
# 既存のコード（変更不要）
result = calculator.calculate_gedig(graph, focal_nodes)  # デフォルトでWake

# Sleep Modeは明示的に指定
result = calculator.calculate_gedig(graph, focal_nodes, mode=GeDIGMode.SLEEP)
```

## 関連ファイル

- `/src/insightspike/algorithms/gedig_core.py` - メイン実装
- `/src/insightspike/algorithms/metrics_selector.py` - メトリクス選択ロジック
- `/src/insightspike/implementations/agents/main_agent.py` - エージェント統合