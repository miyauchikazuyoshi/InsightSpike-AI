"""
InsightSpike-AI geDIG Metrics API Design
=======================================

外部研究者向けの再利用可能なメトリクスAPI設計案
"""

# =============================================================================
# 1. 公開API設計案 (src/insightspike/metrics/__init__.py)
# =============================================================================

"""
# 基本使用例
from insightspike.metrics import compute_delta_ged, compute_delta_ig, compute_fusion_reward

# グラフ間のΔGED計算
delta_ged = compute_delta_ged(graph_before, graph_after)

# 状態間のΔIG計算  
delta_ig = compute_delta_ig(state_before, state_after)

# 融合報酬計算
reward = compute_fusion_reward(
    delta_ged=delta_ged,
    delta_ig=delta_ig, 
    conflict_score=0.1,
    weights={'ged': 0.4, 'ig': 0.3, 'conflict': 0.3}
)
"""

# =============================================================================
# 2. 設定可能な融合報酬スキーム
# =============================================================================

"""
融合報酬式: R(w₁,w₂,w₃) = w₁ × ΔGED + w₂ × ΔIG + w₃ × ConflictPenalty

デフォルト重み:
- w₁ (ΔGED重み): 0.4  # 構造効率性重視
- w₂ (ΔIG重み): 0.3   # 情報利得重視  
- w₃ (Conflict重み): 0.3 # 一貫性重視

設定例:
1. 構造重視設定: w₁=0.6, w₂=0.2, w₃=0.2
2. 情報重視設定: w₁=0.2, w₂=0.6, w₃=0.2
3. バランス設定: w₁=0.33, w₂=0.33, w₃=0.34
"""

# =============================================================================
# 3. 使用例とテストケース
# =============================================================================

def example_usage_cases():
    """実際の使用例"""
    
    # Case 1: 教育システム統合
    """
    from insightspike.metrics import geDIGMetrics
    
    # 学習前後の知識グラフ比較
    learning_progress = geDIGMetrics.analyze_learning_progress(
        knowledge_before=student_kg_before,
        knowledge_after=student_kg_after,
        domain="mathematics"
    )
    
    insight_detected = learning_progress.eureka_spike_detected
    learning_efficiency = learning_progress.delta_ged
    knowledge_gain = learning_progress.delta_ig
    """
    
    # Case 2: 強化学習統合
    """
    from insightspike.metrics import compute_insight_reward
    
    # エージェントの洞察報酬計算
    insight_reward = compute_insight_reward(
        state_trajectory=agent_states,
        action_sequence=actions,
        reward_scheme="balanced"  # "ged_focused", "ig_focused", "balanced"
    )
    """
    
    # Case 3: 研究用途
    """
    from insightspike.metrics.core import DeltaGEDCalculator, DeltaIGCalculator
    
    # カスタム実装でのベンチマーク
    ged_calc = DeltaGEDCalculator(algorithm="exact", max_nodes=50)
    ig_calc = DeltaIGCalculator(method="entropy", discretization_bins=20)
    
    results = []
    for graph_pair in test_dataset:
        ged = ged_calc.compute(graph_pair.before, graph_pair.after)
        ig = ig_calc.compute(graph_pair.before, graph_pair.after)
        results.append({'ged': ged, 'ig': ig, 'label': graph_pair.label})
    """

# =============================================================================
# 4. 設定ファイル形式案
# =============================================================================

"""
# gedig_config.yaml
fusion_reward:
  weights:
    delta_ged: 0.4
    delta_ig: 0.3
    conflict_penalty: 0.3
  
  thresholds:
    eureka_spike_ged: -0.5
    eureka_spike_ig: 0.2
    conflict_threshold: 0.8

algorithms:
  delta_ged:
    method: "approximation"  # "exact", "approximation", "hybrid"
    max_nodes_exact: 30
    cache_enabled: true
    
  delta_ig:
    method: "entropy"  # "entropy", "clustering", "hybrid"
    discretization_bins: 10
    clustering_k: 8

performance:
  parallel_computation: true
  batch_size: 100
  timeout_seconds: 5.0
"""

# =============================================================================
# 5. ドキュメント生成テンプレート
# =============================================================================

"""
## geDIG Metrics API Documentation

### Core Functions

#### `compute_delta_ged(graph_before, graph_after, **kwargs)`

**Purpose**: Calculate Graph Edit Distance change between two knowledge states

**Mathematical Definition**:
```
ΔGED = GED(G_after, ∅) - GED(G_before, ∅)

where GED(G₁, G₂) = min{∑(cost_ops) | transform G₁ → G₂}
```

**Parameters**:
- `graph_before`: Initial graph state (NetworkX, PyG, or dict format)
- `graph_after`: Final graph state
- `algorithm`: "exact", "approximation", or "hybrid" (default: "hybrid")
- `max_nodes`: Maximum nodes for exact calculation (default: 30)

**Returns**: `float` - Negative values indicate simplification (insight)

**Example**:
```python
delta_ged = compute_delta_ged(
    graph_before=student_knowledge_before,
    graph_after=student_knowledge_after,
    algorithm="exact"
)
print(f"Knowledge structure change: {delta_ged:.3f}")
```

#### `compute_delta_ig(state_before, state_after, **kwargs)`

**Purpose**: Calculate Information Gain change between cognitive states

**Mathematical Definition**:
```
ΔIG = H(S_before) - H(S_after)

where H(S) = -∑ p(x) log₂ p(x)  (Shannon entropy)
```

**Parameters**:
- `state_before`: Initial state representation
- `state_after`: Final state representation  
- `method`: "entropy", "clustering", or "hybrid" (default: "entropy")
- `discretization_bins`: Bins for continuous features (default: 10)

**Returns**: `float` - Positive values indicate learning progress

#### `compute_fusion_reward(delta_ged, delta_ig, conflict_score, weights)`

**Purpose**: Calculate combined insight reward using fusion scheme

**Mathematical Definition**:
```
R(w₁,w₂,w₃) = w₁ × ΔGED + w₂ × ΔIG - w₃ × ConflictScore

Default weights: w₁=0.4, w₂=0.3, w₃=0.3
```

### Configuration Examples

#### Research-focused (High precision)
```python
config = {
    'weights': {'ged': 0.5, 'ig': 0.4, 'conflict': 0.1},
    'algorithms': {
        'delta_ged': {'method': 'exact', 'max_nodes': 50},
        'delta_ig': {'method': 'entropy', 'bins': 20}
    }
}
```

#### Production-focused (High speed)  
```python
config = {
    'weights': {'ged': 0.3, 'ig': 0.3, 'conflict': 0.4},
    'algorithms': {
        'delta_ged': {'method': 'approximation'},
        'delta_ig': {'method': 'clustering', 'k': 5}
    }
}
```
"""
