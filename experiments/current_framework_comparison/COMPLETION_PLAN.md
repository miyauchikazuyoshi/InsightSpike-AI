# MainAgentWithQueryTransform 完成計画

## 必要な実装

### 1. ✅ **query_transformation.py モジュール**
基本実装を作成しました：
- `QueryState`: クエリの状態を管理
- `QueryTransformationHistory`: 変換履歴を追跡
- `QueryTransformer`: メッセージパッシングによるクエリ変換

### 2. 🔧 **メッセージパッシングの改善**

#### 現在の実装（簡易版）:
```python
# Attention-based message passing
scores = np.dot(doc_embeddings, query_embedding)
weights = np.exp(scores / temperature)
new_embedding = 0.7 * current + 0.3 * weighted_sum(docs)
```

#### 理想的な実装:
```python
# Graph Neural Network message passing
class QueryGNN(torch.nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
    
    def forward(self, x, edge_index):
        # ノード間でメッセージを伝播
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        out = self.conv2(h, edge_index)
        return out
```

### 3. 🔧 **クエリ再生成の改善**

#### 現在:
```python
# 単純な文字列連結
expanded_query = f"{original} (related to: {keywords})"
```

#### 改善案:
```python
# LLMを使った自然な再生成
prompt = f"""
Original query: {original_query}
Key concepts discovered: {concepts}
Connections found: {connections}

Generate a refined query that captures these insights:
"""
refined_query = llm.generate(prompt)
```

### 4. 🔧 **洞察検出の強化**

```python
class InsightDetector:
    def detect_insight_emergence(self, transformation_history):
        # 1. 埋め込みの急激な変化
        embedding_shift = compute_embedding_shift(history)
        
        # 2. 新しい概念の出現
        new_concepts = extract_new_concepts(history)
        
        # 3. グラフ構造の単純化
        graph_simplification = measure_graph_simplification(history)
        
        return {
            "insight_detected": embedding_shift > threshold,
            "insight_type": classify_insight(new_concepts),
            "confidence": calculate_confidence(all_signals)
        }
```

## 実装手順

### ステップ1: 基本動作確認
```python
# test_query_transform.py
from query_transformation import QueryTransformer

transformer = QueryTransformer()
state = transformer.place_query_on_graph("What is energy?")
print(f"Initial state: {state.to_dict()}")

# ダミーデータでテスト
new_state = transformer.transform_query(state, None, dummy_docs)
print(f"Transformed: {new_state.text}")
```

### ステップ2: MainAgentとの統合
```python
# main_agent_with_query_transform.pyの修正
# 1. import文を修正（相対パスを調整）
from experiments.current_framework_comparison.src.query_transformation import (
    QueryState, QueryTransformationHistory, QueryTransformer
)

# 2. _get_current_knowledge_graph()メソッドを実装
def _get_current_knowledge_graph(self):
    # Layer3から現在のグラフを取得
    if hasattr(self, 'l3_graph'):
        return self.l3_graph.get_current_graph()
    return None
```

### ステップ3: 実験スクリプトの作成
```python
# run_query_transform_experiment.py
from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform

# クエリ変換を有効にして実行
agent = MainAgentWithQueryTransform(config, enable_query_transformation=True)
result = agent.process_question("What is the relationship between energy and information?")

# 変換履歴を可視化
history = result['transformation_history']
for i, state in enumerate(history['states']):
    print(f"Step {i}: {state['text']} (confidence: {state['confidence']:.2f})")
```

## 期待される効果

### Before (通常のMainAgent):
```
Query: "What is energy?"
→ 固定されたクエリで検索
→ 表面的な回答
```

### After (QueryTransform付き):
```
Query: "What is energy?"
→ "What is energy? (related to: capacity, work, conservation)"
→ "How does energy relate to thermodynamics and information?"
→ "What is the fundamental nature of energy in physics?"
→ より深い洞察を含む回答
```

## 完成の定義

1. ✅ query_transformationモジュールが存在
2. 🔧 MainAgentWithQueryTransformが動作
3. 🔧 メッセージパッシングが機能
4. 🔧 クエリが実際に変換される
5. 🔧 変換によって新しい洞察が生まれる
6. 🔧 実験で効果が測定できる

現在、基礎は整いましたが、統合テストと改善が必要です。