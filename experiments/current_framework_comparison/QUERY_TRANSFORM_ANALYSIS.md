# Query Transform Analysis

## 発見事項

### 1. **MainAgentWithQueryTransformが存在する！**
- `/src/insightspike/core/agents/main_agent_with_query_transform.py`
- クエリ変換機能を持つ拡張版MainAgent

### 2. **しかし、使われていない**
- すべての実験スクリプトは通常の`MainAgent`を使用
- `MainAgentWithQueryTransform`をインポートしている場所がない

### 3. **query_transformationモジュールが見つからない**
```python
from ..query_transformation import (
    QueryState,
    QueryTransformationHistory,
    QueryTransformer,
)
```
- このインポートが失敗するはず
- つまり、このクラスは**未完成**か**削除されたモジュール**に依存

## MainAgentWithQueryTransformの機能（実装予定だったもの）

### クエリ変換フロー：
```python
# 1. クエリをグラフ上に配置
initial_state = self.query_transformer.place_query_on_graph(
    question, self._get_current_knowledge_graph()
)

# 2. グラフを通じてクエリを変換
new_query_state = self.query_transformer.transform_query(
    query_state, knowledge_graph, retrieved_documents
)

# 3. 変換されたクエリで検索
query_text = self._get_effective_query(current_state)
retrieved_documents = self.l2_memory.search_episodes(query_text, k=5)
```

### メッセージパッシング（推測）：
- `QueryTransformer.transform_query()`の中で実装されているはず
- グラフノード間でメッセージを伝播させてクエリを洗練
- しかし、`QueryTransformer`クラス自体が存在しない

## 結論

### 現状：
1. **通常のMainAgent** - クエリ変換なし（現在使用中）
2. **MainAgentWithQueryTransform** - クエリ変換あり（未使用・未完成）

### なぜ使われていないか：
- `query_transformation`モジュールが存在しない
- 依存関係が満たされていない
- 開発途中で放置された可能性

### 理想と現実のギャップ：
- **理想**: クエリがグラフ上でメッセージパッシングによって洗練される
- **現実**: 単純なベクトル検索のみ

## 使用するには

```python
# 現在（動かない）
from insightspike.core.agents.main_agent import MainAgent
agent = MainAgent(config)

# 理想（query_transformationが実装されていれば）
from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform
agent = MainAgentWithQueryTransform(config, enable_query_transformation=True)
```

しかし、必要なモジュールが欠けているため、現時点では使用不可能です。