# MainAgentWithQueryTransform 完成への最終ステップ

## 現在の状態

### ✅ 完了
- `query_transformation`モジュールが存在し動作
- `MainAgentWithQueryTransform`が初期化可能
- 基本的なクエリ変換ロジックが実装済み

### ❌ 未解決の問題

#### 1. **次元不一致エラー**
```
mat1 and mat2 shapes cannot be multiplied (4x256 and 384x256)
```
- GNNが期待する入力次元と実際の埋め込み次元が異なる
- MiniLM (384次元) vs GNN期待値 (256次元)

#### 2. **応答生成の失敗**
- LLMが "No response" を返している
- プロンプト構築の問題の可能性

## 解決方法

### 1. GNN次元問題の修正

```python
# layer3_graph_reasoner.py の修正
class GNNMessagePassing(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=384):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 384 -> 256
        self.conv2 = GCNConv(hidden_dim, output_dim)  # 256 -> 384
```

### 2. QueryTransformerの設定調整

```python
# MainAgentWithQueryTransform.__init__ の修正
self.query_transformer = QueryTransformer(
    embedding_model_name="paraphrase-MiniLM-L6-v2",  # 384次元
    use_gnn=False  # GNNを一時的に無効化
)
```

### 3. _get_current_knowledge_graph()の実装

```python
def _get_current_knowledge_graph(self):
    """現在の知識グラフを取得"""
    if hasattr(self, 'l3_graph') and self.l3_graph:
        # 最新のグラフデータを取得
        if hasattr(self.l3_graph, 'current_graph'):
            return self.l3_graph.current_graph
        # または保存されたグラフを読み込む
        try:
            import torch
            return torch.load('data/graph_pyg.pt')
        except:
            pass
    return None
```

### 4. _get_effective_query()の実装

```python
def _get_effective_query(self, query_state: QueryState) -> str:
    """変換されたクエリテキストを取得"""
    if query_state and hasattr(query_state, 'text'):
        return query_state.text
    return self.original_query  # フォールバック
```

## 簡易修正版の作成

最も簡単な解決策は、GNNを無効にして動作させることです：

```python
# run_with_query_transform.py
from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform
from insightspike.core.config import Config

config = Config()
config.llm.model_name = "distilgpt2"  # シンプルなモデル

# GNNを無効化してエージェントを作成
agent = MainAgentWithQueryTransform(
    config, 
    enable_query_transformation=True
)

# GNNを明示的に無効化
if hasattr(agent, 'query_transformer'):
    agent.query_transformer.use_gnn = False
if hasattr(agent, 'l3_graph'):
    agent.l3_graph.use_gnn = False

# 初期化と実行
agent.initialize()
result = agent.process_question("What is energy?")
```

## 完成の定義（改訂版）

### 必須要件（MVP）:
1. ✅ query_transformationモジュールが動作
2. ✅ MainAgentWithQueryTransformが初期化可能
3. 🔧 クエリが実際に変換される（GNNなしでも可）
4. 🔧 エラーなく完全な処理サイクルが実行される
5. 🔧 変換履歴が取得できる

### 理想的な完成形:
1. 🔧 GNNメッセージパッシングが動作
2. 🔧 LLMによる自然なクエリ再生成
3. 🔧 洞察検出と統合
4. 🔧 通常のMainAgentより優れた結果

## 推奨アプローチ

1. **まずGNNなしで動作確認**
   - `use_gnn=False`で基本機能を確認
   - アテンションベースのメッセージパッシングで十分

2. **段階的に機能追加**
   - LLMによるクエリ再生成
   - グラフメトリクスとの統合
   - GNN実装（オプション）

3. **実験で効果測定**
   - 通常のMainAgent vs QueryTransform付き
   - クエリ変換による洞察発見率の向上を確認

現時点では、基本的な仕組みは整っているので、エラーを回避する設定で動作させることが優先です。