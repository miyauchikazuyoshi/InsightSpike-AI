# geDIG Normalization 統合計画

## 現状の問題

### アーキテクチャの不整合
1. **L3GraphReasoner**:
   - `graph_analyzer.calculate_metrics()` でGED/IG計算
   - `delta_ged_func` と `delta_ig_func` を個別に渡している

2. **GeDIGCalculator**:
   - 独立したクラスとして存在
   - 実際には使われていない？

3. **GraphAnalyzer**:
   - GED/IGを別々に計算
   - スパイク判定も独自実装

## 統合方針

### Option 1: GraphAnalyzer内でGeDIGCalculatorを使用（推奨）

```python
# graph_analyzer.py の修正
class GraphAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        # GeDIGCalculatorを初期化
        from insightspike.algorithms.gedig_core_normalize import create_gedig_calculator
        self.gedig_calculator = create_gedig_calculator(config)
    
    def calculate_metrics(self, current_graph, previous_graph, delta_ged_func=None, delta_ig_func=None):
        # PyG Data を NetworkX に変換
        nx_current = self._pyg_to_networkx(current_graph)
        nx_previous = self._pyg_to_networkx(previous_graph) if previous_graph else None
        
        # GeDIGCalculatorを使用
        result = self.gedig_calculator.calculate(nx_previous, nx_current)
        
        # 既存の形式に合わせて返す
        return {
            "delta_ged": result["ged"],
            "delta_ig": result["ig"],
            "graph_size_current": current_graph.num_nodes,
            "graph_size_previous": previous_graph.num_nodes if previous_graph else 0,
            # 正規化版の追加情報
            "normalized_metrics": result.get("normalized_metrics", {}),
            "has_spike": result.get("has_spike", False)
        }
```

### Option 2: L3GraphReasonerで直接使用

```python
# layer3_graph_reasoner.py の修正
def __init__(self, config):
    # ...
    from insightspike.algorithms.gedig_core_normalize import create_gedig_calculator
    self.gedig_calculator = create_gedig_calculator(config)

def analyze_documents(self, documents, context=None):
    # ... グラフ構築 ...
    
    # メトリクス計算
    if self.config.get("normalization", {}).get("enabled", False):
        # 正規化版を使用
        nx_current = self._pyg_to_networkx(current_graph)
        nx_previous = self._pyg_to_networkx(previous_graph)
        gedig_result = self.gedig_calculator.calculate(nx_previous, nx_current)
        
        metrics = {
            "delta_ged": gedig_result["ged"],
            "delta_ig": gedig_result["ig"],
            **gedig_result.get("normalized_metrics", {})
        }
        spike_detected = gedig_result["has_spike"]
    else:
        # 既存の方法
        metrics = self.graph_analyzer.calculate_metrics(...)
        spike_detected = self.graph_analyzer.detect_spike(...)
```

## 後方互換性チェックリスト

### 1. 出力形式の互換性
- [x] `ged` フィールド（生の値）
- [x] `ig` フィールド（生の値）
- [x] `has_spike` フィールド
- [x] `gedig` フィールド（統合指標）
- [ ] `spike_detected` フィールド（L3が期待）
- [ ] `metrics` 辞書形式

### 2. 入力形式の互換性
- [ ] PyG Data形式のサポート
- [x] NetworkX形式のサポート
- [ ] 変換関数の実装

### 3. 設定の互換性
- [x] 既存の設定で動作
- [x] `normalization.enabled: false` で従来版
- [ ] delta_ged_func, delta_ig_func の扱い

## 実装手順

1. **変換関数の追加**:
   ```python
   def _pyg_to_networkx(pyg_graph: Data) -> nx.Graph:
       """Convert PyTorch Geometric Data to NetworkX Graph"""
   ```

2. **GraphAnalyzerの修正**:
   - GeDIGCalculatorを使用するように変更
   - 後方互換性のための条件分岐

3. **テストの追加**:
   - 正規化なし/ありの比較
   - 出力形式の確認
   - スパイク判定の一貫性

4. **設定ファイルの更新**:
   - デフォルトで `normalization.enabled: false`
   - 実験時に明示的に有効化

## リスクと対策

### リスク1: PyG ↔ NetworkX変換のオーバーヘッド
- **対策**: 変換を最小限に抑える、キャッシュの利用

### リスク2: 既存の実験との非互換性
- **対策**: デフォルトで無効化、段階的移行

### リスク3: スパイク判定ロジックの不一致
- **対策**: 両方のロジックをテストで検証