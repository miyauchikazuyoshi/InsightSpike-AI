# スペクトルGED追加の後方互換性保証

## 設計方針

スペクトル評価を追加しても、既存のAPIと動作を完全に維持します。

## 実装戦略

### 1. 設定による制御

```python
class GeDIGCore:
    def __init__(self,
                 # 既存パラメータ
                 node_cost: float = 1.0,
                 edge_cost: float = 1.0,
                 normalization: str = 'sum',
                 efficiency_weight: float = 0.3,
                 
                 # 新規パラメータ（デフォルトでOFF）
                 enable_spectral: bool = False,  # 👈 デフォルトFalse
                 spectral_weight: float = 0.3,
                 
                 # 他のパラメータ...
                 ):
        self.enable_spectral = enable_spectral
        self.spectral_weight = spectral_weight
```

### 2. 既存メソッドの拡張（内部のみ）

```python
def _calculate_normalized_ged(self, g1: nx.Graph, g2: nx.Graph) -> Dict[str, float]:
    """既存の実装に影響を与えずにスペクトル評価を追加"""
    
    # 既存の計算（変更なし）
    result = self._original_normalized_ged_calculation(g1, g2)
    
    # スペクトル評価（有効な場合のみ）
    if self.enable_spectral:
        spectral_score = self._calculate_spectral_improvement(g1, g2)
        
        # 内部的に拡張されたGEDを計算
        enhanced_ged = result['normalized_ged'] - self.spectral_weight * spectral_score
        result['enhanced_ged'] = np.clip(enhanced_ged, -1.0, 1.0)
        
        # structural_improvementも更新（既存フィールドを利用）
        result['structural_improvement'] = (
            result['structural_improvement'] * (1 - self.spectral_weight) +
            spectral_score * self.spectral_weight
        )
    
    return result
```

### 3. 外部インターフェースは不変

```python
def calculate(self, graph_before, graph_after, ...) -> GeDIGResult:
    """公開APIは完全に維持"""
    
    # 内部計算（スペクトル評価を含む可能性）
    ged_result = self._calculate_normalized_ged(g1, g2)
    ig_result = self._calculate_entropy_variance_ig(...)
    
    # GeDIGResultの構築（既存と同じ）
    return GeDIGResult(
        gedig_value=ged_result['structural_improvement'] - ig_result['information_gain'],
        ged_value=ged_result['structural_improvement'],
        ig_value=ig_result['information_gain'],
        structural_improvement=ged_result['structural_improvement'],
        information_integration=ig_result['information_integration'],
        # ...
    )
```

## 段階的な導入計画

### Phase 1: 実験的導入（現在）
```python
# デフォルトでは無効
calculator = GeDIGCore()  # enable_spectral=False
```

### Phase 2: オプトイン
```python
# 明示的に有効化
calculator = GeDIGCore(enable_spectral=True, spectral_weight=0.3)
```

### Phase 3: 検証後のデフォルト化
```python
# 十分な検証後、デフォルトで有効に
class GeDIGCore:
    def __init__(self, enable_spectral: bool = True, ...):
```

## テストによる保証

```python
def test_backward_compatibility():
    """スペクトル評価OFFで既存の動作を保証"""
    
    # 既存の設定
    calculator_old = GeDIGCore(enable_spectral=False)
    
    # 同じ入力
    g1, g2 = create_test_graphs()
    
    # 結果が完全に一致することを確認
    result_old = calculator_old.calculate(g1, g2)
    
    # 新実装でもenable_spectral=Falseなら同じ結果
    calculator_new = GeDIGCore(enable_spectral=False)
    result_new = calculator_new.calculate(g1, g2)
    
    assert result_old.gedig_value == result_new.gedig_value
    assert result_old.ged_value == result_new.ged_value
    assert result_old.ig_value == result_new.ig_value
```

## 利点

1. **完全な後方互換性**
   - 既存のコードは変更不要
   - 同じ入力→同じ出力を保証

2. **段階的な移行**
   - リスクなく新機能を試せる
   - 問題があれば即座に無効化可能

3. **A/Bテストが可能**
   ```python
   # 既存アルゴリズム
   result_a = GeDIGCore(enable_spectral=False).calculate(g1, g2)
   
   # 拡張アルゴリズム
   result_b = GeDIGCore(enable_spectral=True).calculate(g1, g2)
   
   # 比較評価
   compare_results(result_a, result_b)
   ```