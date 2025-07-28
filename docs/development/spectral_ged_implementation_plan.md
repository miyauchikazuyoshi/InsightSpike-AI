# スペクトルGED実装計画（最小限の追加）

## 実装箇所と追加コード

### 1. gedig_core.py への追加箇所

```python
class GeDIGCore:
    def __init__(self,
                 # ... 既存パラメータ ...
                 
                 # 🆕 スペクトル評価パラメータ（ここに追加）
                 enable_spectral: bool = False,
                 spectral_weight: float = 0.3,
                 ):
        # ... 既存の初期化 ...
        
        # 🆕 スペクトル設定を保存
        self.enable_spectral = enable_spectral
        self.spectral_weight = spectral_weight
```

### 2. スペクトル計算メソッドの追加

```python
    # 🆕 新規メソッド（既存コードの後に追加）
    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        """ラプラシアン固有値による構造スコア"""
        if g.number_of_nodes() < 2:
            return 0.0
        
        try:
            L = nx.laplacian_matrix(g).toarray()
            eigvals = np.linalg.eigvalsh(L)
            return np.std(eigvals)  # 固有値の標準偏差
        except:
            return 0.0
```

### 3. 既存の_calculate_normalized_gedメソッドに数行追加

```python
    def _calculate_normalized_ged(self, g1: nx.Graph, g2: nx.Graph) -> Dict[str, float]:
        # ... 既存の計算はそのまま ...
        
        # 🆕 スペクトル評価（ここに追加）
        if self.enable_spectral:
            spectral_before = self._calculate_spectral_score(g1)
            spectral_after = self._calculate_spectral_score(g2)
            spectral_improvement = (spectral_before - spectral_after) / (spectral_before + 1e-10)
            
            # structural_improvementを更新
            structural_improvement = (
                structural_improvement * (1 - self.spectral_weight) +
                np.tanh(spectral_improvement) * self.spectral_weight
            )
        
        return {
            'raw_ged': raw_ged,
            'normalized_ged': normalized_ged,
            'structural_improvement': np.clip(structural_improvement, -1.0, 1.0),
            'efficiency_change': efficiency_change
        }
```

## 2. config.yaml での設定

### config.yaml に追加
```yaml
# Advanced Metrics (geDIG) Settings
metrics:
  use_normalized_ged: true
  use_entropy_variance_ig: false
  use_multihop_gedig: false
  
  # 🆕 スペクトル評価設定
  spectral_evaluation:
    enabled: false        # デフォルトは無効
    weight: 0.3          # 有効時の重み
    
  multihop_config:
    max_hops: 3
    decay_factor: 0.5
```

### 3. 設定の読み込み（MetricsSelector等で）

```python
# metrics_selector.py または適切な場所で
def create_gedig_calculator(config: InsightSpikeConfig):
    """設定に基づいてGeDIGCalculatorを作成"""
    
    # スペクトル設定を読み取り
    spectral_config = config.metrics.get('spectral_evaluation', {})
    
    return GeDIGCore(
        # 既存パラメータ
        enable_multihop=config.metrics.use_multihop_gedig,
        
        # 🆕 スペクトルパラメータ
        enable_spectral=spectral_config.get('enabled', False),
        spectral_weight=spectral_config.get('weight', 0.3)
    )
```

## 実装の簡潔さ

### 追加行数の見積もり
- gedig_core.py: +30行程度
- config.yaml: +4行
- 設定読み込み: +5行

**合計: 約40行の追加で実装可能**

## 実装手順

1. **Step 1**: gedig_core.pyに`_calculate_spectral_score`メソッド追加
2. **Step 2**: `__init__`にパラメータ追加
3. **Step 3**: `_calculate_normalized_ged`に条件分岐追加
4. **Step 4**: config.yamlに設定項目追加
5. **Step 5**: 設定読み込み部分を更新

## テスト方法

```bash
# 1. デフォルト（スペクトルOFF）でテスト
poetry run pytest tests/unit/test_gedig_calculator.py

# 2. config.yamlでスペクトルをONにしてテスト
# spectral_evaluation.enabled: true に変更
poetry run pytest tests/unit/test_gedig_calculator.py

# 3. 結果の比較
# OFFとONで異なる結果が出ることを確認
```