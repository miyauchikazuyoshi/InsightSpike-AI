# スパイク判定の責務配置に関する検討

## 現状の問題
- スパイク判定ロジックが複数箇所に分散している
- GraphAnalyzer、GeDIGCalculator、GeDIGNormalizedCalculatorでそれぞれ異なる実装

## 責務の観点からの分析

### Option 1: geDIG計算器内でスパイク判定（現在の実装）

**メリット**：
- GED/IG値とスパイク判定が密結合（同じ場所で計算）
- 正規化版では統一報酬Rで判定するため、計算器内が自然
- 設定の一元管理が可能

**デメリット**：
- 単一責任原則に反する可能性（計算と判定は別の責務）
- 異なる判定ロジックを試すときに計算器を変更する必要

### Option 2: スパイク判定を別クラスに分離

```python
class SpikeDetector:
    """スパイク判定専用クラス"""
    
    def detect(self, metrics: Dict[str, float]) -> bool:
        # 判定ロジック
        pass

class ThresholdSpikeDetector(SpikeDetector):
    """閾値ベースの判定"""
    
class GradientSpikeDetector(SpikeDetector):
    """勾配ベースの判定"""
    
class ConservationSpikeDetector(SpikeDetector):
    """保存量ベースの判定（正規化版）"""
```

**メリット**：
- 単一責任原則に従う
- 判定ロジックの切り替えが容易
- テストが書きやすい

**デメリット**：
- クラスが増える
- 設定が分散する可能性

### Option 3: 階層的な責務分離（推奨）

```python
# 1. 計算器は値の計算のみ
class GeDIGCore:
    def calculate() -> Dict[str, float]:
        # GED, IG, geDIG値の計算のみ
        return {"ged": ..., "ig": ..., "gedig": ...}

# 2. 正規化計算器も値の計算のみ  
class GeDIGNormalizedCalculator:
    def calculate() -> Dict[str, float]:
        # 正規化された値の計算のみ
        return {"ged": ..., "ig": ..., "reward": ...}

# 3. 上位レイヤーでスパイク判定
class GraphAnalyzer:
    def __init__(self):
        self.calculator = create_gedig_calculator(config)
        self.spike_detector = create_spike_detector(config)
    
    def analyze():
        metrics = self.calculator.calculate(...)
        has_spike = self.spike_detector.detect(metrics)
```

## 推奨案

### 短期的対応（現実的）
- 現在の実装を維持（計算器内でスパイク判定）
- ただし、判定ロジックをプライベートメソッド `_detect_spike` に分離
- 設定で判定方法を切り替え可能に

### 長期的対応（理想的）
- スパイク判定を独立したクラスに分離
- Strategy パターンで判定ロジックを切り替え
- GraphAnalyzerが統合ポイント

## 実装方針

1. **gedig_core_normalize.py** では現状維持
   - 報酬Rの符号で判定は妥当（R < 0）
   - ただし、`_detect_spike` メソッドとして分離済み

2. **将来の拡張性を確保**
   ```python
   def _detect_spike(self, reward: float) -> bool:
       if self.spike_mode == "threshold":
           return reward < self.spike_threshold
       elif self.spike_mode == "gradient":
           # 将来実装
           pass
       elif self.spike_mode == "custom":
           # カスタムdetectorを呼び出し
           return self.custom_detector.detect({"reward": reward})
   ```

3. **出力には判定結果を含める**
   - 計算結果と判定結果をセットで返す
   - 上位レイヤーは結果を使うだけ