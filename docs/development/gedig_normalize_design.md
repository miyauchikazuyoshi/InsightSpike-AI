# geDIG Normalization Design Document

## 概要
geDIG計算の新しい正規化バージョンの設計書。ΔGED + ΔIGを保存量として扱い、報酬計算の恣意性を最小化する。

## 背景と動機

### 従来のgeDIG（gedig_core.py）
```
geDIG = w1 * ΔGED - kT * ΔIG
```
- ΔGEDとΔIGを独立に評価
- 別々の閾値でスパイク判定

### 新しい正規化版geDIG
```
R = λ * Z(ΔIG) - μ * ΔGED_normalized

where:
- ΔGED_normalized = pure_GED / (|E| + β|N|) / (ΔGED + ΔIG)
- Z(ΔIG) = (ΔIG - μ_IG) / σ_IG
```

### 設計思想
1. **保存量仮説**: ΔGED + ΔIG = const（エネルギー保存的な考え方）
2. **二段階正規化**:
   - Stage 1: グラフサイズによる正規化
   - Stage 2: 保存量による比率正規化
3. **パラメータ削減**: 恣意的パラメータを3つ（β, λ, μ）のみに
4. **符号の一貫性**: 報酬Rの符号で探索行動が決定される（R < 0でスパイク）

## インターフェース設計

### 入力（gedig_coreと同じ）
```python
def calculate(
    self,
    graph_before: nx.Graph,
    graph_after: nx.Graph,
    focal_nodes: Optional[Set[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

### 出力形式
```python
{
    "gedig": float,              # 正規化された統合指標
    "ged": float,                # 生のGED値（後方互換性）
    "ig": float,                 # 生のIG値（後方互換性）
    "has_spike": bool,           # スパイク判定
    "normalized_metrics": {
        "ged_normalized": float,  # 正規化後のGED
        "ig_z_score": float,      # Z変換後のIG
        "conservation_sum": float, # ΔGED + ΔIG（保存量）
        "reward": float           # 最終的な報酬値
    },
    "statistics": {
        "ig_mean": float,         # IG平均（Z変換用）
        "ig_std": float,          # IG標準偏差（Z変換用）
        "graph_size_before": {
            "nodes": int,
            "edges": int
        },
        "graph_size_after": {
            "nodes": int,
            "edges": int
        }
    }
}
```

## 設定構造

### config.yaml
```yaml
insightspike:
  algorithms:
    gedig:
      # 既存のgedig_core設定
      w1: 0.5
      kT: 1.0
      thresholds:
        delta_ged: 0.5
        delta_ig: 0.5
      
      # 新しい正規化設定
      normalization:
        enabled: true
        mode: "conservation"  # "conservation" or "legacy"
        
        # サイズ正規化パラメータ
        size_normalization:
          beta: 0.5  # ノード重み係数
          
        # Z変換パラメータ
        z_transform:
          use_running_stats: true  # 実行時統計を使用
          window_size: 100         # 統計計算のウィンドウサイズ
          
        # 報酬関数パラメータ
        reward:
          lambda_ig: 1.0   # IG係数
          mu_ged: 0.5      # GED係数
          
        # スパイク判定
        spike_detection:
          mode: "threshold"  # "threshold" or "gradient"
          threshold: 0.0     # R < 0 でスパイク（符号の一貫性確認）
```

## 実装方針

### 1. クラス構造
```python
class GeDIGNormalizedCalculator:
    """正規化版geDIG計算器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalization_config = config.get("normalization", {})
        
        # 統計情報の保持（Z変換用）
        self.ig_history = deque(maxlen=self.normalization_config.get(
            "z_transform", {}).get("window_size", 100))
        
        # 後方互換性のため、基本的なGED/IG計算は既存実装を利用
        self._base_calculator = GeDIGCalculator(config)
```

### 2. 計算フロー
1. 基本的なGED/IG計算（既存のgedig_coreを利用）
2. サイズ正規化
3. 保存量計算（ΔGED + ΔIG）
4. Z変換（IG）
5. 最終的な報酬計算
6. スパイク判定

### 3. 後方互換性
- 出力に`ged`と`ig`フィールドを含める（生の値）
- `has_spike`の判定ロジックは設定で切り替え可能
- `normalization.enabled: false`で従来の動作

## 使用例

### 基本的な使用
```python
from insightspike.algorithms import GeDIGNormalizedCalculator

# 設定読み込み
config = load_config()
calculator = GeDIGNormalizedCalculator(config.algorithms.gedig)

# 計算実行
result = calculator.calculate(graph_before, graph_after)

# 結果の利用
if result["has_spike"]:
    print(f"Spike detected! Reward: {result['normalized_metrics']['reward']}")
```

### L3GraphReasonerとの統合
```python
# analyze_documents内で
if self.config.get("normalization", {}).get("enabled", False):
    from insightspike.algorithms import GeDIGNormalizedCalculator
    calculator = GeDIGNormalizedCalculator(self.config)
else:
    calculator = self.gedig_calculator
```

## テスト計画

### 単体テスト
1. 基本的な正規化計算
2. 保存量の確認（ΔGED + ΔIG）
3. Z変換の妥当性
4. エッジケース（空グラフ、単一ノード）

### 統合テスト
1. L3GraphReasonerとの連携
2. 設定の切り替え（legacy/conservation）
3. 後方互換性の確認

### 性能テスト
1. 大規模グラフでの計算時間
2. メモリ使用量（統計情報の保持）

## 移行計画

### Phase 1: 実装とテスト
- gedig_core_normalize.pyの実装
- 単体テストの作成
- ドキュメント作成

### Phase 2: 統合
- L3GraphReasonerへの組み込み
- 設定システムの更新
- 既存実験の再実行

### Phase 3: 評価
- 正規化なし vs 正規化ありの比較実験
- パラメータチューニング
- 論文用の図表作成

## リスクと対策

### リスク1: 統計的不安定性
- **問題**: 初期段階でIG履歴が少ない
- **対策**: 最小履歴数の設定、初期値の工夫

### リスク2: 保存量仮説の妥当性
- **問題**: ΔGED + ΔIGが実際に保存されるか
- **対策**: 実験的検証、理論的裏付けの整理

### リスク3: パラメータ調整の困難さ
- **問題**: λ, μ, βの最適値が不明
- **対策**: グリッドサーチ、ベイズ最適化の導入

## 次のステップ
1. この設計書のレビューと承認
2. gedig_core_normalize.pyの実装
3. テストコードの作成
4. 統合と実験