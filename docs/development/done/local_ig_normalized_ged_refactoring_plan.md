---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# 局所IG・正規化GEDリファクタリング計画書

## 作成日: 2025-07-27

## 1. 背景と目的

### 発見された理論的洞察
- **局所的な情報拡散モデル**: 新規情報（高エントロピー）がグラフエッジを通じて拡散し安定化する過程が洞察形成
- **構造と情報の統合**: 類似性ベースで最適な接続を探し、構造改善（ΔGED）と情報統合（ΔIG）を同時に達成
- **情報理論的正当性**: 最大エントロピー原理、自由エネルギー最小化、統合情報理論と整合

### 現在の問題点
1. **IG計算の感度不足**: 類似度ベースのエントロピーでは変化量が0.01〜0.1程度
2. **GEDの正規化不足**: グラフサイズによって値が大きく変動
3. **局所性の欠如**: グローバルな指標のみで局所的な情報統合を捉えられない

## 2. リファクタリング設計

### 2.1 新しいIG計算（LocalInformationGain）

```python
class LocalInformationGain:
    """
    局所的な情報拡散に基づくIG計算
    
    主要な概念:
    1. Surprise（驚き）: ノードが周囲とどれだけ異なるか
    2. Information Diffusion: エッジを通じた情報の拡散
    3. Stabilization: エントロピーの減少として測定
    """
    
    def calculate_local_ig(graph, features_before, features_after, new_nodes):
        # 1. 各ノードの"驚き"を計算
        # 2. 情報拡散シミュレーション
        # 3. 安定化指標の計算
        #    - Global IG: 全体的なエントロピー減少
        #    - Homogenization: 局所的な均一化
        #    - Tension Reduction: エッジ間の緊張緩和
```

### 2.2 正規化されたGED（NormalizedGED）

```python
class NormalizedGED:
    """
    スケール不変の正規化GED
    
    正規化手法:
    1. グラフサイズによる正規化: GED / (n1 + n2 + e1 + e2)
    2. 構造効率の考慮: 効率改善時は負の値
    3. 値域を[-1, 1]に制限
    """
    
    def calculate_normalized_ged(graph1, graph2):
        # 1. 基本的なGED計算
        # 2. グラフサイズで正規化
        # 3. 構造効率の変化を評価
        # 4. [-1, 1]の範囲にクリップ
```

### 2.3 統合スパイクスコア

```python
def calculate_unified_spike_score(delta_ged, delta_ig):
    """
    GEDとIGを統合したスパイクスコア
    
    特徴:
    - 個別の閾値ではなく統合スコアで判定
    - 動的閾値（変化の大きさに応じて調整）
    - 両指標の寄与を適切に重み付け
    """
```

## 3. 実装フェーズ

### Phase 1: コア実装（2日）
- [ ] LocalInformationGainクラスの実装
- [ ] NormalizedGEDクラスの実装
- [ ] 統合スパイクスコア計算の実装

### Phase 2: 統合（1日）
- [ ] MetricsSelectorへの新実装の統合
- [ ] フィーチャーフラグの追加（`use_local_ig`, `use_normalized_ged`）
- [ ] 既存インターフェースとの互換性確保

### Phase 3: テスト（2日）
- [ ] ユニットテスト作成
  - LocalIGの各メソッド
  - NormalizedGEDの正規化ロジック
  - 統合スコア計算
- [ ] 後方互換性テスト
  - 既存の実験が動作することを確認
  - 必要に応じて互換性レイヤーを追加
- [ ] パイプラインテスト
  - MainAgent経由での動作確認
  - スパイク検出の精度評価

### Phase 4: 評価と調整（1日）
- [ ] 数学実験での再評価
- [ ] 閾値の最適化
- [ ] パフォーマンス測定

## 4. 期待される効果

### 4.1 IG感度の向上
- 現在: 0.01〜0.1の変化
- 期待: 0.1〜0.5の変化（5〜50倍の改善）

### 4.2 GEDの安定性
- 現在: グラフサイズで1〜100まで変動
- 期待: [-1, 1]の安定した範囲

### 4.3 スパイク検出精度
- 現在: 数学実験で0%
- 期待: 60%以上の検出率

## 5. 実装詳細

### 5.1 情報拡散アルゴリズム
```python
def diffuse_information(graph, initial_values, alpha=0.15, steps=3):
    """
    PageRankライクな情報拡散
    
    Args:
        graph: ネットワーク構造
        initial_values: 初期"驚き"値
        alpha: 減衰率
        steps: 拡散ステップ数
    """
    # 遷移行列の作成
    # 反復的な情報拡散
    # 収束値の返却
```

### 5.2 構造効率の計算
```python
def calculate_structure_efficiency(graph):
    """
    グラフの構造効率を計算
    
    考慮する指標:
    - Global efficiency: 平均最短経路長の逆数
    - Clustering coefficient: 局所的な密結合度
    - Hub centrality: ハブ構造の形成度
    """
```

## 6. 後方互換性の維持

### 6.1 設定による切り替え
```yaml
# config.yaml
graph:
  # 新しいIG実装を使用
  use_local_ig: true
  # 正規化GEDを使用
  use_normalized_ged: true
  # 統合スコアベースのスパイク検出
  use_unified_spike_score: true
  
  # 後方互換性のための旧閾値
  legacy_spike_ged_threshold: -0.5
  legacy_spike_ig_threshold: 0.2
```

### 6.2 互換性レイヤー
```python
class BackwardCompatibleMetrics:
    """旧形式の値を返す互換性レイヤー"""
    
    def convert_normalized_to_legacy(normalized_ged):
        # [-1, 1] → 旧スケールへの変換
        return normalized_ged * 10
    
    def convert_local_to_legacy_ig(local_ig_result):
        # 新形式 → 旧形式への変換
        return local_ig_result['total_ig'] * 0.2
```

## 7. リスクと対策

### リスク
1. **計算コストの増加**: 情報拡散シミュレーションは計算量が多い
2. **パラメータ調整**: 新しいパラメータの最適値が不明
3. **既存実験への影響**: 結果が大きく変わる可能性

### 対策
1. **最適化**: NumPyベクトル化、疎行列演算の活用
2. **段階的調整**: デフォルトは保守的な値、実験で最適化
3. **フィーチャーフラグ**: 段階的な移行を可能に

## 8. 成功基準

1. **数学実験でのスパイク検出**: 5問中3問以上で検出
2. **パフォーマンス**: 処理時間の増加を2倍以内に抑制
3. **後方互換性**: 既存の全実験が動作

## 9. タイムライン

- Day 1-2: コア実装
- Day 3: 統合作業
- Day 4-5: テスト実装と実行
- Day 6: 評価と最終調整

合計: 6日間

## 10. 参考文献

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Tononi, G. (2008). Consciousness as integrated information
- Shannon, C. E. (1948). A mathematical theory of communication
- Page, L. et al. (1999). The PageRank citation ranking