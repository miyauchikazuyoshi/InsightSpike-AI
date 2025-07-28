# GED符号反転問題リファクタリング計画書

## 1. 現状の問題点

### 1.1 概念的問題
- GED（Graph Edit Distance）は距離なのに負の値を返している
- 「構造の簡素化」を表現するために距離を負にするという概念的矛盾
- コードの可読性と保守性が低下

### 1.2 実装の混乱
```python
# 現在の問題のある実装例
if efficiency_delta > 0.1:
    return -float(ged)  # 距離を負にする？？
```

## 2. 影響範囲調査

### 2.1 コアアルゴリズム
- `/src/insightspike/algorithms/`
  - `graph_edit_distance_fixed.py` ⭐ 実際に使用中
  - `graph_edit_distance.py`
  - `proper_delta_ged.py`
  - `metrics_selector.py`

### 2.2 メトリクス層
- `/src/insightspike/metrics/`
  - `graph_metrics.py` - delta_ged()関数
  - `__init__.py` - GEDIG複合計算、Eureka Spike検出

### 2.3 グラフ解析層
- `/src/insightspike/features/graph_reasoning/`
  - `graph_analyzer.py` - detect_spike()メソッド

### 2.4 エージェント実装
- `/src/insightspike/implementations/`
  - `agents/main_agent.py` - スパイク検出結果の処理
  - `layers/layer3_graph_reasoner.py` - グラフ解析の呼び出し

### 2.5 テストコード
- `/tests/`
  - `integration/test_spike_detection_core.py`
  - `unit/test_graph_metrics.py`
  - その他多数のテスト

### 2.6 実験コード
- `/experiments/`
  - 各実験でスパイク検出を使用
  - 閾値設定（-0.5など）に依存

### 2.7 設定ファイル
- `config.yaml`
- 各種実験設定ファイル
- プリセット設定

## 3. 修正計画

### Phase 1: 新実装の追加（後方互換性維持）
1. `graph_structure_analyzer.py` ✅ 作成済み
2. `improved_gedig_metrics.py` ✅ 作成済み
3. 新しいテストケースの追加

### Phase 2: 段階的移行
1. メトリクスセレクターに新実装を追加
2. フィーチャーフラグで切り替え可能に
3. 実験的に新実装を試用

### Phase 3: 全面移行
1. すべてのスパイク検出を新実装に
2. 閾値の再調整
3. レガシーコードの削除

## 4. 修正内容詳細

### 4.1 新しいメトリクス構造
```python
@dataclass
class GEDIGMetrics:
    # 生のメトリクス（常に正）
    ged: float                    # Graph Edit Distance
    ig: float                     # Information Gain
    
    # 派生メトリクス（正負あり）
    structural_improvement: float  # 正 = 構造改善
    knowledge_coherence: float     # 0-1 = 知識の一貫性
    
    # 複合スコア
    insight_score: float          # 統合洞察スコア
    spike_detected: bool          # スパイク検出
```

### 4.2 スパイク検出の変更
```python
# 旧: GEDが負かつIGが正
spike = (delta_ged <= -0.5 and delta_ig >= 0.2)

# 新: 構造改善度とIGの正規化スコア
spike = (insight_score >= 0.6)
```

## 5. テスト計画

### 5.1 机上計算可能なテストケース

#### ケース1: 正方形＋中心ハブ
```
初期: 4ノード正方形（各ノード次数2）
追加: 中心ハブ（全ノードと接続）

期待値:
- GED = 5（4エッジ追加 + 1ノード追加）
- 構造効率性向上 = +0.4（推定）
- IG = +0.3（クラスタリング向上）
```

#### ケース2: 線形チェーン→スター
```
初期: A-B-C-D（線形）
変更: B中心のスター

期待値:
- GED = 3（エッジの付け替え）
- 構造効率性向上 = +0.6（直径減少）
- IG = +0.2
```

#### ケース3: 分離グラフ→接続
```
初期: (A-B) (C-D)（2つの分離成分）
追加: B-Cエッジ

期待値:
- GED = 1（エッジ追加のみ）
- 構造効率性向上 = +0.8（連結性確立）
- IG = +0.1
```

### 5.2 テスト実装方針
```python
def test_square_to_hub_pattern():
    """正方形→ハブ構造のテスト"""
    # 1. モックベクトル配置
    vectors = create_square_pattern(4)
    graph_before = build_graph(vectors)
    
    # 2. 中心ノード追加
    vectors_with_hub = add_center_hub(vectors)
    graph_after = build_graph(vectors_with_hub)
    
    # 3. メトリクス計算
    metrics = calculate_gedig_metrics(
        graph_before, graph_after,
        vectors, vectors_with_hub
    )
    
    # 4. 期待値との比較
    assert metrics.ged == pytest.approx(5.0, rel=0.1)
    assert metrics.structural_improvement > 0.3
    assert metrics.spike_detected == True
```

## 6. リスクと対策

### 6.1 後方互換性
- リスク: 既存の実験が動かなくなる
- 対策: レガシーラッパーの提供

### 6.2 閾値の再調整
- リスク: スパイク検出感度の変化
- 対策: 段階的移行と検証

### 6.3 パフォーマンス
- リスク: 新実装が遅い
- 対策: ベンチマークテストの実施

## 7. 実装スケジュール

### Week 1: 準備とテスト
- [ ] テストケースの実装
- [ ] 机上計算値の検証
- [ ] ベンチマーク環境構築

### Week 2: コア実装
- [ ] メトリクスセレクターの拡張
- [ ] フィーチャーフラグの実装
- [ ] 単体テストの実装

### Week 3: 統合テスト
- [ ] 既存テストの修正
- [ ] 新旧実装の比較検証
- [ ] パフォーマンステスト

### Week 4: 段階的移行
- [ ] 実験コードでの試用
- [ ] 閾値の調整
- [ ] ドキュメント更新

## 8. なぜ今までスパイクが検出されたか？

### 検証結果
`test_ged_calculation.py`の実行により判明：

1. **効率性向上でGEDが負になる設計**
   - 正方形→ハブ: GED = -5.0（実際は5だが符号反転）
   - 効率性: 0.583 → 0.830（+0.247向上）
   - `efficiency_delta > 0.1`のため、GEDを負にする

2. **グラフが空や小さい時に発生**
   - 検索結果0件 → 合成グラフ（1ノード）
   - 少数ノード → 効率性計算が極端な値
   - 分離グラフ → 接続で大幅な効率向上

3. **IGが正になる条件**
   - ベクトル空間でのクラスタリング改善
   - 新しい知識の追加
   - エピソード数の増加

### 実際のスパイク検出パターン
```python
# 典型的なケース
ΔGED = -5.0  # 構造効率化（負は改善を意味）
ΔIG = 0.3    # 情報増加
→ スパイク検出（両条件を満たす）
```