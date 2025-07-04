# Layer2 エピソード統廃合仕様

## 概要
Layer2 Memory Manager (`L2MemoryManager`) は、エピソードの統合と分裂を管理します。

## 1. エピソード統合 (Integration)

### 統合判定プロセス (`_check_episode_integration`)

新しいエピソードを追加する際、既存エピソードとの統合を検討します：

1. **ベクトル類似度** (50% weight)
   - コサイン類似度を計算
   - 閾値: 0.85 (85%)

2. **コンテンツ重複度** (30% weight)
   - 単語集合の Jaccard 係数
   - 閾値: 0.70 (70%)

3. **C値互換性** (20% weight)
   - C値の差分から計算
   - 閾値: 0.30 (差分)

### 統合条件
以下のすべてを満たす場合に統合：
- ベクトル類似度 ≥ 0.85
- コンテンツ重複度 ≥ 0.70
- 統合スコア ≥ 0.65

### 統合処理 (`_integrate_with_existing`)

1. **ベクトル更新**
   - C値による重み付き平均
   - `integrated_vector = (c1 * v1 + c2 * v2) / (c1 + c2)`

2. **テキスト結合**
   - パイプ記号で連結: `"text1 | text2"`

3. **C値更新**
   - 最大値を採用: `max(c1, c2)`

4. **メタデータ記録**
   - `integration_history` に統合履歴を保存
   - `integration_count` をインクリメント

## 2. エピソード分裂 (Split)

### 分裂条件
- 手動呼び出しのみ（自動分裂なし）
- 2文以上のテキストが必要

### 分裂処理 (`split`)

1. **文単位で分割**
   - ピリオドで区切って分割

2. **新エピソード作成**
   - 元のベクトルに微小ノイズを追加
   - C値を0.8倍に減少
   - `split_from`, `split_part`, `split_total` メタデータを付与

3. **元エピソード削除**
   - 分割後、元のエピソードは削除

## 3. 現在の統合率

実験データから：
- **統合閾値が高い**: 類似度85%、重複70%は非常に厳しい条件
- **実際の統合率**: 約17%以下
- **C値が固定**: すべて0.5のため、C値による重み付けが機能していない

## 4. 問題点と改善案

### 問題点
1. 統合閾値が高すぎる
2. 自動分裂機能がない
3. C値が活用されていない

### 改善案
1. **閾値の調整**
   ```python
   similarity_threshold = 0.70  # 85% → 70%
   content_overlap_threshold = 0.50  # 70% → 50%
   ```

2. **自動分裂の実装**
   - テキスト長が閾値を超えたら自動分裂
   - 複雑度が高いエピソードを分割

3. **C値の動的更新**
   - 使用頻度やフィードバックに基づいて更新

## 5. API使用例

```python
# エピソード追加（自動統合判定）
episode_idx = memory.add_episode(vector, text, c_value)

# 手動統合
result = memory._integrate_with_existing(target_idx, new_vector, new_text, new_c_value)

# 手動分裂
new_indices = memory.split(episode_index)
```

## 6. 設定可能パラメータ

```python
# config.reasoning に設定
episode_integration_similarity_threshold = 0.85
episode_integration_content_threshold = 0.70
episode_integration_c_threshold = 0.30
```