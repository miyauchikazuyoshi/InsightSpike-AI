# コンフリクトベース自動分裂機能仕様

## 概要
グラフ構造のコンフリクトを検出し、エピソードを自動的に分裂させることで、知識グラフの一貫性を保つ機能。

## 1. コンフリクト検出

### 1.1 コンフリクトの種類

#### セマンティックコンフリクト
- 接続ノード間の意味的な相違
- 計算: `1 - cosine_similarity(vec1, vec2)`
- 閾値: 0.7以上で高コンフリクト

#### 方向性コンフリクト
- ベクトルが反対方向を向いている度合い
- 計算: `dot(vec1, -vec2) / (norm(vec1) * norm(vec2))`
- 正の値が大きいほど反対方向

#### クラスタコンフリクト
- 接続ノードが異なるトピッククラスタに属する度合い
- 計算: 近傍ベクトルの分散
- 高分散 = 高コンフリクト

### 1.2 総合コンフリクトスコア
```python
total_conflict = (
    0.5 * semantic_conflict +
    0.3 * directional_conflict +
    0.2 * cluster_conflict
)
```

## 2. 分裂判定条件

### 2.1 自動分裂のトリガー
1. **高コンフリクト**: 総合スコア ≥ 0.7
2. **十分な接続数**: 接続ノード ≥ 3
3. **分裂可能性**: 矛盾グループが2つ以上

### 2.2 追加条件
- **テキスト長**: 500文字以上で分裂候補
- **統合履歴**: 5回以上統合されたエピソードは分裂優先

## 3. 分裂プロセス

### 3.1 グループ識別
```python
# 類似度0.7以上のノードを同一グループに
for neighbor in neighbors:
    if similarity(neighbor1, neighbor2) > 0.7:
        same_group()
```

### 3.2 分裂生成
1. **テキスト分割**
   - 文単位で分割
   - 各グループに関連する文を割り当て

2. **ベクトル調整**
   ```python
   # 各分裂を対応グループ方向に調整
   split_vec = 0.7 * original_vec + 0.3 * group_center_vec
   ```

3. **C値減衰**
   - 分裂後のC値 = 元のC値 × 0.8

## 4. 実装例

```python
class ConflictBasedSplitter:
    def should_split_episode(self, episode_idx, graph):
        # 1. 接続ノードを取得
        neighbors = get_neighbors(episode_idx, graph)
        
        # 2. コンフリクト計算
        conflict = calculate_conflict(episode, neighbors)
        
        # 3. 分裂判定
        if conflict.total > 0.7 and len(neighbors) >= 3:
            groups = identify_groups(neighbors)
            if len(groups) >= 2:
                return True, groups
        
        return False, None
```

## 5. 期待される効果

### 5.1 グラフの改善
- **一貫性向上**: 矛盾する接続の解消
- **クラスタ品質**: より明確なトピッククラスタ
- **検索精度**: 関連性の高いノード同士の接続

### 5.2 メモリ管理の改善
- **統合精度向上**: 分裂により適切な統合先が明確に
- **知識の整理**: 混在したトピックの分離
- **スケーラビリティ**: 大規模化してもグラフ品質を維持

## 6. 設定パラメータ

```python
@dataclass
class SplitConfig:
    conflict_threshold: float = 0.7      # コンフリクト閾値
    min_connections: int = 3             # 最小接続数
    max_splits: int = 3                  # 最大分裂数
    split_decay: float = 0.8            # C値減衰率
    auto_split_enabled: bool = True      # 自動分裂の有効化
```

## 7. 統合との連携

### 7.1 分裂後の統合
- 分裂により生成されたエピソードは、より適切な既存エピソードと統合されやすくなる
- グラフ接続を考慮した統合により、トピックの一貫性が保たれる

### 7.2 フィードバックループ
```
新エピソード追加
    ↓
統合チェック（グラフ考慮）
    ↓
グラフ更新
    ↓
コンフリクト検出
    ↓
必要に応じて分裂
    ↓
グラフ再構築
```

これにより、知識グラフが自己組織化され、品質が継続的に向上します。