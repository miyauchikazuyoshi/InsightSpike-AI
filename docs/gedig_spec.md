# geDIG (Generalized Differential Information Gain) Specification

## 正準定義 (Canonical Definition)

### 基本式
```
geDIG = GED - IG
```

- **GED (Generalized Edit Distance)**: グラフ編集距離の一般化
- **IG (Information Gain)**: 情報利得

### 詳細計算式

#### GED計算
```python
GED = spatial_distance * 0.3 + 
      temporal_distance * 0.3 + 
      type_difference * 0.2 + 
      outcome_difference * 0.2
```

**各項の定義:**
- `spatial_distance`: マンハッタン距離 / (height + width)
- `temporal_distance`: |idx1 - idx2| / max(100, total_episodes)
- `type_difference`: 0.3 if type1 != type2 else 0
- `outcome_difference`: 0.2 if success1 != success2 else 0

#### IG計算
```python
IG = similarity * 0.5
```

- `similarity`: コサイン類似度 [0, 1]

### 値域と解釈
- **geDIG < 0**: 高い情報利得（IG > GED）→ 良好な学習
- **geDIG = 0**: 編集距離と情報利得が均衡
- **geDIG > 0**: 編集距離が大きい → 構造的な差異

## ハイパーパラメータ

| パラメータ名 | 記号 | 既定値 | 範囲 | 意味 |
|------------|------|--------|------|------|
| gedig_threshold | τ_gedig | 0.6 | [0.0, 1.0] | エッジ生成閾値 |
| gedig_weight | w_gedig | 0.3 | [0.0, 1.0] | 評価時の重み |
| spatial_weight | w_s | 0.3 | [0.0, 1.0] | 空間距離の重み |
| temporal_weight | w_t | 0.3 | [0.0, 1.0] | 時間距離の重み |
| type_weight | w_type | 0.2 | [0.0, 1.0] | タイプ差の重み |
| outcome_weight | w_out | 0.2 | [0.0, 1.0] | 成果差の重み |
| ig_factor | α_ig | 0.5 | [0.0, 1.0] | 情報利得の係数 |

## 使用コンテキスト

### 1. グラフエッジ生成時
```python
if gedig_value < gedig_threshold:
    # エッジを生成
    create_edge(node1, node2, weight=similarity, gedig=gedig_value)
```

### 2. エピソード評価時
```python
info_value = similarity - gedig_value
# info_value > 0 なら価値のある関連
```

### 3. メッセージパッシング時
```python
propagation_weight = (1.0 - min(gedig, 1.0)) * decay_factor
# geDIG値が低い（良い）ほど伝播が強い
```

### 4. 深度選択時
```python
if recent_gedig < -0.3:
    depth = 5  # 深い推論
elif recent_gedig < 0:
    depth = 4
elif recent_gedig < 0.3:
    depth = 3
elif recent_gedig < 0.5:
    depth = 2
else:
    depth = 1  # 浅い推論
```

## スパイク検出での役割

geDIG自体はスパイク検出の直接的な指標ではないが、グラフ構造の質を示す補助指標として使用される：

- **良好な学習状態**: 平均geDIG < 0
- **構造的な変化**: geDIG値の急激な変動
- **収束の兆候**: geDIG値の安定化

## 実装参照

### メインコード
- `/src/insightspike/implementations/layers/scalable_graph_builder.py`: _calculate_gedig()メソッド
- `/experiments/pure-movement-episodic-memory/src/pure_memory_agent_optimized.py`: 迷路実験での実装

### 評価での使用
- エピソード間の関連性評価
- グラフ構造の最適化
- 検索深度の適応的調整

## 注意事項

1. **正規化の重要性**: 各距離項は必ず[0, 1]に正規化すること
2. **重みの合計**: GEDの重み合計は1.0を維持
3. **類似度の計算**: コサイン類似度を使用（ユークリッド距離ではない）
4. **temporal_distanceの分母**: エピソード数が少ない初期は100を最小値として使用

## 更新履歴

- 2025-08-08: 初版作成
- 基準実装: pure_memory_agent_optimized.py (11x11迷路で60-80%成功率達成)