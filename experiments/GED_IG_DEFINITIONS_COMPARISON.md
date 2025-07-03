# GED・IG定義の比較：実験実装 vs InsightSpike-AI本実装

## 1. Graph Edit Distance (GED)

### InsightSpike-AI本来の定義
```python
# グラフ編集距離：一つのグラフを別のグラフに変換する最小編集コスト
GED(G1, G2) = min Σ(cost(operation))

# 洞察検出のためのΔGED
ΔGED = GED(G_after, G_reference) - GED(G_before, G_reference)
```

**特徴：**
- NetworkXグラフ構造を使用
- ノード・エッジの追加・削除コストを考慮
- 3つの最適化レベル（FAST/STANDARD/PRECISE）
- **負のΔGED**（構造の単純化）が洞察を示す

### 実験での簡略実装
```python
# 状態間のL2ノルム（ユークリッド距離）
diversity = np.linalg.norm(state - self.prev_state)
```

**問題点：**
- グラフ構造を考慮していない
- 単純な状態差分のみ
- 構造的な単純化を測定できない
- 常に正の値（本来は負値が重要）

## 2. Information Gain (IG)

### InsightSpike-AI本来の定義
```python
# 情報利得：エントロピーの減少量
IG(S, A) = H(S) - Σ(|Sv|/|S|) × H(Sv)

# シャノンエントロピー
H(S) = -Σ p(x) log₂ p(x)

# 洞察検出のためのΔIG
ΔIG = IG_after - IG_before
```

**特徴：**
- 4つのエントロピー計算手法
- データの不確実性の減少を測定
- **正のΔIG**（情報の獲得）が洞察を示す

### 実験での簡略実装
```python
# 訪問回数の逆数（新奇性）
novelty = 1.0 / (1.0 + self.state_visits[state_key])
```

**問題点：**
- エントロピー計算なし
- 単純な訪問頻度のみ
- 情報理論的な基盤がない
- スケールが異なる（0-1の範囲）

## 3. geDIG (ΔGED × ΔIG)

### InsightSpike-AI本来の洞察検出
**EurekaSpike（洞察の瞬間）の条件：**
- ΔGED ≤ -0.5（構造の大幅な単純化）
- ΔIG ≥ 0.2（情報の大幅な獲得）

**脳科学的意味：**
- 知識グラフの再構造化（神経回路の再配線）
- 情報内容の増加（シナプス強化）
- "Aha!"モーメントの検出

### 実験での簡略実装
```python
# 単純な掛け算
intrinsic_reward = novelty * diversity
```

**問題点：**
- 符号が逆（GEDは負値であるべき）
- スケールが統一されていない
- 閾値による洞察判定なし
- 脳科学的な意味が失われている

## 4. 改善提案

### 短期的改善（実験用）
```python
def _calculate_ged_simplified(self, state1, state2):
    """簡略化されたGED（負値を返す）"""
    # 状態の類似性（0-1）
    similarity = 1 - np.linalg.norm(state1 - state2) / (np.linalg.norm(state1) + np.linalg.norm(state2) + 1e-8)
    # 構造の単純化を表現（類似性が高いほど負のGED）
    return similarity - 1.0  # -1.0 to 0.0の範囲

def _calculate_ig_simplified(self, state):
    """簡略化されたIG（エントロピーベース）"""
    # 訪問分布からエントロピーを計算
    visits = np.array(list(self.state_visits.values()))
    if len(visits) == 0:
        return 0.0
    
    probs = visits / visits.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    
    # 新規状態の場合は高いIG
    state_key = tuple(state)
    if state_key not in self.state_visits:
        return entropy * 0.5  # スケール調整
    
    return entropy * (1.0 / (1.0 + self.state_visits[state_key]))
```

### 長期的改善（本実装統合）
1. **実際のInsightSpike-AI APIを使用**
   - NetworkXグラフへの状態変換
   - 正式なGED/IG計算

2. **洞察検出の実装**
   - ΔGED ≤ -0.5 かつ ΔIG ≥ 0.2 の判定
   - EurkaSpike発生時の特別な報酬

3. **脳科学的解釈の追加**
   - 知識構造の再編成をログ記録
   - 学習曲線での洞察モーメント可視化

## まとめ

現在の実験実装は計算効率を優先した簡略版であり、InsightSpike-AIの本来の脳科学的な意味（知識の再構造化による洞察の検出）が失われています。しかし、基本的な「新奇性×多様性」の概念は保持されており、内発的動機付けの効果を示すには十分でした。

今後の実験では、より忠実な実装により、真の「geDIG技術」の効果を検証することが望まれます。