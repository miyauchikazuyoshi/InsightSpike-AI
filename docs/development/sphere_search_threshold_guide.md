---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Sphere Search Threshold Guide

## 推奨閾値範囲（次元考慮版）

### 1. 3D直感ベース（推奨）

**球体検索（intuitive_radius）:**
- **非常に狭い**: 0.2-0.3（体積比: 0.8%-2.7%）
  - ほぼ同一の概念のみ
  - 例：「りんご」→「リンゴ」「apple」
  
- **狭い**: 0.3-0.4（体積比: 2.7%-6.4%）
  - 密接に関連する概念
  - 例：「りんご」→「果物」「赤い」
  
- **標準**: 0.4-0.6（体積比: 6.4%-21.6%）
  - 一般的な関連性
  - 例：「りんご」→「食べ物」「健康」「農業」
  
- **広い**: 0.6-0.8（体積比: 21.6%-51.2%）
  - 緩い関連性も含む
  - 例：「りんご」→「Newton」「iPhone」「赤」

**ドーナツ検索（inner/outer）:**
- **既知フィルタ（内側）**: 0.15-0.25
  - あまりに自明な情報を除外
  
- **関連性（外側）**: 0.4-0.7
  - 処理する価値のある範囲

### 2. 体積比ベース

```python
# 推奨設定
volume_fractions = {
    "very_selective": 0.001,    # 0.1%
    "selective": 0.01,          # 1%
    "normal": 0.1,              # 10%
    "inclusive": 0.3,           # 30%
    "very_inclusive": 0.5       # 50%
}
```

### 3. 次元別の実際の半径

| 3D直感 | 体積比 | 768次元での実半径 | 384次元での実半径 |
|--------|--------|-------------------|-------------------|
| 0.25   | 1.6%   | 0.984            | 0.968            |
| 0.40   | 6.4%   | 0.993            | 0.987            |
| 0.50   | 12.5%  | 0.9954           | 0.991            |
| 0.60   | 21.6%  | 0.997            | 0.994            |
| 0.75   | 42.2%  | 0.9985           | 0.997            |

### 4. タスク別推奨設定

**厳密な検索（学術・医療）:**
```yaml
wake_sleep:
  wake:
    search:
      method: "donut"
      # 3D直感値で指定
      intuitive_inner_radius: 0.1   # 非常に近いものを除外
      intuitive_outer_radius: 0.35  # 厳密な関連性のみ
```

**一般的な質問応答:**
```yaml
wake_sleep:
  wake:
    search:
      method: "sphere"
      intuitive_radius: 0.5  # バランスの良い検索
```

**創造的な探索:**
```yaml
wake_sleep:
  wake:
    search:
      method: "sphere"
      intuitive_radius: 0.7  # 広範囲の関連性
```

### 5. 実装での使い方

```python
from insightspike.query.dimension_aware_sphere_search import DimensionAwareSphereSearch

# 初期化
searcher = DimensionAwareSphereSearch(node_vectors)

# タスクに応じた検索
if task_type == "precise":
    # 厳密: 体積比1%以内
    results = searcher.search_intuitive(query, intuitive_radius=0.3)
    
elif task_type == "balanced":
    # バランス: 体積比12.5%（1/8）
    results = searcher.search_intuitive(query, intuitive_radius=0.5)
    
elif task_type == "creative":
    # 創造的: 体積比~40%
    results = searcher.search_intuitive(query, intuitive_radius=0.7)

# ドーナツ検索の場合
if use_donut:
    results = searcher.adaptive_donut_search(
        query,
        inner_volume_fraction=0.001,  # 0.1%を除外
        outer_volume_fraction=0.1      # 10%以内を検索
    )
```

### 6. チューニングのコツ

1. **開始時は控えめに**
   - intuitive_radius: 0.4-0.5から始める
   - 結果を見て調整

2. **ドメインによる調整**
   - 専門分野: より狭く（0.3-0.4）
   - 一般会話: より広く（0.5-0.7）

3. **フィードバックループ**
   ```python
   # 結果の質を測定
   if too_many_irrelevant:
       intuitive_radius *= 0.9  # 10%狭める
   elif too_few_results:
       intuitive_radius *= 1.1  # 10%広げる
   ```

### 7. 高度な設定

**動的調整:**
```python
class AdaptiveRadiusManager:
    def __init__(self):
        self.base_radius = 0.5
        self.performance_history = []
    
    def adjust_radius(self, query_complexity):
        # 複雑なクエリは広めに
        if query_complexity > 0.7:
            return self.base_radius * 1.2
        # シンプルなクエリは狭めに
        else:
            return self.base_radius * 0.8
```

## まとめ

- **3D直感値 0.3-0.6** が多くの場合で有効
- 高次元でも**体積比**で考えれば直感的
- タスクの性質に応じて調整
- ドーナツ検索は内側0.1-0.2、外側0.4-0.7が標準的

---
*この設定により、768次元空間でも「半径半分」のような直感的な指定が可能になります。*