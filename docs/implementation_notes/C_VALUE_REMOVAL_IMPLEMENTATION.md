# C値削除の実装完了

## 実装内容

### 1. GraphCentricMemoryManager (`layer2_graph_centric.py`)

C値を完全に削除した新しいメモリマネージャー：

```python
@dataclass
class GraphEpisode:
    vec: np.ndarray
    text: str
    metadata: Dict[str, Any]
    # C値なし！
```

### 2. 動的な重要度計算

C値の代わりに、以下の要素から動的に重要度を計算：

```python
def get_importance(self, episode_idx: int) -> float:
    # 1. グラフ次数（接続数）
    graph_degree = count_connections(episode_idx)
    
    # 2. アクセス頻度
    access_score = log1p(access_count) / 10
    
    # 3. 時間的減衰
    time_decay = exp(-(current_time - last_access) / 86400)
    
    # 複合スコア
    importance = 0.4 * graph_degree + 0.3 * access_score + 0.3 * time_decay
```

### 3. 改善された統合処理

```python
# 旧: C値による重み付け（実質無意味）
integrated = (0.5*v1 + 0.5*v2) / 1.0

# 新: グラフ接続強度による重み付け
weight = graph_connection_strength or similarity
integrated = (1-weight)*v1 + weight*v2
```

### 4. 主な変更点

#### 削除されたもの
- Episode.c フィールド
- C値関連の設定（c_min, c_max, c_initial）
- C値による統合判定
- C値の更新処理

#### 追加されたもの
- 動的重要度計算
- アクセス追跡
- 時間的減衰
- グラフ構造ベースの重み付け

## メリット

1. **シンプル化**
   - 不要なパラメータを削除
   - コードが理解しやすい

2. **動的な振る舞い**
   - 重要度がグラフ構造から自動計算
   - アクセスパターンを反映

3. **統一されたアプローチ**
   - グラフ中心の設計に完全移行
   - Self-Attention的な動作を純粋に実現

4. **保守性向上**
   - 管理する状態が減少
   - バグの可能性が低下

## 使用例

```python
# 新しいマネージャー
manager = GraphCentricMemoryManager(dim=384)

# エピソード追加（C値は無視される）
idx = manager.add_episode(vector, text)  # c_value引数は不要

# 重要度は動的に計算
importance = manager.get_importance(idx)

# 検索時に重要度を考慮
results = manager.search_episodes(query, k=5)
```

## 互換性

後方互換性のため、`add_episode`メソッドは`c_value`引数を受け取りますが、単に無視します。これにより、既存のコードを変更せずに新しい実装を使用できます。

## まとめ

C値を削除することで、システムはよりシンプルで理解しやすくなり、グラフ構造の利点を最大限に活用できるようになりました。動的な重要度計算により、実際の使用パターンに基づいた知識管理が可能になります。