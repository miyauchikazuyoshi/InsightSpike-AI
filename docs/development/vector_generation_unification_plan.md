---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# ベクトル生成処理の統一化計画

## 現状分析

### 洞察ベクトル生成（現在の実装）
- **場所**: `layer4_llm_interface.py`の`_create_insight_vector`
- **処理内容**:
  1. 文書の埋め込みベクトルを収集
  2. クエリベクトルとの類似度を計算
  3. クエリベクトル（重み1.0）と文書ベクトル（関連性重み）を統合
  4. 重み付き平均でベクトルを生成

### エピソード分岐ベクトル生成（episode_branching_architecture.md）
- **場所**: `aggregate_messages`関数
- **処理内容**:
  1. 親エピソードベクトル（重み0.4）
  2. 隣接ノードベクトル（重み0.6を分配）
  3. 重み付き平均で文脈特化ベクトルを生成

**本質的に同じプロセス！**

## 統一化の提案

### 1. 汎用ベクトル統合クラスの作成

```python
class VectorIntegrator:
    """
    汎用的なベクトル統合処理を提供
    """
    
    def integrate_vectors(self,
                         vectors: List[np.ndarray],
                         query_vector: Optional[np.ndarray] = None,
                         integration_type: str = "insight",  # "insight", "branching", "summary"
                         weights: Optional[List[float]] = None,
                         query_weight: float = 1.0) -> np.ndarray:
        """
        ベクトルを統合して単一のベクトルを生成
        
        Args:
            vectors: 統合するベクトルのリスト
            query_vector: クエリベクトル（オプション）
            integration_type: 統合タイプ
            weights: 各ベクトルの重み（Noneの場合は自動計算）
            query_weight: クエリベクトルの重み
            
        Returns:
            統合されたベクトル
        """
        pass
```

### 2. 統合タイプごとの設定

```python
INTEGRATION_CONFIGS = {
    "insight": {
        "primary_weight": 1.0,      # クエリベクトルの重み
        "secondary_weights": "similarity",  # 文書は類似度ベース
        "aggregation": "weighted_mean"
    },
    "episode_branching": {
        "primary_weight": 0.4,      # 親エピソードの重み
        "secondary_weights": "equal",  # 隣接ノードは均等分配
        "aggregation": "weighted_mean"
    },
    "context_merging": {
        "primary_weight": 0.5,
        "secondary_weights": "distance",  # 距離ベースの重み
        "aggregation": "weighted_mean"
    }
}
```

### 3. 統一化後の使用例

```python
vector_integrator = VectorIntegrator()

# 洞察ベクトル生成（Layer4）
insight_vector = vector_integrator.integrate_vectors(
    [doc["embedding"] for doc in retrieved_docs],
    primary_vector=query_vector,
    integration_type="insight"
)

# エピソード分岐ベクトル生成（CachedMemoryManager）
branch_vector = vector_integrator.integrate_vectors(
    [graph.nodes[n]['vec'] for n in neighbor_ids],
    primary_vector=parent_episode.vec,
    integration_type="episode_branching"
)

# メッセージパッシング後のノード更新（MessagePassing）
updated_vector = vector_integrator.integrate_vectors(
    neighbor_representations,
    primary_vector=node_representation,
    integration_type="message_passing",
    custom_weights=attention_weights  # カスタム重み
)
```

## 実装手順

1. **VectorIntegratorクラスの作成**
   - `/src/insightspike/core/vector_integrator.py`

2. **既存コードのリファクタリング**
   - `_create_insight_vector`をVectorIntegratorを使うように変更
   - 他の場所で同様の処理があれば統一

3. **設定の外部化**
   - 統合タイプごとの設定をconfigに追加
   - 動的な調整を可能に

4. **テストの作成**
   - 各統合タイプのテスト
   - 後方互換性の確認

## メリット

1. **コードの重複削減**
   - 同じロジックを複数箇所で実装する必要がない

2. **一貫性の確保**
   - すべてのベクトル統合が同じインターフェースを使用

3. **拡張性**
   - 新しい統合タイプを簡単に追加可能
   - パラメータの調整が容易

4. **テストの簡素化**
   - 単一のクラスをテストすればOK

## 影響範囲

- Layer4のコード変更（軽微）
- 新しいコアモジュールの追加
- 既存の動作に影響なし（リファクタリング）

## 実装優先度

中〜低：機能的には問題ないが、コードの保守性向上のため実装する価値あり