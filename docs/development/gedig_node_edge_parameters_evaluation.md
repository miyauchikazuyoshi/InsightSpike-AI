# geDIG ノード/エッジパラメータの実装評価

## ChatGPT提案の構造

### ノード（Episode）パラメータ
```python
@dataclass
class GeDIGNode:
    # 必須フィールド
    node_id: uint32              # 4B - ✅ 必須
    embedding: np.float16[768]   # 1.5KB - ✅ 必須（既に実装済み）
    level: NodeLevel             # enum: raw/episode/centroid - 🤔 要検討
    delta_ig_hist: RingBuffer[8] # 16B - ⚠️ 実装複雑
    access_ts: uint32            # 4B - ✅ 有用
    predictive_entropy: float16  # 2B - 🤔 計算方法が不明確
    
    # オプション
    summary: Optional[str]       # ✅ 既にEpisodeにtext fieldあり
    size_tokens: uint16          # ✅ 圧縮判定に有用
```

### エッジパラメータ
```python
@dataclass 
class GeDIGEdge:
    src: uint32          # 4B - ✅ 必須
    dst: uint32          # 4B - ✅ 必須
    weight_sim: float16  # 2B - ✅ 類似度（既にScalableGraphBuilderで使用）
    cooccur_cnt: uint16  # 2B - 🤔 更新処理が必要
    edge_type: EdgeType  # enum - ⚠️ 型分類の基準が不明確
```

## 実装観点からの評価

### 👍 採用すべきパラメータ

1. **node_id, embedding, access_ts**
   - 既存実装と整合性あり
   - メモリ効率的（fp16使用）
   - LRUエビクションに必須

2. **weight_sim（エッジ）**
   ```python
   # 既にScalableGraphBuilderで実装済み
   similarity = np.dot(embed1, embed2) / (norm1 * norm2)
   if similarity > threshold:
       graph.add_edge(i, j, weight=similarity)
   ```

3. **size_tokens**
   - 圧縮対象の選定に有用
   - 既存のmetadataに追加可能

### 🤔 要検討パラメータ

1. **level (raw/episode/centroid)**
   ```python
   # 問題点：3階層は複雑すぎる？
   # 代案：2階層（active/archived）で十分かも
   class NodeStatus(Enum):
       ACTIVE = "active"      # メモリ内
       COMPRESSED = "compressed"  # 圧縮済み
   ```

2. **predictive_entropy**
   ```python
   # 問題点：計算方法が不明確
   # 代案：既存のuncertainty計算を流用
   def calculate_entropy(self, embedding):
       # Layer1StreamProcessorの実装を使用
       return self._calculate_entropy(embedding)
   ```

3. **delta_ig_hist (履歴)**
   ```python
   # 問題点：RingBufferの実装とメモリ管理
   # 代案：最新N件のみ保持
   class EpisodeWithHistory:
       def __init__(self):
           self.ig_scores = deque(maxlen=5)  # 最新5件
   ```

### ⚠️ 実装が複雑なパラメータ

1. **edge_type分類**
   ```python
   # 問題点：semantic/causal/temporalの判定ロジック
   # 現実的には：
   class SimpleEdgeType(Enum):
       SIMILARITY = "similarity"  # 類似度ベース（既存）
       TEMPORAL = "temporal"      # 時系列（追加可能）
       # causalは判定が困難なので保留
   ```

2. **cooccur_cnt（共起回数）**
   ```python
   # 問題点：インクリメンタル更新の実装
   # いつ・どのように更新？
   def update_cooccurrence(self, node1, node2):
       # 同じコンテキストで参照されたらカウント？
       # 実装の複雑さ vs 効果が不明
   ```

## 実装提案：段階的導入

### Phase 1: 最小限の拡張（すぐ実装可能）
```python
@dataclass
class EnhancedEpisode(Episode):
    """既存Episodeの拡張"""
    access_count: int = 0
    last_access_ts: float = 0.0
    token_count: int = 0
    compression_score: float = 0.0  # 圧縮優先度
    
    def update_access(self):
        self.access_count += 1
        self.last_access_ts = time.time()
```

### Phase 2: グラフ構造の強化
```python
class EnhancedGraphBuilder(ScalableGraphBuilder):
    def build_graph_with_metadata(self, embeddings, episodes):
        # 既存の類似度エッジ
        super().build_graph(embeddings)
        
        # 時系列エッジを追加
        for i in range(len(episodes) - 1):
            self.graph.add_edge(
                i, i+1, 
                weight=0.5,
                edge_type="temporal"
            )
```

### Phase 3: ΔIG/ΔGED計算（効果測定後）
```python
class DeltaMetricsTracker:
    def __init__(self):
        self.ig_history = defaultdict(lambda: deque(maxlen=5))
        
    def calculate_delta_ig(self, episode_id, new_ig):
        history = self.ig_history[episode_id]
        if history:
            delta = new_ig - history[-1]
        else:
            delta = new_ig
        history.append(new_ig)
        return delta
```

## メモリ使用量の現実的な見積もり

```python
# 10万ノードでの使用量
base_episode = 1.5  # KB (embedding)
metadata = 0.1      # KB (追加フィールド)
graph_overhead = 0.05  # KB (エッジ情報)

total_per_node = base_episode + metadata + graph_overhead  # 1.65 KB
total_memory = 100_000 * total_per_node / 1024  # 約161 MB

# 結論：ChatGPTの見積もり（150-200MB）は妥当
```

## 推奨事項

### 採用すべき最小セット
1. **ノード**: node_id, embedding(fp16), access_ts, token_count
2. **エッジ**: src, dst, weight_sim
3. **追加**: compression_score（エビクション用）

### 段階的に追加検討
1. **Phase 2**: temporal edges, access_count
2. **Phase 3**: delta_ig（単純な差分のみ）
3. **将来**: hierarchical levels（必要性が証明されたら）

### 避けるべき複雑性
- 多層的なlevel管理
- 複雑なedge_type分類
- リアルタイムでのcooccur_cnt更新

## 実装例：最小限の拡張

```python
# src/insightspike/core/episode_enhanced.py
from dataclasses import dataclass, field
from typing import Dict, Any
import time
import numpy as np

@dataclass
class EnhancedEpisode:
    """geDIG対応の拡張エピソード"""
    # 既存フィールド
    text: str
    embedding: np.ndarray  # fp16推奨
    c_value: float = 0.5
    
    # geDIG拡張フィールド（最小限）
    episode_id: int = field(default_factory=lambda: int(time.time() * 1000000))
    access_ts: float = field(default_factory=time.time)
    token_count: int = 0
    
    # メタデータ（既存互換）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_node_dict(self) -> Dict[str, Any]:
        """PyG用のノード特徴量辞書"""
        return {
            'x': self.embedding,
            'access_ts': self.access_ts,
            'c_value': self.c_value,
            'token_count': self.token_count,
        }
```

この最小限の拡張なら、既存システムへの影響を最小限に抑えつつ、geDIGの核心機能を実現できます。