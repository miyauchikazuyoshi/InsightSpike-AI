---
status: active
category: edges
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# 理想的なノード・エッジ構造設計

## 設計原則

1. **測定優先**: まず測定し、データに基づいて拡張
2. **段階的拡張**: コアから始めて必要に応じて追加
3. **デバッグ可能**: 各パラメータの影響を個別に検証可能

## 理想の3段階構造

### Stage 1: 測定コア（今すぐ実装）
```python
@dataclass
class MeasurableEpisode:
    """測定可能な最小限のエピソード"""
    # === 既存 ===
    text: str
    embedding: np.float32  # fp16は後で
    c_value: float
    
    # === 測定に必須 ===
    episode_id: int  # 追跡用の一意ID
    created_at: float
    last_accessed_at: float
    access_count: int = 0
    
    # === メトリクス ===
    byte_size: int  # メモリ使用量追跡
    response_time_ms: float = 0.0  # このエピソードの処理時間
    
    def log_access(self, response_time: float):
        """アクセスを記録"""
        self.access_count += 1
        self.last_accessed_at = time.time()
        self.response_time_ms = response_time

@dataclass
class MeasurableEdge:
    """測定可能なエッジ"""
    source_id: int
    target_id: int
    weight: float  # 類似度
    
    # === 使用統計 ===
    traversal_count: int = 0  # 何回このエッジを辿ったか
    last_used: float = 0.0
    
    def log_traversal(self):
        self.traversal_count += 1
        self.last_used = time.time()
```

### Stage 2: 洞察検出層（データ収集後）
```python
@dataclass  
class InsightAwareEpisode(MeasurableEpisode):
    """洞察検出機能を持つエピソード"""
    # === 洞察メトリクス ===
    surprise_score: float = 0.0  # 予想外度（ΔIGの簡易版）
    integration_score: float = 0.0  # 他の知識との統合度
    
    # === 簡易履歴（Ring Bufferは複雑すぎ）===
    recent_surprise_scores: List[float] = field(default_factory=lambda: [0.0] * 3)
    
    def update_surprise(self, context_similarity: float):
        """文脈との差異から驚き度を計算"""
        self.surprise_score = 1.0 - context_similarity
        self.recent_surprise_scores.append(self.surprise_score)
        self.recent_surprise_scores.pop(0)
        
    @property
    def is_insight_spike(self) -> bool:
        """洞察スパイクかどうか"""
        # 急激な surprise_score の上昇
        if len(self.recent_surprise_scores) >= 2:
            return self.recent_surprise_scores[-1] > self.recent_surprise_scores[-2] * 1.5
        return False

@dataclass
class InsightEdge(MeasurableEdge):
    """洞察の関係性を表現するエッジ"""
    edge_type: str = "similarity"  # similarity, temporal, causal
    confidence: float = 1.0  # この関係の確信度
    
    # 因果関係の方向性（causalの場合のみ）
    causality_strength: float = 0.0  # -1.0 ~ 1.0
```

### Stage 3: 自己最適化層（十分なデータ蓄積後）
```python
@dataclass
class SelfOptimizingEpisode(InsightAwareEpisode):
    """自己最適化機能を持つエピソード"""
    # === 圧縮状態 ===
    compression_level: int = 0  # 0: 非圧縮, 1: 軽圧縮, 2: 重圧縮
    original_size: int = 0
    
    # === 予測メトリクス ===
    predicted_access_probability: float = 0.5  # 今後アクセスされる確率
    predicted_insight_potential: float = 0.5  # 新しい洞察を生む可能性
    
    def should_compress(self) -> int:
        """圧縮レベルを決定"""
        if self.predicted_access_probability < 0.1:
            return 2  # 重圧縮
        elif self.predicted_access_probability < 0.3:
            return 1  # 軽圧縮
        return 0  # 非圧縮
        
    def should_evict(self, memory_pressure: float) -> bool:
        """エビクション判定"""
        # メモリ圧力と価値のバランス
        value_score = (
            0.4 * self.c_value +
            0.3 * self.predicted_insight_potential +
            0.3 * self.predicted_access_probability
        )
        return value_score < memory_pressure * 0.5
```

## なぜこの設計が理想的か

### 1. 段階的に検証可能
```python
# Stage 1で十分なケースも多い
basic_memory = MemoryManager(episode_class=MeasurableEpisode)
# → アクセスパターンを分析

# 必要に応じてStage 2へ
if needs_insight_detection:
    insight_memory = MemoryManager(episode_class=InsightAwareEpisode)
# → 洞察検出の効果を測定

# 本当に必要ならStage 3へ
if needs_self_optimization:
    advanced_memory = MemoryManager(episode_class=SelfOptimizingEpisode)
```

### 2. 各パラメータが明確な目的を持つ
- **測定系**: 実際の使用状況を把握
- **洞察系**: InsightSpikeの本質機能
- **最適化系**: リソース効率化

### 3. デバッグが容易
```python
# 問題の切り分けが簡単
if memory_explosion:
    check_stage1_metrics()  # 基本的なアクセスパターン
elif poor_insight_detection:
    check_stage2_metrics()  # surprise_scoreの推移
elif inefficient_resource:
    check_stage3_metrics()  # 圧縮・エビクション判定
```

## 実装ロードマップ

### Phase 1（1週間）
- MeasurableEpisodeの実装
- メトリクス収集開始
- ダッシュボード作成

### Phase 2（1ヶ月後）
- データ分析
- InsightAwareEpisodeの設計調整
- A/Bテスト実施

### Phase 3（3ヶ月後）
- 十分なデータに基づいて最適化層を設計
- 機械学習モデルでpredicted_*を計算

## ChatGPTの提案との違い

| 観点 | ChatGPT | 私の理想 |
|-----|---------|---------|
| 初期複雑度 | 高（5つのメカニズム） | 低（測定のみ） |
| 理論重視度 | 高（ΔGED等） | 低（実データ重視） |
| 実装難易度 | 高 | 段階的に増加 |
| デバッグ性 | 困難 | 容易 |
| 拡張性 | 事前設計 | データ駆動 |

## 結論

理想は「**測定から始めて、データに基づいて進化する構造**」です。

ChatGPTの理論的に優れた設計も、実際のデータなしには机上の空論になりがちです。
まず測定し、本当に必要な機能だけを追加していく。これが実用的な理想です。

```python
# 最初の一歩
class IdealStart:
    """これだけから始める"""
    episode_id: int
    access_count: int
    response_time_ms: float
    
    # これだけで、どのエピソードが本当に価値があるか分かる
```