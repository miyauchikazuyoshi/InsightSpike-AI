---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# メモリ爆発防止設計

## 問題
- 概念分裂：矛盾検出により1エピソード→2エピソード以上
- 洞察追加：スパイク検出ごとに新エピソード
- 結果：エピソード数が指数関数的に増加する可能性

## 現在の制御メカニズム

### 1. C値による自然淘汰
```python
# 低C値エピソードの定期的な刈り取り
def prune_low_value_episodes(self, threshold: float = 0.1):
    """C値が閾値以下のエピソードを削除"""
    pruned = [e for e in self.episodes if e.c_value < threshold]
    # 削除処理
```

### 2. エピソード上限設定
```yaml
memory:
  max_episodes: 10000  # ハードリミット
  prune_threshold: 0.1
  prune_interval: 1000  # 1000エピソードごとに刈り取り
```

## 追加の防止策

### 1. 分裂の抑制
```python
def should_split_episode(self, conflicts: List[Dict]) -> bool:
    """分裂条件を厳格化"""
    # 現在のエピソード数を考慮
    current_count = len(self.episodes)
    max_count = self.config.memory.max_episodes
    
    # 容量に余裕がない場合は分裂を抑制
    if current_count > max_count * 0.8:  # 80%以上使用
        # より厳しい条件（3つ以上の深刻な矛盾）
        return len([c for c in conflicts if c['severity'] > 0.9]) >= 3
    
    # 通常の条件
    return len([c for c in conflicts if c['severity'] > 0.7]) >= 2
```

### 2. 洞察の重複チェック
```python
def should_create_insight_episode(self, insight_text: str) -> bool:
    """類似の洞察が既に存在しないかチェック"""
    # 既存の洞察エピソードを検索
    existing_insights = [
        e for e in self.episodes 
        if e.text.startswith("[INSIGHT]")
    ]
    
    # ベクトル類似度で重複チェック
    insight_vector = self.embedder.embed_text(insight_text)
    for existing in existing_insights:
        similarity = cosine_similarity(insight_vector, existing.vector)
        if similarity > 0.9:  # ほぼ同じ洞察
            # 既存のC値を少し上げる（再発見による確信度向上）
            existing.c_value = min(existing.c_value + 0.05, 1.0)
            return False  # 新規作成しない
    
    return True
```

### 3. 適応的な刈り取り
```python
def adaptive_pruning(self):
    """メモリ使用量に応じて刈り取り閾値を調整"""
    usage_ratio = len(self.episodes) / self.max_episodes
    
    if usage_ratio > 0.9:  # 90%以上
        # 積極的な刈り取り
        self.prune_low_value_episodes(threshold=0.3)
    elif usage_ratio > 0.7:  # 70%以上
        # 中程度の刈り取り
        self.prune_low_value_episodes(threshold=0.2)
    else:
        # 通常の刈り取り
        self.prune_low_value_episodes(threshold=0.1)
```

### 4. エピソード統合の活用
```python
def merge_similar_episodes(self, threshold: float = 0.95):
    """高類似度エピソードを統合してメモリ節約"""
    merged_count = 0
    
    # FAISSで高速に類似ペアを検出
    for i, episode in enumerate(self.episodes):
        if episode.c_value < 0.5:  # 低信頼度は統合対象外
            continue
            
        # k=5で近傍検索
        distances, indices = self.index.search(episode.vector, k=5)
        
        for dist, idx in zip(distances[0][1:], indices[0][1:]):  # 自分自身を除く
            if dist < (1.0 - threshold):  # 類似度が閾値以上
                # 統合実行
                self.merge_episodes([i, idx])
                merged_count += 1
                break
    
    return merged_count
```

### 5. 世代管理
```python
class Episode:
    def __init__(self, text: str, **kwargs):
        self.text = text
        self.generation = kwargs.get('generation', 0)  # 世代番号
        self.last_accessed = time.time()
        # ...

def generational_gc(self, max_generations: int = 10):
    """古い世代のエピソードを優先的に削除"""
    current_gen = max(e.generation for e in self.episodes)
    
    # 古い世代から削除候補を選択
    for gen in range(0, current_gen - max_generations):
        old_episodes = [
            e for e in self.episodes 
            if e.generation == gen and e.c_value < 0.5
        ]
        # 削除処理
```

## メモリ使用量の見積もり

### エピソード1つあたり
- テキスト: ~500 bytes
- ベクトル: 384 * 4 = 1,536 bytes (float32)
- メタデータ: ~200 bytes
- 合計: ~2.2 KB/エピソード

### 10,000エピソードで
- 約22 MB（許容範囲）

### 100,000エピソードで
- 約220 MB（要検討）

## 推奨設定

```yaml
memory:
  # エピソード管理
  max_episodes: 10000
  prune_threshold: 0.15
  prune_interval: 500
  
  # 分裂制御
  split_threshold: 0.8  # より厳しく
  max_splits_per_episode: 3  # 1エピソードの最大分裂回数
  
  # 洞察管理
  insight_similarity_threshold: 0.9  # 重複防止
  max_insights_per_session: 50  # セッションあたりの上限
  
  # 統合促進
  merge_threshold: 0.95
  merge_interval: 1000  # 1000エピソードごとに統合チェック
```

## 結論

メモリ爆発は防げる：
1. **積極的な刈り取り**で低価値エピソードを削除
2. **重複チェック**で無駄な洞察作成を防止
3. **適応的閾値**でメモリ使用量を制御
4. **エピソード統合**で類似知識を圧縮
5. **世代管理**で古い知識を整理

これらの組み合わせで、有用な知識を保持しつつメモリ使用量を制御可能。