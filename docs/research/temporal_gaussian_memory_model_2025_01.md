# 時系列ガウス分布記憶モデル (2025-01-25)

## 概要
差分記憶の効率化に関する議論から生まれた、人間のエピソード記憶メカニズムに基づく新しい記憶モデルの提案。

## 背景
- 差分記憶を実装する際、いちいち前メモリを参照することになり効率が悪い
- 特許申請した階層ベクトル量子化（VQ）による差分記憶の実装を検討
- Layer1でのgeDIG実装と組み合わせた効率化が必要

## 核心的洞察
「差分をトリガーにEPまるっと記憶してるのかもね。んで、時系列キートリガーがあって、それを中心に時間のガウス分布で覚えてる濃度が変わるイメージ。」

## モデルの詳細

### 1. 差分トリガーによる完全エピソード記憶
- 大きな差分（驚き）が検出されたとき、エピソード全体を記憶
- 差分の大きさが閾値を超えた瞬間がキートリガーとなる
- 前メモリとの比較は記憶時のみ、想起時は不要

### 2. 時系列ガウス分布モデル
```python
# 記憶強度の計算
def memory_strength(query_time, key_triggers):
    """
    キートリガーを中心としたガウス分布の重ね合わせ
    各トリガーからの時間距離に応じて記憶強度が減衰
    """
    strength = 0.0
    for trigger in key_triggers:
        time_diff = abs(query_time - trigger["timestamp"])
        sigma = trigger["diff_magnitude"] * TIME_SCALE  # 差分の大きさに比例
        gaussian = np.exp(-(time_diff**2) / (2 * sigma**2))
        strength += gaussian * trigger["diff_magnitude"]
    return strength
```

### 3. 人間の記憶現象との対応

#### フラッシュバルブ記憶
- 大きな驚き（差分）→ 鮮明で詳細な記憶
- 例：「9/11の時何してた？」→ その瞬間の詳細を覚えている

#### 時間的文脈依存
- キーイベント周辺の記憶も引きずられて想起される
- ガウス分布により、キートリガー付近の記憶も強化

#### 記憶の濃淡
- キートリガーに近い時間 = 鮮明な記憶
- 時間的に遠い = ぼんやりした記憶

## 実装上の利点

### 1. 効率性
- 想起時に前メモリ全参照が不要
- キートリガーとの時間距離計算のみで記憶強度を算出
- O(n)からO(k)への計算量削減（k=キートリガー数）

### 2. 生物学的妥当性
- 人間のエピソード記憶の特性を再現
- 感情的・認知的顕著性（差分の大きさ）と記憶強度の相関

### 3. 特許技術との整合性
- 階層VQによる差分エンコーディングと組み合わせ可能
- 信頼度（C値）をガウス分布のパラメータに反映可能

## 実装案

```python
class GaussianTemporalMemory:
    def __init__(self):
        self.episodes = {}  # id -> full episode
        self.key_triggers = []  # 差分トリガーのリスト
        
    def store_if_significant(self, episode, context):
        diff = calculate_diff(episode, context)
        if diff > SURPRISE_THRESHOLD:
            # 完全エピソードを保存
            self.episodes[episode.id] = episode
            # キートリガーとして記録
            self.key_triggers.append({
                "timestamp": now(),
                "diff_magnitude": diff,
                "episode_id": episode.id
            })
            
    def recall(self, query_time):
        # 時間ベースの想起（前メモリ参照不要）
        recalled = []
        for episode_id, episode in self.episodes.items():
            strength = self.memory_strength(query_time, episode.timestamp)
            if strength > RECALL_THRESHOLD:
                recalled.append((episode, strength))
        return sorted(recalled, key=lambda x: x[1], reverse=True)
```

## 今後の検討事項

1. **TIME_SCALEパラメータの最適化**
   - 差分の大きさとガウス分布の広がりの関係

2. **複数モダリティへの拡張**
   - 視覚的驚き、聴覚的驚きなど、モダリティ別の差分計算

3. **忘却メカニズムとの統合**
   - 時間経過によるキートリガーの減衰
   - ガーベジコレクションとの連携

4. **geDIG Layer1との統合**
   - グラフ構造の差分をキートリガーとして利用
   - 構造的驚きと時間的記憶の統合

## 関連ファイル
- `/docs/development/gedig_layer1_roadmap.md` - Layer1でのgeDIG実装計画
- `/docs/research/episode_splitting_merging_discussion_2025_01.md` - エピソード管理の議論
- 特許申請書類（階層ベクトル量子化によるメモリ圧縮）