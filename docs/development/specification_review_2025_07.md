# InsightSpike 仕様再検討事項

Date: 2025-07-27

## 1. 設定システムの仕様再検討

### 現状の問題
- 3つの設定形式が混在（dict、Pydantic、legacy object）
- 各レイヤーが異なる形式を期待
- 変換ロジックが未実装

### 仕様の選択肢

**Option A: Pydantic一本化**
```python
# すべてPydanticモデルで統一
config = InsightSpikeConfig(
    processing=ProcessingConfig(...),
    memory=MemoryConfig(...),
    l4_config=LLMConfig(...)
)
# アクセスは常にオブジェクト形式
config.processing.enable_learning
```

**Option B: Dict一本化**
```python
# すべてdictで統一
config = {
    "processing": {...},
    "memory": {...}
}
# アクセスは常にdict形式
config.get("processing", {}).get("enable_learning")
```

**推奨**: Option A（Pydantic一本化）
- 型安全性
- IDE補完
- バリデーション
- 後方互換性は変換レイヤーで対応

## 2. エピソードのC値フィールド仕様

### 現状の問題
- `c`と`c_value`が混在
- 保存時と表示時で異なるフィールド使用
- 意味が不明確（confidence? correlation?）

### 仕様の明確化
```python
class Episode:
    text: str
    confidence: float  # 0.0-1.0, 信頼度
    timestamp: float
    metadata: dict
    vector: np.ndarray
    
    # 廃止: c, c_value, vec
```

### 移行計画
1. `c_value` → `confidence`に統一
2. 古いフィールドの読み込み互換性維持
3. 保存時は新形式のみ

## 3. CachedMemoryManagerの仕様

### 現状の問題
- `episodes`プロパティがキャッシュのみ返す
- 全エピソード取得の方法が不明確

### 仕様の明確化
```python
class CachedMemoryManager:
    @property
    def cached_episodes(self) -> List[Episode]:
        """キャッシュされたエピソードのみ"""
        
    def get_all_episodes(self) -> List[Episode]:
        """全エピソードを取得（遅い）"""
        
    def get_episode_count(self) -> int:
        """総数のみ取得（速い）"""
```

## 4. スパイク検出の仕様

### 現状の問題
- エージェントは`has_spike=False`と報告
- でもエピソードは作成される（真のスパイク）
- 検出基準が不明確

### 仕様の明確化

**スパイクの定義**：
1. **メトリクススパイク**: GED/IG/Conflictの閾値超え
2. **エピソードスパイク**: 新規洞察エピソード生成
3. **ユーザースパイク**: ユーザーが有用と判断

**検出ロジック**：
```python
spike_detected = (
    metrics_spike or      # メトリクスベース
    episode_spike or      # エピソード生成ベース
    confidence > 0.8      # 高信頼度
)
```

## 5. LLMプロバイダーの仕様

### 現状の問題
- MockProviderが実際の洞察を生成しない
- LocalProviderが未実装
- エラー時のフォールバック不明確

### 仕様の明確化

**プロバイダー階層**：
1. Primary: 設定されたプロバイダー
2. Fallback: CleanLLMProvider
3. Emergency: 固定レスポンス

**MockProviderの改善**：
```python
class ImprovedMockProvider:
    def generate_response(self, context, question):
        # 実際にコンテキストを解析
        # 簡単な洞察を生成
        if "relationship" in question:
            return f"Mock insight: Concepts are related through {random_relation}"
```

## 6. グラフ構築の仕様

### 現状の問題
- ScalableGraphManagerが使用されない理由不明
- Layer2との統合が不完全

### 仕様の確認事項
- ScalableGraphManagerは必須？オプション？
- どの条件で使用される？
- パフォーマンス要件は？

## 7. メモリ使用量の仕様

### 現状の問題
- 500MBで警告（低すぎる？）
- キャッシュサイズが1に減る（厳しすぎる？）

### 仕様の見直し
```python
MEMORY_THRESHOLDS = {
    "warning": 1024,  # 1GB
    "critical": 2048,  # 2GB
    "reduce_cache": 1536  # 1.5GB
}
```

## 8. データ永続化の仕様

### 現状の意図
- 実験中：一時ディレクトリ使用
- 実験後：スナップショット保存
- 本番：？

### 仕様の明確化
```python
class DataStoreConfig:
    mode: Literal["experiment", "production", "test"]
    base_path: Path
    auto_snapshot: bool
    snapshot_interval: int  # episodes
```

## 即座に決定が必要な仕様

1. **設定システム**: Pydantic一本化？
2. **C値の名前**: `confidence`？`c_value`？
3. **スパイク検出基準**: エピソード生成も含める？
4. **メモリ閾値**: 1GB？2GB？

## 実装優先順位

1. **Phase 1**: 仕様決定（1日）
2. **Phase 2**: 基本的なバグ修正（2-3日）
3. **Phase 3**: 仕様に基づくリファクタリング（1週間）
4. **Phase 4**: テスト追加（3-5日）

これらの仕様を決定してから実装に入りましょう。