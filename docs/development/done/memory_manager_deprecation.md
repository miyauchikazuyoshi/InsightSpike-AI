# L2MemoryManager廃止とCachedMemoryManager移行

## 実施日
2025-07-26

## 背景
L2MemoryManagerは全エピソードを`self.episodes`リストに保持するため、メモリ爆発を引き起こす設計でした。これを解決するため、CachedMemoryManagerへの移行を実施しました。

## 変更内容

### 1. インポートパスの変更
以下のファイルでL2MemoryManagerの使用を停止し、CachedMemoryManagerに置き換えました：

- `/src/insightspike/implementations/layers/__init__.py`
  - L2MemoryManagerのインポートをコメントアウト
  - CachedMemoryManagerをエクスポートに追加

- `/src/insightspike/__init__.py`
  - L2MemoryManagerをCachedMemoryManagerのエイリアスとして設定（後方互換性）
  - 非推奨の警告を追加

### 2. エージェントの更新

#### MainAgent (`main_agent.py`)
```python
# 変更前
from ..layers.layer2_compatibility import CompatibleL2MemoryManager as Memory

# 変更後
from ..layers.cached_memory_manager import CachedMemoryManager as Memory
```

DataStoreが必須となり、初期化時にDataStoreがない場合はエラーを発生させます。

#### ConfigurableAgent (`configurable_agent.py`)
全てのモードでCachedMemoryManagerを使用するよう変更：
- BASIC/ENHANCED: キャッシュサイズ100
- GRAPH_CENTRIC: キャッシュサイズ150

### 3. 設定ファイルの更新 (`config.yaml`)
新しいエージェント設定セクションを追加：
```yaml
agent:
  type: main_refactored           # 推奨
  datastore:
    enabled: true                 # 常に有効
    type: sqlite
    path: data/insight_facts.db

memory:
  cache_size: 100                 # メモリ爆発防止
```

## 重要な注意事項

### DataStoreが必須に
すべてのエージェントでDataStoreが必須となりました。これにより：
- メモリ使用量が一定（エピソード数に依存しない）
- 長時間稼働でも安定
- 無制限のエピソード保存が可能

### 後方互換性
- L2MemoryManagerという名前は残っていますが、内部的にはCachedMemoryManagerを使用
- 既存のコードは動作しますが、非推奨です

### パフォーマンス
- キャッシュヒット時: < 0.1ms
- キャッシュミス時: 5-10ms（SSDアクセス）
- FAISSインデックスによる高速検索は維持

## 移行方法

### 新規プロジェクト
```python
from insightspike.core.datastore import DataStore
from insightspike.implementations.agents import MainAgentRefactored

# DataStoreを作成
datastore = DataStore(storage_path="data/insight_facts.db")

# エージェントを作成
agent = MainAgentRefactored(config=config, datastore=datastore)
```

### 既存プロジェクト
1. 設定ファイルで`agent.type: main_refactored`を設定
2. DataStoreインスタンスを作成してエージェントに渡す
3. テストを実行して動作確認

## 次のステップ

1. **古い実装の物理削除**（保留中）
   - `layer2_memory_manager.py`
   - `layer2_compatibility.py`
   - 関連テストファイル

2. **全テストの更新**
   - CachedMemoryManagerを使用するようテストを更新
   - メモリ爆発テストの追加

3. **ドキュメントの更新**
   - ユーザーガイドの更新
   - APIドキュメントの更新