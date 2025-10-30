---
status: active
category: memory
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# メモリ爆発問題の解決計画

## 問題の診断結果

他のAIの指摘は完全に正しいです。現在の実装には以下の問題があります：

### 1. 新旧実装の混在
```python
# MainAgent.__init__
self.datastore = datastore      # 新実装（ほぼ未使用）
self.l2_memory = Memory()       # 旧実装（実際に使用中）
```

### 2. 全データのメモリ保持
```python
# L2MemoryManager
self.episodes: List[Episode] = []  # 全エピソードをメモリに保持
self.max_episodes = 10000         # デフォルトで1万件
```

### 3. DataStoreの不完全な統合
- save/load時のみDataStore使用
- 実行時は全てメモリベースの処理

## 解決方針

### Phase 1: 即座の対処（メモリリーク防止）

```python
# 1. L2MemoryManagerにキャッシュ機能を追加
class CachedL2MemoryManager(L2MemoryManager):
    def __init__(self, datastore, cache_size=100):
        super().__init__()
        self.datastore = datastore
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
    def get_episode(self, episode_id):
        if episode_id in self.cache:
            return self.cache[episode_id]
            
        # DataStoreから取得
        episode = self.datastore.get_episode(episode_id)
        
        # キャッシュに追加（LRU）
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[episode_id] = episode
        
        return episode
```

### Phase 2: DataStore完全統合（1週間）

```python
# MainAgentの修正
class MainAgent:
    def __init__(self, config, datastore):
        self.datastore = datastore
        # L2MemoryManagerは削除
        
    def add_knowledge(self, text):
        # 直接DataStoreに保存
        episode_id = self.datastore.add_episode(text)
        return episode_id
        
    def process_question(self, question):
        # DataStoreから検索
        results = self.datastore.search_episodes_by_vector(
            question_embedding, 
            top_k=10
        )
        # メモリには最小限のデータのみ保持
```

### Phase 3: 古い実装の完全削除（2週間）

1. **削除対象ファイル**
   - `layer2_memory_manager.py`（旧メモリ管理）
   - `knowledge_graph_memory.py`（旧グラフ管理）
   - その他の旧実装

2. **リファクタリング**
   - MainAgentからL2MemoryManagerへの全参照を削除
   - DataStoreインターフェースのみを使用
   - テストの更新

## 具体的な実装手順

### Step 1: メモリ使用量の可視化
```python
# memory_monitor.py
import psutil
import logging

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
        
    def check_memory(self):
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        if memory_mb > self.threshold_mb:
            logging.warning(f"High memory usage: {memory_mb:.0f} MB")
        return memory_mb
```

### Step 2: DataStore専用のMainAgent
```python
# datastore_main_agent.py
class DataStoreMainAgent:
    """DataStoreのみを使用する新しいMainAgent"""
    
    def __init__(self, config, datastore):
        self.config = config
        self.datastore = datastore
        self.memory_monitor = MemoryMonitor()
        
    def add_knowledge(self, text, metadata=None):
        # メモリチェック
        self.memory_monitor.check_memory()
        
        # DataStoreに直接保存
        episode = Episode(text=text, metadata=metadata)
        return self.datastore.add_episode(episode)
        
    def search_knowledge(self, query, top_k=10):
        # DataStoreから直接検索
        return self.datastore.search_episodes_by_text(query, top_k)
```

### Step 3: 移行スクリプト
```python
# migrate_to_datastore.py
def migrate_agent():
    # 旧エージェントからデータを抽出
    old_agent = MainAgent(config)
    old_agent.load()
    
    # 新DataStoreに移行
    datastore = SQLiteDataStore("data/insight.db")
    
    for episode in old_agent.l2_memory.episodes:
        datastore.add_episode(episode)
        
    # 新エージェントを作成
    new_agent = DataStoreMainAgent(config, datastore)
    return new_agent
```

## 期待される効果

### メモリ使用量の比較

| 実装 | 1万エピソード | 10万エピソード | 100万エピソード |
|-----|--------------|----------------|-----------------|
| 現在 | 200 MB | 2 GB | 20 GB（クラッシュ） |
| 改善後 | 50 MB | 50 MB | 50 MB |

### パフォーマンス

- **検索速度**: SQLiteのインデックスにより高速化
- **起動時間**: データの遅延ロードにより大幅短縮
- **スケーラビリティ**: 事実上無制限のエピソード数

## 結論

他のAIの診断は的確です。現在の実装は確実にメモリ爆発を起こします。
ただし、段階的な移行により、システムを稼働させながら問題を解決できます。

最優先事項：
1. **メモリモニタリングの追加**（今すぐ）
2. **キャッシュ層の実装**（1日）
3. **DataStore完全移行**（1週間）