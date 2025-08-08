# DataStore統合仕様

## 1. メインコード統合

### 使用クラス
```python
from insightspike.index import IntegratedVectorGraphIndex
from insightspike.implementations.datastore.factory import DataStoreFactory
```

## 2. DataStore永続化設計

### 2.1 ディレクトリ構造
```
data/experiments/pure_movement_50x50/
├── session_20250106_123456/
│   ├── episodes/
│   │   ├── episode_0000.json
│   │   ├── episode_0001.json
│   │   └── ...
│   ├── artifacts/
│   │   ├── session_info.json
│   │   ├── final_result.json
│   │   └── navigation_path.json
│   └── index/
│       ├── vectors.npy
│       └── graph.pkl
```

### 2.2 エピソード保存形式
```json
{
  "episode_id": 1234,
  "vector": [0.1, 0.2, 0.5, 1.0, -1.0, 0.05, 0.0],
  "metadata": {
    "type": "movement",
    "position": [10, 15],
    "action": "right",
    "success": true,
    "is_wall": false,
    "timestamp": 1704538456.789
  }
}
```

### 2.3 セッション情報
```json
{
  "session_id": "20250106_123456",
  "maze_size": [51, 51],
  "start": [1, 1],
  "goal": [49, 49],
  "start_time": "2025-01-06T12:34:56",
  "config": {
    "max_depth": 5,
    "dimension": 7,
    "similarity_threshold": 0.4
  }
}
```

## 3. ログ記録システム

### 3.1 リアルタイムログ
```python
# 1000ステップごとに進捗を記録
progress_log = {
    "step": 1000,
    "position": [x, y],
    "distance_to_goal": 15,
    "wall_hit_rate": 0.23,
    "episodes_created": 5423,
    "timestamp": "2025-01-06T12:35:10"
}
```

### 3.2 最終結果
```python
final_result = {
    "success": True,
    "steps": 12345,
    "total_time": 123.45,
    "total_episodes": 61725,
    "wall_hits": 2840,
    "wall_hit_rate": 0.23,
    "path_length": 9505,
    "metrics": {
        "search_times": [...],
        "hop_usage": {"1-hop": 12345, ...},
        "episode_types": {"movement": 12345, "visual": 49380}
    }
}
```

## 4. メインコードIndexの利点

### 4.1 IntegratedVectorGraphIndex機能
- **ベクトル検索**: 高速な類似検索
- **グラフ構造**: エピソード間の関係管理
- **空間インデックス**: 位置ベースの検索
- **永続化サポート**: save/load機能

### 4.2 DataStoreの利点
- **自動永続化**: エピソードの自動保存
- **メタデータ管理**: 豊富なメタデータサポート
- **アーティファクト保存**: 結果やパスの保存
- **セッション管理**: 実験ごとの分離

## 5. 実験後の解析

### 5.1 エピソード読み込み
```python
# 保存されたエピソードを読み込み
datastore = DataStoreFactory.create(
    "filesystem",
    base_path="data/experiments/pure_movement_50x50"
)

# 全エピソードを取得
episodes = datastore.list_episodes()

# 特定のエピソードを取得
episode = datastore.get_episode("episode_1234")
```

### 5.2 結果の可視化
```python
# 結果を読み込み
result = datastore.load_artifact(f"result_{session_id}")

# パスを読み込み
path_data = datastore.load_artifact(f"path_{session_id}")
path = path_data['path']

# 可視化
visualize_navigation(maze, path, result['visit_counts'])
```

### 5.3 複数実験の比較
```python
# 複数セッションの結果を比較
sessions = datastore.list_sessions()
results = []

for session in sessions:
    result = datastore.load_artifact(f"result_{session}")
    results.append(result)

# 統計分析
avg_success_rate = sum(r['success'] for r in results) / len(results)
avg_steps = np.mean([r['steps'] for r in results if r['success']])
```

## 6. 実装のポイント

### 6.1 メモリ管理
```python
# 大規模実験でのメモリ管理
MAX_EPISODES_IN_MEMORY = 10000

if self.episode_id % 1000 == 0:
    # 定期的にディスクにフラッシュ
    self.datastore.flush()
    
    # 古いエピソードをメモリから削除
    if len(self.index) > MAX_EPISODES_IN_MEMORY:
        self.index.prune_old_entries(keep=8000)
```

### 6.2 エラーハンドリング
```python
try:
    # エピソード保存
    self.datastore.save_episode(episode_data)
except Exception as e:
    print(f"Warning: Failed to save episode: {e}")
    # 実験は継続
```

### 6.3 再開可能性
```python
# 実験の途中結果を定期保存
if step % 5000 == 0:
    checkpoint = {
        'step': step,
        'position': self.position,
        'episode_id': self.episode_id,
        'metrics': self.metrics
    }
    self.datastore.save_artifact(
        f"checkpoint_{self.session_id}_{step}",
        checkpoint
    )
```

## 7. 利点まとめ

1. **完全なログ**: 全エピソードが永続化される
2. **再現性**: セッション情報で実験を再現可能
3. **解析可能**: 実験後に詳細な解析が可能
4. **スケーラブル**: 大規模実験でも安定動作
5. **統合済み**: InsightSpikeメインコードと完全統合