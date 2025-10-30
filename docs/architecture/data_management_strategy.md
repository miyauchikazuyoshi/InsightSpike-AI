# データ管理戦略

## 概要

InsightSpike-AIは、2025年7月に「スナップショット方式」から「トランザクション方式」へと完全に移行しました。この文書では、現在のデータ管理戦略と実装の詳細を記録します。

## アーキテクチャの進化

### 旧：スナップショット方式（〜2025年7月）
```
起動時：全データをメモリにロード
　↓
実行中：メモリ上で操作
　↓
終了時：全データをディスクに保存
```

**問題点：**
- メモリ使用量がデータ量に比例
- 起動時間がデータ量に比例
- クラッシュ時のデータ損失リスク
- スケーラビリティの限界

### 新：トランザクション方式（2025年7月〜）
```
起動時：アプリケーションのみ起動（データロードなし）
　↓
実行中：必要なデータのみDBから取得・即座に永続化
　↓
終了時：特別な処理不要（すでに永続化済み）
```

**利点：**
- メモリ使用量一定（ワーキングセットのみ）
- 起動時間一定（O(1)）
- データ損失リスクゼロ
- 無限のスケーラビリティ

## データ構造

### 1. ディレクトリ構造
```
data/
├── sqlite/                    # SQLiteデータベース
│   └── insightspike.db       # 統合データベース
│
├── knowledge_base/           # 静的な知識データ
│   ├── initial/             # 初期投入用データ
│   └── samples/             # サンプルデータ
│
├── models/                   # MLモデルキャッシュ
├── logs/                     # アプリケーションログ
└── cache/                    # 一時キャッシュ
```

### 2. データベーススキーマ

#### episodes テーブル
```sql
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB,                    -- 384次元の埋め込みベクトル
    vector_norm REAL,               -- ベクトルのノルム値（⚡ NEW: 統合インデックス用）
    c_value REAL DEFAULT 0.5,       -- 確信度
    metadata TEXT,                  -- JSON形式のメタデータ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- パフォーマンス用インデックス
CREATE INDEX idx_episodes_namespace ON episodes(namespace);
CREATE INDEX idx_episodes_created_at ON episodes(created_at);
```

#### graph_nodes テーブル
```sql
CREATE TABLE graph_nodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    node_type TEXT,                 -- concept, episode, etc.
    attributes TEXT,                -- JSON形式の属性
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### graph_edges テーブル
```sql
CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT,                 -- related, subset, etc.
    attributes TEXT,                -- JSON形式の属性（weight等）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
);
```

#### queries テーブル ⚡ **NEW**
```sql
CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB,                    -- 384次元の埋め込みベクトル
    has_spike BOOLEAN DEFAULT FALSE,
    spike_episode_id TEXT,
    response TEXT,
    metadata TEXT,                  -- JSON形式のメタデータ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- パフォーマンス用インデックス
CREATE INDEX idx_queries_timestamp ON queries(created_at);
CREATE INDEX idx_queries_has_spike ON queries(has_spike);
CREATE INDEX idx_queries_namespace ON queries(namespace);
```

## コンポーネント責務

### 1. DataStore層
**SQLiteDataStore**
- 役割：データの永続化を担当
- 責務：
  - エピソードの保存・検索
  - グラフの保存・読み込み
  - ベクトル検索（統合インデックス連携）⚡ **UPDATED**
  - トランザクション管理
  - **クエリの保存・検索** ⚡ **NEW**

**EnhancedFileSystemDataStore** ⚡ **NEW**
- 役割：統合インデックスを使用した高速DataStore
- 責務：
  - IntegratedVectorGraphIndexの管理
  - 既存APIとの後方互換性維持
  - 段階的移行のサポート（shadow/partial/full）

### 2. ワーキングメモリ層
**L2WorkingMemoryManager**
- 役割：現在の思考に必要なデータのみ管理
- 責務：
  - 最大100件程度のワーキングセット管理
  - LRU方式でのメモリ管理
  - DataStoreからの必要データ取得

### 3. エージェント層
**DataStoreMainAgent**
- 役割：ユーザーリクエストの処理
- 責務：
  - 知識追加時の即座の永続化
  - 質問処理時の必要データ取得
  - インサイト検出と記録

## データフロー

### 知識追加フロー
```
1. ユーザー入力
    ↓
2. MainAgent.add_knowledge()
    ↓
3. エピソード作成
    ↓
4. DataStore.save_episodes() ← 即座にDBに保存
    ↓
5. 完了通知
```

### 質問処理フロー
```
1. ユーザー質問
    ↓
2. MainAgent.process_question()
    ↓
3. DataStore.search_episodes_by_vector() ← 必要な分だけ取得
    ↓
4. ワーキングメモリで推論
    ↓
5. 結果返却（新しいインサイトは即座に保存）
    ↓
6. DataStore.save_queries() ← クエリと結果を保存 ⚡ NEW
```

## パフォーマンス特性

### メモリ使用量
- **定数**: ワーキングセットサイズ（デフォルト100件）に固定
- **計算式**: `メモリ = アプリ基本 + (ワーキングセットサイズ × エピソードサイズ)`

### 検索性能
- **ベクトル検索**: 統合インデックスによりO(1) ⚡ **IMPROVED**
  - 旧: FAISSインデックスでO(log n) + 正規化O(n)
  - 新: 事前正規化済みベクトルでO(1)
- **テキスト検索**: SQLiteのFTS（Full Text Search）によりO(log n)
- **グラフクエリ**: インデックス付きでO(1)〜O(log n)
- **空間検索**: 位置インデックスによりO(log n) ⚡ **NEW**

### スケーラビリティ
- **エピソード数**: 実質無制限（ディスク容量まで）
- **並行アクセス**: SQLiteのWALモードで複数プロセス対応
- **バックアップ**: 単一ファイルコピーで完了

## ベストプラクティス

### 1. ネームスペースの活用
```python
# 異なる知識ドメインを分離
datastore.save_episodes(episodes, namespace="medical")
datastore.save_episodes(episodes, namespace="technical")
```

### 2. バッチ処理の推奨
```python
# 良い例：バッチで保存
episodes = [create_episode(text) for text in texts]
datastore.save_episodes(episodes)

# 悪い例：1件ずつ保存
for text in texts:
    episode = create_episode(text)
    datastore.save_episodes([episode])  # 非効率
```

### 3. ワーキングセットの調整
```python
# メモリが潤沢な環境
L2WorkingMemoryManager(datastore, max_working_size=500)

# メモリ制約のある環境
L2WorkingMemoryManager(datastore, max_working_size=50)
```

## 移行ツール

### レガシーデータの移行
```bash
# 旧形式データをSQLiteに移行
python scripts/migrate_to_sqlite.py \
    --legacy-dir ./data/old \
    --target-db ./data/sqlite/insightspike.db
```

### データ構造の整理
```bash
# スナップショット形式のディレクトリを整理
python scripts/migrate_data_structure.py --auto-delete
```

## 監視とメンテナンス

### データベース統計の確認
```python
stats = datastore.get_stats()
print(f"Total episodes: {stats['episodes']['total']}")
print(f"Total queries: {stats['queries']['total']}")  # ⚡ NEW
print(f"Spike rate: {stats['queries']['spike_rate'] * 100:.1f}%")  # ⚡ NEW
print(f"DB size: {stats['db_size_bytes'] / 1024 / 1024:.1f} MB")
```

### バックアップ
```bash
# SQLiteの単純バックアップ
cp data/sqlite/insightspike.db data/sqlite/backup_$(date +%Y%m%d).db

# オンラインバックアップ（アプリ実行中も可能）
sqlite3 data/sqlite/insightspike.db ".backup data/sqlite/backup.db"
```

### インデックスの最適化
```sql
-- 定期的なインデックス再構築
REINDEX;

-- データベースの最適化
VACUUM;
```

## 今後の拡張可能性

### 1. 統合インデックスの拡張 ⚡ **NEW**
- **現在実装済み**:
  - 事前正規化によるO(1)検索
  - 空間インデックスによる位置ベース検索
  - グラフ構造の統合管理
  - FAISSの自動切り替え（大規模データ対応）
- **今後の拡張**:
  - 分散インデックスのサポート
  - GPUアクセラレーション
  - より高度な近似最近傍探索

### 2. 分散データベース対応
- PostgreSQLやMySQLへの移行パス
- 複数ノードでのレプリケーション

### 3. ベクトルデータベース統合
- Pinecone, Weaviate, Qdrantなどの専用ベクトルDB
- より高度なベクトル検索機能（統合インデックスとの併用）

### 4. グラフデータベース連携
- Neo4jなどの専用グラフDB
- より複雑なグラフクエリの実行

## まとめ

このデータ管理戦略により、InsightSpike-AIは：

1. **スケーラブル**: 数百万エピソードまで対応可能
2. **高信頼性**: データ損失リスクゼロ
3. **高性能**: 必要なデータのみ処理
4. **運用容易**: 単一DBファイルで管理

研究プロトタイプから、実運用可能なプロダクションシステムへと進化しました。
