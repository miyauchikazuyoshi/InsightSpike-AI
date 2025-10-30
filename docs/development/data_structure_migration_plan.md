---
status: active
category: infra
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
next_step: Complete legacy datastore -> integrated index migration with checksum validation.

2025-09-08 Update (Phase 1 complete)
------------------------------------

- Added migration/validation scripts:
  - `scripts/migrate_datastore_to_integrated_index.py`: Scans a FilesystemStore root, exports episode metadata and prepares integrated index stubs
  - `scripts/validate_integrated_index.py`: Computes checksums for episodes/metadata and verifies integrity
- Non-destructive: migration runs in dry-run by default and writes reports to `data/migration/` (configurable)

Usage:
```bash
# Dry-run migration preview
poetry run python scripts/migrate_datastore_to_integrated_index.py --root data/datastore --out data/migration --dry-run

# Validate checksums
poetry run python scripts/validate_integrated_index.py --root data/datastore --report data/migration/checksums.json
```

Next steps:
- Phase 2: Wire integrated index metadata into FilesystemStore and ensure transparent fallback
- Phase 3: Add CI job to validate checksums on sample dataset
---

# データ構造移行計画

## 概要
スナップショット方式からトランザクション方式への移行に伴うdata/フォルダ構造の整理計画。

## 削除対象ディレクトリ/ファイル

### 1. core/ ディレクトリ
- **理由**: スナップショット方式の中核。すべてSQLiteに移行済み
- **内容**:
  - `episodes.json` → SQLiteの`episodes`テーブルへ
  - `graph_pyg.pt` → SQLiteの`nodes`/`edges`テーブルへ  
  - `index.faiss` → SQLiteDataStoreのFAISSインデックスへ

### 2. db/ ディレクトリ  
- **理由**: 複数の個別DBファイルは`sqlite/insightspike.db`に統合
- **内容**:
  - `insight_facts.db` → 統合DB内のテーブルへ
  - `unknown_learning.db` → 統合DB内のテーブルへ

### 3. processed/ ディレクトリ
- **理由**: 中間ファイルは不要。処理結果は直接DBへ
- **新方式**: プロセッシングパイプラインは直接DBに書き込む

### 4. experiments/ ディレクトリ（data/内）
- **理由**: 実験データは`/experiments/`配下で管理すべき
- **移動先**: プロジェクトルートの`experiments/`へ

### 5. backup/ と clean_backup/
- **理由**: スナップショット方式のバックアップ
- **新方式**: SQLiteの`BACKUP`コマンドまたはファイルコピー

## 新規作成ディレクトリ

### 1. sqlite/
```bash
data/sqlite/
├── insightspike.db      # メインDB（すべてのテーブル）
├── insightspike.db-wal  # Write-Ahead Log（自動生成）
└── insightspike.db-shm  # 共有メモリ（自動生成）
```

### 2. knowledge_base/（raw/を改名）
```bash
data/knowledge_base/
├── initial/             # 初期データ
│   ├── episodes.jsonl   # 大量データ対応
│   ├── concepts.csv     # 概念定義
│   └── relations.csv    # 関係定義
└── samples/            # テスト用サンプル
```

## 移行手順

### Phase 1: データマイグレーション
```bash
# 1. 既存データをSQLiteに移行
python scripts/migrate_to_sqlite.py --legacy-dir ./data --target-db ./data/sqlite/insightspike.db

# 2. 移行確認
sqlite3 data/sqlite/insightspike.db "SELECT COUNT(*) FROM episodes;"
```

### Phase 2: ディレクトリ整理
```bash
# 1. 新ディレクトリ作成
mkdir -p data/sqlite
mkdir -p data/knowledge_base/initial
mkdir -p data/knowledge_base/samples

# 2. サンプルデータ移動
mv data/samples/* data/knowledge_base/samples/
mv data/raw/*.txt data/knowledge_base/initial/

# 3. 不要ディレクトリ削除（バックアップ後）
# tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
# rm -rf data/core data/db data/processed data/backup data/clean_backup
```

### Phase 3: 設定更新
```yaml
# config.yaml の更新
data:
  base_path: "./data"
  sqlite:
    db_path: "./data/sqlite/insightspike.db"
    wal_mode: true
  knowledge_base:
    initial_data: "./data/knowledge_base/initial"
    samples: "./data/knowledge_base/samples"
```

## 利点

1. **シンプルな構造**: 役割が明確で理解しやすい
2. **原子性**: すべてのデータ操作がトランザクションで保護される
3. **バックアップ容易性**: 単一のDBファイルをコピーするだけ
4. **並行アクセス**: SQLiteのWALモードで複数プロセス対応
5. **データ整合性**: 外部キー制約やトリガーで整合性保証

## 注意事項

- 移行前に必ずフルバックアップを取る
- 実験中のデータは別途保存しておく
- clean_backup/のデータは初期状態として保存しておく価値がある
