# Clean Data Backup

## 📋 概要

このフォルダには、InsightSpike-AIプロジェクトのクリーンな初期状態のデータファイルが保存されています。

## 📄 バックアップファイル

| ファイル | 元ファイル | 説明 |
|---------|-----------|------|
| `episodes_clean.json` | `episodes.json` | クリーンなエピソードメモリ（5エピソード） |
| `graph_pyg_clean.pt` | `graph_pyg.pt` | クリーンなPyTorchグラフ（1ノード） |
| `index_clean.faiss` | `index.faiss` | クリーンなFAISSインデックス（5ベクトル） |
| `insight_facts_clean.db` | `insight_facts.db` | クリーンな洞察データベース |
| `unknown_learning_clean.db` | `unknown_learning.db` | クリーンな学習データベース |

## 🔧 復元方法

```bash
# dataディレクトリから実行
# 全ファイルを一括復元（新しい構造に対応）
cp clean_backup/episodes_clean.json core/episodes.json
cp clean_backup/graph_pyg_clean.pt core/graph_pyg.pt
cp clean_backup/index_clean.faiss core/index.faiss
cp clean_backup/insight_facts_clean.db db/insight_facts.db
cp clean_backup/unknown_learning_clean.db db/unknown_learning.db
```

## 📊 バックアップ作成時の状態

- **作成日時**: 2025年7月1日 20:34
- **エピソード数**: 5
- **ベクトル次元**: 384
- **グラフノード数**: 1
- **データ整合性**: ✅ 完全

## 🎯 使用場面

- 実験前のデータリセット
- 開発中の誤操作からの復旧
- 新機能テスト前の初期化
- ベンチマーク実行前の状態統一

## ⚠️ 注意事項

- このバックアップは手動で更新してください
- 実験データが蓄積された後は、新しいクリーンポイントを作成することを推奨
- ファイルの整合性を保つため、個別ファイルではなく一括復元を推奨

---

*InsightSpike-AI Project - Clean Data Backup*
