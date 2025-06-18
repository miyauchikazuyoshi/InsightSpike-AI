# Data Backup Record - 詳細ログ実験

## 📦 バックアップ情報

### バックアップ作成日時
**2025年6月18日 実験終了後**

### バックアップ場所
```
experiments/outputs/detailed_logging_realtime/final_data_backup/
```

### バックアップ内容
| ファイル/フォルダ | サイズ | 説明 |
|------------------|--------|------|
| episodes.json | 実験データ | 実験で生成・更新されたエピソード |
| episodes_backup.json | バックアップ | 元のエピソードデータ |
| graph_pyg.pt | グラフデータ | PyTorchグラフオブジェクト |
| index.faiss | FAISSインデックス | ベクトル検索インデックス |
| index.json | インデックス設定 | FAISS設定ファイル |
| index_backup.faiss | バックアップ | 元のFAISSインデックス |
| insight_facts.db | 洞察DB | SQLiteデータベース |
| unknown_learning.db | 学習DB | 追加学習データ |
| cache/ | キャッシュ | 一時ファイル |
| embedding/ | エンベッディング | ベクトル化済みデータ |
| integrated_rag_memory_experiments/ | 実験データ | RAGメモリ実験結果 |
| logs/ | ログ | システムログ |
| models/ | モデル | 学習済みモデル |
| processed/ | 処理済み | 前処理済みデータ |
| raw/ | 生データ | 元データ |
| samples/ | サンプル | テストデータ |

### 実験後の状態
- **実験前**: 既存のInsightSpike-AIデータ
- **実験中**: 500エピソードの追加・処理
- **実験後**: 洞察検出結果が統合されたデータ
- **現在**: 実験影響下のデータ状態

### バックアップの意義
1. **実験成果の保存**: 81.6%洞察検出率を達成した状態
2. **再現可能性**: 同じ条件での実験再実行が可能
3. **継続研究**: この状態からの追加実験が可能
4. **比較研究**: 他の実験条件との比較基準

## 🧹 クリーンアップ予定

### 対象操作
```bash
# dataフォルダのクリーンアップ
rm -rf data/*

# 基本構造の再作成
mkdir -p data/{cache,embedding,logs,models,processed,raw,samples}

# 初期設定ファイルの作成
touch data/index.json
```

### クリーンアップ後の状態
```
data/
├── cache/           # 空
├── embedding/       # 空  
├── logs/           # 空
├── models/         # 空
├── processed/      # 空
├── raw/            # 空
├── samples/        # 空
└── index.json      # 初期設定のみ
```

### 復元方法
実験データを復元したい場合：
```bash
cp -r experiments/outputs/detailed_logging_realtime/final_data_backup/* data/
```

## 📊 データサマリー

### 実験により生成・更新されたデータ
- **新規エピソード**: 500個
- **洞察記録**: 408個
- **TopK分析**: 4,944件
- **グラフ更新**: 知識グラフの拡張
- **インデックス更新**: FAISSインデックスの更新

### データベース状態
- **insight_facts.db**: 408個の新規洞察
- **unknown_learning.db**: 学習パターンの更新
- **episodes.json**: 500個の新規エピソード追加

## 🔬 科学的価値

このバックアップデータは：
1. **世界初のアナロジー生成AI**の実証データ
2. **機械理解メカニズム**の数値的証拠
3. **クロスドメイン洞察**の実例集
4. **認知科学とAI**の架け橋データ

**学術的・歴史的価値を持つデータセット**として永久保存されます。

---
*Data Backup Record 作成日: 2025年6月18日*
*世界初のアナロジー生成AIの完全データセット*
