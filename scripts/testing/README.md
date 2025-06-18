# テストスクリプト

## 概要
システムコンポーネント・データ同期・セットアップのテスト用スクリプト集

## 含まれるスクリプト

### 🔄 `test_data_sync.py`
- **機能**: データ同期機能のテスト
- **用途**: データ整合性・同期プロセスの検証
- **実行**: `python test_data_sync.py`

### 🦙 `test_tinyllama_setup.py`
- **機能**: TinyLlama モデルセットアップのテスト
- **用途**: モデル読み込み・初期化の検証
- **実行**: `python test_tinyllama_setup.py`

### 🛡️ `safe_component_test.py`
- **機能**: 安全なコンポーネントテスト
- **用途**: 各システムコンポーネントの動作確認
- **実行**: `python safe_component_test.py`

## 使用方法

```bash
# データ同期テスト
cd /path/to/InsightSpike-AI
python scripts/testing/test_data_sync.py

# TinyLlamaセットアップテスト
python scripts/testing/test_tinyllama_setup.py

# 安全コンポーネントテスト
python scripts/testing/safe_component_test.py
```

## テスト対象

- **データ同期**: ファイル同期・データベース整合性
- **モデルセットアップ**: LLMモデル読み込み・設定
- **コンポーネント**: メモリ管理・洞察検出・グラフ処理

## テスト結果

テスト結果は標準出力に表示され、エラーがあれば詳細な診断情報が出力されます。

## 注意事項

- テスト実行前にデータのバックアップを推奨
- 一部のテストはネットワーク接続が必要
- モデルテストには十分なメモリが必要

---
*InsightSpike-AI Project - Testing Scripts*
