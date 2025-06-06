# Tests

このディレクトリには各種テストファイルが整理されています。

## ディレクトリ構造

- `unit/` - ユニットテスト（個別モジュールのテスト）
- `integration/` - 統合テスト（システム全体のテスト）
- `debug/` - デバッグ用スクリプト（開発時のみ使用）

## 主要テストファイル

### 本番テスト
- `test_dependency_commands.py` - 依存関係コマンドのテスト
- `test_dependency_resolver.py` - 依存関係解決のテスト
- `test_mvp_integration.py` - MVP統合テスト
- `test_platform_utils.py` - プラットフォームユーティリティのテスト
- `test_poetry_integration.py` - Poetry統合テスト

### 統合テスト（integration/）
- `test_mvp_integration.py` - メイン統合テスト（保持）
- その他のtest_*.pyファイル - 実験的統合テスト

### デバッグスクリプト（debug/）
- `debug_deps.py` - 依存関係デバッグ
- `debug_list.py` - リスト表示デバッグ

## 実行方法

```bash
# 全テスト実行
pytest

# ユニットテストのみ
pytest tests/unit/

# 統合テストのみ
pytest tests/integration/test_mvp_integration.py
```
