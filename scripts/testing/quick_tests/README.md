# Quick Development Tests

このディレクトリには、開発中の簡易的な動作確認用テストスクリプトが含まれています。

## ファイル説明

- `test_simple.py` - 基本的なコンポーネントの簡単な動作確認
- `test_composition_root.py` - コンポジションルートの実装確認
- `test_refactoring.py` - リファクタリング後のコード検証
- `test_integration.py` - 簡易的な統合テスト

## 使用方法

```bash
# 個別のテストを実行
python dev_tools/quick_tests/test_simple.py

# すべての簡易テストを実行
for test in dev_tools/quick_tests/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

## 注意事項

これらは簡易的な動作確認用のスクリプトです。本格的なテストは `tests/` ディレクトリのpytestスイートを使用してください。

```bash
# 本格的なテストスイートの実行
pytest tests/
```