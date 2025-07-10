# テストスクリプト

## 概要
システムコンポーネント・データ同期・セットアップのテスト用スクリプト集

## 含まれるスクリプト


### 🛡️ `safe_component_test.py`
- **機能**: 安全なコンポーネントテスト
- **用途**: 各システムコンポーネントの動作確認
- **実行**: `python safe_component_test.py`

## 使用方法

```bash
# 安全コンポーネントテスト
cd /path/to/InsightSpike-AI
python scripts/testing/safe_component_test.py

# 完全な洞察システムテスト
python scripts/testing/test_complete_insight_system.py

# 大規模統合テスト
python scripts/testing/large_scale_integration_test.py

# セーフモードテスト
python scripts/testing/test_safe_mode.py
```

## 含まれるスクリプト

### 🧪 `test_complete_insight_system.py`  
- **機能**: 洞察システム全体のエンドツーエンドテスト
- **用途**: 洞察検出・登録・検索の統合テスト
- **実行**: `python test_complete_insight_system.py`

### 📈 `large_scale_integration_test.py`
- **機能**: 大規模データでの統合テスト
- **用途**: パフォーマンス・スケーラビリティ検証
- **実行**: `python large_scale_integration_test.py`

### 🔒 `test_safe_mode.py`
- **機能**: セーフモード（モックLLM）でのテスト
- **用途**: LLMなしでのシステム動作確認
- **実行**: `python test_safe_mode.py`

## テスト対象

- **コンポーネント**: メモリ管理・洞察検出・グラフ処理
- **統合機能**: エンドツーエンドのワークフロー
- **パフォーマンス**: 大規模データ処理・応答時間
- **セーフモード**: モックプロバイダーでの動作

## テスト結果

テスト結果は標準出力に表示され、エラーがあれば詳細な診断情報が出力されます。

## 注意事項

- テスト実行前にデータのバックアップを推奨
- 一部のテストはネットワーク接続が必要
- モデルテストには十分なメモリが必要

---
*InsightSpike-AI Project - Testing Scripts*
