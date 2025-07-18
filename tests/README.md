# InsightSpike Test Suite (新設計)

このディレクトリには、新しいアーキテクチャに基づいたテストスイートが含まれています。

## テスト構造

```
tests_new/
├── unit/                    # ユニットテスト
│   ├── config/             # 設定システムのテスト
│   ├── layers/             # 各レイヤーのテスト
│   └── core/               # コア機能のテスト
├── integration/            # 統合テスト
│   ├── cli/               # CLIコマンドのテスト
│   ├── workflow/          # ワークフローのテスト
│   └── datastore/         # データストアのテスト
├── e2e/                    # エンドツーエンドテスト
│   ├── scenarios/         # 実使用シナリオ
│   └── performance/       # パフォーマンステスト
└── fixtures/               # テストフィクスチャ
    ├── configs/           # テスト用設定
    └── data/              # テストデータ
```

## テスト方針

1. **Pydantic設定を前提**
   - 全てのテストは新しい設定システムを使用
   - ConfigConverterは使用しない

2. **実際のコンポーネントを使用**
   - モックは最小限に
   - 実際の動作を確認

3. **明確なテストケース**
   - 各テストは1つの機能を検証
   - 分かりやすいテスト名

## 実行方法

```bash
# 全テストを実行
pytest tests_new/

# ユニットテストのみ
pytest tests_new/unit/

# 統合テストのみ
pytest tests_new/integration/

# カバレッジ付き
pytest tests_new/ --cov=insightspike --cov-report=html
```