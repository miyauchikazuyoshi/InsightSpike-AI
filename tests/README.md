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

## マーカー語彙（フィルタ用）

ルート（tests/pytest.ini）で定義:

- `slow` — 遅いテスト
- `integration` — 統合テスト
- `e2e` — エンドツーエンド
- `requires_gpu` — GPU 必須
- `performance` — パフォーマンス/安定性
- `reproducibility` — 再現性/決定性

実験系（例: experiments/*）で使用されることがあるマーカー（語彙揺れの目安）:

- `gpu` ≈ `requires_gpu`
- `ablation` — アブレーション
- `longterm` — 長時間

フィルタ例:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 INSIGHTSPIKE_LITE_MODE=1 INSIGHTSPIKE_MIN_IMPORT=1 \
  pytest -q -m "not slow and not integration and not performance and not benchmark and not calibration"
```
