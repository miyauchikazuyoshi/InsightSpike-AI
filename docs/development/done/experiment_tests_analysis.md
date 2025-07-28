# Experiment Tests Analysis - Mathematical Concept Evolution v2

## 分析結果

### 実験ディレクトリのテストファイル一覧

1. **test_spike_detection.py** - スパイク検出の詳細テスト
2. **test_adaptive_loop.py** - アダプティブループの実装テスト
3. **test_llm_response.py** - LLMレスポンステスト
4. **test_spike_with_consolidation.py** - 統合付きスパイクテスト
5. **test_process_question_calls.py** - プロセス質問呼び出しテスト
6. **test_cycle_vs_adaptive.py** - サイクルvs適応的処理の比較
7. **test_optimized_process.py** - 最適化プロセステスト
8. **test_with_claude.py** - Claude APIテスト
9. **test_anthropic_direct.py** - Anthropic直接テスト
10. **test_with_bypass.py** - バイパステスト
11. **test_simple_add_knowledge.py** - シンプルな知識追加テスト

## CIへの流用可能性

### ✅ CIに適している（整理して移動すべき）

1. **test_spike_detection.py**
   - 基本的なスパイク検出機能のテスト
   - `tests/integration/test_spike_detection.py`として整理

2. **test_adaptive_loop.py**
   - アダプティブループ機能のテスト
   - `tests/integration/test_adaptive_loop.py`として整理

3. **test_simple_add_knowledge.py**
   - 知識追加の基本機能テスト
   - `tests/unit/test_knowledge_management.py`として整理

### ⚠️ 条件付きでCI可能（モック化が必要）

4. **test_llm_response.py**
   - LLMプロバイダーのモック化が必要
   - `tests/integration/test_llm_integration.py`として整理

5. **test_process_question_calls.py**
   - 質問処理フローのテスト
   - `tests/integration/test_question_processing.py`として整理

### ❌ CIに不適（実験専用）

6. **test_with_claude.py**
7. **test_anthropic_direct.py**
   - 実際のAPIキーが必要
   - 実験環境でのみ実行

8. **test_cycle_vs_adaptive.py**
9. **test_optimized_process.py**
   - パフォーマンス比較のための実験的テスト
   - ベンチマークスイートとして別管理

10. **test_spike_with_consolidation.py**
11. **test_with_bypass.py**
    - 実験的な機能のテスト
    - 安定化後にCIへ移行

## 推奨アクション

### 1. 即座に整理すべきテスト ✅ 完了

```bash
# スパイク検出テストを移動 ✅
# tests/integration/test_spike_detection_core.py として作成済み

# アダプティブループテストを移動 ✅
# tests/integration/test_adaptive_loop_feature.py として作成済み

# 知識管理テストを移動 ✅
# tests/unit/test_knowledge_management_basic.py として作成済み
```

**実施内容:**
- 実験ディレクトリのテストをCI向けに完全にリファクタリング
- PyTestフィクスチャを使用した適切なテスト構造に変換
- テンポラリディレクトリを使用したDataStore分離
- パラメトリックテストの追加
- エラーハンドリングとエッジケースのカバレッジ向上

### 2. リファクタリングが必要なテスト

- **test_llm_response.py**
  - MockProviderを使用するように修正
  - 外部依存を削除

- **test_process_question_calls.py**
  - 単体テストとして分割
  - 決定的な結果を返すように修正

### 3. 実験ディレクトリに残すテスト

```
experiments/mathematical_concept_evolution_v2/
├── test_with_claude.py          # API実験用
├── test_anthropic_direct.py     # API実験用
├── test_cycle_vs_adaptive.py    # パフォーマンス比較
├── test_optimized_process.py    # 最適化実験
└── test_spike_with_consolidation.py  # 実験的機能
```

## テスト整理の優先順位

1. **高優先度**（今すぐ整理）
   - test_spike_detection.py
   - test_adaptive_loop.py
   - test_simple_add_knowledge.py

2. **中優先度**（リファクタリング後）
   - test_llm_response.py
   - test_process_question_calls.py

3. **低優先度**（実験完了後）
   - その他の実験的テスト