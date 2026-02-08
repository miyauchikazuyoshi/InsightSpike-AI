# src リファクタリング実行計画（2026-02）

**作成日**: 2026-02-07  
**Status**: Draft（実行前）  
**対象**: `src/insightspike/**/*.py`

---

## 1. 目的

`src` 配下で「現役コード / 実験コード / レガシーコード」が混在している状態を解消し、以下を達成する。

1. 構造と責務の明確化（どのファイルが何のために存在するかを明文化）
2. 重複実装の統合（MainAgent 系、L2 系、Datastore 系）
3. 大型ファイルの責務分割（変更衝突と回帰を減らす）
4. 未到達コードの隔離（archive / experiments への移管）

---

## 2. 監査成果物

- 構造・責務・改善提案 CSV:  
  `docs/design/src_structure_responsibility_improvement_sheet.csv`
- 集計サマリ:  
  `docs/design/src_structure_responsibility_improvement_summary.txt`
- 互換性・実験影響評価:  
  `docs/design/src_refactoring_compatibility_impact_2026_02.md`

### 2.1 現状メトリクス（CSV集計）

- 対象 Python ファイル数: `298`
- 優先度 `P0`: `7`
- 優先度 `P1`: `9`
- 優先度 `P2`: `185`
- 優先度 `P3`: `97`
- runtime から未到達: `188`
- runtime/test 双方から未到達: `101`

---

## 3. 優先度定義

- `P0`: 直ちに着手。責務過多/重複の中核で、全体品質に直接影響
- `P1`: P0 後に着手。統合・分割で設計整合を回復
- `P2`: 所有者確定と用途明記。archive/move 判定を進める
- `P3`: 維持管理（型・テスト・doc の補強）

---

## 4. 直近の重点対象（P0/P1）

### 4.1 P0（最優先）

1. `src/insightspike/implementations/agents/main_agent.py`
2. `src/insightspike/cli/spike.py`
3. `src/insightspike/implementations/layers/layer2_memory_manager.py`
4. `src/insightspike/implementations/layers/layer4_llm_interface.py`
5. `src/insightspike/implementations/datastore/sqlite_store.py`
6. `src/insightspike/implementations/layers/cached_memory_manager.py`
7. `src/insightspike/implementations/agents/main_agent_refactored.py`

### 4.2 P1（次点）

1. `src/insightspike/implementations/agents/datastore_main_agent.py`
2. `src/insightspike/implementations/agents/datastore_agent.py`
3. `src/insightspike/implementations/agents/slim_main_agent.py`
4. `src/insightspike/implementations/layers/layer2_working_memory.py`
5. `src/insightspike/implementations/layers/layer4_prompt_builder.py`
6. `src/insightspike/implementations/datastore/sqlite_store_graph.py`
7. `src/insightspike/algorithms/gedig_core.py`

---

## 5. 実行フェーズ（Phase 1/2 のみ）

## Phase 1: 中核分割（1.5週）

1. `main_agent.py` を package 化し、責務を以下へ分割
2. `cli/spike.py` のコマンド群を `cli/commands/` へ再配置
3. `layer2_memory_manager.py` を検索/統合/aging/prune で分離
4. `layer4_llm_interface.py` を provider registry / adapter / prompt orchestration に分割
5. `sqlite_store.py` を schema / queries / graph / migration に分割

完了条件:
- 既存公開 API の互換が維持
- 主要 CLI 操作（query/embed/insights/stats）が回帰なし

## Phase 2: 重複統合（1週）

1. `main_agent_refactored.py` の差分を吸収し archive 化
2. `datastore_*_agent.py`, `slim_main_agent.py` をモード戦略に統合
3. `cached_memory_manager.py` / `layer2_working_memory.py` の重複機能を L2 戦略へ吸収
4. `sqlite_store_graph.py` を `sqlite_store.py` へ統合

完了条件:
- Agent 系の主実装を 1 系統へ収束
- Datastore/L2 の重複ロジック削減（CSV の `merge` 対象を解消）

## 6. CSV の使い方

`docs/design/src_structure_responsibility_improvement_sheet.csv` の主要カラム:

- `responsibility_summary`: そのファイルの責務の要約
- `runtime_reachable` / `test_reachable`: 到達性（静的 import ベース）
- `recommended_action`: `split` / `merge` / `archive_or_move` など
- `merge_with`: 統合先候補
- `split_recommendation`: 分離すべき責務案
- `improvement_proposal`: 実行時の改善方針

運用ルール:

1. PR は CSV の該当行を更新してから実装変更に入る
2. `recommended_action` が `merge` のものは統合先を固定してから着手
3. `archive_or_move` は owner 合意なしに削除しない

---

## 7. リスクと対策

1. 静的解析のみでは動的 import を過小評価する可能性  
対策: `archive_or_move` は必ず owner レビューを挟む

2. 分割で API 回帰が発生する可能性  
対策: 互換 facade を先に置き、段階的に内部を差し替える

3. 実験資産の消失  
対策: 削除でなく `experiments/` または `archive/` へ移管

---

## 8. 受け入れ基準

1. P0 の 7 ファイルが `split/merge` 計画に従って処理済み
2. Agent/L2/Datastore の重複系が統合され、主経路が明確
3. 既存公開 API と主要 CLI 操作（query/embed/insights/stats）の互換性が維持される
4. CSV と実装状態が同期される
