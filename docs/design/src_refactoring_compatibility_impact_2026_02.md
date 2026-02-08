# src リファクタリング互換性・実験影響評価（2026-02）

**作成日**: 2026-02-07  
**対象計画**: `docs/design/src_refactoring_plan_2026_02.md`  
**根拠データ**: `docs/design/src_structure_responsibility_improvement_sheet.csv`

---

## 1. 結論（要約）

- 全体として、**P0/P1の着眼点は概ね妥当**。
- ただし、`P2` の一部は「runtime+tests基準」で評価しているため、**実験用途を過小評価**している。
- 後方互換性は、分割そのものよりも **import path / 公開シンボル / CLI alias / SQLite契約** を壊すことが主リスク。

---

## 2. 計画・CSVの妥当性評価

### 2.1 主要数値

- 対象ファイル: 298
- `P0`: 7 / `P1`: 9 / `P2`: 185 / `P3`: 97
- `runtime_unreachable`: 188
- `runtime_and_test_unreachable`: 101

### 2.2 実験到達性を加味した再評価

`experiments/_archive_before_20260201_refactor` を除く現行実験を到達判定に追加した結果:

- experiments 起点で到達する `insightspike` モジュール: 59
- 既存CSVで `runtime_and_test_unreachable` のうち、実験から到達するもの: 2
  - `src/insightspike/algorithms/gedig/__init__.py`
  - `src/insightspike/algorithms/gedig/attention.py`

**解釈**:
- P0/P1の大枠は有効。
- ただし P2 の `archive_or_move` / `test_only_or_promote` は、実験依存を見て最終判定すべき。

---

## 3. 後方互換性リスク

## 3.1 互換性の主要面

1. CLI エントリポイント
- `pyproject.toml` の script:
  - `insightspike = insightspike.cli.legacy:main`
  - `spike = insightspike.__main__:run_app`
- `python -m insightspike.cli.spike` 実行経路（テスト利用）

2. import path 互換
- 直接importされている主要内部モジュール:
  - `insightspike.implementations.agents.main_agent`
  - `insightspike.implementations.layers.layer2_memory_manager`
  - `insightspike.implementations.layers.layer4_llm_interface`
  - `insightspike.algorithms.gedig_core`

3. CLI コマンド/alias 互換
- `query/embed/stats/insights/insights-search`
- alias: `chat/q/e/ask/learn/l`

4. DataStore 契約互換
- `save_episodes/load_episodes/search_episodes_by_vector`
- `save_graph/load_graph`
- `save_queries/load_queries`
- SQLite table契約（`episodes`, `queries` など）

## 3.2 P0/P1ごとの互換リスク

- `main_agent.py`（P0）: **高**
  - テスト・統合側の参照が最も多い。`MainAgent`, `CycleResult` の公開位置・挙動維持が必須。

- `cli/spike.py`（P0）: **高**
  - `-m insightspike.cli.spike` 実行および alias互換を壊しやすい。

- `layer2_memory_manager.py`（P0）: **中〜高**
  - L2生成・設定読み込み・EmbeddingManager patch点に注意。

- `layer4_llm_interface.py`（P0）: **中〜高**
  - `L4LLMInterface`, `LLMConfig`, `LLMProviderType`, `get_llm_provider` の import 互換必須。

- `sqlite_store.py`（P0）: **高**
  - 永続化契約が多数参照される。スキーマ変更は migration なし禁止。

- `gedig_core.py`（P1）: **高（実験直撃）**
  - maze実験の現行コードが直接依存。

---

## 4. 実験への影響（現行 experiments のみ）

対象: `experiments/` 配下（`_archive_before_20260201_refactor` 除外）

## 4.1 直接影響が大きいもの

1. `experiments/maze/run_experiment_query.py`
- 依存: `insightspike.algorithms.gedig_core.GeDIGCore`
- 影響: `gedig_core` 分割で import/挙動が変わると即停止

2. `experiments/maze/qhlib/evaluator.py`
- 依存: `insightspike.algorithms.gedig_core.GeDIGCore`
- 影響: 同上

## 4.2 間接影響（gedig package exports）

1. `experiments/transformer/extract_and_score.py`
2. `experiments/transformer/intervene_eval.py`

- 依存: `from insightspike.algorithms.gedig import AttentionGeDIGConfig, AttentionGeDIGCalculator`
- 影響: `algorithms/gedig/__init__.py` の export変更で停止

## 4.3 補足（P2誤判定になりやすい例）

- `src/insightspike/environments/maze.py`
- `src/insightspike/environments/proper_maze_generator.py`
- `src/insightspike/metrics/psz.py`

これらは runtime roots からは未到達扱いでも、現行実験が利用しているため、短期で archive しない。

---

## 5. 実施ルール（互換を壊さないための最低条件）

1. **import path互換レイヤーを先に置く**
- 分割後も旧パスで import できる shim を残す

2. **公開APIシンボル固定**
- 互換対象:
  - `MainAgent`, `CycleResult`
  - `L2MemoryManager`, `MemoryConfig`, `MemoryMode`
  - `L4LLMInterface`, `LLMConfig`, `LLMProviderType`, `get_llm_provider`
  - `GeDIGCore`

3. **CLIコマンド・alias固定**
- `query/embed/stats/insights/insights-search` と `chat/q/e/ask/learn/l`

4. **SQLite契約固定**
- メソッド名・戻り型・主要テーブル名を維持
- スキーマ変更は migration を同時導入

5. **実験依存モジュールは保護対象化**
- `gedig_core`
- `algorithms.gedig` の export
- `environments.maze`, `environments.proper_maze_generator`, `metrics.psz`

---

## 6. 推奨アクション（この順）

1. Phase 1 着手前に「互換対象シンボル一覧」を固定（ドキュメント化）
2. `gedig_core` は最初に shim 方針を確定（maze実験保護）
3. `algorithms.gedig.__init__` の export を凍結して transformer 実験を保護
4. P2の `archive_or_move` は実験到達性チェックを経てから実行

---

## 7. archive 実験への影響範囲（追加調査）

対象: `experiments/_archive_before_20260201_refactor`（`external/` 配下除外）

### 7.1 スコープ集計

- archive 配下の Python ファイル数（`external/` 除外）: `142`
- `insightspike` / `src.insightspike` を import しているファイル数: `22`
- 該当 import 行数: `39`

### 7.2 P0/P1 直撃範囲

- P0/P1のうち archive 実験が直接参照しているのは **`insightspike.algorithms.gedig_core`（P1）のみ**。
- `main_agent.py` / `cli/spike.py` / `layer2_memory_manager.py` / `layer4_llm_interface.py` / `sqlite_store.py` など、他のP0/P1対象への直接依存は未検出。
- `python -m insightspike...` や `spike query` 等の CLI 直接呼び出しは archive 側では未検出。

`gedig_core` を直接 import する archive ファイル:

1. `experiments/_archive_before_20260201_refactor/exp2to4_lite/src/gedig_scoring.py`
2. `experiments/_archive_before_20260201_refactor/hotpotqa-benchmark/src/hotpotqa_adapter.py`
3. `experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/qhlib/evaluator.py`
4. `experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/run_experiment_query.py`
5. `experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/tests/test_evaluator_ig_fallback.py`
6. `experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/tests/test_evaluator_ig_linkset.py`
7. `experiments/_archive_before_20260201_refactor/rag-dynamic-db-v3-lite/src/gedig_scoring.py`
8. `experiments/_archive_before_20260201_refactor/structural_similarity/science_history_simulation.py`

### 7.3 プロジェクト別の再現不能リスク

高（`gedig_core` 依存が実行経路にあり、フォールバックが弱い/なし）:

1. `maze-query-hub-prototype`
2. `hotpotqa-benchmark`
3. `structural_similarity`

中（`gedig_core` 依存はあるが lite fallback 実装あり）:

1. `exp2to4_lite`
2. `rag-dynamic-db-v3-lite`

低（今回の Phase 1/2 対象に直接依存しない）:

1. `rag_reranking`
2. `neuro_pruning`
3. `isomorphism_discovery`（`src.insightspike.algorithms.isomorphism_discovery` を参照）
4. `demo_flash_gedig.py`
5. `flash_gedig_validate.py`

### 7.4 影響が小さい（src 直接依存が未検出）archive ディレクトリ

1. `ablation_study`
2. `maze-sleep-phase2`
3. `preliminary`
4. `rag_cross_genre`
5. `structural_similarity_maze`
6. `transformer_gedig`

### 7.5 判断（再現不能を許容する場合）

- 「archive 実験はベストエフォート（再現保証なし）」方針なら、Phase 1/2 はそのまま進行可能。
- ただし、`gedig_core` の import path と `GeDIGCore.calculate` の戻り値契約（`gedig_value`, `hop_results`）を壊すと、上記の高リスク群は再現不能になる可能性が高い。

---

## 8. archive 実験の回収方針（再現可能化ガイド）

### 8.1 回収レベル定義

- `R1`（実行再現）: 主要エントリーポイントが起動し、完走する
- `R2`（結果再現）: 既存README/レポートと同種メトリクスを再生成できる

### 8.2 共通方針

1. `experiments/_archive_before_20260201_refactor` は「削除せず固定」し、回収は互換レイヤーで吸収する。
2. `src` 側で分割・移動する際は、旧 import path に shim を残す（少なくとも `R1` 期間）。
3. `gedig_core` 互換は別枠で保護する（`calculate` 入力引数・`gedig_value`/`hop_results` 契約を維持）。
4. 重依存（OpenAI/HF/datasets）は「実行再現」と「数値再現」を分離して判定する。

### 8.3 実験別回収プラン

| 対象 | 目標 | 重要互換点 | 回収方針 | 最低確認コマンド |
|---|---|---|---|---|
| `demo_flash_gedig.py` | R1 | `insightspike.gedig.compute_f_score`, `FlashGeDIGLoss` | `insightspike.gedig` の exportを固定。分割時は `__init__.py` で再公開。 | `python experiments/_archive_before_20260201_refactor/demo_flash_gedig.py` |
| `flash_gedig_validate.py` | R1 | `insightspike.gedig.compute_f_score` | 同上。`torch` 依存のため CPU 実行経路を維持。 | `python experiments/_archive_before_20260201_refactor/flash_gedig_validate.py --device cpu` |
| `exp2to4_lite` | R1→R2 | `GeDIGCore`, `GeDIGResult`, `build_linkset_info` | `insightspike.algorithms.gedig_core` と `insightspike.algorithms.linkset_adapter` に shim を残す。fallback（lite）を壊さない。 | `python -m experiments._archive_before_20260201_refactor.exp2to4_lite.src.run_experiment --config experiments/_archive_before_20260201_refactor/exp2to4_lite/configs/exp23_smoke.yaml` |
| `hotpotqa-benchmark` | R1（R2は外部API依存） | `GeDIGCore`, `decide_gates`, `StructuralSimilarityConfig`, `StructuralSimilarityEvaluator` | `gedig_core` と `gating`/`structural_similarity`/`config.models` を互換維持。OpenAI依存は mock/offline モードを補助線として許容。 | `python experiments/_archive_before_20260201_refactor/hotpotqa-benchmark/scripts/run_gedig.py --data experiments/_archive_before_20260201_refactor/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl` |
| `isomorphism_discovery` | R1 | `src.insightspike.algorithms.isomorphism_discovery` 旧import | 旧 `src.insightspike` import を `insightspike` import に置換、または import fallback を追加して両対応。 | `python experiments/_archive_before_20260201_refactor/isomorphism_discovery/benchmark.py` |
| `maze-query-hub-prototype` | R1（重要） | `GeDIGCore` API + `core.metrics`, `gedig.selector`, `sp_distcache`, `environments.maze`, `metrics.psz`, `layer3_graph_reasoner` | 最優先で `gedig_core` 契約を固定。必要なら `qhlib/compat_insightspike.py` を設置し、参照先を一元吸収。 | `python experiments/_archive_before_20260201_refactor/maze-query-hub-prototype/run_experiment_query.py --maze-size 25 --max-steps 20 --seeds 1 --output /tmp/mq_summary.json --step-log /tmp/mq_steps.json` |
| `rag-dynamic-db-v3-lite` | R1→R2 | `GeDIGCore`, `GeDIGResult` | 既存lite fallbackを維持。`INSIGHTSPIKE_MIN_IMPORT=1` 前提で軽量再現導線を残す。 | `python experiments/_archive_before_20260201_refactor/rag-dynamic-db-v3-lite/src/run_experiment.py --config experiments/_archive_before_20260201_refactor/rag-dynamic-db-v3-lite/configs/experiment_geDIG_vs_baselines.yaml` |
| `rag_reranking` | R1 | `StructureReranker`, `compute_f_score` | `insightspike.rag.reranker` と `insightspike.gedig` exportを固定。 | `python experiments/_archive_before_20260201_refactor/rag_reranking/eval_rerank.py` |
| `neuro_pruning` | R1 | `compute_f_score` | `insightspike.gedig` export維持。`datasets/transformers` のバージョンは固定化。 | `python experiments/_archive_before_20260201_refactor/neuro_pruning/prune_by_structure.py --help` |
| `structural_similarity` | R1→R2 | `StructuralSimilarityConfig`, `StructuralSimilarityEvaluator`, `GeDIGCore(structural_similarity_config=...)` | `GeDIGCore` の `structural_similarity_config` 引数と `result.gedig_value` を互換維持。 | `python -m experiments._archive_before_20260201_refactor.structural_similarity.analogy_benchmark` |
| `rag_cross_genre` | R2（間接依存） | 実体は `exp2to4_lite` 実行経路 | `exp2to4_lite` 回収を前提に再実行。データ生成スクリプトは独立維持。 | `python experiments/_archive_before_20260201_refactor/rag_cross_genre/scripts/generate_dataset.py --output /tmp/cross_genre_sample.jsonl --num-queries 20` |
| `structural_similarity_maze` | R1 | ローカルモジュール完結（`extended_gedig` 等） | `src` 非依存のため現状凍結。相対importのまま実行できるよう作業ディレクトリだけ固定。 | `python experiments/_archive_before_20260201_refactor/structural_similarity_maze/src/run_comparison.py` |
| `transformer_gedig` | R1 | `src` 直接依存なし（実験スクリプト群） | アーカイブ内完結として凍結。モデル/データ依存の再取得手順をREADMEに追記してR2へ。 | `python experiments/_archive_before_20260201_refactor/transformer_gedig/analyze_smoke.py --help` |
| `ablation_study`, `maze-sleep-phase2`, `preliminary` | R0 | 実行コードほぼ無し/資料中心 | コード回収対象から外し、資料アーカイブとして固定。必要時のみ個別復元。 | N/A |

### 8.4 実装順序（推奨）

1. `gedig_core` 互換 shim（最優先）
2. `insightspike.gedig` export固定（Flash/neuro/rerank保護）
3. `src.insightspike` 旧importの吸収（isomorphism保護）
4. archive高リスク3件（`maze-query-hub-prototype` / `hotpotqa-benchmark` / `structural_similarity`）の smoke 実行
5. 残りを `R1` ベースで段階回収
