# Adapter Migration Plan — 既存実験コードの統合

## 概要

`src/gedig/` の core + adapter が完成 (44/44 tests pass)。
次は既存実験コードを adapter 経由に切り替える。

**原則:**
- 既存コードは即座に削除しない
- `use_unified=True` フラグで新旧切り替え
- 数値等価性テストに合格したら切り替え
- 旧コードは DeprecationWarning → 2リリース後に削除

---

## Phase 4a: Transformer 移行 (最低リスク)

### 対象ファイル
- `experiments/transformer/thermodynamic_gedig.py`
  - `DifferentiableGeDIG` クラス (lines 46-234)
  - `run_training_time_intervention_comparison()` (lines 1002-1244)

### 移行手順

1. **`DifferentiableGeDIG.forward()` に adapter 分岐を追加**
   ```python
   def forward(self, before_attention, after_attention, mask=None):
       if self.use_unified:
           from gedig.adapters.transformer import TransformerFEval
           adapter = TransformerFEval(
               lambda_param=self.lambda_param,
               gamma=self.gamma,
               percentile=self.percentile,
               temperature=self.temperature,
               use_betti=self.use_betti,
           )
           result = adapter.compute(before_attention, after_attention, mask)
           return {
               "F": result.F, "F_mean": result.F_mean,
               "delta_epc": result.delta_epc, "delta_h": result.delta_h,
               "delta_sp": result.delta_sp, "delta_b1": result.delta_b1,
               "use_betti": result.use_betti,
           }
       # ... existing code ...
   ```

2. **CLI フラグ追加**: `--use-unified`

3. **`__init__` にフラグ追加**: `use_unified: bool = False`

### テスト条件

| # | テスト | 合格基準 | 方法 |
|---|--------|---------|------|
| T1 | 数値等価性 (SP版) | \|F_old - F_new\| < 1e-4 | 同一 attention (2,4,32,32) × 100 samples |
| T2 | 数値等価性 (β₁版) | \|F_old - F_new\| < 1e-4 | 同上、use_betti=True |
| T3 | 勾配等価性 | \|grad_old - grad_new\| < 1e-3 | autograd.gradcheck |
| T4 | Exp4 再現 | 同一 conclusion (negative_better) | SP版 + β₁版 両方 |
| T5 | 速度回帰なし | new/old < 1.5x | 100 forward の wall time |

### 切り替え判定
- T1-T5 全 pass → `use_unified=True` をデフォルトに
- 旧コードに `DeprecationWarning` 追加
- 1ヶ月後: 旧コード削除

---

## Phase 4b: RAG 移行 (中リスク)

### 対象ファイル
- `experiments/hotpotqa_v2/src/unified_graph.py`
  - `compute_qkv_attention()` (lines 378-593) — F-eval 部分のみ
  - `graph_attention_propagation()` (lines 600-640)
- `experiments/hotpotqa_v2/src/gedig_scoring.py`
  - `entity_graph_feval_scores()` (lines 816-848)

### 移行手順

1. **`compute_qkv_attention()` 内の F-eval ループを adapter に委譲**
   ```python
   if use_unified:
       from gedig.adapters.rag import RAGFEval
       rag_f = RAGFEval(f_lambda=config.f_lambda, d_k=3.0)
       for u, v in graph.edges():
           f_val = rag_f.compute_edge_f(cost, q_vec, k_vec)
           ...
   ```

2. **`graph_attention_propagation()` を adapter に委譲**
   ```python
   if use_unified:
       from gedig.adapters.rag import RAGFEval
       rag_f = RAGFEval()
       propagated = rag_f.propagate(graph, init_rel, n_iterations, alpha)
   ```

3. **CLI フラグ**: `--use-unified-feval`

### テスト条件

| # | テスト | 合格基準 | 方法 |
|---|--------|---------|------|
| R1 | per-edge F値等価性 | \|f_old - f_new\| < 1e-6 | HotpotQA 10q の全エッジ |
| R2 | AG/DG 分類一致 | 100% 一致 | 同上 |
| R3 | propagation 等価性 | \|rel_old - rel_new\| < 1e-6 | 全ノード relevance |
| R4 | HotpotQA R@2 一致 | \|R@2_old - R@2_new\| < 0.01 | 100q |
| R5 | HotpotQA SF_F1 一致 | \|SF_old - SF_new\| < 0.01 | 100q |
| R6 | BRIGHT nDCG 一致 | \|nDCG_old - nDCG_new\| < 0.005 | 50q biology |

### 切り替え判定
- R1-R6 全 pass → `use_unified_feval=True` をデフォルトに
- unified_graph.py の旧 F-eval コードに DeprecationWarning
- entity_graph.py / gedig_scoring.py は当面維持 (AGHT 以外のモードで使用)

---

## Phase 4c: Maze 移行 (高リスク)

### 対象ファイル
- `experiments/maze/qhlib/evaluator.py`
  - `evaluate_multihop()` — F-eval 計算部分
- `experiments/maze/run_experiment_query.py`
  - AG/DG fire 判定 (lines 2433-2435)
  - Sleep propagation 呼び出し (lines 4354-4411)

### 移行手順

1. **`evaluator.py` に adapter 分岐追加**
   ```python
   if use_unified:
       from gedig.adapters.maze import MazeFEval
       maze_f = MazeFEval(lambda_param=..., gamma=..., theta_ag=..., theta_dg=...)
       result = maze_f.evaluate_hop(before_graph, after_graph)
       g_value = result.f_value
   ```

2. **Sleep propagation を adapter に委譲**
   ```python
   if use_unified:
       propagated = maze_f.sleep_propagate(graph, rewards, gamma, n_iters)
   ```

3. **CLI フラグ**: `--use-unified-feval`

### テスト条件

| # | テスト | 合格基準 | 方法 |
|---|--------|---------|------|
| M1 | g0 値等価性 | \|g0_old - g0_new\| < 1e-4 | 9x9 maze, seed 0, 50 steps |
| M2 | AG/DG fire 一致 | 100% 一致 | 同上 |
| M3 | sleep propagated 等価性 | \|p_old - p_new\| < 1e-4 | 全ノード |
| M4 | 15x15 success rate | \|rate_old - rate_new\| < 5% | 5 seeds × 500 steps |
| M5 | 25x25 avg steps | \|steps_old - steps_new\| < 10% | 3 seeds × 500 steps |

### 切り替え判定
- M1-M5 全 pass → `use_unified_feval=True` をデフォルトに
- evaluator.py の旧コードに DeprecationWarning

---

## Phase 5: 旧コード削除スケジュール

```
Week 0:  Phase 4a (transformer) 移行 + テスト
Week 1:  Phase 4b (RAG) 移行 + テスト
Week 2:  Phase 4c (maze) 移行 + テスト
Week 3:  全実験で use_unified=True デフォルト化
Week 4:  旧コードに DeprecationWarning 追加
Week 8:  旧コード削除 (破壊的変更)
```

### 削除対象 (Week 8)

| ファイル | 削除対象 | 残すもの |
|---------|---------|---------|
| `thermodynamic_gedig.py` | `_compute_soft_edges`, `_compute_entropy`, `_compute_path_efficiency`, `_compute_betti_1` | `forward()` (adapter 委譲のみ) |
| `unified_graph.py` | F-eval ループ内の手動 Q/K 計算 | `build_unified_graph()`, `score_documents()` |
| `gedig_scoring.py` | `_local_ged()`, `_local_entropy()`, `_local_sp_gain()` | `GeDIGScoringEngine` (adapter 経由) |
| `evaluator.py` | 旧 `_compute_sp_gain_norm()` | hop-series ロジック (adapter 経由) |

---

## 実行コマンド

### Phase 4a テスト
```bash
# T1: 数値等価性
PYTHONPATH=src .venv/bin/python3 -m pytest src/gedig/tests/test_transformer_equiv.py -v

# T4: Exp4 再現
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python3 experiments/transformer/thermodynamic_gedig.py \
  --experiment training --num-samples 200 --beta 0.1 --use-unified \
  --output-dir experiments/transformer/results/thermodynamic_exp4_unified
```

### Phase 4b テスト
```bash
# R4: HotpotQA 100q
.venv/bin/python3 experiments/hotpotqa_v2/scripts/eval_hotpotqa_aght.py \
  --limit 100 --use-unified-feval \
  --output experiments/hotpotqa_v2/results/v37_unified_hotpotqa
```

### Phase 4c テスト
```bash
# M4: 15x15 maze
PYTHONPATH=src INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 \
.venv/bin/python3 experiments/maze/run_experiment_query.py \
  --maze-size 15 --max-steps 500 --seed 0 --use-unified-feval
```
