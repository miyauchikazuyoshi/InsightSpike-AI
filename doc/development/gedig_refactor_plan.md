# geDIG Core リファクタ手順書

本ドキュメントは、`GeDIGCore` を論文 *geDIG ɿಈత஌ࣝάϥϑͷঢ়ଶΛଌΔ*（2025-10-22 版）に揃えるための
作業手順とテスト計画をまとめたものです。対象は `src/insightspike/algorithms/gedig_core.py`
およびそれに付随するメトリクス・実験コード一式です。

---

## リファクタ概要

### 目的
- g₀, gₕ (h ≥ 1) を論文本来の式  
  `g_h = ΔGED_norm − λ (ΔH_norm + γ ΔSP_rel)`  
  で評価できるようにし、構造コストが IG の逆数に潰れる現状を解消する。
- GED/IG/SP の正規化を、after サブグラフに基づく定義 (`C_max(S_h)`, `log(|S_cand|+1)`) に統一。
- multi-hop ステップでの ΔSP_rel（候補エッジの貪欲追加）を明示的に計算・記録し、ゲート判定 b(t)＝min{g₀, g_min} の再現性を向上。

### 影響範囲
- `src/insightspike/algorithms/gedig_core.py`
- `src/insightspike/algorithms/core/metrics.py`
- `experiments/maze-query-hub-prototype/run_experiment_query.py`（ログ項目の受け渡し）
- Layer3 Graph Reasoner や `src/insightspike/implementations/agents/main_agent.py` を含む、geDIG を参照するアプリ層
- HTML ビューア／JSON ログ畳み込みコード

---

## 進捗ログ

| 日付 | 担当 | 作業内容 | 備考 |
| ---- | ---- | -------- | ---- |
| 2025-10-22 | assistant | 現行コードの調査（`normalized_ged`, `GeDIGCore.calculate`) を実施し、仕様差分を整理。 | 本ドキュメント作成・影響範囲を反映。 |
| 2025-10-22 | assistant | `GeDIGCore` を更新し、g₀/gₕ を `ΔGED_norm − λ(ΔH_norm + γΔSP_rel)` で算出する新ロジックへ移行。`StepRecord` に ΔSP を追加。 | `version="onegauge_v1"` 系へ更新、HTML/JSON 連携の項目を追記予定。 |
| 2025-10-22 | assistant | 単体テスト（`test_gedig_core_local_norm.py` など）を更新・実行し、新分母（after グラフ基準）と `log(|S_cand|+1)` 分母を検証。 | `pytest tests/unit/test_gedig_core_local_norm.py tests/unit/test_core_metrics.py tests/unit/test_entropy_ig_fixed_den.py` |

---

## 実装手順

1. **論文式の抽出**
   - §3.2（Eq.(1)）・§3.5〜3.7（Eq.(10)〜(16)）を基準とした演算順序をコメント化しておく。
   - 計算に必要な入力：`S_cand`, `S_link`, `S_h`, `C_max(S_h)`, `ΔH_norm`, `ΔSP_rel`。

2. **GED 正規化を刷新**
   - `core/metrics.py::normalized_ged` を差し替え、after サブグラフ `S_h` を受け取り、
     `ΔGED_norm = (c_node * |ΔV| + c_edge * |ΔE|) / (c_node + c_edge * |S_h.edges|)` を返すようにする。
   - 効率・スペクトル項（`efficiency_weight`, `enable_spectral`）は初期リファクタでは無効化。
     必要であれば後段でオプトイン実装を検討する。

3. **IG 正規化を統一**
   - `entropy_ig` を ΔH_norm を返す API に整理。分母は `log(|S_cand| + 1)`.
   - multi-hop の際も同じ分母で統一（対象候補数が hop ごとに変わる場合はその数を利用）。

4. **ΔSP_rel 計算モジュールの新設**
   - 候補エッジ集合 `E_cand` から Eq.(13)(14) の greedy 追加を行う helper を `GeDIGCore` に追加。
   - 0-hop では ΔSP_rel＝0 を返す。
   - multi-hop 判定で `γ * ΔSP_rel` を足した値を ΔH_norm と合算する。

5. **GeDIGCore 内の合成ロジック更新**
   - `calculate()` と `_calculate_multihop()` を前述の指標で再構成。
   - `HopResult` に ΔGED_norm / ΔH_norm / ΔSP_rel をフィールド追加。
   - g₀/g_min の値、b(t)、AG/DG 判定ロジックの書式を論文アルゴリズム 1 に合わせて更新。

6. **インタフェース調整**
   - `StepRecord`・JSON ログ・HTML ビューアで新しいフィールドを可視化（ΔGED_norm, ΔH_norm, ΔSP_rel, g₀, g_min）。
   - ΔSP 系列を記録し、Maze ビューアや集計で確認できるようにする。
   - 旧式との互換が必要なモジュールには一時的なトグル（`use_legacy_metrics`）を導入。

7. **デフォルト設定の切り替え**
   - リファクタ後の計算をデフォルトにし、旧ロジックは非推奨として後日削除する。
   - CLI やコンフィグファイルから λ, γ を調整可能に整備。

---

## テスト計画

### 単体テスト（Unit）
1. `tests/unit/test_gedig_core_metrics.py`（新規）
   - 小さな before/after グラフで ΔGED_norm, C_max を検証（ノード追加・削除で期待値が出るか）。
   - `entropy_ig` が `log(|S_cand|+1)` を分母に採用しているか確認（|S_cand|=1,2 の境界）。
2. `tests/unit/test_sp_gain.py`（新規）
   - Greedy ΔSP_rel の手続きが Eq.(13)(14) 通りに進むか、複数ケースで検証。
3. `tests/unit/test_gedig_core_refactor.py`
   - g₀, g₁ を手計算できる最小グラフに対して `calculate()` の戻り値を比較。
   - 0-hop で ΔSP_rel=0 になること、multi-hop で ΔSP_rel>0 のとき g_min が低下することを確認。

### 結合テスト（Integration）
1. `tests/integration/test_query_hub_pipeline.py`
   - `run_experiment_query.py` を 5〜10 ステップのみ回すモードを追加し、JSON ログの g₀/g_min/ΔSP_rel が手動算出と一致するか比較。
2. Layer3 Reasoner / `main_agent` 系の smoke test。
   - `tests/unit/test_layer3_graph_reasoner.py` の期待値を新式に合わせて更新。
   - `tests/unit/test_main_agent.py`（新規予定）で geDIG 出力を用いた意思決定が変わらないか確認。

### パイプラインテスト（End-to-End）
1. **Maze Query-Hub**
   - 15×15/seed0 を λ=0.5/1.0/2.0 で回し、旧実装との比較レポートを出力。
   - 行き止まり（ΔSP=0, ΔH>0）で g₀ が正値になること、T 字路（multi-hop）で ΔSP_rel が効くことを確認。
2. **RAG PoC**
   - 500 クエリ・Phase1/Phase2 を切り替え、`w/o ΔSP`, `w/o ΔGED` など ablation を再実行。
   - Operating Curves (Accuracy vs. FMR) が論文の結果と整合するかを確認。
3. **Regression**
   - 既存の `tests/unit/test_layer3_graph_reasoner.py` などで g₀ の期待値が変わる箇所についてアサーションを更新。
   - `src/insightspike/implementations/agents/main_agent.py` を用いた end-to-end 評価（既存の Maze/RAG パイプラインから main agent を呼び出すシナリオ）を最低 1 ケース実行し、学習済みポリシーが異常終了しないことを確認。

---

## 戦略メモ

- リファクタ作業は「メトリクス純化 → GeDIGCore → 実験コード → 可視化 → テスト」という順で行うと、影響範囲を段階的に把握しやすい。
- `use_legacy_formula` / `use_multihop_sp_gain` 等の既存フラグは段階的に削除予定だが、互換性検証が終わるまでは残しておく。
- 大規模変更のため、feature ブランチで進め、Maze/RAG 両方の e2e レポートを添えてレビューに回すこと。
