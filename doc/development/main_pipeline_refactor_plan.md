# Main Pipeline geDIG リファクタ計画

## 目的
- geDIG Core の最新仕様（`g_h = ΔGED_norm − λ (ΔH_norm + γΔSP_rel)`）に合わせ、アプリケーション層（Layer3 Reasoner / main_agent / RAG パイプライン）が依存するロジックを更新する。
- `structural_improvement` など旧 API に依存した設計を整理し、ΔGED_norm / ΔSP_rel など明示的な指標を共有する。
- RAG 実験および洞察ベクトル生成実験の処理フローを論文フロー（Phase1/Phase2）の最新定義に揃える。

## スコープ
- `src/insightspike/implementations/agents/main_agent.py`
- `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
- `src/insightspike/algorithms/gedig/selector.py`（呼び出し経路の調整）
- RAG 系パイプライン（`experiments/rag-dynamic-db-v3-lite/` ほか）
- エネルギーモデル／スパイク検出 (`layer3_graph_reasoner`, `maze_experimental` 系)

## 実装ステップ
1. **指標整理と API 調整**
   - geDIG Core に `delta_ged_norm` / `delta_sp_rel` / `delta_h_norm` を返すプロパティを追加し、アプリ層で直接参照できるようにする。
   - `structural_improvement` の互換問題を解決：暫定的に `-ΔGED_norm` を返しつつ、呼び出し側の置き換えを完了した段階でフィールド自体を廃止する。

2. **Layer3 Reasoner 更新**
   - スパイク判定・energy 計算が `structural_improvement > 0` など旧ロジックに依存している部分を `delta_ged_norm` / `delta_sp_rel` に基づく判定へ置き換える。
   - geDIG ゲート（AG/DG）の閾値比較を g₀/g_min に一本化。旧式 `ΔgeDIG` 閾値との重複を撤廃。

3. **main_agent / プランナー更新**
   - レイヤー連携時に g₀/g_min を渡す箇所を見直し、`candidate_selection` の ΔSP 表示や `delta_ged_norm` を UI/ログへ反映。
   - Backtrack/Prune ロジックを最新の b(t)=min{g₀,g_min} に沿って再実装。

4. **RAG パイプラインの整合**
   - Phase1 の「オンライン geDIG ヒューリスティック」を `ΔGED_norm` / `ΔSP_rel` で再評価。
   - Phase2 の「洞察ベクトル生成（MDL フロー）」について、Eq.(13)(14) の greedy ΔSP 設計と矛盾する箇所を洗い出す。
   - README や config テンプレートに正規化手順（`log(|S_cand|+1)` 等）を追記。

5. **テスト整備**
   - Unit: Layer3 Reasoner／main_agent の新ロジックに対するテスト (`tests/unit/test_layer3_graph_reasoner.py`, 新規 `test_main_agent_decision.py`)。
   - Integration: Maze pipeline の短縮ランを用いた g₀/g_min / b(t) 検証、RAG パイプラインの A/B 比較。
   - End-to-End: 代表的な Maze seed と RAG config での再実験・HTML レポート再生成。

## README / フローの矛盾ポイント
現状の README（root・実験系）には以下の齟齬があるため、メインコード更新と合わせて改訂が必要。

1. `geDIG = GED - k*IG (k=0.5)` と記載されているが、最新仕様では `ΔGED_norm` と `ΔH_norm`（`log(|S_cand|+1)` 分母）および `ΔSP_rel` を用いるため、定数 k の説明が不正確。
2. ΔgeDIG のしきい値判定（ΔgeDIG < −τ 等）に言及しているが、実装は AG/DG ゲート（g₀/g_min）ベースに移行済み。
3. Center node 中心のグラフ構造を前提にした説明が残っている（Query Hub プロトタイプではクエリノードがハブ）。
4. Phase2 の洞察ベクトル生成フローに MDL／ΔSP_rel の具体式が未反映で、メモリ拡張手順と論文式が対応していない。

これらの記述もリファクタ完了時に更新・追記する。

## 進捗ログ

| 日付 | 担当 | 作業内容 | 備考 |
| ---- | ---- | -------- | ---- |
| 2025-10-22 | assistant | geDIGCore に `delta_ged_norm` / `delta_sp_rel` / `delta_h_norm` を追加し、Layer3 Reasoner・main_agent へ伝搬。既存メトリクス辞書を拡張し、エピソード管理の判定を正規化半径ベースに更新。 | 互換性維持のため `delta_ged` は暫定的に負値で保持（従来判定用）、`delta_ged_norm` / `delta_h_norm` / `delta_sp` を新規公開。 |
| 2025-10-22 | assistant | geDIGCore の報酬・スパイク判定を `delta_ged_norm` / `delta_h_norm` / `delta_sp_rel` ベースに刷新し、`structural_improvement` 依存を解消。 | Layer3/main_agent が参照するスコアも新指標へ移行。 |
| 2025-10-22 | assistant | GraphAnalyzer のユニットテストを追加し、`delta_h`／`delta_sp` が確実に出力されることを検証。 | `tests/unit/test_graph_analyzer_metrics.py` 追加。 |
| 2025-10-22 | assistant | Layer3 Reasoner の query-centric 経路を簡易検証するユニットテストを追加（lite モード時は skip）。 | `tests/unit/test_layer3_query_metrics.py` 追加。 |
| 2025-10-22 | assistant | Maze クエリのスモークテストを実施し、g₀/g_min/ΔH/ΔSP が JSON ログに反映されることを確認。 | `run_experiment_query.py --max-steps 30` の結果（tmp_smoke_*）を確認。 |

| 2025-10-23 | assistant | GeDIGCore に linkset モードを追加し、`--linkset-mode` フラグから S_link ベースの ΔGED/ΔH を評価可能に。迷路プロトタイプで新指標（link_delta_*）を記録。 | 今後は旧フローと併用しながら比較可能。 |
## 今後の進め方
- まず `structural_improvement` の互換対応と Layer3 Reasoner の閾値見直しを実施し、Maze 実験でリグレッション確認。
- 続いて RAG パイプラインの整理と README 改訂を行い、Phase1/Phase2 のコードフローと文書フローを完全同期させる。
- 各段階で `doc/development/gedig_refactor_plan.md` へ進捗を追記し、テスト結果（パス/未対応）を明示する。
