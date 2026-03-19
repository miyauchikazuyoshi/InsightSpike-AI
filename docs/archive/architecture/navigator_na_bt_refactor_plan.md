# Navigator NA/BT リファクタリング計画書（2025-10-17）

## 目的
- Navigator の時間軸処理を「観測→思考→移動→事後処理」の明確なフェーズに再編し、NA/BT 判定が期待するタイミングで発火するようにする。
- geDIG 計算および HTML 可視化で参照されるメトリクスが、同一ステップ内で整合した値になるように責務を分離する。
- 既存の NA（No-Advance）・BT（Backtrack）フローを段階的探索へ拡張できるコード構造を整備し、将来的な hop 深度制御や SP 利得活用に備える。

## 背景と現状課題
- `_think_phase` 内で NA フラグを edge 側に付与しているが、実際の移動は `_move_phase` 側で完了するため HTML 上では 1 ステップ遅れに見える。
- 観測データの登録・クエリ生成・候補ノード探索・仮想エッジ生成が混在しており、どの時点でグラフが更新されたのか追跡しづらい。
- geDIG の IG 成分が一部ステップで 0 のまま固定されるなど、評価対象の候補セットがフェーズ間で失われている疑いがある。
- NA/BT が未発火の場合と発火した場合で移動ロジックの入口が分岐し、デバッグ時に責務が追いづらい。

## スコープ
- 対象モジュール：`experiments/maze-online-phase1-querylog/src/navigator.py`（run ループ、フェーズ分割、NA/BT 判定、仮想配線）
- 付随修正：`experiments/maze-online-phase1-querylog/src/gedig_adapter.py`（必要ならば geDIG 呼び出し API 調整）、`vector_proc.py`（クエリ生成のインターフェイス整理）
- 可視化：`experiments/maze-online-phase1-querylog/visualization/generate_report.py`（タイムライン同期、NA/BT マーカーの整合）

## タイムラインフェーズ仕様

| フェーズ | 対応メソッド（予定） | 主な責務 | 入力 | 出力 / 副作用 |
|----------|----------------------|-----------|------|----------------|
| **Observation** | `_observe_phase` | 周囲観測、エピソード登録、`local_entries` 構築 | 現在位置、環境 | `StepObservation`（観測エピソード、クエリ素案、死角判定） |
| **Query Injection** | `_build_query_context`（新設） | 観測時に生成したクエリを確定し、`entry_set.query` と `weighted_query` を準備 | Observation | `QueryContext`（クエリベクトル、重み付け一貫性） |
| **Thinking / Evaluation** | `_think_phase` | 仮想グラフ構築、geDIG 評価、候補ソーティング、NA/BT 判定、計画立案 | Observation, QueryContext, `prev_graph` | `StepDecision`（NA/BT 判定、選択候補、計画結果、評価コンポーネント） |
| **Action Execution** | `_move_phase` | 実移動の決定・実行、仮想→実エッジ反映、訪問回数更新 | StepObservation, StepDecision | 実グラフ更新、環境移動、`edge_events` 追加 |
| **Post-Move Logging** | `_finalize_step`（新設） | ステップログ追記、NA/BT/HP の視覚化用イベント整備、メトリクス更新 | StepObservation, StepDecision, Move結果 | `stats` 更新、HTML 用タスク整列 |

### フェーズ間データの扱い
- `StepObservation` は観測結果のみを保持し、移動候補に手を触れない。
- `QueryContext` を介してクエリベクトルと重み付けを共有し、`_think_phase` 以降で再計算を避ける。
- `StepDecision` は意思決定の最終結果のみ（NA/BT 判定、選択枝、スコア）を渡し、実際の移動は `_move_phase` に限定。
- `_move_phase` の戻り値は「移動が完了したか」「run ループを抜けるか」のみを返し、副作用は `_finalize_step` で一元化。

## シナリオ別フローチャート

### 1. NA 未発火シナリオ
```
Observation → Query Injection → Thinking
  └─ geDIG 評価 (hop0)
  └─ NA 判定: false
  └─ 候補ソート → selected_branch 決定
→ Action Execution
  └─ selected_branch に従って移動
→ Post-Move Logging
  └─ edge_events に移動結果を記録
  └─ HTML タイムラインへ同一ステップで反映
```

### 2. NA 発火 & BT 成功シナリオ
```
Observation → Query Injection → Thinking
  ├─ geDIG 評価 (hop0) → g₀ > θ_NA
  ├─ NA 判定: true（step 時点で node_events に `na_trigger` を記録）
  ├─ multihop geDIG で hop 深度を 1→n と段階的に探索
  ├─ SP 利得 > 閾値 → BT ターゲット確定
→ Action Execution
  └─ `_pending_moves` と BT 経路に従って移動開始
→ Post-Move Logging
  ├─ BT 開始ステップに `bt_trigger` マーカー追加
  └─ NA/BT 判定値をタイムラインに反映
```

### 3. NA 発火 & BT 不成立シナリオ
```
Observation → Query Injection → Thinking
  ├─ geDIG 評価 (hop0) → g₀ > θ_NA
  ├─ NA 判定: true
  ├─ multihop geDIG 走査 → SP 利得なし
  ├─ fallback: 未探索 branch 探索 or local 再挑戦
→ Action Execution
  └─ fallback で選んだ候補へ移動／待機
→ Post-Move Logging
  └─ NA 発火情報と fallback 理由を記録
```

## データ更新タイミングの仕様
- NA フラグは **Thinking** フェーズで `node_events` に書き込み、`position` はトリガー時点のノード座標を用いる。移動後のエッジにフラグを付ける場合は `_finalize_step` で当該ステップの `edge_events` に追記。
- 仮想エッジ（threshold 以下の候補）は **Thinking** フェーズの仮想グラフのみ更新。実エッジへの昇格は **Action Execution** フェーズの結果次第で行う。
- g₀/gₘᵢₙ/IG などのメトリクスは **Thinking** フェーズの出力を `StepDecision.metrics`（新設予定）にまとめ、**Post-Move Logging** でそのまま HTML に渡す。

## リファクタリング実行ステップ
1. **構造整理**（完了: 2025-10-17）: `run()` を Observation → Query → Thinking → Action → Post-Move のハーネスに整理し、`QueryContext`／`StepMovement` を導入。`StepDecision` に `metrics` を追加。
2. **NA/BT 判定の整理**（完了: 2025-10-17）: Thinking フェーズで NA/BT 判定を完結させ、`StepDecision.events` にトリガ情報を格納。`_finalize_step` が node イベントへ反映する形に統一。
3. **仮想→実エッジ昇格の見直し**（進行中: 2025-10-17）: `_cleanup_virtual_node()` を昇格対応へ拡張し、仮想ノードの属性/接続を実ノードへ移しつつ `virtual_promoted` イベントを付与。未選択候補は `observed_cache` に残す方針を継続しつつ、遠方 branch の保持状況を継続検証。
4. **可視化連携**：`generate_report.py` でタイムライン更新順を `step` 単位にそろえ、`na_trigger`・`bt_trigger` を node イベントとしても描画する。
5. **挙動検証**：15×15（seed 19, 20）および 25×25 で実験を再実行し、g₀ の立ち上がりタイミングと HTML 表示が一致することを確認。疑似的ユニットテストも追加検討。

## 検証計画
- **数値検証**：`stats.g0_history` と HTML 表示値が一致するかをステップ単位で確認するスクリプトを追加（オプション）。
- **イベント整合性**：`node_events` と `edge_events` の `step` が単調増加 & タイムラインと同期していることをチェック。
- **パフォーマンス**：仮想配線の検索対象を広げるため、25×25 の走行時間を測定し、既存ベースラインと比較。

## リスクと緩和策
- **フェーズ分割による回帰**：既存テスト（手動）に加え、最小限の自動テストで run() の基本シナリオをカバー。
- **仮想ノード肥大化**：未使用仮想ノードのクリーンアップ方針を `_finalize_step` 後に検討（現段階では retain し、追ってメモリ監視）。
- **HTML への影響**：可視化スクリプト更新時は Node.js テスト（`npm run test:viz` 相当）でリグレッション確認。

## 進捗メモ（2025-10-17）
- size15 / seed19-20, size25 / seed19-20 を `θ_NA=0.59, θ_BT=1.48` で再実行し、`results/na_bt_refactor_reports` に HTML を再生成。
- `run_data.json` の `na_trigger` イベントは `position_before/after` を保持し、タイムラインと一致した位置で描画できる状態を確認。
- `g0_components` に IG 分母・エントロピー差分が埋まっていることを確認（例: seed19 step15 で `entropy_after`=0.8085、`ig_delta`=-0.8085）。
- `bt_trigger` / `virtual_promoted` イベントを `_finalize_step` / deferred edge 仕組みで記録し、HTML へ整合的に反映。
- スナップショット検査用ユーティリティ `scripts/check_g0_snapshot.py` を追加し、`run_data.json` / 生 JSON 双方に対して g₀/gₘᵢₙ/IG 整合性を自動チェック可能にした（size15_seed19 / size25_seed19-20 で実行済み）。

---

この計画書の完成をもって、次ステップ（フェーズ分割と NA/BT 判定処理の実装）へ着手する。
