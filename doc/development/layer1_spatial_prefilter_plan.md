# Layer1 空間プリフィルタ導入計画（グリッド索引＋リング探索）

目的は、候補集合（特にメモリ由来の方向ノード）を「空間的に近いもの」に事前絞り込みし、重い距離計算やSP評価に到達する前に母集団を縮小して全体の計算量を抑えること。迷路実験ではセル座標に基づく索引を用い、RAG+SentenceBERT では既存のANN/インデックスに委ねる（本計画の導入はデフォルトで無影響）。

## スコープ
- 対象: 迷路・QueryHub 実験（`experiments/maze-query-hub-prototype`）、将来的な Layer1 一般化（オプション）。
- 非対象: ANN 実装の入替（Faiss/hnswlib）そのもの、SentenceBERT 埋め込みの学習・推論ロジック。

## 設計概要
- SpatialGridIndex（済）
  - API 例: `add(anchor=(r,c), node_id)`, `iter_cells_rect/iter_nodes_ellipse`。
  - dict[(r,c)]→set(node_id)、Add-only。長方形/楕円ウィンドウ対応。
- リング探索プリフィルタ（済）
  - 現在位置 p=(r,c) の近傍セルのみ候補化。Rr=ceil(r_link*H), Rc=ceil(r_link*W)。
  - フォールバック条件（索引無/巨大窓）では全走査。
- Layer1 ベクトルプリフィルタ（新規・実装済）
  - `qhlib/l1index.py` WeightedL2Index で abs_vector を重み付きで保持し、weighted L2 のtop‑K検索。
  - ランナーに `--layer1-prefilter`（ONでL1）と `--l1-cap`（Top‑K）を追加。ON時はリングを使わずL1で候補化。
- TwoThresholdCandidateSelector は温存（距離/半径/Top‑K 判定はそのまま）。
- 評価範囲は A_core（現在Q）と A_topL（hop0採用dir）のk‑hopユニオン（両側から拡張）。

## 互換性/フラグ
- 迷路: 既定はリング（rect/ellipse切替）。`--layer1-prefilter` でL1ベクトルプリフィルタに切替可能。
- メインコード（RAG+SentenceBERT）: 既定OFF（no-op）。将来的にVectorIndexへ昇格可能。

## 実装タスク（フェーズ）
1) 抽出（Phase A: 完了）
2) Ecand 配線（Phase B: 完了）
3) セレクタ・フック（Phase C: 完了）
4) Layer1 プリフィルタ（Phase C’: 完了）
5) 計測（Phase D: 進行中）
6) 文書化/清掃（Phase E: 本書）

## テスト計画（詳細）

### 単体テスト（迷路・共通）
- SpatialGridIndex
  - 追加/重複登録/境界（0/H-1, 0/W-1）での neighbors 正常性。
  - 空索引/巨大リング時のフォールバック経路を経由しても候補が全取得されること。
- リング探索プリフィルタ
  - Rr=Rc=0 で現在セルのみ返すこと。
  - リングを十分大きくした場合、従来の全走査結果と候補集合が一致すること（順序は不問）。
  - 楕円チェック ON/OFF で候補数が単調非増（ON の方が小さい）であること。
- 候補レコード整合
  - pos_key、relative_delta、abs/vector、visit_count 等が従来と一致。
  - 強制リンク（forced）付与の挙動に変化がないこと。
- build_ecand
  - mem_count/qpast_count の集計、重複除去、prev_graph に存在しない direction は除外のルールが守られる。

### 単体テスト（Core/Evaluator）
- IG 符号オプション（before_after / after_before）切替時の符号/値の整合性。
- EPC 増分 GED vs 完全再計算 GED の一致（正規化分母が固定台に従うこと）。
- AG 早期打ち切り: `g0 ≤ θAG` で hop>0 の評価をスキップし、ホップ系列の最小が `g0` になる。
- DG コミット: best_hop 条件・予算の下でのみコミットされる。
- SP（Core 準拠）
  - 固定ペア/union/trim の境界設定で Lb>0 の相対短縮が [0,1] にクリップされること（負は0）。

### パイプライン/統合テスト（迷路）
- 40 ステップ標準（Strict DS 表示）
  - 浮遊ノードなし、Temporal/Per-hop 整合、bestHop/g_min一致。
  - 候補/リング/距離計算件数ログの妥当性。
- 120/200 ステップ（θAG=0.45/0.50, max_hops=12/20）
  - 楕円/矩形/L1 の cand_ms/eval_ms/距離評価件数の比較。
  - g0は主に行き止まりで発火、DGで早期停止の設計通り。
- A/B 検証（回帰）
  - プリフィルタOFF vs L1上位大（cap大）で選択一致。
  - L1 vs 楕円/矩形でプロファイル改善の確認（速度/件数）。

### パイプライン/統合テスト（RAG+SentenceBERT）
- Dynamic RAG v2/v3-lite の最小構成を実行
  - 例外なし、指標（正答率/信頼度/処理時間）が既存結果±許容差内。
  - geDIG 呼び出しは従来経路（Selector→Core）を通り、Layer1 プリフィルタは無効化（デフォルト）のまま。

### 後方互換性テスト
- 既存ログ/設定（迷路/RAG）でのビルド・実行がすべて成功。
- プリフィルタをOFFにすれば完全に従来と一致（A/B 比較）。

## 計測/モニタリング
- 各ステップで以下を記録（迷路）
  - 候補数（obs/mem/forced）、Ecand 数、リングセル数、距離計算件数。
  - g0、g_min、bestHop、ΔGED/ΔIG/ΔSP。
  - 実行時間ブレークダウン（候補生成/評価/スナップショット）。

## ロールアウト戦略
- 迷路: 既定はリング（rect/ellipse切替）。`--layer1-prefilter` でL1。`--l1-cap`でTop‑K調整。
- RAG: 既定OFF。configで無効のまま。

## リスクと対策
- 取りこぼし（リングが小さ過ぎる）
  - 大リング/フォールバック/楕円ONの3段構えで回避。RAG側は既定無効。
- 索引の一貫性
  - 追加のみ・削除不要の設計。重複登録は set 管理で自然に吸収。
- 表示・整合性
  - Strict DS モードを既定化し、UI は DS のみを信頼（浮遊ノードの再発防止）。

## スケジュール（目安）
- Phase A/B（抽出＋Ecand配線）: 1–2日
- Phase C（セレクタ・フック、任意）: 0.5日
- Phase D（計測・調整）: 0.5日
- テスト一式・A/B 検証: 1–2日

## 完了の定義 (DoD)
- 単体・統合テストが全て緑。
- 40/200 ステップの代表ジョブでレポート/HTML 整合。
- RAG v2/v3-lite の最小ケースが回帰なし（既存指標±許容差）。
- README/計画書更新、CLI/設定の記述反映。

## バックアウトプラン
- 迷路: フラグで OFF、または runner を以前の全走査実装へ戻す。
- RAG: 既定OFFのため影響なし。設定キーを削除しても挙動は従来通り。

---
参考ファイル
- 迷路 PoC（実装済み）: `experiments/maze-query-hub-prototype/run_experiment_query.py`
- エッジ生成: `experiments/maze-query-hub-prototype/qhlib/edges.py`
- セレクタ: `src/insightspike/algorithms/gedig/selector.py`
- レポート生成: `experiments/maze-query-hub-prototype/build_reports.py`


## 進行状況（2025-10-29）

- 実装フェーズの進捗
  - Phase A（SpatialGridIndex 抽出）: 完了
  - Phase B（Ecand ring 配線）: 完了（`ring_center/ring_size/ellipse` を追加。後方互換）
  - Phase C（Selector prefilter フック）: 完了（`prefilter_fn` 追加。既定は no‑op）
  - Phase C'（Layer1 ベクトルプリフィルタ）: 完了（`qhlib/l1index.py`、`--layer1-prefilter`, `--l1-cap`）
  - Phase D（計測）: 進行中（Step Infoに `time_ms_candidates/eval`, `ring_cells/nodes`, `dist_evals` を表示）
  - Phase E（文書/清掃）: 本ドキュメント更新中

- Evaluator/可視化の安定化（関連）
  - AGゲート境界: `g0 < θAG` のみスキップ（`g0 == θAG` は評価）
  - DG判定: マルチホップのみ（`best_hop>=1` かつ `gmin_mh < θDG`）
  - HTML: g0（青折れ線）を前面レイヤに、g_min（赤点）は hop0 を非表示、SPリング（紫破線）を重ね描き、強制線（赤破線）をDS Strictでも常時オーバーレイ

- AB 実行ログ（120 step, seed=0, `sp_cand_topk=64`）
  - θAG=0.45, H=12
    - Rect: `results/_ab_rect_gate045_h12_steps.json`
    - Ellipse: `results/_ab_ellipse_gate045_h12_steps.json`
      - プロファイル例: Rect vs Ellipse → cand_ms: 0.83 vs 0.69 ms, eval_ms: 126.97 vs 84.04 ms, dist_evals: 4352 vs 3208
  - θAG=0.50, H=12
    - Ellipse: `results/_ab_ellipse_gate05_h12_steps.json`
  - θAG=0.45, H=20（深い行き止まり対策）
    - Rect: `results/_ab_rect_gate045_h20_steps.json`
    - Ellipse: `results/_ab_ellipse_gate045_h20_steps.json`
  - L1プリフィルタ（40 step スモーク）: `results/_l1pref_40_steps.json`

- 次アクション（AB 続き）
  - H=20, θAG=0.45 で L1 プリフィルタ（`--layer1-prefilter --l1-cap 128`）の120 stepを追加実行し、リング（Rect/Ellipse）と比較
  - 必要に応じて `sp_cand_topk=32` で同条件AB
  - Evaluator に SP 近似キャッシュ（`sp_cache=cached`）配線を追加し、速度/挙動のAB比較
  - HTMLに表示トグル（AG-only g_min, SPリングON/OFF, 線/点サイズ）を追加（視認性向上）
  - ユニット/統合テスト雛形の投入（Spatial/L1/Selector/Evaluator）
