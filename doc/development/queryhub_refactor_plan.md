# Query‑Hub Refactoring Plan (責務分離計画)

目的（Goals）
- エージェント実行の責務分離と見通し改善（モノリシックな実行スクリプトからの脱却）
- シークエンス（観測→候補→評価→コミット→永続化）の責務分離（エッジ生成と数値計算の分離）
- データ責務分離（迷路呼び出し、データストア、Step ID、各種数値メトリクスの永続化）
- 表示用データの責務分離（UIはステップ完了時点のStep IDを受け取り、DSから都度引く）
- DS（データストア）を唯一の真実源（SSOT）とし、steps.jsonは軽量メタのみにする（スナップショット縮減）

非目標（Non‑Goals）
- geDIGCoreのアルゴリズム自体の改変（本計画ではインターフェース化と呼び出し点の整理を優先）
- UIの大幅刷新（段階的に必要箇所のみ対応）

全体像（Architecture Overview）
- Orchestrator（薄いランナー）
  - ループ/乱数/CLI設定の受け渡しのみ。シークエンス本体に委譲。
- Sequence Engine（Step 実行）
  - 観測/候補/評価/コミット/永続化のフロー制御。純粋な組立てに徹する。
- Edge Builder（候補・配線）
  - 候補集合（S_link/S_cand/forced/Ecand等）の生成。Hop0配線セットとDG仮想採用セットの構築。
- Evaluator（数値計算）
  - geDIG/IG/SP/g(h)/g0/gmin等の計算。2フェーズ評価（graph_pre→eval_after）を標準化。
- Commit Policy（実コミット）
  - DGポリシー（threshold/always/never）とcommit_budgetでコミットセットを確定し、実グラフへ反映。
- Data Store Adapter（DS永続化）
  - SQLite/オンメモリの抽象グラフ、Timeline、Stepメタの保存/読込。Step IDをキーに状態を復元。
- Presentation Extractor（UI抽出）
  - Step IDを受け、当該時点のグラフ/メトリクス/候補等をDSから再構築してUI用JSONへ整形。
- Environment Adapter（迷路）
  - SimpleMaze等の環境APIを安定化。観測/遷移/可能行動のインターフェースを明示。

モジュール配置（Proposed Module Map：実験ローカル）
- `experiments/maze-query-hub-prototype/qhlib/`
  - `orchestrator.py` … 乱数/CLI→Engine呼び出し（将来）
  - `engine.py` … 1 step実行（観測→候補→評価→コミット→永続化）（将来）
  - `edges.py` … 候補生成（S_link/S_cand/forced/Ecand）、Hop0配線セット構築（将来）
  - `evaluator.py` … geDIG/IG/SP/g(h)/g0/gmin（2フェーズ: graph_pre→eval_after）（将来）
  - `commit.py` … DGポリシー/commit_budget適用、実グラフ反映（将来）
  - `store.py` … DS永続化（SQLite/episodeローカル）。Step IDの採番/読込（将来）
  - `present.py` … UI抽出（Step ID→UI JSON、snapshot-level整形）（将来）
  - `env_adapter.py` … Maze環境アダプタ（将来）
  - `timeline.py` … Timelineユーティリティ（Q_prev→dir 等）[済]
  - `ids.py` … ノード/エッジIDユーティリティ（`row,col,dir` 等）[済]

備考：実験ローカルにまとめる理由
- メイン実装（`src/insightspike/...`）を汚さず、実験ごとの大胆な試行錯誤を許容
- 別実験で独自のリファクタをしても相互干渉を避けられる
- コア（GeDIGCore）/迷路などはメインを利用し、実験側は呼び出しの責務に集中

データ契約（Data Contracts）
- NodeID: `(row:int, col:int, dir:int)`（dir=-1はQ）
- Edge: `Tuple[NodeID, NodeID]`、型属性: `link|forced_candidate|timeline|action|graph`
- StepID: episode内連番（0..N-1）。DSの主キーに用いる。
- StepMeta: `g0, gmin, delta_ged/ig/sp, ag_fire, dg_fire, best_hop, commit_items, dg_committed_edges` 等
- Snapshot-level: `minimal|standard|full`
  - minimal: DS真実＋StepMeta（候補/Per-hopなし）
  - standard: 候補/Per-hop/診断を含む
  - full: デバッグ向けにpre/eval/postの全グラフ配列

シーケンス（Reference Timeline）
1) 観測取得（EnvAdapter）
2) 候補生成（EdgeBuilder）
3) 評価用グラフ構築（graph_pre→eval_after）
4) Evaluator（g0/g(h)/IG/SP/gmin等）
5) CommitPolicy（Hop0＋DG実コミットをgraph_commitへ、2フェーズ分離）
6) DataStore永続化（抽象グラフ、Timeline、StepMeta）
7) Snapshot整形（snapshot-levelに応じて最終Payload）

UI抽出（Presentation Extractor）
- 入力: Step ID、Config
- 処理: DSからグラフ/Meta復元→UIスキーマへ変換（候補はstandard/full時のみ）
- 出力: HTMLが期待するJSON断片（grids/abstract/timeline/metricsなど）

移行計画（Migration Plan）
- Phase 0: 実験ローカルユーティリティの切出し（`qhlib/timeline.py`,`qhlib/ids.py`）[済]
- Phase 1: CommitPolicyの分離（`qhlib/commit.py`）＋ランナーで2フェーズ制御の委譲
- Phase 2: Evaluatorの分離（`qhlib/evaluator.py`）＋EdgeBuilder（`qhlib/edges.py`）
- Phase 3: DataStore Adapter（`qhlib/store.py`）＋Presentation Extractor（`qhlib/present.py`）
- Phase 4: Orchestrator（`qhlib/orchestrator.py`）でランナー薄型化（`run_experiment_query.py`は薄いエントリ）
- Phase 5: tests（ユニット/結合）、snapshot-level/DS主義の回帰確認

テスト計画（Testing）
- EdgeBuilder: 候補件数/優先順位、フォールバックの優先（未訪問dir→グローバルdir）
- Evaluator: g0/g(h)/SPの合成値、IG符号（after_before）切替
- CommitPolicy: threshold/always/never、commit_budgetの上限動作
- Store: Step IDごとの復元整合性（DSのみでUIを再構成できること）
- Present: minimal/standard/fullの差分、表示項目の欠損防止

テスト計画（詳細）
- ユニットテスト（experiments/maze-query-hub-prototype/tests/ 配下; pytest）
  - qhlib/edges.py: S_link/S_cand/forced/Ecand生成、Top‑L抽出、未訪問→グローバル優先のフォールバック
  - qhlib/evaluator.py: hop0/hop>0のg(h)、ΔGED正規化（Cmax）、IG（after_before）、SP（fixed-before-pairs/union切替）
  - qhlib/commit.py: DGポリシー（threshold/always/never）、commit_budgetの適用、Q↔dir限定コミット
  - qhlib/timeline.py: Q_prev→dirのみ、オプションでdir→Q_next/pair
  - qhlib/ids.py: canonical key生成、クエリ/同セル判定
- 結合テスト
  - Engine（40 steps, seeds=0..2, size=25, gh-mode=greedy, snapshot-level=standard）でJSON/DSが生成されること
  - minimal/standard の両出力に対し、HTML生成が成功し、主要フィールド（g0,gmin,ΔGED/IG,候補/Per-hop）が存在
- 回帰テスト（旧コード比較）
  - 旧ランナー（legacy）と新エンジン（qhlib）を同条件で実行し、許容範囲内で一致
    - しきい値: |ΔGED|, |ΔIG|, |g0|, |gmin| それぞれ1e-6〜1e-4程度（浮動誤差想定）
    - 準同値: SP>0判定の一致、best_hopの一致（±1は許容）
  - 比較スクリプト（experiments/.../scripts/compare_runs.py）で差分レポート（step単位/集計）
- パフォーマンス/メモリ
  - minimalで200 steps（25×25, seed=0）完走時間/メモリのベンチ
  - standardで40 steps（診断あり）が実用時間内（<60s）で実行

テストマトリクス（縮約）
- maze_size: 15, 25
- seeds: 0..2
- snapshot-level: minimal, standard
- gh-mode: greedy（標準）, radius（参照）
- dg-commit-policy: threshold（標準）, always（対照）
- sp_scope: union（標準）, auto（参照）

運用指針（Performance/Operational）
- 既定は `snapshot-level=minimal` + DS主義（steps.jsonは軽量メタ）
- 重い診断（Per-hop/候補/SP Debug）は短縮ラン（例: 40 steps）を `snapshot-level=standard` で回す
- タイムラインは `Q_prev→dir` のみ標準保存（`dir→Q_next`/`Q_prev↔Q_next` は抑制）
- IDは `row,col,dir` に統一し、UIはこのIDをホバーで表示（デバッグしやすさ）

既知の論点（Open Items）
- SPのpair集合定義（fixed-before-pairs vs union）をEvaluatorで切替可能に
- g_min（overall vs mh-only）をUI側に併記するか（見やすさの意見収集）
- DSスキーマのバージョニング（将来互換）

移行計画（旧コード→新エンジン）
- フラグ駆動の段階移行
  - `--engine legacy|new`（デフォルトlegacy）で切り替え。新実装はqhlibのEngine/Commit/Evaluatorを呼ぶ。
  - 機能/CLI互換を維持（既存オプションはそのまま）。
- シャドーモード（初期）
  - legacyで実行しつつ、裏でqhlib Evaluator/Commitを走らせ、差分のみログへ（DSはlegacyを書き込み）。
- パリティ検証（40 steps, 25×25, seeds=0..2）
  - 旧/新のJSONとDSをcompare_runs.pyで比較。許容差内ならOK。
  - 乖離が出た箇所は、候補集合/評価subgraph/SPペア/正規化分母/IG方向の各ポイントを切り分けて修正。
- 段階ロールアウト
  - `--engine new` を既定に（legacyは残置）。
  - 200 steps minimalの実運用を新で走らせる。
  - 安定後にlegacyパス/コードを整理（ドキュメントに移行完了を明記）。

データ移行（DSスキーマ）
- metaテーブルに `schema_version` を保存し、マイナーバージョンアップは後方互換（NULL許容）で追加。
- 旧 runs/steps.json から DS を再構築するワンショットスクリプト（scripts/import_legacy_steps.py）を用意。
- namespace（run毎）で論理分離。実験名/パラメタを含むキーで衝突回避。

所感（My Opinion）
- まずはCommit/Evaluator/EdgeBuilderを分けることで“どこでSPが立っていないか”の切り分けが容易になります。現状のバグ/見落としは、候補定義と評価グラフの境界（graph_pre/ eval_after/ graph_commit）が混濁しやすいところに集中しているので、責務分離の効果は大きいです。
- DS主義（steps.json最小）に寄せるのは賛成です。UIはStep IDだけを頼りにDSから再構成できるべきで、ログ巨大化を避けられます。
- UIは当面現行のままでも、Present層で必要な断片に整形できれば十分です。最終的にはHTMLから直接DSを引く構成にも発展可能（将来）。

進捗（Progress）
- [x] Phase 1: commit.py 抽出（DGポリシー/2フェーズコミット）完了。ランナーへ導入済み。
- [x] Phase 2: evaluator.py / edges.py 抽出
      - edges: ランナーからEcand構築を移管（qhlib/edges.py）
      - evaluator: 外部評価でhop_series/gmin/bestHop等を上書き適用（qhlib/evaluator.py）
- [x] Phase 3: store 抽出（最小）
      - SQLite へのノード/エッジ/タイムライン/強制候補の永続化を qhlib/store.py に移管
      - ランナーから呼び出して totals とDSスナップショットを反映
      - HTMLコントロールに SP boundary/scope/expand の表示を追加

- [x] Phase 4: Sequence 安定化 + Strict DS 同期（phase4a/4b）
      - Strict DS 表示: 右paneはDSエッジ端点のみ（候補/ショートカット/タイムライン/中心注入を抑制）
      - 左paneもStrict DS時はpre-stepクエリリングをDS端点にある場合のみ描画、post-stepエージェントを非表示
      - Evaluator出力の採用統一: 外部Evaluatorのgmin/bestHop/Δ(min)をインライン再計算で上書きしない
      - HTMLの一元管理: render開始時にstepIdxスナップショットを取得し、全パネルが同一rec（同一stepIdx）を参照
      - build_reportsでpre-stepクエリ（query_node_pre_derived）を後付け補完（再ラン不要の救済）
      - present.load_ds_graph_upto_stepでtimelineエッジ除外（次Q漏れ防止）
      - 40ステップ×1seedの再ラン（_phase4b）で浮遊解消と数値一致を確認

Phase 4 完了報告（Verification & How to reproduce）
- 実行物（例）
  - steps: `experiments/maze-query-hub-prototype/results/_phase4b/steps.json`
  - summary: `experiments/maze-query-hub-prototype/results/_phase4b/summary.json`
  - HTML: `experiments/maze-query-hub-prototype/results/_phase4b/interactive.html`
- 再現コマンド（例）
  1) 実験実行（短縮版・1seed・40steps・Strict DS 永続化ON）
     - `python experiments/maze-query-hub-prototype/run_experiment_query.py \
        --maze-size 25 --max-steps 40 --seeds 1 \
        --lambda-weight 0.5 --max-hops 15 \
        --norm-base link --linkset-mode \
        --theta-ag 0.2 --theta-dg 0.15 --top-link 1 \
        --output experiments/maze-query-hub-prototype/results/_phase4b/summary.json \
        --step-log experiments/maze-query-hub-prototype/results/_phase4b/steps.json \
        --persist-graph-sqlite experiments/maze-query-hub-prototype/results/_phase4b/graph.sqlite \
        --persist-forced-candidates`
  2) HTML生成（DSグラフ参照でStrict DS描画）
     - `python experiments/maze-query-hub-prototype/build_reports.py \
        --summary experiments/maze-query-hub-prototype/results/_phase4b/summary.json \
        --steps   experiments/maze-query-hub-prototype/results/_phase4b/steps.json \
        --sqlite  experiments/maze-query-hub-prototype/results/_phase4b/graph.sqlite \
        --out     experiments/maze-query-hub-prototype/results/_phase4b/interactive.html`

確認ポイント（Phase 4 達成条件）
- 抽象/グリッドとも「浮遊ノードなし」：Strict DS ON のまま、現在Qは常にDS内のエッジ端点として描画される。
- 数値パネルの整合：Temporal、Δ(min)、Per-hop の各ウインドウが同一ステップの hop_series に一致。
- 次Qリークなし：TimelineエッジはStrict DS経路では参照せず、次ステップ情報が混入しない。

次アクション（Next Steps）
1) Present層の拡張（正式化）
   - qhlib/present.py を拡張し、Step ID→pre/eval/postのUI断片をDSから再構成（strict/relaxed両モード）
   - build_reports: --strict/--relaxed スイッチ対応、DS→UI断片の採用を切替

---

Phase 5: g0 キャリブレーション（新規）

目的
- 「通路では g0 を低く、行き止まりで g0 を高く」なるよう正規化と閾値を校正し、無駄発火/無駄探索を抑制する。

方針（A/B計画）
- 正規化の見直し（候補台固定 vs after側エッジ）
  - 現状: `Cmax = node_cost + edge_cost * max(1, |S_link|)`（candidate_base）。raw_ged が Cmax を上回り 1.0 にクリップされやすい→ g0 高止まり。
  - 代案A: `edges_after` 正規化に切替（サブグラフ after のエッジ数を分母）。
  - 代案B: candidate_base のまま `Cmax = node_cost + edge_cost * (|S_link| + α)` に平滑化（α>0）し、1.0 クリップを回避。
  - 代案C: hop0 の ΔGED を定数近似（例: `ged_hop0_const=True` で固定）し、IG側で通路/行き止まりの差を立てる（参考値: 現行の λ=1, IG=linkset）。
- θAG の分位基準化
  - ランごとに g0 分布から `θAG := Pq(g0)` を設定（q=0.90〜0.95 推奨）。通路ではMHに入らず、行き止まりでのみ MH 評価へ。
  - 実装: サマリ生成時に分位を記録し、次ランの既定値へ（CLI上書き可）。
- IG/SP 寄与の調整
  - linkset IGの分母・符号（before_after を既定）と sp_beta をスイープ（sp_beta∈{0.1,0.2,0.3}）。
  - 目的: 通路では ΔH≈0, ΔSP=0 で g0 が下がり、袋小路では ΔSP>0 で g0 が顕著に下がる設計を強化。

測定項目
- 40/120/200 step で g0 の平均/分位、AG発火率、平均評価hop数、DG発火率、実行時間。

成功基準
- 通路区間の g0 が顕著に低下（現行 0.98 付近→ 0.2〜0.5 帯へ）。
- 行き止まり直前/到達時に g0 が高く、AG→DG のシーケンスが設計通りに可視化。
- 早期停止（AG/DG）で平均評価hop数が減少、実行時間が短縮。

実装メモ（反映済み/関連）
- Evaluator: DG/δSP の早期停止、EPC 増分、cached_incr（端点SSSP増分）＋円環時の厳密検証（ΔE>ΔV かつ ΔSP_fast≥τsp）。
- Runner: `--sp-cache-mode {core,cached,cached_incr}`, `--sp-verify-threshold`、L1 プリフィルタ、リング楕円、`sp_cand_topk`。

完了事項（検討結了）
- IG の符号（H の向き）は既定を after_before に統一（Core 既定に準拠）。一時的な切替は環境変数 `MAZE_GEDIG_IG_DELTA=before_after` の明示指定時のみ許可し、ログに記録する方針。
- 0‑hop 定義の厳密化：実行経路（Q_prev→dir→Q_now）の強制結線はデフォルトで行わない（`timeline_to_graph` 明示時のみ注入）。

---

変更履歴（抜粋・実装済み）
- Phase 4: Sequence 安定化 + Strict DS 同期（完了）
- Evaluator: 外部評価採用の一元化、AG境界 (`<`)、DG（best_hop>=1 ∧ gmin_mh<θDG）
- SP: Core準拠固定ペアに復帰、cached_incr 実装（円環時のみ厳密検証）、早期終了（δSP≤0 または g(h)<θDG）
- 候補: SpatialGridIndex（矩形/楕円）、L1 WeightedL2Index、Ecand上限（sp_cand_topk）
- HTML: Temporal/Per-hop の一元参照、g0前面・g_min赤点（hop0除外）、SPリング、強制線常時表示
2) steps.json の軽量化
   - 最小メタ（g0/gmin/bestHop/Δ系、counts など）のみ出力、候補/Per-hop等はpresentで復元
   - 互換運用のための --light-steps フラグを追加
3) T0/T1/T2 のコード分離とCLI化
   - T0(pre)保存APIの明示、T1(評価)は保存しない、T2(post)は次ステップ用のみ
   - --persist-pre-only / --persist-timeline-edges の明文化
4) HTML: SPスコープの可視化強化
   - アンカー＋半径（expand）オーバレイ、strict時はpreのみ
5) 検証
   - 200ステップ×複数seed（Strict DS）で整合性を確認
   - compare_runs（legacy/new）でJSON/DSの整合を自動チェック
