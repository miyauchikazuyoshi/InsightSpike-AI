# geDIG v2 実装リファクタ計画（理論更新同期）

目的
- 論文の現行定義（F = ΔGED − λ·ΔIG、固定分母 log K⋆、局所Cmax、二段しきい値、マルチホップ運用）に `src/` 実装を同期する。
- 既存のAPI/テスト互換を保ちつつ、段階的に切替可能なフラグを提供する。

非目標
- 実験セット全体の刷新は含めない（回帰検証は行う）。
- 既存の代替IG/スペクトル系の削除は行わない（オプショナル継続）。

スコープと差分要約（現状→あるべき）
- ΔIG 正規化分母: `log(candidate_total)` 等の策略 → 1-hop 基準の固定分母 `log K⋆`（K⋆=min(|V₁(q;θcand)|, Kcap)）。
- ΔGED 正規化: グローバル分母 → 局所Cmax（0-hop: 1 + K⋆）。
- 二段しきい値: 未実装 → `θcand > θlink` による候補誘導とコミット。
- マルチホップ: 既存（decay集約、SP任意） → 分母固定のままIG集計、SPは補助（βで構造項に反映: 現状の `use_multihop_sp_gain` 継続）。
- 安全策: K⋆<2, 空集合, サンプリング系のフォールバック → 実装に明示。

設計変更（モジュール別）
1) `src/insightspike/algorithms/core/metrics.py`
- [変更] `entropy_ig(...)` に固定分母を外部注入できる引数を追加：
  - `fixed_den: float | None = None`（nats基準の対数、通常は `math.log(K_star)`）。
  - 与えられた場合は `denom = max(fixed_den, epsilon)` を優先する。
- [変更] `candidate_total` ベースの分母ロジックは既定の後方互換パスとして残す（`norm_strategy`）。
- [追加] `kstar_guard: int = 2` に満たないときは ΔIG=0 を返す安全策。

2) `src/insightspike/algorithms/gedig_core.py`
- [変更] `calculate(...)` の引数に以下を追加（キーワード受け付け）:
  - `k_star: Optional[int] = None`（1-hop 基準サイズ）
  - `l1_candidates: Optional[int] = None`（既存；Cmax=1+K⋆に使用）
  - `ig_fixed_den: Optional[float] = None`（`log K⋆` をそのまま渡す場合）
- [変更] `_calculate_entropy_variance_ig(...)` 呼出しに `fixed_den` を配線。
- [変更] Cmax 局所正規化: 既存の `use_local_normalization` と `l1_candidates` を活用（`norm_override = 1 + K⋆`）。
- [維持] SP補助構造項（`use_multihop_sp_gain` と `sp_beta`）は現仕様を維持。

3) 二段しきい値セレクタ（新規）
- [新規] `src/insightspike/algorithms/gedig/selector.py` 近傍に `TwoThresholdCandidateSelector` を追加：
  - 入力: `query_vec`, `index`, `theta_cand`, `theta_link`, `k_cap`。
  - 出力: `(S_cand, S_link, K_star)` とスコア（類似度）。
  - 仕様: `S_link ⊆ S_cand`, `K_star = min(|S_cand|, k_cap)`。
  - 備考: 既存の `search/similarity_search.py` と `index` 実装に依存しない軽量API（バックエンド差し替え容易に）。

4) コンフィグ/デフォルト
- [追加] `config/models.py` / `GraphConfig` or `MetricsConfig` に以下を追加：
  - `theta_cand: float = 0.4`, `theta_link: float = 0.35`（例）
  - `k_cap: int = 32`、`top_m: Optional[int] = None`（簡易 Top-M 切詰）
  - `ig_denominator: Literal['fixed_kstar','legacy'] = 'fixed_kstar'`
  - `use_local_normalization: bool = True`

5) 呼出し側配線（最小配線）
- パス: `implementations/agents/main_agent.py` → geDIG呼出し前に `TwoThresholdCandidateSelector` で `K⋆` を計算し、
  `gedig_core.calculate(..., k_star=K⋆, l1_candidates=K⋆, ig_fixed_den=log(K⋆))` を渡す。
- 既存の retriever/top-k 結果がある場合は流用し `S_cand` を再フィルタ（θcand）して `K⋆` を得る。

6) マルチホップ整合
- `fixed_den` は hop によらず一定（0-hopのK⋆）。現状の `_calculate_multihop` にてそのまま各hopへ同じ `fixed_den` を渡す。
- （距離減衰レンズは今回スコープ外。必要になった場合は別Issueで検討する）

後方互換/フラグ
- 既定は新方式（`ig_denominator=fixed_kstar`, `use_local_normalization=True`）。
- 旧方式へ戻すには `ig_denominator=legacy`、`use_local_normalization=False` を設定。
- 環境変数（既存の `MAZE_GEDIG_*`）は現行動作を優先。

バリデーション計画（テスト）
1) 単体テスト
- ΔIG: `K⋆<2`、`S_cand=∅`、`fixed_den` 有無でゼロ割回避とスケール不変を確認。
- ΔGED: `norm_override=1+K⋆` が効くことを検証（0-hop変化のO(1)更新）。
- 二段しきい値: `θlink<θcand`、`S_link ⊆ S_cand` を保証。

2) 回帰テスト
- 既存の `calculate_simple` 結果の符号・単調性が保たれること。
- マルチホップ: `decay_factor` による集約が従来通り。

計測/パフォーマンス
- ΔIG計算は `fixed_den` の導入のみで計算量不変。
- ΔGEDの `norm_override` は O(1)。SPは既存のサンプリング・ノード上限のガードを維持。

段階導入（マイルストーン）
- M1: core/metrics と gedig_core の固定分母・Cmax対応（フラグ付き）。
- M2: 二段しきい値セレクタの導入と main_agent 配線。
- M3: 単体/回帰テストの整備と既存実験の煙テスト。
- M4: オプションの Top-M 切詰の追加。

既知のリスク/未決事項
- 距離減衰レンズ（γ）については計画から除外し、必要なら別途仕様化する。
- 既存RAG/迷路での `θcand/θlink` と `k_cap` 既定値の調整（推奨: k_cap∈{32,64}）。
- 監視/ロガーの指標名（ig_raw vs ig_value）の表記整合。

完了条件
- 主要パスで `ΔIG = ΔH/log K⋆` と `Cmax=1+K⋆` が有効。
- K⋆未提供でも旧方式にフォールバックし動作継続。
- 単体・回帰テストが合格し、既存ベンチがエラーなく走る。

---

# 実装計画（詳細）

優先度順のロードマップと受け入れ基準（DoD）

M1 固定分母・局所Cmax（基盤）
- 変更点
  - metrics.entropy_ig: `fixed_den`引数追加、denom決定ロジックに分岐を追加（優先: fixed_den > norm_strategy）。
  - gedig_core.calculate: `k_star`, `ig_fixed_den` を受け取り、0-hop/マルチホップ双方で配線。
  - `use_local_normalization=True`時、`norm_override = 1 + k_star` を設定（未指定は従来ロジック）。
- 受け入れ基準
  - K⋆=5, ΔH=0.5 で IG=0.5/log(5) を返すこと。
  - K⋆<2、features空、extra_vectorsのみ等の退避ケースで例外なくΔIG=0を返すこと。
  - 旧パス（fixed_den未指定）で従来値に近似（誤差<1e-6 or 既存テスト合格）。

M2 二段しきい値 + K⋆検出配線
- 変更点
  - 新規 TwoThresholdCandidateSelector の追加（S_cand, S_link, K⋆を返す純関数）。
  - main_agent（or 呼び出し箇所）で selectorを噛ませ、`k_star` と `ig_fixed_den=log(K⋆)` を geDIG に渡す。
- 受け入れ基準
  - `θlink < θcand` を満たし、`S_link ⊆ S_cand` が保証される。
  - `k_cap`超過時でもK⋆=k_capで安定。
  - 既存の検索結果からの再フィルタで性能劣化がない（Top-K→フィルタ→再ソートなし）。

M3 テスト・回帰・CI
- 変更点
  - 単体テスト追加（下記）。
  - 回帰テストで既存APIの破壊を検出。
  - CI: pytestのスイート時間を10–15%以内に維持（重いSPはサンプリングガードのまま）。
- 受け入れ基準
  - すべての新規/既存テスト合格。
  - 代表的パイプライン（迷路/RAGのスモーク）で例外なし実行。

M4 Top‑M（任意）
- 変更点
  - selectorにTop‑M固定（M=k_cap）オプション。
- Top‑Mによる候補固定のみ実装（距離減衰レンズはスコープ外）。
- 受け入れ基準
  - デフォルト無効時の完全後方互換。

---

# テスト計画（ユニット＋パイプライン）

テスト方針
- 変更最小の範囲で追加。既存テストの意味を壊さない。
- まずモジュール単体 → 低次統合（gedig_core）→ 高次パイプライン（簡易グラフ・小規模データ）で段階検証。

1) 単体テスト
- ファイル案
  - `tests/unit/test_entropy_ig_fixed_den.py`
  - `tests/unit/test_gedig_core_local_norm.py`
  - `tests/unit/test_two_threshold_selector.py`
- ケース
  - entropy_ig
    - fixed_den 指定: K⋆=5、ΔH=0.5 → IG=≈0.5/log(5)。
    - fixed_den 未指定: 現行 norm_strategy=before/logn が従来通りに働く。
    - K⋆<2、features空、extra_vectorsのみ → IG=0。
  - gedig_core
    - `use_local_normalization=True, l1_candidates=K⋆` で `normalization_den=1+K⋆` が `normalized_ged` に反映。
    - `ig_fixed_den` が hop=0/1/2 でも一定で使われる（hop結果の `ig_den` を検査）。
  - selector
    - ダミー類似度行列/episodesで `θcand>θlink` を満たす集合を生成、`S_link ⊆ S_cand` を検証。
    - K⋆=min(|S_cand|, k_cap) の境界（0,1,k_cap−1,k_cap,k_cap+1）。

2) 低次統合テスト
- ファイル案
  - `tests/integration/test_gedig_fixed_den_pipeline.py`
- ケース
  - 小グラフ（5–9ノード、特徴は手作業）で 0-hop: K⋆=5、1-hop: K実サイズ増でもIG分母が不変→スコア比較可能。
  - SP補助は有効/無効の両方で geDIG の符号変化が理にかなう（定性的検証: 構造改善があればFが下がる）。

3) パイプライン・スモーク
- 迷路（最小）
  - `environments/maze.py` を用いたミニ実行（5–10ステップ）。例外なく走行し geDIG が毎ステップ返る。
  - `MAZE_GEDIG_*` の環境フラグで旧/新方式を切替、クラッシュしないこと。
- RAG（最小）
  - 5–10件のダミー文書と簡易クエリで、selector→K⋆→geDIG流れの通電確認。

実行コマンド
- 単体: `pytest -q tests/unit -k 'entropy_ig_fixed_den or two_threshold'`
- 統合: `pytest -q tests/integration -k gedig_fixed_den_pipeline`
- 全体: `pytest -q`

カバレッジ目標
- 新規/変更行の分岐網羅 > 85%。既存全体カバレッジの維持はベストエフォート。

---

# テスト実施ログ
- 2024-10-18 `pytest -q tests/unit/test_entropy_ig_fixed_den.py tests/unit/test_two_threshold_selector.py tests/unit/test_gedig_core_local_norm.py`
- 2024-10-18 `pytest -q tests/test_minimal_healthcheck.py`

---

# 後方互換・影響評価（BC/Compat）

API互換
- 追加はすべてオプショナル引数（`k_star`, `ig_fixed_den`）。既存呼出しは無変更で動作。
- デフォルト挙動
  - 互換性最重視のため、固定分母は「明示指定時のみ有効」とする（初期段階）。
  - 構成ファイルで `ig_denominator=fixed_kstar` を選んだパスのみ新方式を有効化。
  - `use_local_normalization` も既定は現状維持（False）。段階移行時に既定変更を検討。

挙動差の影響
- IGのスケール安定化により、multi-hop 比較が堅くなる一方、過去の閾値/λの最適値が微調整を要する可能性。
- GEDの局所正規化により、0-hop の小変更が過大評価/過小評価されにくくなる（既存しきい値に軽微な影響）。

緩和策
- フラグ駆動：`ig_denominator=legacy|fixed_kstar`、`use_local_normalization=True|False`、環境変数 `MAZE_GEDIG_*` を併用。
- ロギング：新方式が有効なとき、結果ダンプに `ig_norm_den`, `ged_norm_den`, `k_star` を必ず含め、診断容易化。
- フォールバック：K⋆ が未算出/異常時は自動で旧方式へ回帰。

移行ガイド（短期）
- 実験系では `ig_denominator=fixed_kstar`, `use_local_normalization=True` に切替え、分位校正でしきい値再調整。
- 本番系/安定検証中はデフォルト据置（legacy）で回帰確認後に段階導入。

---

# 進捗トラッカー
- [x] M1: 固定分母（log K⋆）と局所Cmaxを実装し、既存パスと後方互換を確認（コード反映済、検証はM3で実施）
- [x] M2: 二段しきい値セレクタ導入とK⋆配線（selector追加、config既定値、MainAgent/L3 Reasonerへ配線済）
- [x] M3: ユニット/統合/パイプラインテストの追加とCI確認（unit3件＋healthcheckスモーク実行）
- [x] M4: Top-M固定追加済（距離減衰レンズは今回スコープ外）

---

# 次フェーズ計画（フォローアップ）
- [ ] 実験系（maze/RAG）で `ig_denominator=fixed_kstar` を有効化し、AG/DG 閾値・λ を改めて校正する（分位ベースで再調整、指標ログを収集）。
- [ ] マルチホップ有効時のエンドツーエンド検証（Layer3 → GeDIGCore の `k_star` 配線と SP 補助が期待通りに作用するか）をスモールデータで確認。
- [ ] 既存レポート／ダッシュボードに `candidate_selection` サマリ（k⋆、θ、top_m 等）を追加し、監視の可視性を確保。
- [ ] 迷路オンライン実験ログの再生成（`experiments/maze-online-phase1*`）で旧スケールとの比較差分をまとめ、ドキュメントへ反映。
- [ ] Graph Reasoner のユニット／統合テストを拡張し、`candidate_selection` を含む新フィールドの後方互換確認を自動化。
