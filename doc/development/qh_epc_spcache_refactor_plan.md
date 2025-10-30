# Query‑Hub: EPC + SP Cache リファクタ計画（総合）

目的
- マルチホップ評価の高速化（EPC: Edit Path Cost の加法化、式(12)の増分化）
- SP の計算簡略化（Strict DS + 距離キャッシュで局所更新）
- 既存の挙動（特に `_diag_std_autowire_rlink004`）を壊さず段階導入（フラグでON/OFF）

不変条件（Non‑regression）
- IGの符号は現行（after_before）を基本とし、アブレーションで before_after も確認
- 既存ベースライン（r_link=0.04, autowire-all, 40 steps 標準ログ）に対し：
  - hopシリーズ（g, ΔGED, IG, H, ΔSP）が一致（許容誤差内）すること
  - Candidate Snapshot の内容・k★・選択行動に有意な変化がないこと
  - 抽象グラフの「浮遊なし」「Strict DS準拠」を維持

ベースライン（固定参照）
- 実験成果物: `experiments/maze-query-hub-prototype/results/_diag_std_autowire_rlink004/interactive.html`
- 主要フラグ: `--link-autowire-all`（既定ON）、`--theta-link 0.04`、`--snapshot-level standard`、`--norm-base link`、`--linkset-mode`、`--max-hops 15`、`--top-link 1`
- IG符号: 現行既定（after_before）を維持

最新の実装状況（2025-10-29）
- Evaluator 早期停止: δSP≤0 または g(h)<θDG で貪欲を中断（`--eval-all-hops` で無効化可）。
- EPC 増分: hop>0 の ΔGED を raw_ged0 + edge_cost×採用本数（分母は固定台 Cmax）で近似。
- SP cached_incr: 端点SSSPでΔSPを増分合成するモードを追加（`--sp-cache --sp-cache-mode cached_incr`）。
  - 円環検知: SP評価サブグラフで `add_edges > add_nodes`（ΔE>ΔV）が立つ hop を“サイクル疑い”とする。
  - 検証閾値: ΔSP_fast ≥ `--sp-verify-threshold`（既定0.05）のとき、その hop は Core厳密SPで置換（精度優先のときのみ発火）。
- CLI/Runner: `--sp-cache-mode {core,cached,cached_incr}`, `--sp-pair-samples`, `--sp-verify-threshold` を配線。
- 速度所感（40 steps, H=20, θAG=0.45, θDG=0.30）
  - core ≈ 55s → cached_incr ≈ 30s（約1.7×）。H=10 なら ≈ 13s（さらに短縮）。

## SP 計算フロー（Before→After→保存）フローチャート

ここでは、提案の「DS上の距離辞書（固定ペア＋Lb）再利用」と、現行実装済みの after 計算短縮（cached_incr: 端点SSSP増分＋円環時の厳密検証）をまとめた実行フローを示す。

- 前提（記号）
  - anchors_core = 現在Q、anchors_top = 0-hop採用Top‑L（dir/過去Q）
  - eff_hop = hop + sp_hop_expand（サブグラフ評価半径）
  - signature = distcache.signature(subgraph_before, anchors_core, eff_hop, scope, boundary)

1) SPbefore 準備（固定ペア＋平均距離 Lb）
  - 入力: prev_graph, anchors_core, anchors_top_before, eff_hop, scope/boundary
  - サブグラフ抽出: before_sub = extract_khop_union(prev_graph, anchors_core, anchors_top_before, eff_hop)
  - 署名作成: sig = signature(before_sub, …)
  - DS参照（ヒット時は再利用）
    - if DS.contains(sig):
      - pairs = DS.load_pairs(sig)  // [(u,v,d_before)] サンプル
      - Lb = DS.load_Lb(sig)
    - else:
      - pairs = sample_all_pairs(before_sub)  // 小規模は全ペア、大規模はサンプル
      - Lb = mean(d_before)
      - DS.save(sig, pairs, Lb)

2) 候補エッジ評価（貪欲; cached_incr）
  - for hop = 1..H:
    - eff_hop = hop + expand
    - after_sub_try = after_graph(hop−1) に候補 e=(u,v) を仮追加したサブグラフの eff_hop 版（構造のみ反映）
    - dSP_fast の推定（端点SSSP＋固定ペア）
      - du = SSSP(before_sub, u), dv = SSSP(before_sub, v)
      - La_new = min(現La, du[a]+1+dv[b], dv[a]+1+du[b]) を各ペアで適用（平均化）
      - SP_fast = max(0, (Lb − mean(La_new)) / Lb)
    - 円環検知＋必要時の厳密検証
      - suspected_cycle := (|E_after| − |E_before|) > (|V_after| − |V_before|)
      - if suspected_cycle && (SP_fast − SP_prev) ≥ τ_sp:
        - SP_try = Core._compute_sp_gain_norm(before_sub, after_sub_try)
      - else:
        - SP_try = SP_fast
    - 候補の中から δSP を最大化する e* を選択
    - 採用: after_graph(hop) = after_graph(hop−1) + e*
    - La_state を e* で増分更新（以降の dSP_fast 計算に使う）
    - g(h) を合成: GED(h)（EPC増分 or 定数）, IG(base_IG) + γ·SP(h)
    - 早期終了: δSP≤0 または g(h) < θDG（診断時は継続）

3) SPafter 保存（次ステップの SPbefore 用）
  - after_sub = extract_khop_union(after_graph(best_hop), anchors_core, anchors_top_after, eff_hop)
  - pairs_after = sample_all_pairs(after_sub)（または採用済み La_state をそのまま保存）
  - Lb_after = mean(d_after)  // 次ステップの before で Lb として再利用
  - DS.save(sig_after, pairs_after, Lb_after)

4) 失敗時・不一致時のフォールバック
  - 署名不一致（アンカー/境界/半径が変わる）や DS ミス時は従来計算（Core厳密 or cached_incrで都度生成）
  - DS は軽量メタ（pairsサンプル＋平均）に限定し、完全APSP辞書は保存しない（容量抑制）

実装メモ（適用先）
- DS: `qhlib/store.py` に sp_pairsets/sp_pairs テーブルを追加（sig, Lb, pairs）
- Evaluator: `qhlib/evaluator.py`（cached_incr パス）で Before参照→After保存を組み込み
- 既存最適化: union k-hop ノード集合のキャッシュ（before/after）と EPC（GED増分）を併用

ログ/計測
- 署名ヒット率（before 再利用率）、SSSP回数、pairs件数、cand_ms/eval_ms、早期終了率（AG/DG）を記録してA/B比較

### 2回目以降（反復時）のフロー補足

“初回（DS miss）→保存”の後は、次ステップ以降で DS hit により前処理が簡略化される。

- 0) 事前（状態）
  - DS には直前ステップの after_sub に対する `sig_after(h,eff)` が保存済み（pairs_after, Lb_after）。
  - 次ステップの before_sub は多くのケースで「直前 after_sub と同一署名」になりやすい（anchors/eff_hop/境界が一致する場合）。

- 1) SPbefore（2回目以降）
  - before_sub 抽出→ `sig_before(h,eff)` を生成
  - DS.lookup(sig_before)
    - hit: pairs_before = DS.pairs(sig_before), Lb_before = DS.Lb(sig_before)
      - 固定ペア生成と平均距離の再計算をスキップ
    - miss: 初回と同様に生成→DS.save
  - 備考: eff_hop ごとに署名が異なるため、hop評価で必要な eff について逐次 lookup（キャッシュ）する。

- 2) 候補評価（cached_incr; 変更なし）
  - dSP_fast 用のLa_stateはプロセス内キャッシュで保持（一時）
  - 円環検知＋必要時のみ厳密検証
  - 採用→La_state更新→g(h)合成→早期終了

- 3) SPafter（2回目以降の保存/更新）
  - after_sub 抽出→ `sig_after(h_best,eff)` を生成
  - DS.upsert(sig_after, pairs_after, Lb_after)
    - 既に同一署名が存在する場合は上書き（rename/inflate なし）
    - 署名キーが異なる場合は新規追加（古い署名は残る。運用でTTL/上限を検討）
  - 保存内容の選択肢
    - 最小: (pairsサンプル, Lb)
    - 拡張: La_state（after側ペア距離ベクトル）も保存しておけば、次回 before の平均距離再計算を完全スキップ可能（メモリ/容量トレードオフ）

- 4) 反復の安定化のコツ
  - anchors/eff_hop/境界が頻繁に変わると署名が一致しにくい→ `anchor_recent_q` を小さめ（例:1–2）にして評価窓を安定化
  - eff_hop の取り扱いは hopごとに独立署名（`(h+expand)`）で管理
 - UI/ログに署名ヒット率・DS参照/保存件数を記録し、DS効果を可視化

## SP Pairset DS API 仕様（保存・再利用のための拡張）

背景/目的
- SPbefore の固定ペア生成と平均距離 Lb 計算を、直前 step の SPafter 結果から再利用して高速化する。
- evaluator（cached_incr パス）から透過的に利用できる抽象 API を定義し、実験→本体へ移植可能な形にする。

データモデル（SQLite テーブル）
- sp_pairsets
  - signature TEXT PRIMARY KEY  … サブグラフ署名（下記参照）
  - namespace TEXT              … 実験/ラン識別子（例: maze_query_hub）
  - scope TEXT                  … 'union' | 'auto' など
  - boundary TEXT               … 'trim' | 'induced' | 'nodes'
  - eff_hop INTEGER             … hop + expand
  - node_count INTEGER, edge_count INTEGER
  - lb_avg REAL                 … 連結ペアの平均距離（before/after側）
  - pair_count INTEGER          … 保存ペア数（サンプル）
  - meta JSON                   … 予備（anchors ハッシュなど）
  - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- sp_pairs
  - signature TEXT              … FK(sp_pairsets.signature)
  - idx INTEGER                 … 0..pair_count-1（PKの一部）
  - u_id TEXT, v_id TEXT        … 'r,c,dir'
  - d_before REAL               … before側距離
  - PRIMARY KEY(signature, idx)

署名仕様（SignatureBuilder）
- 入力: before/after サブグラフ, anchors_core, eff_hop, scope, boundary
- 決定要素のハッシュ: ソート済みノード列, ソート済み無向エッジ列(u|v), anchors_core, eff_hop, scope, boundary
- 形式: sha256(nodes|edges|A|H|S|B)
- 注意: eff_hop ごとに別署名。境界/スコープ/アンカーが変わるとミス（安全）

抽象 API（移植前提のインターフェース）
```
class SPPairsetService(Protocol):
    def load(self, signature: str) -> Optional[Pairset]: ...
    def save(self, pairset: Pairset) -> None: ...
    def upsert(self, pairset: Pairset) -> None: ...
    def stats(self) -> dict[str, Any]: ...  # hits/misses/capacity等

@dataclass
class Pairset:
    signature: str
    lb_avg: float
    pairs: list[dict]   # {u_id, v_id, d_before}
    node_count: int
    edge_count: int
    scope: str
    boundary: str
    eff_hop: int
    meta: dict[str, Any] = field(default_factory=dict)
```

実装バリエーション
- InMemoryPairsetService … PoC/テスト用の辞書実装
- SQLitePairsetService  … 上記スキーマに保存（capacity/TTLは将来オプション）
- NoopPairsetService    … 互換運用（DS不使用）

Evaluator 統合ポイント（cached_incr）
- Before（参照）: signature を作り service.load(sig) → ヒット時は pairs/Lb 再利用、ミス時は生成して service.save
- After（保存）: after_sub の signature を作り pairs/Lb を service.upsert で保存（次 step の Before が命中）

Runner/CLI への追加（案）
- `--sp-ds-sqlite <path>`    … DS保存を有効化（指定が無い場合は Noop）
- `--sp-ds-capacity <int>`   … 上限（将来）
- `--sp-ds-ttl <sec>`        … TTL（将来）

後方互換と導入手順
- 既定は Noop（DS なし）で完全互換。`--sp-ds-sqlite` を指定したときのみ保存/参照を有効化。
- DS スキーマ追加は実験側の SQLite のみ影響。present/build_reports/Strict DS 表示には影響しない。
- 本体へ移植時は `src/insightspike/algorithms/` に抽象 API を置き、実験側はその API に依存する。



段階導入（Phases）
1) Phase A: SP距離キャッシュ（DS活用）
   - 追加: `experiments/maze-query-hub-prototype/qhlib/spcache.py`
   - モード定義:
     - `ds_exact`（厳密）… アンカー近傍サブグラフの全ペア距離を辞書保存し、更新はBFS×2＋距離辞書の局所更新で正確に行う
     - `ds/cached`（推定）… 固定beforeペア＋端点SSSPで相対短縮を近似（速度優先）
   - 仕様（厳密: ds_exact）
     - 用語: K=アンカー近傍サブグラフのノード数、anchor_sig=(anchors, hop_expand, scope, boundary)
     - データ構造:
       - `pair_dist[sig]: {(a,b)->d}`（a<b正規化）… 全ペア最短距離
       - `lb_avg[sig]: float` … before側の平均距離（SPの分母）
     - オペレーション:
       - T0（クエリ生成直後）: `register_all_pairs(sig, subgraph)` … APSP（各ノードBFSの合成）で `pair_dist` と `lb_avg` を構築
       - T1（0-hop Q↔dir 配線）: visit=0 の方向ノード d について `register_node_pairs(sig, subgraph, d)` … 新規ノード d のBFSで (d,x) 距離だけ登録（SPは評価せずΔSP=0）
       - T2（マルチホップ試行: visit>=1 への接続）: `exact_update_on_edge(sig, g_before, u, v)` … du=BFS(u), dv=BFS(v) から `la(a,b)=min(dab, du[a]+1+dv[b], dv[a]+1+du[b])` で距離辞書を更新し、`ΔSP=max(0,(Lb−La)/Lb)` を厳密計算
     - 複雑度: 従来 O(K^2) 寄り → 初期APSP後は BFS×2 + O(|affected|) で更新（実運用は O(nk) 想定）。メモリは O(K^2) だが速度優先で許容。
   - evaluator から `--sp-cache --sp-cache-mode ds`（推定）/ `ds_exact`（厳密）で切替（40 stepsの非破壊はcore/ds_exact、200 stepsの速度検証はds/ds_exact）
   - 検証: 40 steps を `present-mode none` で再生成し、hopシリーズ/ΔSPが一致（許容差内）かを比較。200 steps は実行時間/しきい発火率を確認。

2) Phase B: EPC 増分化（式(12)）＋貪欲PQ
   - 追加: `experiments/maze-query-hub-prototype/qhlib/evaluator_epc.py`
     - ΔEPCはCmax固定で各候補eのδEPCを前計算し、加法的に合成
     - δSP(e)は Phase A の距離キャッシュから取得、δH(e)はリンクセット局所で前計算
     - PQで `g_gain(e)=δEPC(e)−λ·(δH+γ·δSP)` を最大化しながら h 本採用（影響候補のみ再計算）
   - フラグ: `--epc-eval` でON
   - 検証: 40 steps の hopシリーズが従来と一致（許容差内）、200 steps で時間短縮

3) Phase C: Adaptive hops + 早期停止（AG/DG）
   - AG: g0 ≤ θAG なら重いマルチホップをスキップ
   - DG: g(h) < θDG 到達で早期終了、予算上限あり
   - 検証: 200 steps × 複数seedで時間短縮と精度（成功率/分岐挙動）

4) Phase D: Core への還元（任意）
   - 安定後に `src/insightspike/algorithms/gedig_core.py` へEPC/距離キャッシュをオプション移植
   - 既存モードとの互換維持（デフォルトは従来）

実装箇所とスコープ
- 実験側（安全地帯）: `experiments/maze-query-hub-prototype/qhlib/` に追加
- ランナー: `run_experiment_query.py` で新モードをフラグ配線（既定は従来挙動）
- HTML: 既存テンプレートでDS準拠（Strict DS）・診断（present none）を切替可能なまま

仕様追補（SP辞書登録のルール）
- 訪問回数0（visit=0）のノード接続では、SPは評価せず（ΔSP=0）、辞書登録のみ実施（T0/T1で `register_all_pairs`/`register_node_pairs`）。
- 訪問回数≥1 のノード接続時のみ、厳密更新（BFS×2＋距離辞書更新）でSPを評価（T2）。
- これにより、マルチホップ時の最短経路計測は「初回の O(K^2) 構築＋各hopの O(nk) 更新」で回る（O(K^2)→O(nk) 化）。

早期終了（実装済み）
- δSP ≤ 0 の時点で以降の hop を省略（診断が必要なときは `--eval-all-hops` を指定）
- g(h) < θDG 到達でも省略（`--theta-dg` で設定）。コミットは `best_hop>=1` かつ `gmin_mh<θDG` のときのみ（thresholdポリシー）。

追加提案（SPの時間短縮：インクリメンタルBFS）
- 直感：SP(h)→SP(h+1) の差分は「外縁で新しく追加されたノード/エッジ」を媒介する最短路の分だけ再計算すればよい。
- 手順案：
  - 前ステップの評価サブグラフ Gh の全ペア距離（もしくは平均値と件数）を保持。
  - hop拡張で Gh+1 の節点集合が増えるとき、外縁 ΔV からのみ BFS を実行。
  - u∈ΔV の BFS 距離 d(u,·) を用い、`la(a,b)=min(lb(a,b), d(a,u)+1+d(u,b))` を全ペアに対して更新。
  - 平均距離 La を再計算し、ΔSP = max(0, (Lb−La)/Lb) を得る。
- 性質：既存節点間の短縮も必ず外縁を経由するため、外縁起点の BFS だけで十分。計算量は O(|ΔV|·(E+V)) 程度。
- 実装方針：evaluator に `sp_cache_mode=cached_incr` を追加し、距離辞書と平均値を hopごとに保持。A/B（Core厳密 vs 増分BFS）で速度/一致性を評価。

検証プロトコル
- Baseline lock: `_diag_std_autowire_rlink004` を固定参照。比較は hopシリーズ（数値）、Candidate Snapshot（タグ・本数）、抽象グラフ（視覚）
- 逐次テスト：Phase A→B→C の各導入後に 40 steps 標準＋200 steps 最適化を回し、非破壊性と速度を確認
- しきい値（AG/DG）は分位に基づく提案を併記（例: θAG=P80(g0), θDG=P20(gmin)）

想定フラグ（初期案）
- `--sp-cache --sp-cache-mode ds`（Phase A）
- `--epc-eval`（Phase B）
- `--adaptive-hops` + `--dg-commit-policy threshold`（Phase C）

備考
- IG符号は現行の after_before を既定。アブレーションで before_after も用意。
- r_link=0.04（半径ではなく simしきい値0.04ではない点に注意）。半径上限の方は `--link-radius`。
