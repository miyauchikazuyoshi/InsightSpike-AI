# Layer1/Layer2 候補エッジ提案と Layer3 SPエンジン（core/cached/cached_incr）仕様

目的: 迷路で実績のある「固定ペア再利用＋場合分け再計算」に基づく ΔSP 評価を、汎用パイプライン（L1/L2/L3）でも再現可能にする。責務分離（L1/L2: 候補生成、L3: 統合評価とゲーティング）を徹底し、実運用で切り替え可能な SP エンジン（core/cached/cached_incr）を定義する。

---

## 背景とゴール
- 背景: 迷路PoCでは、固定ペア集合（before に対する representative sampling）を辞書登録し、端点SSSPの再利用＋場合分けで ΔSP を高速・安定に再計算している。
- ゴール: RAG/汎用経路でも同等の手法を使えるように、L1/L2 で候補エッジ（Ecand）を生成し、L3 が ΔH/ΔEPC/ΔSP を統合して AG/DG を判定できる状態にする。

---

## 責務分離
### Layer1（＋ Layer2）
- クエリ中心の「中心ノード（centers）」選定。
- 類似度・しきい値・Top‑K 等で「候補エッジ集合（Ecand）」を生成：
  - 形式: `candidate_edges: List[Tuple[int, int, Dict]]` 例: `[(center_idx, node_idx, {"score": 0.82})]`
  - ノードIDは L3 の `current_graph` の index に一致
- 既存の `candidate_selection`（k★、l1_candidates、log_k_star 等）も維持し、必要に応じて L3 に渡す。

### Layer3
- 受け取った `centers` / `candidate_edges` / `candidate_selection` を用いて、ΔH/ΔEPC/ΔSP の統合評価を実行。
- SP エンジン選択（core/cached/cached_incr）ノブを解釈し、ΔSP の評価方法を切り替える。
- GateDecider（AG/DG）でゲーティングし、採用/棄却のイベントを制御。

---

## L3 → SP エンジン（core/cached/cached_incr）
### 共通ノブ（config / env）
- `graph.sp_engine`: `core` | `cached` | `cached_incr`（既定: core）
- `graph.lambda_weight`: λ（既定: 1.0） / env `INSIGHTSPIKE_GEDIG_LAMBDA`
- `graph.sp_beta`: ΔSP 重み γ（既定: 0.2） / env `INSIGHTSPIKE_SP_BETA`
- `graph.sp_scope_mode`: `auto` | `union`（SP 評価スコープ）
- `graph.sp_boundary_mode`: `induced` | `trim` | `nodes`（境界処理）
- `graph.sp_hop_expand`: 追加 hop 数（core の multi‑hop 用）
- `metrics.ig_denominator`: `fixed_kstar` 等（IG 分母モード）
- env 専用:
  - `INSIGHTSPIKE_SP_ENGINE`: 上記 `graph.sp_engine` の env 上書き
  - `INSIGHTSPIKE_SP_PAIR_SAMPLES`: 固定ペアサンプル数（既定: 200）
  - `INSIGHTSPIKE_SP_REGISTRY`: PairSet永続ファイル（JSON）パス（指定がなければプロセス内メモリ）

### core（既定）
- GeDIGCore による multi‑hop ΔSP（平均最短路の相対短縮）をそのまま採用。
- 正攻法・最も厳密だが重い。

### cached（between‑graphs 比較）
- hop0 は GeDIGCore で ΔEPC_norm/ΔH を算出。
- ΔSP は DistanceCache で between‑graphs（g_before vs g_after）の固定ペア比較により再計算。
  - 固定ペアは before に対する representative sampling（辞書登録で再利用）。
  - after は SSSP で pairs の平均距離を再計算。
- gmin ≈ ΔEPC_norm − λ(ΔH + γ·ΔSP_rel) で再合成（hop0に準じた簡易最小値）。
- 候補エッジが無い場合（RAG等の全体評価に適合）。

### cached_incr（候補エッジ単位の増分）
- 前提: L1/L2 から `candidate_edges` が提供されていること。
- アルゴリズム（迷路Evaluatorの簡約版）：
  1) before サブグラフと anchors（centers）から、固定ペア集合（PairSet）を取得（レジストリ再利用）。
  2) 各候補エッジ (u,v) について、端点SSSP（before）を再利用し、la′ の候補（dab, au+1+vb, av+1+ub）で平均最短路を更新 → ΔSP を求める。
  3) 貪欲（best‑first）に数本（Top‑K）採用し、g(h) と gmin を更新（λ, γ で ΔH/ΔSP を統合）。
  4) 早期打ち切り（ΔSP が十分小さい、cycle/leaf の分岐）と SSSP のキャッシュを併用し、P95/P99 を短縮。
- フォールバック: `candidate_edges` が無い場合は `cached` に自動ダウンシフト。

---

## レジストリ（PairSet 辞書登録）
### 概要
- 署名: `signature = hash(nodes,edges) + meta(|A|,hop,scope,boundary)`
- 実装: `src/insightspike/algorithms/sp_distcache.py`
  - In‑proc（_MemoryRegistry）と File（_FileRegistry; JSON）を用意
  - env `INSIGHTSPIKE_SP_REGISTRY=/path/to/pairsets.json` を指定すると永続化

### JSON スキーマ（簡易）
```json
{
  "<signature>": {
    "lb_avg": 1.42,
    "pairs": [["u","v", 3.0], ["x","y", 5.0], ...]
  },
  "<signature2>": { ... }
}
```

---

## L1/L2 → L3 インタフェース（context）
### 入力（例）
```python
context = {
  "centers": [0, 3, 8],
  "candidate_edges": [ (0, 12, {"score": 0.83}), (0, 17, {"score": 0.79}), ... ],
  "candidate_selection": {
     "k_star": 5,
     "l1_candidates": 20,
     "log_k_star": 1.609
  }
}
```
### NormSpec（ノルム仕様）の受け渡し（推奨）
```python
context["norm_spec"] = {
  "metric": "cosine",                # or "l2"
  "radius_mode": "intuitive",        # or "absolute"
  "intuitive": {"outer": 0.6, "inner": 0.2},
  # absolute の場合: {"outer": 0.995, "inner": 0.98}
  "dimension": 384,
  "scope": "sphere",                  # or "donut"
  "effective": {                      # 実際に使用したしきい値
    "theta_link": 0.35,
    "theta_cand": 0.45
  }
}
```
Layer3 は `norm_spec` を metrics にエコーし（`metrics["norm_spec"]`）、正規化台（Cmax= c_node + |S_link|·c_edge）と `candidate_selection` の値とあわせて再現に用いる。将来、半径自体を正規化台に反映する設計へ拡張する場合の伏線となる。
### 出力（L3 の metrics 例）
```python
metrics = {
  "delta_ged": -0.34,
  "delta_ged_norm": 0.34,
  "delta_ig": 0.27,      # ΔH + γ·ΔSP_rel (cached/cached_incr)
  "delta_h": 0.21,
  "delta_sp": 0.29,
  "g0": 0.41,
  "gmin": 0.18,
  "sp_engine": "cached_incr"   # or "core" / "cached"
}
```

---

## バリデーション指針
- 小規模グラフで core vs cached/cached_incr の gmin 相関を確認（閾値/順位一致）。
- 速度指標（P95/P99）を記録し、core 比での短縮率を算出。
- フォールバック確認: `candidate_edges` 無し → cached、ノブ未指定 → core。
- レジストリ動作: 同 signature のペア集合再利用（プロセス間は FileRegistry で確認）。

---

## 実装ロードマップ
1) L2 に `propose_candidate_edges(graph, centers, top_k, theta_link)` を追加（Top‑K/しきい値ベース）。
2) L3 に `cached_incr` 分岐を実装（`context['candidate_edges']` を利用、無ければ `cached`）。
3) ノブ（config/env）をまとめ、README/QUICKSTART に反映。
4) 最小ユニット＋スモークテストの追加（core/cached/cached_incr を切り替えて整合性チェック）。

進捗（実装状況）
- [x] DistanceCache レジストリ（Memory/File）実装、between-graphs ΔSP 対応
- [x] Layer3 SP エンジン切替（core/cached）、sp_engine を metrics に出力
- [x] NormalizedConfig に `graph.sp_engine`, `graph.norm_spec` を取り込み
- [x] L2 ヘルパ `propose_candidate_edges_from_graph` を追加
- [x] Layer3 cached_incr の簡易実装（candidate_edges があれば貪欲/予算採用、なければ cached）
- [x] NormSpec の実値（effective θ_link/θ_cand 等）を L1/L2 相当（ExplorationLoop/Agent 経由）で context に格納（WakeSleep.SphereSearchConfig を優先して導出）
- [x] candidate_edges 未提供時の自動候補生成（L3 内で graph.x と centers から提案）
- [x] torch 非依存の GraphBuilder フォールバック（numpy + Data スタブ）
- [x] 非 query-centric 経路でも metrics に `sp_engine` を明示

-未完了/次タスク
- [x] cached_incr の逐次適用（相互作用考慮）を実装（Greedy sequential; budget制御）。切替ノブは今後必要に応じ追加。
- [ ] candidate_edges の堅牢化拡張（重複/空/無効index/予算境界を網羅するテスト追加）
- [ ] config ノブの整理と API リファレンス反映（`graph.cached_incr_budget`, 自動候補 `candidate_topk`/`theta_link`）
- [ ] 図・アーキテクチャ更新（centers/Ecand/norm_spec データフロー、SP scope/boundary）
- [x] Layer3 で `metrics["norm_spec"]` へのエコー（context→metrics。無い場合は config.graph.norm_spec を採用）

---

## Config 設定（config.yaml）
`graph` セクションで SP エンジンと NormSpec を設定できるようにする。

```yaml
graph:
  sp_engine: core            # core | cached | cached_incr
  lambda_weight: 1.0         # λ
  sp_beta: 0.2               # γ（ΔSP 重み）
  sp_scope_mode: auto        # auto | union
  sp_boundary_mode: trim     # induced | trim | nodes
  cached_incr_budget: 1      # 候補エッジ採用の最大本数（逐次）
  # NormSpec（ノルム仕様）
  norm_spec:
    metric: cosine           # cosine | l2
    radius_mode: intuitive   # intuitive | absolute
    intuitive:
      outer: 0.6
      inner: 0.2
    # absolute:
    #   outer: 0.995
    #   inner: 0.980
    dimension: 384
    scope: sphere            # sphere | donut
    # effective は L1/L2 側で決定して context に書き戻す
```

ENV での上書き（任意）:
```
INSIGHTSPIKE_SP_ENGINE=cached
INSIGHTSPIKE_SP_PAIR_SAMPLES=200
INSIGHTSPIKE_SP_REGISTRY=/path/to/pairsets.json
INSIGHTSPIKE_GEDIG_LAMBDA=1.0
INSIGHTSPIKE_SP_BETA=0.2
INSIGHTSPIKE_SP_BUDGET=2
INSIGHTSPIKE_CAND_TOPK=10
```

NormalizedConfig では、`graph.sp_engine` と `graph.norm_spec.*` を読み出し、L1/L2/L3 へ一貫して渡す。現行の SphereSearchConfig（intuitive_radius 等）からの移行は後方互換のブリッジで吸収する。

---

## テスト計画
### ユニットテスト
- GateDecider/PSZ: 閾値条件の境界値での判定、PSZ の P50 計算
- DistanceCache:
  - MemoryRegistry/FileRegistry の load/save 循環（署名一致でペア集合再利用）
  - between‑graphs ΔSP_rel が [0,1] に収まり、エッジ追加で非負に寄る傾向
- 正規化メタ（NormSpec）: context に載せて L3 metrics にエコーされる（実装後）

### 結合（パイプライン）テスト
- L1/L2 モックで centers/candidate_edges を生成 → L3 を core/cached/cached_incr で切替
  - core vs cached/cached_incr の gmin 相関を閾値で検証（順位の上位 k が一致等）
  - 速度目標（P95/P99）で cached/cached_incr が core より優位
- フォールバック: candidate_edges 無し → cached に自動ダウンシフト

### 非機能テスト
- レジストリの永続ファイルが成長しすぎない（簡易サイズチェック）
- ENV と config.yaml の優先順位が意図どおり（ENV > config > 既定）

---

## ドキュメント更新（アーキテクチャ / API / 図）

本仕様に合わせ、以下を更新する。

- アーキテクチャドキュメント（docs/architecture/*）
  - L1/L2/L3の責務分離に NormSpec と SP エンジン切替（core/cached/cached_incr）を追記
  - クエリ中心パスにおける `centers`/`candidate_edges`/`candidate_selection`/`norm_spec` のデータフロー図
  - 正規化台（Cmax）と IG 分母モード（fixed_kstar 等）の整合に関する注意

- APIリファレンス（docs/api-reference/*）
  - config.yaml の `graph.sp_engine`, `graph.lambda_weight`, `graph.sp_beta`, `graph.sp_scope_mode`, `graph.sp_boundary_mode`, `graph.norm_spec.*` を追加
  - 環境変数の優先順位（INSIGHTSPIKE_SP_ENGINE / _SP_PAIR_SAMPLES / _SP_REGISTRY / _GEDIG_LAMBDA / _SP_BETA）を明記
  - L3 `analyze_documents` context の拡張（`candidate_edges`, `norm_spec`）

- 図（docs/images/* / docs/architecture/*）
  - コンポーネント図：L1/L2→L3 の context 受け渡し（NormSpec と Ecand を明示）
  - フローチャート：SP エンジンの分岐（core→multi‑hop / cached→between / cached_incr→増分）
  - 正規化関連の図：S_link / k★ / norm_base と IG 分母の対応

チェックリスト
1) アーキテクチャ章のデータフロー図を更新（mermaid/plantUML いずれか）
2) API サマリーに新規オプションを追記し、既存の設定例に `graph.norm_spec` を追加
3) 迷路/RAG のミニチュアルート図を更新して、core/cached/cached_incr の切替位置を明確化
