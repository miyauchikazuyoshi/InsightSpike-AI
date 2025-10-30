# トポロジ署名（Topology Signature）による SP 前処理再利用 — 提案（保留）

Status: Proposed (on hold) / Owner: QH / Date: 2025‑10‑29

## 背景と目的

- SP 計算（固定ペア＋平均距離 Lb、cached_incr の端点 SSSP 等）は、評価サブグラフの規模に比例して重くなる。
- 現行の DS 再利用（厳格署名: ノード/エッジ/anchors/eff_hop/scope/boundary を完全一致）は“正しさ優先”で安全だが、跨ステップで一致しづらい（迷路のように窓が滑っていく場合）。
- トポロジ（同型）ベースの署名で“座標や絶対IDに依存しない”指紋を作れば、跨ステップ/跨ランでも固定ペア＋Lb を再利用できる可能性がある。

本提案は、正しさを後退させない範囲で「トポロジ署名（topo_v1）」を追加し、A/B 評価で効果を測る計画である。なお、迷路はほぼ直線グラフでパターンが少なく、効果判定が難しいため“保留”とする（複雑なグラフで SP 問題に直面した時に再検討）。

## 要件（非機能含む）

1. 正しさ優先：署名が一致したときのみ再利用し、不一致/曖昧時は現行（厳格署名 or 再計算）にフォールバックする。
2. 評価条件を含む：eff_hop / scope（union 等）/ boundary（trim 等）/ anchors の役割を署名に反映する。
3. 可搬性：抽象 API（`SPPairsetService` / `SignatureBuilder`）にぶら下げる構成で、実験→本体移植が容易。
4. 安全な導入：既定は厳格署名のまま。topo_v1 はフラグで A/B 運用。

## トポロジ署名（topo_v1）の設計概要

アンカー Q を根とした局所サブグラフを“座標非依存”にカノニカル化し、その構造指紋を署名とする。保存時は固定ペア（カノニカル ID 表現）と Lb を記録し、再利用時は“実ノード ↔ カノニカル ID”の写像を構築して適用する。

### カノニカル化手順

1) multi‑source BFS による層分け：
   - L0 = anchors_core（通常 Q）、L1…Lk（eff_hop まで）。
2) 初期色（C0）付与：
   - node_type（query/dir）、anchor_role（core/top/none）、degree（層内/全体）などの組をハッシュ化。
3) 近傍色による反復洗練（1‑WL/Weisfeiler‑Lehman）：
   - T≒eff_hop 反復。Ct+1(u) = hash( Ct(u), multiset{ Ct(v) | v∈N(u) } )。
4) 層内の安定順序付け：
   - ソートキー: (色 → degree → node_type → 近傍色列ハッシュ) で層ごとに rank を付与。
5) カノニカル ID とエッジ：
   - CN(u)=(layer, rank[, type]) を ID とし、エッジCN(u)–CN(v)を整列して列挙。
6) 署名生成：
   - sha256( 層サイズ列 | カノニカルエッジ列 | アンカー役割分布 | eff_hop | scope | boundary )。

### 保存フォーマット（Pairset / DS スキーマの拡張）

- 既存 `sp_pairsets` / `sp_pairs` を流用（`meta.scheme = 'topo_v1'` を付与）。
- 保存：
  - signature（topo_v1）
  - lb_avg（連結ペアの平均距離）
  - pairs（`(u_canon, v_canon, d_before)` の配列）
  - node_count, edge_count, eff_hop, scope, boundary
  - meta（層サイズ/役割分布/カノニカル化パラメータ/`scheme='topo_v1'`）

### 再利用（SPbefore）

1. 評価サブグラフに対してカノニカル化→(signature, mapping f: 実ノード→カノニカル ID)。
2. DS で signature を lookup。
3. ヒット時、pairsを f^{-1} で実ノードペアに戻し、Lb/pairs を再利用（固定ペア生成をスキップ）。
4. ミス/曖昧時は厳格署名→生成→保存。

### cached_incr との整合

- δSP_fast は端点 u/v の SSSP を使うため、座標非依存でも“実グラフ”でSSSPは必要。
- ただし、今回導入済みの“leaf 追加スキップ（dv側 SSSP を打たない）”や“円環時のみ厳密検証”で、BFS回数はさらに削減可能。

## 安全性とフォールバック

- 署名には eff_hop/scope/boundary/アンカー役割を含める。
- カノニカル化の tie‑break が曖昧な場合や、写像に失敗した場合は必ずフォールバック（厳格署名 or 再計算）。
- 検証段階ではダブルチェック（例：層サイズ/次数分布/近傍色分布ハッシュの一致）を追加可能。

## API / フラグ

- 既存の `SPPairsetService` 抽象に依存（InMemory/SQLite/Noop）。
- 新フラグ（実験用想定）
  - `--sp-ds-sqlite <path>` … DS 有効化
  - `--sp-ds-namespace <ns>`
  - `--sp-sig-mode {strict, topo_v1}` … 署名モード切替（既定: strict）

## 評価計画（A/B）

- ベンチセット：
  - 迷路（直線中心; ヒット率評価には不利）と、より複雑な形状（分岐/円環が多いグラフ）を用意。
- 指標：
  - 署名ヒット率、固定ペア生成時間、端点 SSSP 回数、cand_ms/eval_ms、総所要、DG/SP ゲートの発火率。
- 期待：
  - 円環検出/ショートカットの多い形状で、before 再利用が効きやすくなる。
  - 正しさを損なわず、BFS回数（特に固定ペア生成側）を低減。

## ロールバック / 互換

- 既定は strict（現行）。topo_v1 はフラグを切れば完全無効。
- DS への保存も `scheme` を見て区別できるため、混在しても問題ない（読み出し側でモード判定）。

## 備考（今回保留の理由）

- 迷路はほぼ直線グラフでパターンが少なく、判定が難しい。複雑な形状で SP 問題（特に円環の早期検出）が必要になったタイミングで再検討する。
- 現時点では「正しく計算すること」を第一目的に、leaf スキップや厳格署名 DS（既存）、円環時限定検証の最適化を優先する。

