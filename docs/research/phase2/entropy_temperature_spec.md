# Phase 2 Spec: Softmax Shannon Entropy（Boltzmann分布）と温度パラメータ

この仕様書は、geDIG の情報側指標 $\Delta H_{\mathrm{norm}}$（Shannon entropy 差分）を **Softmax Shannon（= Boltzmann/Gibbs 分布）**として定義し直し、Phase 2（sleep/offline）で導入する「ゴール価値伝播→行動選択バイアス（softmax）」と **同一の数式骨格で整合**させるための設計を記述する。

## 1. 目的（Why）
- **構造（$\Delta$EPC, $\Delta$SP）と情報（$\Delta H$）の役割分離**を保ったまま、エントロピー計測を理論的に自然な形（Boltzmann）に寄せる。
- Phase 2 の「バイアス付き softmax 行動選択」と、Phase 1 の「エントロピーによる曖昧さ計測」を **同じ確率モデル（softmax/temperature）**で統一する。
- 既存の実装（正の重みを線形正規化）と **後方互換**を維持しつつ、二重指数などの不自然な変形を避ける。

## 2. 前提（Scope / Constraints）
- $\Delta$の向きは paper/実装と同じく **after−before** を維持する。
  - $\Delta H_{\mathrm{norm}}<0$ はエントロピー低下（秩序化/確信↑）。
  - $\Delta H_{\mathrm{norm}}>0$ はエントロピー上昇（拡散/不確実性↑）。
- Phase 2 の maze への適用は **strict no-peeking**（ゴール到達後のみ情報伝播）を前提とする。

## 3. 記号と入力データ
候補集合 $S=\{i\}$（linkset / candidate pool / 行動候補）に対して、候補 $i$ ごとに以下のいずれかのスカラーが得られるとする。
- 距離 $d_i\ge0$（小さいほど良い）
- 類似度 $\mathrm{sim}_i>0$（大きいほど良い、ただし「指数化済み」の場合がある）
- ロジット（スコア）$s_i\in\mathbb{R}$（大きいほど良い）
- 正の重み $w_i>0$（正規化前の重み）

本仕様では、確率化に使う温度を **$\tau$** と書く（Phase 2 の行動選択温度と同型）。geDIG のトレードオフ重み **$\lambda$** とは別パラメータとして扱う（同一視はしない）。

## 4. 正規形（Canonical）: Softmax による確率化
### 4.1 推奨（ロジット形式）
候補 $i$ のロジット $s_i$ があるとき、
\[
p_i(\tau)=\frac{\exp(s_i/\tau)}{\sum_j\exp(s_j/\tau)}=\mathrm{softmax}(s/\tau)_i,\qquad \tau>0
\]
とする。これは統計力学の Gibbs/Boltzmann 分布 $p\propto\exp(-E/\tau)$ と同値（$s=-E$）であり、Phase 2 の「バイアスを加算して softmax」をそのまま共有できる。

**例（推奨変換）**
- 距離 $d_i$ がある場合: $s_i=-d_i$（または $s_i=-d_i/T_0$ のように尺度を揃える）
- cosine 類似度 $\cos_i$ がある場合: $s_i=\beta\cos_i$（$\beta$ はスケール）
- 既に正の重み $w_i$ がある場合: $s_i=\log w_i$

### 4.2 後方互換（重み形式; 4.1 と等価）
正の重み $w_i$ を直接持つ場合、
\[
p_i(\tau)=\frac{w_i^{1/\tau}}{\sum_j w_j^{1/\tau}}
\]
としてよい。これは 4.1 の $s_i=\log w_i$ と同値であり、
- $\tau=1$ で **現状の線形正規化** $p_i=w_i/\sum w$ を再現する
- $\tau\to\infty$ で一様分布へ近づく
- $\tau\to0$ で最良候補へ集中する

## 5. Shannon entropy と $\Delta H_{\mathrm{norm}}$
\[
H(S;\tau)=-\sum_{i\in S} p_i(\tau)\log p_i(\tau)
\]
\[
\Delta H_{\mathrm{norm}}=\frac{H_{\mathrm{after}}(\tau)-H_{\mathrm{before}}(\tau)}{\log K}
\]
ここで $K=|S_{\mathrm{after}}|$（分母は固定尺）。符号規約は after−before を維持する。

### 5.1 解釈（運用）
- $\Delta H_{\mathrm{norm}}<0$: 分布がシャープ化（確信↑、曖昧さ↓）
- $\Delta H_{\mathrm{norm}}>0$: 分布がフラット化（不確実性↑、多様化/均質化）
- 「秩序化の利得」を正で読みたい場合は $-\Delta H_{\mathrm{norm}}>0$ を利得と呼ぶ（式自体は反転しない）。

## 6. Phase 2 仕様との整合（Maze: ゴール価値伝播→softmax）
Phase 2 では、ゴール到達後に経験遷移グラフ上で価値/ポテンシャル $V(Q)$ を計算し、Phase 1 の行動選択に **加算バイアス**として注入する（詳細は `docs/research/phase2/draft_specification.md` の「Maze Application」参照）。

行動 $a$ のロジットを
\[
\mathrm{logit}(a)=\mathrm{base\_logit}(a)+\beta\,\phi_{\mathrm{goal}}(Q,a)
\]
として、
\[
p(a\mid Q)\propto \exp(\mathrm{logit}(a)/\tau)
\]
でサンプリング/選択する。この形は 4.1 の正規形そのものであり、**Phase 2 の追加項が「softmax の前に加算」**として自然に入る。

## 7. 実装ガイド（既存実装への当てはめ）
### 7.1 Maze（距離→類似度が指数化済みのケース）
迷路では、しばしば
\[
\mathrm{sim}(a)=\exp(-d(a)/T_0)
\]
のように **指数化済みの類似度**を作る。この場合、確率化を
- 非推奨: $p\propto \exp(\mathrm{sim}/\tau)$（二重指数になりやすい）
- 推奨: $p=\mathrm{softmax}(\log \mathrm{sim}/\tau)$（= $p=\mathrm{softmax}(-d/(T_0\tau))$）

として統一する。

### 7.2 Linkset（正の重みを線形正規化しているケース）
候補の `similarity` を正の重み $w$ と見なしている場合は、4.2 の
\[
p\propto w^{1/\tau}
\]
で softmax 化を導入できる。$\tau=1$ を既定にすると挙動は現状互換のまま、温度スキャンを可能にできる。

## 8. 設定・ログ（再現性）
最低限ログすべき項目：
- `entropy_method`（`logit_softmax` / `weight_power` など）
- `entropy_temperature_tau`
- `H_before`, `H_after`, `delta_h_norm`, `K`
- （maze）`base_logit` の定義（`-distance/T0` か `log(sim)` か）
- `embedding_version/index_version/graph_version`（Phase 2 で分布が変わるため）

## 9. 検証（最低限のアブレーション）
- `tau=1`（現状互換） vs `tau≠1`（温度付き）
- maze: success/steps/AG・DG発火率の変化、no-peeking 逸脱がないこと
- rag: FMR/Acc/Latency と $s_{\mathrm{PSZ}}$ の operating curve 上の変化
- 追加: $\lambda$ と $\tau$ は独立ノブとして sweep（「\tau \propto 1/\lambda」仮説は *後付け検証* として扱う）

## 10. 非目標（Non-goals）
- Von Neumann entropy を $\Delta H$ の主指標には採用しない（構造側指標と役割が被りやすい）。必要なら診断用に別ログで扱う。

## 11. 簡易スキャン結果（Phase 1 Maze, linkset IG）
リンクセット IG（base=pool, norm=cand, $\theta_{\mathrm{cand}}=\theta_{\mathrm{link}}=10$, cand/link radius=100, max\_steps=10, maze=15×15, seeds=1, $k^\*=20.8$ 平均）で $\tau$ を掃引した簡易結果。

| $\tau$ | $\overline{\Delta H_{\mathrm{norm}}}$ | median $\Delta H_{\mathrm{norm}}$ | $\overline{H_\mathrm{after}}$ | $g0$ mean | $gmin$ mean |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.2505 | 0.2215 | 0.6931 | 0.5316 | 0.4624 |
| 1.0 | 0.2504 | 0.2215 | 0.6932 | 0.5316 | 0.4624 |
| 2.0 | 0.2408 | 0.2110 | 0.7249 | 0.5413 | 0.4720 |

所見:
- $\tau$ を上げると分布がフラット化し、$\Delta H_{\mathrm{norm}}$ が小さく（=情報利得が弱く）なる一方、$H_\mathrm{after}$ は増える。
- $g0/gmin$ は $\tau$ 依存でわずかに動く。$\tau$ 上昇で IG 抑制→コスト側優勢になり $g$ がやや上がる傾向。
- $\tau=1$ が現行互換で、ハイパスイープが後方互換を壊さないことを確認。
