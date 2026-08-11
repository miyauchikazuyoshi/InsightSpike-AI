# geDIG の多尺度粗視化・くりこみ仮説

- **Date:** 2026-08-12
- **Status:** DRAFT / Hypothesis / Research design
- **Canonical reference:** `docs/gedig_spec.md`（geDIG v4）

> [!IMPORTANT]
> 本メモは、シナプス・ニューロン・局所回路・機能ユニットの間で geDIG 型評価則が粗視化可能かを検証する仮説である。脳に普遍的な固定点があること、同一の geDIG 値が全スケールで保存されること、Visual Packet が神経系での粗視化を実証することは、現時点では主張しない。

## 0. 中心仮説

検証したいのはスカラー \(F\) の数値的不変性ではなく、欠落自由度を defect（欠損項）または有限記憶として明示したときに、geDIG の**汎関数族と意思決定則が粗視化に対して近似的に閉じること**である。

スケール \(\ell\) の状態を \(X_\ell\)、粗視化を \(C_\ell\)、局所編集・介入を \(U_a\)、その macro action を \(\bar a=\Pi_\ell(a)\) とする。最小の検証対象は、

\[
C_\ell(U_aX_\ell)\approx U_{\bar a}(C_\ell X_\ell)
\]

という状態遷移の近似可換性と、

\[
F_{\ell+1}(CX,CX')
=Z_\ell\,\mathbb E[F_\ell(X,X')\mid
CX,CX',\Pi_\ell(a)=\bar a]+\varepsilon_{F,\ell}
\]

という functional-family closure である。これは後述する共通の fiber-average 変換 \(T_\ell\) が \(S=(E,H,B)\) に誘導する関係の略記であり、\(F\) だけに別の写像を fit しない。\(Z_\ell>0\) は単位・正規化の変更、\(\varepsilon_F\) は粗視化 defect を表す。実用上は一点一致より、十分な margin を持つ AG/DG 判定の保存を重視する。

## 1. 正準式、符号、用語

### 1.1 現行正準形

本メモは geDIG v4 を正準形とする。

\[
F=\Delta EPC_{\mathrm{norm}}
-\lambda(\Delta H_{\mathrm{norm}}+\gamma B),
\qquad B:=\Delta SP_{\mathrm{rel}},
\]
\[
\Delta SP_{\mathrm{rel}}
=\frac{L_{\mathrm{before}}-L_{\mathrm{after}}}{L_{\mathrm{before}}}.
\]

現行正準の第 3 項 \(B\) は **\(\Delta SP_{\mathrm{rel}}\)** である。Betti 数 \(\beta_1\) や persistence は候補観測量であり、正準項ではない。導入時は置換・拡張・比較対象のどれかを明示し、別記号にする。

### 1.2 \(\Delta H\) は符号付き候補分布エントロピー変化

\[
\Delta H_{\mathrm{raw}}:=H_{\mathrm{after}}-H_{\mathrm{before}},
\qquad
\Delta H_{\mathrm{norm}}:=\frac{\Delta H_{\mathrm{raw}}}{d(K_{\mathrm{after}})},
\quad
d(K):=
\begin{cases}
\log K,&K\ge2,\\
\epsilon_H,&K<2.
\end{cases}
\]

数学量としては**符号付き候補分布エントロピー変化**であり、構築・探索 profile ではこれを**符号付き伸展利得**と解釈する。正なら after 側で候補質量がより均質に分布し、複数候補が同程度に選択可能になる方向、負なら一部候補へ集中する方向を表す。候補数の増加は正値の必要条件ではない。これを「不確実性減少」と呼ぶと符号を逆転させるため、その呼称を使わない。正準仕様どおり分母は after の候補数 \(K_{\mathrm{after}}\) から定め、before/after の双方へ同じ分母を使う。\(K<2\) の guard \(\epsilon_H>0\) は固定し、報告する。

この読みは、迷路実験で扱う**構築・探索用 profile**である。\(\Delta H>0\) は探索可能性・選択肢の均質化を増やし、式中で減算されるため、他項が同じなら \(F\) を下げる。判断・収束時には集中が有用な場合もあるが、それを利得とみなすなら符号規約または目的関数を変えた別 profile として事前登録する。profile 切替は外生的な状態機械と観測可能な条件で確証実験前に固定し、結果を見て profile や \(\Delta H\) の意味を反転させない。

### 1.3 AG は \(\Delta H\) 単独ではない

正準仕様の 0-hop 値は、

\[
g_0=\Delta EPC_{\mathrm{norm}}-\lambda\Delta H_{\mathrm{norm}}
\]

である。AG は entropy または novelty 単独の閾値ではなく、編集コストと符号付き伸展利得の組を評価する。multi-hop 候補 \(h\) を含む DG 側も、

\[
g_{\min}=\min_h\{\Delta EPC_{\mathrm{norm}}
-\lambda(\Delta H_{\mathrm{norm}}+\gamma\Delta SP_{\mathrm{rel}}^{(h)})\}
\]

という全項の値である。現行 canonical event flow は、

\[
AG=[g_0>\theta_{AG}]\quad\text{（探索を開く）},
\qquad
DG=[\min(g_0,g_{\min})\le\theta_{DG}]\quad\text{（commitする）}
\]

であり、AG は永続状態を確定しない。AG は局所評価が不十分な高 score 側で探索を開き、DG は検証後の低 score 側で確定するため、不等号の向きが異なる。multi-hop 必須条件や論理結合を変更する adapter は非正準 profile と明記し、閾値とともに記録する。

### 1.4 固定すべき gauge

- before/after の向き、EPC primitive と重み
- entropy の確率変数、候補集合 \(K\)、\(\epsilon_H\)
- shortest-path のノード対、到達不能処理、辺距離
- \(\lambda,\gamma\)、AG/DG 閾値、時間窓と遅延

gauge の違いをスケール効果と誤認しない。

### 1.5 Regime-conditioned gauge 仮説

既存の[三つの読み方メモ](gedig_formula_three_readings_20260306.md)は探索相と秩序相、[Transformer 相転移メモ](insight_transformer_phase_transition_landscape.md)は entropy と topology、[β₁ routing メモ](beta1_navigation_routing_20260208.md)は閾値掃引による構造レイヤー境界を論じている。本メモでは一歩進め、測るべき確率変数と carrier graph 自体が regime に依存する可能性を、未検証の拡張仮説として扱う。

相または regime の状態を \(z\in\mathcal Z\)、version 固定した observable registry の参照を \(r_H(z),r_B(z)\) とし、

\[
F_{\ell,z}=E_{\ell,z}-\lambda_{\ell,z}
\left[s_z\Delta H_{\ell}^{r_H(z)}
+\gamma_{\ell,z}B_{\ell}^{r_B(z)}\right]
\]

と置く。registry entry は、entropy の確率変数・候補集合・after 側 \(K\)・guard、構造の carrier graph・SP pair・到達不能処理、正規化、閾値、計算費を固定する。情報温度 \(\lambda\) と entropy 分布を作る温度 \(\tau_H\) は別パラメータとして記録する。geDIG v4 の正準 arm は \(s_z=+1\)、\(B=\Delta SP_{rel}\) である。\(s_z=-1\) による集中利得、または \(B\) の \(\beta_1\) / persistence / curl への置換は、正準の読み替えでなく別の phase-profile 拡張として比較する。

最初から latent phase を仮定しない。検証順は、(1) 介入前に指定する外生 regime、(2) `O_<=t` だけを使い別 split で凍結した causal selector、(3) `q_t(z)=P(z_t | O_<=t)` を持つ latent filter とする。selector には hysteresis、minimum dwell、switch cost を設ける。同じ遷移の `H,B,F`、未来 event、reward、正解から最小の `F_z` を選ぶことと、offline smoothing を因果推論として使うことを禁止する。候補編集は生成時の `regime_id/profile_id` を保持し、途中で regime が変わっても同じ profile で DG または rollback まで完了する。全 registry entry を shadow-log し、系の変化と測定器切替による数値ジャンプを分離する。

粗視化では `z` も状態の一部として、

\[
\widetilde C_\ell(X,z)=(C_\ell X,\Phi_\ell z),
\qquad
F_{\ell+1,\Phi_\ell z}
\approx Z_{\ell,z}\,\mathbb E[F_{\ell,z}\mid
CX,CX',\Pi_\ell(a),z]
+\varepsilon_{\ell,z}
\]

を regime 内と境界近傍で別々に検証する。粗視化窓に複数 regime が混ざる場合は単一 label に潰さず、占有率・遷移回数・posterior を保持する。`z` を捨てると `H(Y)=E_z[H(Y|z)]+I(Y;z)` の相情報が entropy defect になる。

比較は fixed v4、固定 observables で係数・閾値だけを regime scheduling する arm、calibration で選ぶ static-best、oracle `z`、causal selector、latent filter、offline-smoother 上限、random/shuffled `z` とする。oracle でも fixed gauge に勝たなければ regime 依存仮説、oracle だけ勝てば selector、smoother だけ勝てば因果的 regime 推定を棄却する。shadow-log 上の固定 gauges は変わらず、切替後の `F` だけが跳ぶ場合は測定 artifact とする。臨界性・秩序変数・hysteresis の独立証拠が出るまでは「相転移を検出した」でなく **regime-conditioned gauge** と呼ぶ。

## 2. 多尺度状態と観測量

状態を、

\[
X_\ell=(G_\ell,a_\ell,\mu_\ell,\tau_\ell,m_\ell,z_\ell)\in\mathcal X_\ell
\]

とする。

- \(G_\ell\): 有向・符号付き・重み付き graph / hypergraph / cell complex
- \(a_\ell\): ノード・辺・cell 上の活動、発火、同期など
- \(\mu_\ell\): 候補、信念、予測、経路に対する分布
- \(\tau_\ell=(t,\Delta t_\ell)\): 時刻と時間窓
- \(m_\ell\): 消去自由度を近似する任意の有限記憶状態
- \(z_\ell\): 外生または因果的に推定した regime と profile provenance

神経 graph では通常ニューロンまたは compartment が node、synapse が edge である。object type はスケールごとに明示する。

スカラー化前に、

\[
S_\ell=(E_\ell,H_\ell,B_\ell)
=(\Delta EPC_{\ell,\mathrm{norm}},
\Delta H_{\ell,\mathrm{norm}},
\Delta SP_{\ell,\mathrm{rel}})
\]

を保存する。まず 3 成分の変換則、最後に \(F\) と gate を検証する。成分 defect の偶然の相殺を closure の証拠にしない。

## 3. 粗視化 \(C\)、介入 \(U\)、可換図

\[
C_\ell:\mathcal X_\ell\to\mathcal X_{\ell+1},
\qquad U_a:X_\ell\mapsto X_\ell',
\qquad \bar a=\Pi_\ell(a).
\]

\(C_\ell\) は最低限、partition / cover、edge aggregation、activity pooling、time binning、\(\mu\) の pushforward、path-pair 対応、filtration、必要なら \(m\) の推定を定義する。partition だけでは geDIG の粗視化にならない。

```text
          U_a
X_ell -------------> X'_ell
 |                       |
 | C_ell                 | C_ell
 v                       v
X_ell+1 -----------> X'_ell+1
          U_a_bar
```

状態レベルの commutator residual を、

\[
\delta_{\mathrm{comm},\ell}(X,a)
:=d_{\ell+1}(C_\ell U_aX,U_{\Pi_\ell(a)}C_\ell X)
\]

とする。距離は graph、活動、分布の積距離として事前定義する。

観測量については、

```text
(X_ell, X'_ell) -------- S_ell --------> R^3
       |                                    |
       | C_ell x C_ell                      | T_ell
       v                                    v
(X_ell+1, X'_ell+1) ---- S_ell+1 -------> R^3
```

の近似可換性を調べる。primary estimand は、同じ macro before/after と macro action の fiber 上で、

\[
T_\ell[S_\ell]
:=A_\ell\,\mathbb E[S_\ell\mid
C_\ell X=\bar X,\ C_\ell X'=\bar X',\ \Pi_\ell(a)=\bar a]
\]

とする。\(A_\ell\) は確証実験前に固定する component-wise rescaling である。paired micro/macro intervention はこの条件付き期待値の Monte Carlo 標本として扱い、全成分と \(F\) に同じ fiber 条件を使う。exact toy の singleton fiber だけは対応する一標本との比較に退化する。

micro dynamics が Markov でも、内部状態を消去した macro dynamics は一般に非 Markov になる。有限履歴・latent state・memory kernel を比較してよいが、記憶次元と履歴長を事前登録する。無制限の記憶で residual を吸収して反証不能にしない。

## 4. 成分ごとの変換則と defect

### 4.1 EPC defect

粗視化では block 内 edit が消え、複数 edit が 1 個に潰れ、相反する edit が相殺され、1 macro edit が多数の micro edit を代表しうる。よって EPC は一般に加法的でない。

raw edit cost と v4 の正規化分母を、

\[
P_\ell:=\Delta EPC_{\ell,raw},
\qquad M_\ell:=M_{\ell,\mathrm{candidate\_base}},
\qquad E_\ell:=\frac{P_\ell}{M_\ell}
\]

と分ける。raw cost、分母、normalized cost の defect をそれぞれ、

\[
D_{P,\ell}:=P_{\ell+1}-T_{P,\ell}[P_\ell],\qquad
D_{M,\ell}:=M_{\ell+1}-T_{M,\ell}[M_\ell],
\]
\[
D_{E,\ell}:=E_{\ell+1}-T_{E,\ell}[E_\ell]
\]

として測る。\(D_E\) には raw edit の非加法性と candidate-base 分母の変化がともに入るため、\(D_P,D_M\) を併記する。exact toy case では block 間 edit のみを許し、block size に基づく既知 rescaling を与える。

### 4.2 raw \(H\) の chain rule

micro 変数を \(Z\)、macro label を \(Q=C(Z)\) とすると、

\[
H(Z)=H(Q)+H(Z\mid Q).
\]

before/after 差について、

\[
\Delta H^{micro}_{raw}=\Delta H^{macro}_{raw}+\Delta H_{within},
\]
\[
\Delta H_{within}:=H_{after}(Z\mid Q)-H_{before}(Z\mid Q).
\]

したがって \(\Delta H^{macro}_{raw}=\Delta H^{micro}_{raw}-\Delta H_{within}\)。これは raw entropy の厳密式であり、内部自由度の entropy 変化を捨てれば \(\Delta H\) は保存されない。

一般の fiber では同じ条件付き期待値を使い、\(D^{raw}_{H,\ell}:=\Delta H^{macro}_{raw}-T^{raw}_{H,\ell}[\Delta H^{micro}_{raw}]\) とする。上の chain rule は各 paired sample 内で評価してから fiber 平均を取る。

### 4.3 \(\log K\) 正規化 defect

chain rule は raw \(H\) には成立するが、候補数が変わる normalized \(H\) にはそのまま成立しない。\(d_\ell=d(K_{\ell,\mathrm{after}})\) とすると、singleton fiber の paired case では、

\[
D^{norm}_{H,\ell}
:=\Delta H^{macro}_{norm}-\Delta H^{micro}_{norm}
=\Delta H^{micro}_{raw}
\left(\frac1{d_{\ell+1}}-\frac1{d_\ell}\right)
-\frac{\Delta H_{within}}{d_{\ell+1}}.
\]

第 1 項は \(K\) 変更、第 2 項は block 内 entropy 消去による defect である。raw entropy、\(K\)、within-block entropy、normalized entropyをすべて保存する。

### 4.4 正準 \(B=\Delta SP_{rel}\) の defect

\[
B_\ell=\frac{L_{\ell,b}-L_{\ell,a}}{L_{\ell,b}},
\qquad D_{B,\ell}:=B_{\ell+1}-T_{B,\ell}[B_\ell].
\]

singleton fiber の paired case で macro 距離が \(L_{\ell+1,b}=qL_{\ell,b}+\eta_b\)、\(L_{\ell+1,a}=qL_{\ell,a}+\eta_a\) なら、

\[
B_{\ell+1}=\frac{q(L_{\ell,b}-L_{\ell,a})+\eta_b-\eta_a}
{qL_{\ell,b}+\eta_b}.
\]

\(\eta_a=\eta_b=0\) の一様 rescaling では保存されるが、一般の quotient では保存されない。対象 pair、到達不能処理、分母 guard を固定し、quasi-isometry / path-distortion bound と距離分布を調べる。

### 4.5 \(\beta_1\) / persistence 候補

候補は別記号 \(B^\beta\) または \(B^{PH}\) とする。quotient は cycle を生成も消去もする。保存を主張するには good cover / Nerve theorem、relative homology、filtration の functoriality、persistence stability の適用条件を確認し、

\[
D_{\beta,\ell}:=B^\beta_{\ell+1}(CX,CX')-T_{\beta,\ell}B^\beta_\ell(X,X')
\]

を測る。安定性の実証なしに正準 \(B\) を置換しない。

### 4.6 全 \(F\) と margin 付き gate 保存

\(F_{\ell+1}-T_{F,\ell}[F_\ell]\) には \(D_E,D_H,D_B\) と \(\lambda,\gamma\) の不一致が入る。\(T_F\) は同じ fiber 条件と成分変換 \(T_E,T_H,T_B\) から誘導し、別 fit しない。scalar residual だけでなく成分 residual を報告する。

向き付き margin を、

\[
m_{AG,\ell}:=g_{0,\ell}-\theta_{AG,\ell},
\qquad
m_{DG,\ell}:=\theta_{DG,\ell}-\min(g_{0,\ell},g_{\min,\ell})
\]

と置く。どちらも正なら対応する gate が発火する。各 \(r\in\{AG,DG\}\) について \(Z_{r,\ell}>0\) かつ

\[
|m_{r,\ell+1}-Z_{r,\ell}m_{r,\ell}|\le\epsilon_{r,\ell}
\]

なら、\(|Z_{r,\ell}m_{r,\ell}|>\epsilon_{r,\ell}\) の領域で判定が保存される。閾値近傍には gate 別の abstention band を置き、margin 別 agreement を報告する。

## 5. closure と十分統計量

汎関数族を、

\[
\mathcal F=\{F_\theta=E-\lambda(H+\gamma B)\}_{\theta\in\Theta}
\]

とする。粗視化後も同じ有限次元族、または明示した最小拡張族に留まることを closure と呼ぶ。許す変換 \(T_\ell\) と defect 関数族、memory 次元、最大履歴長、fit 回数を確証実験前に固定し、calibration data で一度だけ推定して held-out で評価する。\(\varepsilon_F\) は sample ごとに合わせる補正項でなく、説明されずに残った residual として測る。必要な記憶・観測量・非線形項がスケールごとに無制限に増えるなら finite-dimensional closure を棄却する。

時刻 \(t\) までに利用可能な量から、さらに後の未来または介入応答 \(Y_{t+1:t+h}\) を予測する十分性を、

\[
P(Y_{t+1:t+h}\mid X_{\ell,\le t},a_t)\approx
P(Y_{t+1:t+h}\mid C_\ell X_{\ell,\le t},
S_{\ell+1,\le t},\bar a_t,m_{\ell+1,\le t})
\]

で評価する。同一遷移の結果を予測する試験では、after state から計算した \(S\) や memory を入力へ含めない。指標は conditional mutual information、predictive KL、held-out log-loss とする。比較順は、macro state のみ、+ \((E,H,B)\)、+ scalar \(F\)、+ 最小 memory、単純な degree / energy / prediction-error baseline とする。3 成分が十分でも 1 scalar が十分とは限らない。

## 6. parameter flow、semigroup、fixed point

\[
\theta_\ell=(\lambda_\ell,\gamma_\ell,w^{EPC}_\ell,
\mathrm{normRule}_{H,\ell},\epsilon_{H,\ell},
\theta_{AG,\ell},\theta_{DG,\ell},\tau_\ell,\phi^{mem}_\ell),
\qquad \theta_{\ell+1}=R_\ell(\theta_\ell)+\xi_\ell.
\]

主張は段階化する。

1. **Pairwise transport:** 隣接 2 スケールで held-out data に再現する \(R_\ell\) を推定する。この段階では RG flow と呼ばない。
2. **Composition:** nested partition と associative aggregation のもとで、\(R_{\ell+1}\circ R_\ell\approx R_{\ell\to\ell+2}\) を検証する。direct / composed 差 \(\delta_R\) が安定して小さいときだけ近似 semigroup 候補と呼ぶ。
3. **Fixed point:** 同型の粗視化を反復できる 3 スケール以上で \(R(\theta^*)=\theta^*\) と dimensionless な \((E,H,B)\)、defect、gate statistics の同時安定性を調べる。
4. **Universality:** 異なる回路・個体・task・partition が relevant parameter と scaling exponent を共有して初めて universality-class candidate と呼ぶ。

overlapping / non-nested cover では semigroup でなく category / groupoid 的 path dependence がありうる。有限スケールの plateau や単一 dataset の parameter collapse を fixed point / universality の証拠にしない。

## 7. 数学・実験計画

### 7.1 最初の定理

block-homogeneous directed graph、hierarchical SBM、exchangeable block spiking model、または一様 path rescaling を持つ tree-like graph を使う。

最小の exact theorem は、指定した block homogeneity、候補分布の条件付き不変性、edit 対応、path rescaling のもとで \(D_E,D_H,D_B=0\) または既知上界を持ち、汎関数族が閉じる、という形にする。仮定を外した反例も併記する。

近似版では、

\[
|F_{\ell+1}-T_{F,\ell}[F_\ell]|
\le c_E\varepsilon_E+c_H\varepsilon_H+c_P\varepsilon_{path}
+c_\theta\|\theta_{\ell+1}-R_\ell\theta_\ell\|
\]

と gate-flip probability の境界を狙う。係数が graph size とともに無制限に発散しない条件を求める。

### 7.2 実験 phase

1. **Toy:** exact class、raw/normalized logger、defect calculator、可換性 test。
2. **Synthetic intervention:** edge edit、刺激、抑制、時間窓を held-out にして closure と十分性を評価。
3. **Partition robustness:** anatomical、activity-based、random matched、overlapping cover を比較。
4. **Real data:** toy/synthetic で Claim L2 を満たした後、connectome + activity へ進む。観測的 connectome 解析だけで許されるのは L0–L1 までであり、神経可塑性機構の検証には時間整合した活動記録と介入が必要である。

主要指標は \(\delta_{comm}\)、\(D_E,D_H,D_B\)、\(\varepsilon_F\)、sufficiency gap、margin-conditioned gate agreement、\(R\) の split/seed 再現性、\(\delta_R\)、必要 memory 次元である。

## 8. 必須 counterexample suite

1. **同じ quotient、異なる内部 cycle:** macro graph は同じだが block 内 \(\beta_1\) と未来 dynamics が異なる。
2. **cycle の生成・消去:** quotient が micro にない cycle を生成、または micro cycle を block 内へ消す。
3. **同じ pooled activity、異なる E/I timing:** 同じ平均発火だが同期・遅延が異なり未来応答も異なる。
4. **entropy 符号反転:** macro entropy と within-block entropy の変化が競合し、\(\Delta H\) の符号が scale 間で反転する。
5. **EPC 非加法:** 多数の block 内 edit が macro で消え、または 1 macro edit が多数 micro edit を要する。
6. **非 Markov 性:** 同じ macro state に写る 2 micro state が異なる次状態を持つ。
7. **gate flip:** \(g-\theta\) を 0 に近づけ、小 defect で判定を反転させる。
8. **partition 非一意:** 複数の妥当な partition で \(R\)、defect、gate が大きく変わる。
9. **同じ \(S/F\)、異なる効用:** 観測量は同じだが介入応答が異なり、十分統計量仮説を破る。

## 9. Claim ladder

- **L0 — Analogy:** 同じ形式の観測量を複数神経スケールで定義できる。RG、固定点、普遍則とは呼ばない。
- **L1 — Cross-scale observables:** dimensionless に較正した \(E,H,B\) が held-out prediction で単純 baseline を上回る。まだ closure とは呼ばない。
- **L2 — Approximate closure:** 指定した \(C,U\) で commutator と成分 defect に上界があり、margin 外で gate が保存される。ここで初めて「近似粗視化可能」と呼ぶ。
- **L3 — Reproducible flow:** 3 スケール以上で \(R\) が再現し、direct / composed path が整合する。ここで初めて「RG-like flow」「近似 semigroup」を候補として議論する。
- **L4 — Fixed point / universality candidate:** 複数初期条件・独立系から dimensionless statistics が同一領域へ収束し、有限サイズ効果と区別できる。

## 10. 反証条件

次が再現的に成立した場合、対応する強い claim を棄却・縮小する。

- 最小 memory を加えても \(\delta_{comm}\) が random partition と同程度。
- 成分 defect が scale とともに増大し、有限個の項で予測不能。
- margin を広げても AG/DG agreement が calibrated baseline を上回らない。
- macro state + \((E,H,B)\) に micro state の大きい条件付き情報が残る。
- \(R_\ell\) が seed、sample、partition、介入ごとに再現しない。
- direct / composed flow の差が測定誤差を一貫して超える。
- fixed-point plateau が scale、初期条件、held-out task の変更で消える。
- \(\beta_1\)/persistence が filtration の小変更で符号反転し、安定性境界もない。
- degree、energy、prediction error などの単純 baseline が同等以上。
- counterexample 型 degeneracy が実データの主要質量を占める。

L2 以上が反証されても L0–L1 の記述的価値まで自動的には否定されない。

## 11. Visual Packet との境界

Visual Packet / TSD は、本理論の**応用リンクまたは将来の工学的 testbed**に限定する。局所幾何から大域構造への集約、時間窓付き packet、transport / decay / revision は \(X,C,U\) と defect を具体化する別ドメインの例になりうる。具体的な反証プロトコルは、[vector-based-cnn-ocr 側の対応メモ](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr/blob/main/docs/notes/theory/visual_packet_multiscale_gedig_20260812.md)へ分離する。

しかし、その成功は、神経スケール間の closure、同一 \(R\) や fixed point、geometry/topology の同型、AG/DG と脳機能の一対一対応を証明しない。Visual Packet を神経粗視化仮説の根拠・生物学的 validation として使わない。

## 12. 最小成果物と結論

必要な成果物は、次である。

- **数学:** 型付き \(X,C,U,S\)、raw \(H\) chain rule、\(\log K\) defect、EPC/B defect、margin 付き gate 保存、各仮定を破る反例。
- **コード:** coarse-graining interface、gauge record、raw/normalized logger、defect/commutator/path-consistency evaluator。
- **実験:** exact toy class、counterexample suite、synthetic intervention、3-scale transport、random/degree/energy/predictive-error baseline。

中心文は以下とする。

> geDIG の粗視化可能性とは、\(F\) の数値が全スケールで保存されることではない。捨てた自由度を defect または有限記憶として明示した後に、観測量の汎関数族と margin 付き意思決定が閉じることである。

最初の到達目標は L2、すなわち限定した graph/dynamics class における近似 closure と margin 付き gate 保存である。parameter flow、semigroup、fixed point、universality は、その成立後に順に検証する段階的主張とする。
