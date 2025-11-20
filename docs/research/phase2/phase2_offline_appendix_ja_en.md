# Phase 2: Offline Optimization / Sleep Phase Appendix  

このドキュメントは、geDIG v5 論文で「Phase 2（オフライン全体最適化）」と呼んでいる設計案のうち、  
より神経科学メタファ寄りのアイデア（睡眠フェーズ / 神経調節物質など）を、日本語と英語で補足するための付録です。  
論文本文の主張や実験結果は Phase 1 のみを対象としており、ここで述べる内容は **将来の拡張候補** です。

---

## 日本語版

### 1. 概要

Phase 2 は、オンラインの Phase 1 で蓄積されたログやグラフ構造を、入力を止めたオフライン環境で再利用しながら  
「\textbf{全体整合の回復と圧縮}" を行うフェーズです。  
覚醒 / 睡眠の比喩で言えば、Phase 1 が覚醒（オンライン更新）、Phase 2 が睡眠（オフライン最適化）に対応します。

主な狙いは次のとおりです。

- **記憶の統合（Memory Consolidation）**: 重要な接続を強化し、弱くノイズ的な接続を剪定する。
- **グラフ最適化（Graph Optimization）**: 検索効率の観点から、冗長な枝を減らし、橋構造を整理する。
- **洞察の発見（Insight Discovery）**: 離れたノード間の「見落としていたショートカット」を、シグナル伝播や共起パターンから発見する。
- **資源／エネルギ効率（Energy / Resource Efficiency）**: 「代謝コスト」を意識した重み更新や剪定を行い、限られた記憶容量での運用を支援する。

### 2. 目的関数と Phase 1 との関係

Phase 1 では、$\F = \Delta\mathrm{EPC}_{\mathrm{norm}} - \lambda\,\Delta\mathrm{IG}_{\mathrm{norm}}$ を用いて  
「その場の編集（ローカル更新）」を判断します。  
Phase 2 では、同じログとエッジ特徴量を使いながら、より大きな単位での再配線・圧縮を行うことを想定します。

典型的な目的関数の例は、v5 本文にあるような

> $\min_{G'\subseteq G}\; \alpha\,\mathrm{GED}_{\min}(G,G') + \beta\,H(G') + \gamma\,|E(G')|$  
> subject to 一貫性 / 到達可能性 / 重要度制約

のような「最小編集距離 + エントロピー + エッジ数」の組み合わせであり、  
Phase 2 ではこれを直接解く代わりに、\textbf{ヘッブ学習・シグナル伝播・剪定／再配線の組み合わせ}で近似する、という立場を取ります。

### 3. コア概念

#### 3.1 神経調節物質メタファ

Phase 2 の操作を設計するために、以下のような「神経調節物質（neuromodulator）」をメタファとして導入します。  
生理学的な写像を主張するものではなく、\textbf{制御ロジックを整理するための比喩}です。

- **GABA（抑制・スパース化） — 「原始的なブレーキ」**  
  - 役割: 近傍内での抑制・スパース化（lateral inhibition）。  
  - トリガ: アクティブノード数が閾値 $K_{\max}$ を超える、あるいは活性分布のエントロピーが高すぎるとき。  
  - 効果: 温度を下げた softmax で分布を尖らせる、Top-$k$ でハードに切る、Winner 周辺のノードを抑制する。

- **ドーパミン（報酬・強化） — 「探索者」**  
  - 役割: 正の洞察（DG によるゲイン確認）や矛盾検出に対する強化信号。  
  - トリガ: $\Delta\mathrm{geDIG}$ がしきい値 $\theta_{\mathrm{insight}}$ を超える、あるいは明確な矛盾が見つかったとき。  
  - 効果: その経路に沿った STDP 的更新（重みの強化 / 弱化）を行い、Phase 2 の rewiring に反映する。

- **アセチルコリン（注意・状態切替） — 「モード切替」**  
  - 役割: 覚醒（外部入力重視）/睡眠（内部再生重視）のスイッチ。  
  - トリガ: 新規入力の予測誤差が高いとき（AG オープン）／外部入力が少ないときや Sleep Pressure が高いとき。  
  - 効果: 覚醒時は感覚入力の重みを上げ、睡眠時は内部再生（リプレイ）の重みを上げる。

- **コルチゾール（ストレス・適応） — 「あきらめと保護」**  
  - 役割: 「行き詰まり」や慢性的エラーに対するリソース制御。  
  - トリガ: 一定回数以上 AG が開きっぱなしになる／パスコストが予算を超える。  
  - 効果: 短期的には探索の温度を上げるが、慢性的な場合は枝を打ち切り、学習率を下げて保護側に振る。

#### 3.2 シグナル伝播（Signal Propagation）

- シードノード: 直近に多く参照されたノード、あるいは高重要度ノードをシードとする。  
- 伝播: 各ステップで活性が係数 $\gamma$ だけ減衰しながら近傍に広がる。  
- 停止条件: 活性がしきい値 $\theta$ を下回ったときに停止する。  
- 結果: 高活性ノード・エッジのパターンから、「保持すべき構造」「ショートカット候補」を抽出する。

#### 3.3 ヘッブ学習（Hebbian Learning）

> 「一緒に発火する細胞は結びつく」

エッジ $(i,j)$ の重み更新を

> $\Delta w_{ij} = \eta \cdot (s_i \cdot s_j)$

のように定義し、$w_{ij}$ を $[0,1]$ の範囲にクリップする。  
Phase 1 の AG/DG ログや $\F$ の寄与も、$w_{ij}$ に対する「信用度」として組み込むことができる。

#### 3.4 睡眠サイクル（NREM / REM）

Phase 2 を、NREM と REM に対応する二つのモードに分ける：

1. **NREM（局所統合・剪定）**  
   - 焦点: 既存の強い接続の強化と、弱い／古い接続の剪定。  
   - 作用: $w_{ij}$ が小さく、かつ古いエッジを落とす／最近よく使われているエッジを強化する。

2. **REM（大域探索・洞察生成）**  
   - 焦点: グローバルな再配線とショートカット生成。  
   - 作用: 減衰を弱めたシグナル伝播を行い、高共起ノード間に新しいエッジを張る（ただし Phase 1 の $\F$ や FMR 統計を見ながら慎重に適用）。

### 4. アーキテクチャ（概念）

簡略化したコンポーネント構成は次のように整理できる：

- `SleepController`: Phase 2 全体の進行を管理（入眠トリガ、シード選択、伝播、ヘッブ更新、剪定／再配線）。  
- `L2MemoryManager`: 永続的な知識グラフ（NetworkX 等）を保持し、Phase 1 の一時的な構造と同期する。  
- `SignalPropagator`: グラフ上のシグナル伝播と活性計算を担当。  
- `EdgeAttributes`: 各エッジの強さ・最終活性時刻・使用回数・$\F$ への寄与などを保持。

### 5. プロセスフロー（Phase 1 との関係）

1. **覚醒フェーズ（Phase 1, Online）**  
   - クエリ処理・エピソード統合を通じて、AG/DG ログや $\F$ の履歴を蓄積する。  
   - `L2MemoryManager` は、エッジの `usage_count` や `last_activated` を更新する。

2. **睡眠フェーズ（Phase 2, Offline）**  
   - トリガ: 明示的なコマンド（`agent.sleep()`）や「N エピソード毎」など。  
   - ステップ:  
     1. シード選択（最近よく使われたノードや、高重要度ノードを選ぶ）。  
     2. シグナル伝播（減衰付き活性拡散）。  
     3. ヘッブ更新（共起パターンに基づく重み更新）。  
     4. 再配線（REM）：高共起だが未接続のノード間にショートカットを張る。  
     5. 剪定（NREM）：重みが閾値未満・長期間未使用のエッジを削除する。

### 6. 将来拡張の例

- **Dreaming**: 統合されたグラフを使ってエピソードを生成し直す「夢」のような生成リプレイ。  
- **Forgetting**: 干渉しやすい記憶や低価値な記憶を、意図的に忘却するメカニズム。  
- **Multi-Agent Sleep**: 複数エージェント間での同期睡眠・知識共有。

---

## English Version

### 1. Overview

Phase 2 is an offline optimization phase that reuses logs and graph structures collected during Phase 1  
to perform **global consistency restoration and compression** under resource constraints.  
In the awake/sleep metaphor, Phase 1 corresponds to online, query-centric operation (awake),  
while Phase 2 corresponds to offline replay and restructuring (sleep).

The main goals are:

- **Memory consolidation**: Strengthen important connections and prune weak/noisy ones.
- **Graph optimization**: Reorganize the graph to improve retrieval efficiency and robustness.
- **Insight discovery**: Identify non-obvious shortcuts between distant nodes via signal propagation and co-activation.
- **Energy / resource efficiency**: Account for “metabolic” cost and keep the memory graph compact and useful.

### 2. Objective and relation to Phase 1

In Phase 1, the unified gauge

> $\F = \Delta\mathrm{EPC}_{\mathrm{norm}} - \lambda\,\Delta\mathrm{IG}_{\mathrm{norm}}$

controls local edits under online constraints.  
Phase 2 uses the same logs and edge features but aims at larger-scale rewiring and compression.

A typical objective (as in the v5 manuscript) is:

> $\min_{G'\subseteq G} \alpha\,\mathrm{GED}_{\min}(G,G') + \beta\,H(G') + \gamma\,|E(G')|$  
> subject to consistency / reachability / importance constraints,

where $\mathrm{GED}_{\min}$ is a (regularized) graph edit distance, $H$ aggregates local entropies,  
and $|E(G')|$ controls sparsity.  
Phase 2 does not attempt to solve this exactly; instead, it approximates it via  
**Hebbian-style updates, signal propagation, and pruning/rewiring heuristics**.

### 3. Core concepts

#### 3.1 Neuromodulator-inspired control (metaphor)

We borrow a few neuromodulator names as metaphors for control logic.  
These are **operational analogies only**, not biological claims.

- **GABA (inhibition & sparsity) – the primitive brake**  
  - Role: lateral inhibition and sparsification within a neighborhood.  
  - Trigger: “too many candidates” (active node count $> K_{\max}$, or high entropy in the activation distribution).  
  - Effect: sharpen the distribution via low-temperature softmax, apply Top-$k$ filtering,  
    and inhibit neighbors of the current winner node.

- **Dopamine (reward & reinforcement) – the seeker**  
  - Role: global reinforcement signal for insight discovery or contradiction detection.  
  - Trigger: positive $\Delta\mathrm{geDIG}$ above a threshold $\theta_{\mathrm{insight}}$, or strong contradictions.  
  - Effect: modulate plasticity (STDP-like updates) along the path that led to the outcome.

- **Acetylcholine (attention & state switching) – the modulator**  
  - Role: switch between encoding (wake) and consolidation (sleep).  
  - Trigger: high prediction error (AG open) for new inputs vs. low external input / high sleep pressure.  
  - Effect: during wake, boost sensory input weights; during sleep, boost recurrent weights for replay.

- **Cortisol (stress & adaptation) – the regulator**  
  - Role: resource management and “giving up” when stuck.  
  - Trigger: AG remains open for many cycles, or path cost consistently exceeds a budget.  
  - Effect: short term: increase exploration temperature; chronic: prune the branch and reduce learning rate.

#### 3.2 Signal propagation

- **Seeds**: recently used nodes, high-importance nodes, or sampled nodes for exploration.  
- **Propagation**: activation decays by a factor $\gamma$ at each hop and spreads over neighbors.  
- **Thresholding**: propagation stops when activation falls below a threshold $\theta$.  
- This yields patterns of high-activation nodes/edges, which feed into consolidation and shortcut discovery.

#### 3.3 Hebbian learning

We use a simple Hebbian rule

> $\Delta w_{ij} = \eta \cdot (s_i \cdot s_j)$

for edge weights $w_{ij}$, with learning rate $\eta$ and activations $s_i, s_j$,  
and clamp $w_{ij}$ to $[0,1]$. Phase 1 logs (e.g., contributions to $\F$, FMR events)  
can be injected as prior confidence on edges.

#### 3.4 Sleep cycle (NREM / REM)

We split Phase 2 into two modes:

1. **NREM (local consolidation and pruning)**  
   - Focus: strengthening strong, frequently used edges; pruning weak, old, or noisy edges.  
   - Effect: reduce redundancy and remove stale connections while preserving important local structure.

2. **REM (global exploration and shortcut creation)**  
   - Focus: exploring the graph via replay and adding long-range shortcuts when justified.  
   - Effect: run weaker-decay propagation, and create edges between highly co-activated but previously unconnected nodes,
     subject to constraints derived from Phase 1 ($\F$, FMR, PSZ behavior).

### 4. Architecture (conceptual)

At a high level:

- **`SleepController`**: orchestrates Phase 2 (triggers, seed selection, propagation, Hebbian updates, pruning/rewiring).  
- **`L2MemoryManager`**: maintains the persistent knowledge graph (e.g., NetworkX), and syncs it with transient Phase 1 structures.  
- **`SignalPropagator`**: handles activation propagation and decay.  
- **`EdgeAttributes`**: stores per-edge metadata (strength, last_activated, usage_count, contribution to $\F$, etc.).

### 5. Process flow (relative to Phase 1)

1. **Wake (Phase 1, online)**  
   - Queries and episodes are processed; AG/DG decisions and $\F$ trajectories are logged.  
   - The memory manager updates usage counts and timestamps for visited nodes and edges.

2. **Sleep (Phase 2, offline)**  
   - Trigger: explicit call (e.g., `agent.sleep()`) or policy (every $N$ episodes).  
   - Steps:  
     1. Select seeds (recent / important nodes).  
     2. Propagate signals (decaying activation).  
     3. Apply Hebbian updates to edge weights.  
     4. Rewire (REM): create shortcuts between highly co-activated nodes.  
     5. Prune (NREM): remove low-weight, long-unused edges.

### 6. Configuration sketch

An example configuration block (for illustration only) might look like:

```yaml
sleep:
  enable_auto_sleep: true
  interval_episodes: 100
  learning_rate: 0.1
  decay_factor: 0.8
  prune_threshold: 0.2
  rem_probability: 0.3
```

### 7. Future extensions

- **Dreaming**: generative replay of episodes based on the consolidated graph.  
- **Forgetting**: active forgetting of interfering or low-value memories.  
- **Multi-agent sleep**: synchronized sleep / consolidation across multiple agents.

