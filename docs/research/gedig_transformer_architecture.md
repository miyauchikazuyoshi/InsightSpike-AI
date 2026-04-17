# geDIG Transformer 統合アーキテクチャ（統合版）

**最終更新**: 2026-04-17  
**ステータス**: 骨格作成済。本文統合は順次。  
**Part**: 4（Transformer 統合）  
**前提**: [Part 1 コア理論](gedig_core_theory_unified.md) §7-9、[Part 2 認知・推論](gedig_cognitive_architecture.md) §3, §6  
**正準参照**: `docs/gedig_spec.md`

> **この文書の位置づけ**: Part 1 で確立したスカラー直接扱い戦略と三項分解を、
> **現代 AI の中核である Transformer**にどう適用するかを論じる統合ノート。
> Part 1 §8.6 で記録した negative result（`delta_r2_struct` 全モデル負値）への対処方針も含む。
> 素材は `geDIG_transformer_discussion_20260416.md` / `dynamic_transformer_spec.md` /
> `splatting_attention_duality_for_gedig.md` / `insightspike_as_gnn_transformer.md` の 4 本。

---

## 目次

1. [はじめに — Part 1-2 の帰結としての Transformer 適用](#1-はじめに--part-1-2-の帰結としての-transformer-適用)
2. [Transformer を geDIG レンズで解釈する](#2-transformer-を-gedig-レンズで解釈する)
3. [動的学習 Transformer の設計候補（6 パターン）](#3-動的学習-transformer-の設計候補6-パターン)
4. [β₁ 次元フリー性の活用 — Transformer 統合の鍵](#4-β₁-次元フリー性の活用--transformer-統合の鍵)
5. [GNN-Transformer 統合](#5-gnn-transformer-統合)
6. [Hallucination と Nominalization の再解釈](#6-hallucination-と-nominalization-の再解釈)
7. [実験計画と現状 — Part 1 §8.6 negative result への対処](#7-実験計画と現状--part-1-86-negative-result-への対処)
8. [未解決問題と棄却条件](#8-未解決問題と棄却条件)

付録:
- A. [統合前の素材ファイル対応表](#付録-a-統合前の素材ファイル対応表)
- B. [命名・用語の決めごと](#付録-b-命名用語の決めごと)
- C. [Part 1 / Part 2 への接続点](#付録-c-part-1--part-2-への接続点)

---

## 1. はじめに — Part 1-2 の帰結としての Transformer 適用

> **3行要約**:  
> Part 1 で確立した**スカラー直接扱い**戦略と**三項分解**は、現代 AI の中核である Transformer に適用可能か。  
> Part 1 §8.6 で記録した **negative result**（`delta_r2_struct` 全モデル負）の解釈と修正方針を中心に議論する。  
> 本 Part の主張は「**Transformer に動的学習を導入する**」ことだが、現段階では**概念的スケッチと設計候補**に留まる。工学的実証は今後の課題。

### 1.1 Part 1-2 からの論理的接続

Part 1 で確立した要素:
- スカラー F による構造制御（§1-7）
- β₁ の次元フリー性（[insight_beta1_dimension_free.md](thinking/insight_beta1_dimension_free.md)）
- Transformer 実験の negative result（§8.6）と解釈候補（§9.5）

Part 2 で確立した要素:
- curl 検出の階層的定義（§3）、特に階層 3: attention flow の解析（§3.3）
- 自律的発見機 = LLM + curl プラグイン（§6.3）

本 Part 4 は、これらを**工学的実装の観点で深掘り**する。

### 1.2 本 Part の位置づけと範囲

**範囲内**:
- Transformer を geDIG レンズで解釈する方法
- 動的学習 Transformer の設計候補（6 パターン）
- Part 1 §8.6 の negative result への対処方針

**範囲外**:
- 具体的な実装コード（`experiments/transformer/` を参照）
- 大規模モデル検証の結果（今後の実験に依存）
- 「動的学習 Transformer」の完全な工学的実現（長期目標）

### 1.3 重要な留保

現状で Transformer 統合は **「概念的スケッチ + 設計候補」**レベルで、
**「十分検証された仮説」ではない**。Part 1 §9.5 で記録した通り、`delta_r2_struct` が全モデルで負値であることは、
現時点での geDIG の Transformer 適用が**仮説段階**であることを正直に示している。

本 Part はこの現状を踏まえ、**修正方針**と**設計候補**を明示することを目的とする。

**統合素材**:
- Part 1 §7-9（特に §8.6 negative result、§9.5 解釈候補）
- Part 2 §3, §6（curl 検出、自律的発見機）
- `geDIG_transformer_discussion_20260416.md` §全体

---

## 2. Transformer を geDIG レンズで解釈する

> **3行要約**:  
> Transformer の各層は、geDIG の F 動態として解釈できる。Attention graph を知識グラフとして扱い、層別の F 値変化を観測する。  
> Transformer Circuits (Elhage, Olsson) の発見と整合する対応が観察される（attention = edge、FFN = node update）。  
> Splatting (where) と Attention (what) の双対性が、geDIG の二段ゲート（AG/DG）と構造的に一致する。

### 2.1 Transformer Circuits との対応

Anthropic の Transformer Circuits 研究（Elhage, Olsson et al.）との対応表:

| Transformer Circuits | geDIG レンズ | 意味 |
|---|---|---|
| Attention head | エッジ候補生成器 | 構造的な connectivity の提案 |
| Attention weight | エッジの重み | F の EPC 項に寄与 |
| FFN (Feed-Forward) | ノード更新 | 情報統合（ΔH に寄与） |
| Residual stream | 累積的な構造情報 | 層間の F の継承 |
| Skip connection | ΔEPC の保存 | 構造変化の保持 |

これは `geDIG_transformer_discussion_20260416.md` §4 の対応表を基にしている。

### 2.2 attention graph の構築

Transformer の attention weight `A ∈ ℝ^{T×T}` を**閾値化**してグラフ化する:

```
edge(i, j) ⟺ A[i,j] > threshold
```

閾値選択の選択肢:
- **Percentile-based**: 上位 `p%` を edge に
- **Absolute threshold**: `A[i,j] > θ_abs`
- **k-NN**: 各トークンについて上位 `k` 個を edge に

Part 1 §8.6 で記録した通り、閾値設計は**結果に敏感**で、ベースラインとの順位が入れ替わるケースがある。

### 2.3 層別 F 動態の観察

各層 `l` で F 値を計算:
```
F(l) = ΔEPC(l) - λ(ΔH(l) + γ·Δβ₁(l))
```

観察すべき指標:
- **F の層別軌跡**: 深い層で F が減少するか
- **AG/DG の発火頻度**: 層位置との相関
- **β₁ の層別値**: 構造複雑度の階層性

Part 1 §8.6 では `delta_r2_struct` を指標としたが、β₁ 直接指標に切り替える方針（§7.2）。

### 2.4 Splatting (where) / Attention (what) の双対性

`splatting_attention_duality_for_gedig.md` より:

- **Splatting (where)**: 位置情報の分散（spatial attention）
- **Attention (what)**: 内容情報の選択（content attention）

この 2 つは geDIG の二段ゲートと自然に対応:
- **where** ≒ **AG**（どこが曖昧か → 探索開始）
- **what** ≒ **DG**（何が確定したか → 統合確定）

つまり Transformer 内部にも**二段構造**が暗黙的に存在しており、geDIG はそれを**明示化**する枠組みとして機能する。

**統合素材**:
- `geDIG_transformer_discussion_20260416.md` §4（Transformer Circuits 対応表）
- `splatting_attention_duality_for_gedig.md`（where/what 双対）
- Part 2 §3.3（curl の attention flow 解析）

---

## 3. 動的学習 Transformer の設計候補（6 パターン）

> **3行要約**:  
> 「動的学習 Transformer」を実現する 6 つの設計候補を、**実装容易度と影響度**で並べる。  
> **案① Hallucination Detector** は最小コストの PoC（1-2 週間）、**案⑤ Wake-Sleep Transformer** が最も novel。  
> **案⑥ Concept-Addition Layer** は nominalization 問題への究極解だが、現時点では**封印**（crank リスク回避）。

### 3.1 案① Hallucination Detector（難易度★☆☆、1-2 週間）

**設計**:
- 推論時に F 値をモニタ
- AG ゲート不在 = hallucination 候補として検出
- 訓練不要、既存 LLM に後段として挿入可能

**利点**:
- β₁ 非微分性問題を回避（gate 判定のみ）
- 既存実装資産で即検証可能
- 1-2 週間で PoC

**欠点**:
- 「検出」止まりで、根本解決にはならない

**差別化**: LLM-as-judge / SelfCheckGPT より**構造的根拠**を持つ unfamiliarity 指標。

### 3.2 案② geDIG-Regularized Training（難易度★★☆、1-2 ヶ月）

**設計**:
- 通常の loss に `λ·ΔH + γ·Δβ₁` を正則化として加算
- ΔH は微分可能、Δβ₁ は differentiable persistent homology (Gudhi / TopologyLayer) で代替

**利点**:
- Information Bottleneck の位相版として位置付けられる
- 訓練過程に組み込める

**欠点**:
- differentiable PH は計算コストが重い
- 小規模モデルから段階的検証が必要

### 3.3 案③ Adaptive Computation Time (ACT) 版

**設計**:
- F > θ_DG なら次の層へ、F < θ_AG なら early exit
- Graves (2016) の ACT の geDIG 版

**利点**:
- PyTorch の halting 機構に乗る
- 計算量が入力の難しさに比例（Part 1 §5.7.3）

**欠点**:
- halting の学習が不安定になりやすい
- 閾値調整が必要

### 3.4 案④ MoE / Dynamic Routing with geDIG Gate

**設計**:
- AG: router entropy 高 → expert 動員数増加
- DG: F 値で expert 統合の閾値判定
- 既存 MoE 基盤（Mixtral, Switch Transformer）に乗せる

**利点**:
- 「動的学習 Transformer」に最も近い
- Titans (Google 2024) との**差別化**: memory-as-context と直交する制御軸（expert routing）

**欠点**:
- 計算資源が必要
- 大規模モデルでの検証コスト

### 3.5 案⑤ Wake-Sleep Transformer（難易度★★★★、4-6 ヶ月）

**設計**:
- **Wake**: 通常推論 + attention graph の蓄積
- **Sleep**: graph rewire + propagation（`graph_persistent_dg/sleep_propagate.py` の拡張）
- **Wake'**: rewired memory で推論

**利点**:
- 迷路実装の資産がそのまま活きる
- **geDIG 独自性が最も強い**設計
- Titans / RWKV と明確に違う「**offline consolidation**」アーキテクチャ

**欠点**:
- Sleep 時の embedding 更新が通常の backprop に乗らない（カスタム学習ループ必須）
- 実装コストが最大

**これが本命**。既存の動的学習 Transformer 研究と**最も差別化できる**設計。

### 3.6 案⑥ Concept-Addition Layer（難易度★★★★★、封印）

**設計**:
- Nominalization 問題の解決案
- DG 発火時に新 token embedding を追加し、語彙を動的に拡張

**利点**:
- LLM の固定語彙制約を破る
- 人間の語彙獲得に最も近い

**欠点**:
- Continual learning の未解決問題
- 現時点では**実装的に非現実的**

**判断**: 本 Part では**封印**。「可能性を示唆」レベルに留め、踏み込むと crank 側に寄る。

### 3.7 実装容易度と優先順

| 案 | PoC 可否 | 論文化難易度 | インパクト | 本研究での優先度 |
|---|---|---|---|---|
| ① Hallucination Detector | **即可** | Workshop | 中 | **最初に着手** |
| ② Regularization | 可 | Conference 短編 | 中 | 第 2 候補 |
| ③ ACT 版 | 可 | Conference | 中 | 余力があれば |
| ④ MoE gate | 可（要計算資源） | Conference | **高** | 本命候補 B |
| ⑤ Wake-Sleep | 困難 | Conference | **高** | **本命候補 A** |
| ⑥ Concept-Addition | 非現実的 | Top venue | 極大 | **封印** |

**推奨戦略**:
- ① を 1-2 週間でやって「存在証明」を出す
- ④ か ⑤ のどちらかを本命として 6 ヶ月で実装

**統合素材**:
- `dynamic_transformer_spec.md`（動的再構成仕様、案②③④）
- `insightspike_as_gnn_transformer.md`（GNN-Transformer、案⑤の基盤）
- Part 1 §9.2（β₁ 非微分性の C 案 Wake/Sleep 分離）

---

## 4. β₁ 次元フリー性の活用 — Transformer 統合の鍵

> **3行要約**:  
> Part 1 §8.6 の negative result の主因は、**structural probe 依存の SP 指標が高次元 Transformer で不安定**な可能性。  
> β₁ は**次元フリー**（[insight_beta1_dimension_free.md](thinking/insight_beta1_dimension_free.md)）で、768-4096 次元の attention graph でも O(V+E) で計算可能。  
> β₁ 直接指標への切り替えが Transformer 統合の**最優先修正事項**。

### 4.1 高次元空間の curse of dimensionality 回避

Transformer の hidden state は典型的に 768-4096 次元。この高次元で構造を捉える手段:

- **Cosine similarity**: 高次元で discriminative power が落ちる
- **Structural probe (Hewitt & Manning)**: 低次元射影に依存、probe 学習の不安定性
- **Attention weight**: 確率分布化されており、位相情報が落ちる
- **β₁**: **次元に依存しない**、O(V+E) で計算可能

### 4.2 structural probe 依存の問題

Part 1 §8.6 の実装は structural probe に依存していた（`metrics.py` の SP 項）:
```
SP(l) = spearman_corr(depth_vectors[l], depth_vectors[l-1])
```

問題:
- depth vector が高次元で不安定
- probe 学習に追加の教師信号が必要（geDIG の self-contained 主張と矛盾）
- `delta_r2_struct` が全モデルで負値になる原因

### 4.3 β₁ 直接指標への切り替え

`betti1_engineering_spec.md` Part B の仕様に従い、attention graph から直接 β₁ を計算:

```python
def _betti_1_from_distance_matrix(dist_mat, threshold=None, k_neighbors=0):
    # 距離行列を閾値化 or k-NN でグラフ化
    # β₁ = E - V + C を計算
    return E - V + C
```

実装は `experiments/transformer/` 配下。`k=5` の k-NN グラフがデフォルト推奨。

### 4.4 検証可能な仮説 H_dim

> **H_dim**: β₁ 直接指標は hidden dim を変えても安定した結果を返す。
> structural probe 依存の SP 指標より**次元呪いの影響を受けない**。

実験:
- 同一モデル・異なる dim projection（64, 128, 256, 512, 768）
- β₁ の値と変動係数（CV）を測定
- cosine similarity、structural probe と比較

**成功条件**: β₁ の CV < 0.1（次元によらず安定）。

### 4.5 Part 1 §9.5 との接続

Part 1 §9.5 で Transformer negative result の**解釈候補 4 つ**を示した:
1. 指標の問題（structural probe 依存）
2. モデル規模の問題
3. 理論の問題
4. 閾値設計の問題

本節 §4 は **解釈 1 への対処**を具体化する。解釈 2-4 は §7 で継続検討。

**統合素材**:
- [thinking/insight_beta1_dimension_free.md](thinking/insight_beta1_dimension_free.md)（β₁ 次元フリー性の詳細）
- `thinking/betti1_engineering_spec.md` Part B（Transformer 向け β₁ 実装仕様）
- Part 1 §8.6, §9.5（negative result と解釈）

---

## 5. GNN-Transformer 統合

> **3行要約**:  
> Transformer の attention は、GNN の message passing と数学的に等価である。  
> geDIG レンズでこれを統一すると、**スパース接続による効率化**と**構造的解釈可能性**の両立が見える。  
> これは Part 1 §5.6.1 NN 剪定理論（Lottery Ticket）の Transformer 拡張として位置付けられる。

### 5.1 attention = message passing の等価性

`insightspike_as_gnn_transformer.md` より:

- Attention: `out[i] = Σ_j A[i,j] · V[j]`
- GNN message passing: `out[i] = Σ_{j ∈ N(i)} msg(V[j])`

**両者は同じ計算**（A[i,j] = 1 iff j ∈ N(i) の場合）。Transformer は**全結合 GNN** と見なせる。

### 5.2 スパース接続による効率化

全結合（β₁ 最大）の Transformer は冗長。Lottery Ticket Hypothesis (Part 1 §5.6.1) の含意:
- β₁ を最小化する部分グラフ = 効率的な attention pattern
- 剪定後: 計算量 O(T²) → O(T·k)（k = 平均次数）

**geDIG-guided pruning**:
1. 学習中に F 値を監視
2. F > 0 が続く attention edge を剪定
3. β₁ が最小の部分グラフに収束

### 5.3 構造的解釈可能性

スパースな attention graph は**人間が読める**:
- どのトークン間に edge があるか
- どこに β₁ のサイクルが生じているか
- AG/DG がどこで発火したか

これは mechanistic interpretability（Transformer Circuits）の延長線上。

### 5.4 案④・案⑤ との接続

- **案④ MoE gate** (§3.4): expert の動的選択にスパース化を組み合わせる
- **案⑤ Wake-Sleep** (§3.5): Sleep 相で大規模 attention graph を剪定

**統合素材**:
- `insightspike_as_gnn_transformer.md`（全体）
- Part 1 §5.6.1（Lottery Ticket Hypothesis）

---

## 6. Hallucination と Nominalization の再解釈

> **3行要約**:  
> **Hallucination = AG ゲート不在**の現象として再解釈できる。geDIG は AG 発火の明示化により、hallucination を**検出可能**にする。  
> **Nominalization 問題**: LLM は訓練後に新トークン（概念）を生成できない。geDIG の AND 蒸留（Part 2 §2.2）はこの限界への**工学的回答**候補。  
> **Concept-Reuse Asymmetry Test** ([Part 2 §7.4](gedig_cognitive_architecture.md)) が棄却条件の中心。

### 6.1 Hallucination = AG ゲート不在

**観察**: LLM が hallucinate する時、モデル自身は**確信を持って出力している**。

geDIG 的解釈:
- AG（曖昧性検知）が発火していない
- モデルは「知らない」を検知できない
- 出力の信頼度が構造的に担保されない

**geDIG の答え**:
- F 値を推論時にモニタ（案① Hallucination Detector）
- AG 発火がないのに出力が確定的 = hallucination 候補

これは`geDIG_transformer_discussion_20260416.md` §3.3 の主張と整合。

### 6.2 Nominalization 問題

`geDIG_transformer_discussion_20260416.md` §9.2 より:

> LLM は訓練後に**新トークン（概念）を生成できない**。
> 人間はエピソードから新語彙を動的に立ち上げる。

geDIG の Part 2 §2.2 AND 蒸留は、**この限界への工学的回答**として設計されている:
- Sleep 相で複数エピソードの共通項抽出
- 新概念が emergent に立ち上がる
- これが案⑥ Concept-Addition Layer の理論的基盤

ただし案⑥ は **封印**（§3.6）。nominalization 問題は**理論的には解ける**が、**工学的には時期尚早**。

### 6.3 Concept-Reuse Asymmetry Test（棄却条件の中心）

Part 2 §7.4 で記録した通り、nominalization 主張の**棄却条件**:

- LLM と人間被験者（n=10）で Task B/C の性能差を測定
- **Task B**: 既存語彙の再利用
- **Task C**: 新語彙の動的生成

**予想**:
- LLM: Task B は可能、Task C は困難
- 人間: Task B, C ともに可能

もしこの非対称が観測されれば:
- LLM の nominalization 非対応性が実証
- geDIG の AND 蒸留の必要性が裏付けられる

もし非対称が観測されなければ:
- LLM は既に暗黙的に nominalization している
- geDIG の主張の独自性が弱まる

**実施時期**: 2026 年下半期予定。

### 6.4 Part 2 §7.4 との関係

Part 2 §7.4 で nominalization 問題を open problem として記録した。
本 Part 4 §6 は、それを **Transformer 統合の観点で工学的に位置付け**直す。

**統合素材**:
- `geDIG_transformer_discussion_20260416.md` §3.3（hallucination）, §9.2（nominalization）
- Part 2 §2.2 (AND 蒸留), §7.4 (Concept-Reuse Test)
- Part 1 §9.6（棄却条件の一部として）

---

## 7. 実験計画と現状 — Part 1 §8.6 negative result への対処

> **3行要約**:  
> Part 1 §8.6 の negative result（`delta_r2_struct` 全モデル負値）は、**指標選択の問題**である可能性が高い。  
> 修正方針の優先順: (1) β₁ 直接指標への切替（§4）、(2) 大規模モデル検証、(3) 閾値設計の sensitivity analysis。  
> 正直な報告として、**Transformer 統合は「仮説段階」**であることを明示する。

### 7.1 現状の negative result（Part 1 §8.6 再掲）

| メトリクス | distilgpt2 | gpt2 | gpt2-medium |
|---|---|---|---|
| `delta_r2_struct` | **-0.73** | **-0.18** | **-0.16** |

すべて負値。**構造指標が他の指標より多くを説明する**という期待と逆方向。

### 7.2 修正方針（優先順）

**優先度 1**: β₁ 直接指標への切り替え（§4）
- 期間: 2-4 週間
- 成功基準: `delta_r2_struct` が正に反転する
- 失敗時: 他の修正方針へ

**優先度 2**: 大規模モデル検証
- 期間: 2-3 ヶ月（GPU 資源確保次第）
- 対象: Llama 8B, 70B
- 成功基準: 大規模モデルで同一 (λ, γ) が機能する
- 失敗時: 「小規模モデルでは構造情報が希薄」仮説が支持される

**優先度 3**: 閾値設計の sensitivity analysis
- 期間: 1 ヶ月
- 成功基準: 閾値選択にロバストな結果

**優先度 4**: partial positive の拡張
- DistilBERT/SST-2 の +0.33pt を他タスクで再現
- F 正則化の最適 α の一般化

### 7.3 大規模モデル検証計画

- **モデル**: Llama 3.1 8B、Llama 3.3 70B（または同等スケール）
- **タスク**: HotpotQA, MMLU, 構造依存タスク
- **指標**: β₁ 直接指標（§4.3）
- **必要資源**: H100 8 枚 × 2 ヶ月
- **調達先**: クラウド（AWS, GCP）または共同研究機関

### 7.4 失敗条件の明文化

以下が観測されたら **Transformer 統合は棚上げ**:

1. β₁ 直接指標でも `delta_r2_struct` が負のまま
2. 大規模モデル（Llama 70B）でも同じ
3. 3 つ以上の独立タスクで partial positive が再現できない

この場合、geDIG は **Transformer 以外のドメイン**（迷路、RAG、専門システム）に集中する。

**統合素材**:
- Part 1 §8.6, §9.5（現状の negative result）
- `experiments/transformer/inference_gedig_v3_layerwise_redefine/`（現行実装）
- `betti1_engineering_spec.md` Part B（β₁ 切替仕様）

---

## 8. 未解決問題と棄却条件

> **3行要約**:  
> Part 4 の主張は **「動的学習 Transformer は可能性として設計できる」** レベルで、Part 1 の工学的実証より**仮説度が高い**。  
> 主な open problem: β₁ 非微分性、計算資源、既存研究（Titans, RWKV 等）との差別化実証。  
> 棄却条件を Part 1 §9.6 と同じスタイルで明文化する。

### 8.1 load-bearing な仮定

**T1**: Transformer の attention graph は geDIG が想定する「知識グラフ」と互換
- 検証状況: 部分的（Transformer Circuits の対応は観察済）
- 崩れたら: attention graph の別解釈が必要

**T2**: β₁ 直接指標が Transformer で安定した結果を返す
- 検証状況: 未実施（§7.2 優先度 1）
- 崩れたら: 他の次元フリー位相指標を検討

**T3**: 大規模モデルで同一 (λ, γ) が機能する
- 検証状況: 未実施（§7.3）
- 崩れたら: スケール依存の正則化が必要

**T4**: 動的学習 Transformer の設計候補（§3）のどれか 1 つは工学的に実装可能
- 検証状況: 案①は即実装可能
- 崩れたら: geDIG は推論時モニタ止まり

### 8.2 β₁ 非微分性の Transformer 特有の課題

Part 1 §9.2 で一般的な対処案（A/B/C 案）を示したが、Transformer 特有の課題:

- 学習中の β₁ の監視 = 勾配が流れない項
- Wake / Sleep 分離（C 案）が Transformer にも適用可能か
- differentiable PH の計算コストが 768 次元で現実的か

**推奨アプローチ**: 案①②③は β₁ を gate 判定のみに使う（A 案）、案④⑤は Wake/Sleep 分離（C 案）。

### 8.3 既存研究との差別化

動的学習 Transformer は 2024-2025 の超ホット領域。本 Part は既存研究への言及を強化すべき:

| 競合研究 | 本研究との差別化 |
|---|---|
| Titans (Google 2024) | memory-as-context vs **構造的制御軸**（AG/DG） |
| Test-Time Training (Sun+ 2024) | パラメータ更新 vs **構造更新** |
| RWKV, Mamba | state space vs **グラフ位相制御** |
| Hypernetworks, MAML | meta-learning vs **Wake-Sleep consolidation** |
| Adaptive Computation Time (Graves 2016) | halting probability vs **F 符号判定** |

これは Part 1 §2.3 の対比表の Transformer 版。**論文 §2 Related Work の下書き**に使える。

### 8.4 棄却条件の明文化

| クレーム | 棄却条件 |
|---|---|
| **Transformer への適用可能性**（§2） | β₁ 直接指標 + 大規模モデルでも delta_r2 負のまま |
| **動的学習 Transformer の実現可能性**（§3） | 3 年以内に案①-⑤ のどれも動作 PoC が出ない |
| **Hallucination Detector の有効性**（§3.1） | 3 タスク以上で baseline を超えられない |
| **Wake-Sleep Transformer**（§3.5）| 迷路での有効性が Transformer に転用できない |
| **β₁ 次元フリー性の Transformer 適用**（§4） | 次元を変えると β₁ 結果が大きく変動（CV > 0.3） |
| **Nominalization 非対応性**（§6.2） | Concept-Reuse Asymmetry Test で LLM が Task C も可能 |

### 8.5 Part 1 §9 / Part 2 §7 との関係

- Part 1 §9: 工学的・定量的な棄却条件
- Part 2 §7: 認識論的・仮説的な棄却条件
- **Part 4 §8**: Transformer 統合特有の工学的棄却条件

三者は補完関係にある。外部レビュアーによる批判的レビューが、特に Part 4 の主張には必須。

**統合素材**:
- `geDIG_transformer_discussion_20260416.md` §9（critical self-review）
- Part 1 §9（棄却可能性のスタイル）
- Part 2 §7（nominalization, Concept-Reuse Test）

---

## 付録 A: 統合前の素材ファイル対応表

| 旧ファイル | 本ノートの対応節 | 統合後の扱い |
|---|---|---|
| `geDIG_transformer_discussion_20260416.md` | §2, §6, §8 | **維持**（§9 critical review が現役、長期参照） |
| `dynamic_transformer_spec.md` | §3 | 統合後 `_archive/` へ退避候補 |
| `splatting_attention_duality_for_gedig.md` | §2.4 | 統合後 `_archive/` へ退避候補 |
| `insightspike_as_gnn_transformer.md` | §5 | 統合後 `_archive/` へ退避候補 |

退避判断は本文統合が完了してから。

---

## 付録 B: 命名・用語の決めごと

Part 1 付録 C / Part 2 付録 B の規則を継承しつつ、Part 4 固有の用語:

| 用語 | 採用する表記 | 避ける表記 |
|---|---|---|
| 動的学習 Transformer | 「動的学習 Transformer」または "dynamic-learning Transformer" | 「動的 Transformer」の揺れを避ける |
| attention graph | 「attention graph」 | 「アテンショングラフ」の揺れを避ける |
| Wake-Sleep Transformer | 「Wake-Sleep Transformer」（Part 1 の Wake-Sleep-Wake と区別） | 「WSW Transformer」等の略称は使わない |
| 案①〜案⑥ | 「案 X」（§3 の番号付け） | 「パターン」「アプローチ」等の揺れを避ける |

---

## 付録 C: Part 1 / Part 2 への接続点

### Part 1 → Part 4

- Part 1 §5（β₁ 採用根拠）→ Part 4 §4（次元フリー性の活用）
- Part 1 §7.5（帰納 vs 演繹）→ Part 4 §3（動的学習 Transformer 設計）
- Part 1 §8.6（Transformer negative result）→ Part 4 §7（対処方針）
- Part 1 §9.5（解釈候補）→ Part 4 §4, §7（具体化）

### Part 2 → Part 4

- Part 2 §3.3（curl の attention flow 解析）→ Part 4 §2.4（where/what 双対）
- Part 2 §6.3（LLM への curl プラグイン）→ Part 4 §3.1（案① Hallucination Detector）
- Part 2 §2.2（AND 蒸留）→ Part 4 §3.6（案⑥ Concept-Addition、封印）
- Part 2 §7.4（Concept-Reuse Test）→ Part 4 §6.3（棄却条件の中心）

### Part 4 → 他 Part（将来）

- Part 5 応用・実装: 案①-⑤ の具体的実装
- Part 7 自己批判: 外部レビュアーによる Transformer 統合の批判的レビュー
