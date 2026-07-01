# v7 Tier 1 Action Plan — 今日までの整理を v7 に反映する計画

**最終更新**: 2026-04-17  
**前提**: [v7/plan.md](plan.md)（v7 全体計画、12 章 + Phase 0-3）  
**位置付け**: v7 plan は**実験ベースのロードマップ**。本メモは**今日までの研究整理（思想・理論・図版）を v7 に反映する計画**。両者は補完関係。

---

## 0. 背景

2026-04-17 のセッションで以下を整理:

1. [Part 1 コア理論統合ノート §1-9](../../research/gedig_core_theory_unified.md) — スカラー直接扱い宣言、Figure 1、三項独立性、β₁ 採用根拠、3 つの読み方、Helmholtz 正しい対応、Wake-Sleep-Wake、実験、棄却条件
2. [Part 2 認知・推論アーキテクチャ骨格](../../research/gedig_cognitive_architecture.md) — curl 階層的定義、AG/DG 神経基盤、TCL、自律的発見機
3. [Part 4 Transformer 統合骨格](../../research/gedig_transformer_architecture.md) — 動的学習 6 設計候補、β₁ 次元フリー性
4. [overview.md](../../research/overview.md) — 1 ページ外向け資料
5. **気づきメモ 5 本** — Bourbaki, β₁ 次元フリー, 三項直交性, 形態形成 (💭), Transformer 相転移 landscape, **連続・確率パラダイム批判**
6. **実験プロトコル 1 本** — [H_grokking-curl](../../research/thinking/experiment_grokking_curl.md)（🧪 最優先実験）
7. **ランディングページ更新** — Helmholtz Correspondence 刷新、OCR 数字追加、Zenodo DOI
8. **略称統一** — `graph edit Distance and Information Gain` (論文 v5/v6 と整合)

本メモは、これらが **v7 plan のどこに反映されるか** を Tier 分けで整理する。

---

## 1. Tier 分け

### Tier 1: 実験データ**不要**（即実行、1-2 日で完了）

論文の「顔」を変える作業。v7 plan の構造は維持したまま、思想と図版を強化する。

### Tier 2: 実験データ**必要**（v7 plan Phase 0-2）

[v7/plan.md](plan.md) の Phase 0-2 を実行。数週間単位。

### Tier 3: Future Work（実装コスト大、GPU 必要 or 時間必要）

論文の本体には入れず、§11 Limitations and Future Work で触れる。

---

## 2. Tier 1 の詳細（本メモの主内容）

### A. §1 Introduction 全面書き直し（4-6 時間）

**現状（v7 plan §1）**:
> §1 Introduction
> - F = ΔEPC - λ(ΔH + γΔβ₁) — β₁ への一般化
> - 「1つの方程式、3つのドメイン」
> - AG/DG = attention の閾値判定

**拡張案**:

```
§1.1 問題提起: 連続・確率パラダイムの限界
     - 現代 AI/物理/認知科学は構造情報を確率分布に押し込めている
     - 空間軸の押し込め: KL ダイバージェンスで Case A/B を区別できない
     - 時間軸の押し込め: FEP 予測誤差最小化は離散イベントを捉えない
     - 誘因: 微分と確率が人類とコンピュータにとって便利すぎた
     - 7 現象が見えなくなる（閃き、相転移、Grokking、概念獲得、等）

§1.2 本研究のアプローチ: スカラー直接扱い
     - 構造量を構造量のまま、単一スカラー F で制御
     - 3 項 (EPC / ΔH / Δβ₁) は現代数学の 3 基本空間（計量/測度/位相）の原子
     - [Figure 1: matchstick, Case A/B/C] — 三項の独立性と KL の盲点

§1.3 貢献の一覧
     - 1 方程式で 3 ドメイン (maze / RAG / Transformer)
     - AGHT (Analytical Heterogeneous Graph Transformer)
     - Transformer F-regularization (negative_better)
     - 統一コア実装 (src/gedig/, 71 tests)
     - （略称明示: geDIG = graph edit Distance and Information Gain）

§1.4 本稿の構成
```

**素材**:
- [insight_continuous_probabilistic_paradigm_critique.md](../../research/thinking/insight_continuous_probabilistic_paradigm_critique.md)
- [Part 1 §1 戦略宣言](../../research/gedig_core_theory_unified.md)
- [overview.md](../../research/overview.md)
- [insight_bourbaki_three_structures.md](../../research/thinking/insight_bourbaki_three_structures.md)
- [matchstick_figure_v2.html](../../research/thinking/matchstick_figure_v2.html)

### B. §2.1 F 定義直後に Figure 1 を挿入（3 時間、図版化含む）

**現状（v7 plan §2.1）**:
> Definition of F (v6 §1.1 を更新、SP → β₁)

**拡張**:

```
§2.1 Canonical Definition
     F = ΔEPC - λ(ΔH + γ·Δβ₁)

§2.1.1 Three Independent Atoms
     - ΔEPC: combinatorial (metric)
     - ΔH: information-theoretic (measure)
     - Δβ₁: algebraic-topological (topology)

§2.1.2 Visual Proof of Independence [Figure 1]
     - Case A/B/C: same EPC=1, different Δβ₁ ∈ {+1, 0, -1}
     - KL only sees ΔH → cannot distinguish A (insight) from B (routine)
     - geDIG distinguishes via Δβ₁
     - Necessity of all three terms (with explicit counterexample)
```

**作業**:
- matchstick HTML を PDF/PNG 化（TikZ で LaTeX 直書きに変換 or スクリーンショット）
- 既存の fig_scripts/ で再現コード作成

### C. §2.4 FEP-MDL Bridge → Helmholtz Correspondence に改称（1 時間）

**現状（v7 plan §2.4）**:
> FEP-MDL Bridge (v6 §9 から移動)

**変更後**:

```
§2.4 Helmholtz Correspondence

     - F の 3 つの読み方:
       1. EPC - (H + B) = cost - gain (canonical, economic)
       2. (EPC - B) - H = U - TS (Helmholtz-like, physical)
       3. (EPC - H) - B = internal_state - topological_order (biological)
     - 読み 2 が正しい Helmholtz 対応:
       U ↔ (ΔEPC - Δβ₁)  内部エネルギー
       T ↔ λ              情報温度
       S ↔ ΔH             エントロピー
     - FEP (variational free energy) との関係:
       - 操作的対応であり、literal な等価性ではない
       - geDIG は FEP より離散・微視的な操作を提供（粒度差）
     - 構造 ≡ 確率 の数学的厳密化は open problem (→ §11 Limitations)
     - λ を情報温度として動的制御する可能性 (simulated annealing)
```

**素材**:
- [Part 1 §6](../../research/gedig_core_theory_unified.md)
- [three_readings.md §4.3-4.4](../../research/thinking/gedig_formula_three_readings_20260306.md)

### D. §10 Related Work 拡充（2 時間）

**現状（v7 plan §10）**:
> §10.1 Graph Attention Networks (GAT, HGT, Graphormer)
> §10.2 Reasoning-intensive retrieval (BRIGHT, IRCoT)
> §10.3 Topological data analysis in NLP
> §10.4 Knowledge graph construction

**追加する新規 2026 研究**:

```
§10.3 Transformer × Phase Transition × Topology (新規セクション 追加)
     - Özönder 2025 "Attention to Order" (arXiv 2510.07401):
       attention entropy で 2D Ising 相転移を検出
       → geDIG は β₁ を加えることで Case A/B の区別を可能に
     - Sun & Haghighat 2025 "Transformer = O(N) model" (arXiv 2501.16241):
       O(N) 連続場として再定式化
       → geDIG は離散グラフとしての補完
     - TAG-DS 2025 (Betti-Fiedler partition as grokking proxy):
       β₁ が grokking の proxy として機能
       → geDIG は β₁ を free energy 項として統合、単なる proxy を超える
     - Grokking as Dimensional Phase Transition (arXiv 2604.04655):
       d_eff の相転移、Part 1 §5.7.2 (d_eff = β₁/V + 1) と整合
     - T3former, TopoFormer (2025):
       TDA + Transformer (feature engineering)
       → geDIG は自由エネルギー項として扱う点で異なる
```

[対比表](../../research/thinking/insight_transformer_phase_transition_landscape.md)を §10.3 に挿入。

### E. §11 Limitations and Future Work 拡充（1 時間）

**現状（v7 plan §11）**:
> §11.1 Statistical significance (single seed → 3+ seeds)
> §11.2 Scale (DistilBERT → GPT-2/LLaMA)
> §11.3 ARC Prize application
> §11.4 F-regularization at pre-training scale

**追加**:

```
§11.5 Open Theoretical Problems
     - 構造 ≡ 確率 の等価性の数学的厳密化
       (情報幾何 / MDL / MaxEnt 理論家に委ねる open problem)
     - 三項の直交性の厳密化
       (例示的独立 → 統計的独立 → 情報幾何的直交 の 3 段階)
     - 離散 curl の Hodge 分解との接続
     - β₁ の高次元スケーリング法則

§11.6 High-Priority Experiments (未実施)
     - H_grokking-curl: Nanda 2023 再現 + β₁ + curl(attention)
       → 既存 β₁ proxy (TAG-DS 2025) の拡張
       → curl (Part 2 §3.3 階層 3) の初工学検証
     - H_ising-bkt: Özönder 再現 + β₁ augmentation
     - Concept-Reuse Asymmetry Test (nominalization)
```

**素材**:
- [Part 1 §9.6 棄却条件表](../../research/gedig_core_theory_unified.md)
- [Part 1 付録 D](../../research/gedig_core_theory_unified.md)
- [experiment_grokking_curl.md](../../research/thinking/experiment_grokking_curl.md)
- [insight_three_terms_orthogonality.md](../../research/thinking/insight_three_terms_orthogonality.md)

### F. 略称統一（30 分）

v7 全文検索・置換:
- "Generalized Differential Information Gain" → "graph edit Distance and Information Gain"
- v5/v6 の論文はこの表記なので、**v7 を統一することで過去論文との整合性を回復**

**対象**:
- v7 plan.md（必要なら）
- v7 の新規 tex file（本文書き始める際）

---

## 3. Tier 1 実施順序（推奨）

所要時間ベースで効率順:

```
Day 1:
  09:00-09:30  F. 略称統一（全文確認、置換）
  09:30-10:30  C. §2.4 Helmholtz 改称（内容決まっているので素早い）
  10:30-12:30  D. §10 Related Work 拡充（素材は landscape メモ）
  13:30-14:30  E. §11 Limitations 拡充（素材は §9.6 + 付録 D）
  14:30-17:30  B. §2.1 Figure 1 挿入（TikZ or PNG 化）
               matchstick HTML → LaTeX TikZ 変換

Day 2:
  09:00-15:00  A. §1 Introduction 全面書き直し（最難関）
               - 連続・確率パラダイム批判を織り込む
               - Figure 1 forward reference
               - 3 層の思想的根拠の調和

                       Tier 1 完了
```

---

## 4. ターゲット会議別の Tier 対応

[v7/plan.md](plan.md) の 4 ターゲットと本 Tier の対応:

| ターゲット | 〆切 | 必要な Tier | 思想的厚み |
|---|---|---|---|
| **JSAI 2026** | 2026-02-18（既に過ぎている可能性） | Tier 1 の F のみ（短編なので） | 限定的 |
| **EMNLP 2026** | ~June 2026 | Tier 1 全部 + Phase 1 | 中 |
| **NeurIPS 2026** | ~May 2026 | Tier 1 全部 + Phase 1-2 | **高** |
| **ICLR 2027** | ~Oct 2026 | Tier 1 全部 + Phase 1-2-3 | **最高** |

**推奨**: Tier 1 を先に完了 → どのターゲットでも戦える状態を作る → その後 Phase 0-2 の実験着手 → ターゲット決定。

---

## 5. Tier 2: 実験データ必要（v7 plan Phase 0-2 と連動）

### Phase 0 (PREREQUISITE): β₁ への統一
- maze evaluator.py を SP → β₁ に置換
- transformer adapter の use_betti=True デフォルト化
- gedig_spec.md 更新
- maze 60 seeds 再現（paired SP vs β₁）

### Phase 1 (CRITICAL): Exp4 3-seed 再現
- T1: negative_better (β₁ ベース)、3 seeds
- 失敗時: v7 から Exp4 を外す

### Phase 2 (HIGH): 8 タスク並列
- T3: β sweep (4 点)
- T4: NLI 再現
- T5: BERT-base 再現
- R1: BRIGHT 3-seed
- R4: BRIGHT 他ドメイン
- H1+H2: HotpotQA 3-seed
- M1+M2: Maze β₁ 3-seed

詳細は [v7/plan.md](plan.md) 参照。

---

## 6. Tier 3: Future Work（論文本体外、§11 で言及）

- **H_grokking-curl** ([experiment_grokking_curl.md](../../research/thinking/experiment_grokking_curl.md))
  - Nanda 2023 再現 + β₁ + curl(attention)
  - 10 週間、GPU 必要
  - Part 4 §7 (Transformer 修正方針) の中核的検証
- **ARC Prize application**
- **F-regularization at pre-training scale** (GPT-2 pretrain + F regularizer)
- **Concept-Reuse Asymmetry Test** (nominalization、§11.6 で言及)
- **形態形成一般性** (💭 妄想、論文には入れない、個別エッセイで)

---

## 7. 本メモの使い方

- **論文書き直し時**: 本メモの A-F を順に実行
- **進捗管理**: 本メモに完了チェックボックスを入れて更新
- **変更要求時**: 本メモを更新してから作業
- **外部共有時**: overview.md と組み合わせて全体像を説明

---

## 8. 進捗トラッキング（2026-07-02 更新）

- [x] F. 略称統一 — draft_sections/ の全ドラフトで "graph edit Distance and Information Gain" に統一済み。**tex 起稿時に旧称の全文検索を 1 回実行すること**
- [x] C. §2.4 Helmholtz 改称 — **ドラフト完了**: [draft_sections/section2_4_helmholtz_draft.md](draft_sections/section2_4_helmholtz_draft.md)（β₁ 版の U 対応、3 読み方、canonical 固定。v6.1 Lemma 1/2 の扱いは著者判断待ち）
- [x] D. §10 Related Work 拡充 — **ドラフト完了**: [draft_sections/section10_11_additions_draft.md](draft_sections/section10_11_additions_draft.md)（⚠ arXiv 番号は投稿前に全件原典照合）
- [x] E. §11 Limitations 拡充 — **ドラフト完了**: 同上ファイル（§11.6 に maze sleep ablation prereg を追加 — tier1 計画にない 2026-07-02 の判断）
- [x] B. §2.1 Figure 1 挿入 — **図版化完了**: [draft_sections/figure1_matchstick.tex](draft_sections/figure1_matchstick.tex)（TikZ standalone、pdflatex ビルド確認済み、EN ラベル、色規則は poster/HTML と統一）。本文への挿入は v7 tex 起稿時
- [x] A. §1 Introduction 全面書き直し — **ドラフト完了**: [draft_sections/section1_introduction_draft.md](draft_sections/section1_introduction_draft.md)（監査後の誠実版貢献リスト。未決事項 3 点は編集メモ参照）

**注**: [x] は「著者レビュー待ちのドラフト完了」であり、論文反映済みという意味ではない。ドラフトはすべて英語本文 + 日本語編集メモの形式。貢献リストは 2026-06-10 監査と整合させ、negative_better は Phase 1 T1（3-seed 再現）が通るまで断定しない方針で書かれている。

Phase 0-2 は本メモのスコープ外（[v7/plan.md](plan.md) で管理）。

---

## 9. 関連リンク

### v7 本体
- [v7/plan.md](plan.md) — v7 全体計画（12 章 + Phase 0-3）
- [v7/restructured_draft.tex](restructured_draft.tex) — 2025-09 の草稿（v7 の種として参考）

### 今日の整理（本メモが v7 に反映する素材）
- [Part 1 コア理論統合](../../research/gedig_core_theory_unified.md)
- [Part 2 認知骨格](../../research/gedig_cognitive_architecture.md)
- [Part 4 Transformer 骨格](../../research/gedig_transformer_architecture.md)
- [overview.md](../../research/overview.md) — 1 ページ外向け資料
- [insight_continuous_probabilistic_paradigm_critique.md](../../research/thinking/insight_continuous_probabilistic_paradigm_critique.md) — §1 書き直しの思想的核
- [matchstick_figure_v2.html](../../research/thinking/matchstick_figure_v2.html) — Figure 1 の元データ
- [insight_transformer_phase_transition_landscape.md](../../research/thinking/insight_transformer_phase_transition_landscape.md) — §10.3 の素材
- [experiment_grokking_curl.md](../../research/thinking/experiment_grokking_curl.md) — §11.6 の素材

### 前版
- [v6/geDIG_onegauge_improved_v6.tex](../v6/geDIG_onegauge_improved_v6.tex) — 現行版（Zenodo DOI 固定）
- [v6/arxiv_en/](../v6/arxiv_en/) — 英語版 arxiv
