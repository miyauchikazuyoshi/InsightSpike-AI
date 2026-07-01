# v7 §1 Introduction — 全面書き直しドラフト

> **作成**: 2026-07-02（Tier 1 作業 A、[tier1_action_plan.md](../tier1_action_plan.md) §2.A）
> **状態**: DRAFT — 著者レビュー待ち。英語本文 + 日本語編集メモ。
> **素材**: [insight_continuous_probabilistic_paradigm_critique.md](../../../research/thinking/insight_continuous_probabilistic_paradigm_critique.md)（思想的核）、[gedig_core_theory_unified.md](../../../research/gedig_core_theory_unified.md) §1–§2、[insight_bourbaki_three_structures.md](../../../research/thinking/insight_bourbaki_three_structures.md)、v6.1 EN §1
> **監査整合**: 貢献リストは 2026-06-10 監査（f_sign / PER / oracle）後の誠実版。negative_better は断定しない（Phase 1 T1 の 3-seed 再現が通るまで）。

---

## 編集メモ（日本語 — 本文には入れない）

1. **v6.1 からの最大の変更**: v6.1 §1 は「動的 KG に受容規範がない」という工学的問題提起から始まる（9 行、约 400 語）。v7 では、その手前に「なぜ既存手法では規範が作れないのか＝構造情報の確率への押し込め」というパラダイム批判の層を敷く。工学的問題提起（when to accept）は §1.1 の後半に保持する。
2. **crank 回避の 3 点**（素材メモ §6 の指示に従う）: (i) 補完であって代替ではないと明言、(ii) スケールによる住み分け、(iii) 反証可能な差分（Figure 1 + 事前登録）。すべて §1.1 末尾の Scope and positioning 段落に入れた。
3. **「微分と確率が便利すぎた」の思想層**は本文に出さない（素材メモ §8 の留保どおり）。§1.1 の脚注 1 に一言だけ置き、詳細は discussion/appendix 行き。
4. **時間軸批判（FEP の粒度）**は §1.1 で 1 段落。工学的裏付けは AG/DG の離散発火（§1.2）に接続。
5. **略称統一**（Tier 1 作業 F）: geDIG = "graph edit Distance and Information Gain"（v5/v6 と整合）。本ドラフトで統一済み。
6. **未決事項（著者判断待ち）**:
   - (a) Figure 1 の ΔH の具体値（+0.4 / +0.3 / −0.2）を本文に書くか、図のキャプションに落とすか
   - (b) 貢献リストに maze stage-2（sleep）を入れるか — sleep ablation（[docs/prereg/maze_sleep_ablation.md](../../../prereg/maze_sleep_ablation.md)）の結果が出るまで「pre-registered, in progress」の一行に留めるのが誠実
   - (c) §1.3 の BRIGHT の数値は biology 単一ドメイン 0.439 か、フル 3 ドメイン 0.19 を先に出すか（README は両方併記の誠実スタイル — 論文も踏襲を推奨）

---

## §1 Introduction (English draft)

### §1.1 The Problem: Structural Information Squeezed into Probability

Two observations motivate this work — one spatial, one temporal.

**The spatial squeeze.** When contemporary AI and ML methods handle graph structure or topological information, they almost invariably route it through a probability-distribution representation. FEP / Active Inference maps structure onto probabilistic models over Markov blankets; VAEs map it into latent distributions; GNN message passing aggregates neighborhoods into expectations; the Information Bottleneck reduces structure to a mutual information; graphical models encode it as conditional-independence relations. The conversion buys real computational advantages — differentiability, Bayesian inference — but it drops information that is specifically structural, above all topology.

Figure 1 makes the loss concrete. Three graph edits with *identical* edit cost (EPC = 1) diverge completely in topology: closing a triangle (Δβ₁ = +1), extending a path (Δβ₁ = 0), and deleting a cycle edge (Δβ₁ = −1). Their entropy changes are of the same order, so any criterion built on distribution change alone — KL divergence, ELBO, prediction error, mutual information — cannot distinguish the insight-like Case A (a new independent cycle: two previously separate paths now confirm each other) from the routine Case B (one more edge on a growing path). A topological invariant separates them immediately. Section 2 tabulates seven representative methods against this test; the six probability-side methods all fail it.

**The temporal squeeze.** The same compression happens on the time axis. Prediction-error minimization describes learning as a statistically converging average. But insight, grokking, and concept acquisition are *discrete events*: something happens at a specific moment — a sudden generalization at one epoch, a new concept that emerges rather than accretes. A framework whose primitive is an expectation over trials cannot represent the timestamp of a single structural commitment.¹

**The engineering gap this creates.** These two squeezes surface as a concrete, unsolved problem in dynamic knowledge graphs: while Retrieval-Augmented Generation has thoroughly optimized *what to retrieve*, there is still no normative criterion for *when to accept* a new piece of structure into memory — when to commit an edge, when to restructure, when to discard. A criterion for "accept iff the structural gain justifies the cost" must be able to *see* structural gain in the first place; a probability-only gauge cannot.

**Scope and positioning.** We emphasize what this argument is not. It is a *complement*, not a replacement: statistical mechanics, Shannon theory, and FEP are correct — and remain the right tools — for large-scale, long-run behavior. geDIG targets the opposite regime: small-scale, single-event decisions — one edit, one commitment, one moment. The two regimes divide by scale, much as thermodynamics and statistical mechanics do. And the difference is testable rather than rhetorical: Figure 1 is a concrete counterexample on the spatial axis (§2.1), and every control claim in this paper carries a pre-registered falsification condition (§9; the project's registry includes recorded defeats, not only wins).

> ¹ One may ask why the squeeze happened at all. Differentiation and probability have been overwhelmingly convenient for both humans and computers — 350 years of analytic machinery, and hardware built around continuous algebra. We keep this observation out of the technical argument; the engineering claim below stands on its own.

### §1.2 Our Approach: One Scalar over Three Structural Atoms

geDIG (graph edit **D**istance and **I**nformation **G**ain) keeps structural quantities structural. Its single gauge is

$$\mathcal{F} \;=\; \Delta \mathrm{EPC} \;-\; \lambda\,(\Delta H + \gamma\, \Delta \beta_1),$$

where ΔEPC is the (normalized) edit-path cost of a candidate restructuring, ΔH the Shannon-entropy change, and Δβ₁ the change in the first Betti number — the count of independent cycles. **F < 0** means the information gain exceeds the structural cost: the system should commit the change. v6 of this framework used shortest-path shortening (ΔSP) as the structural-gain term; v7 generalizes it to Δβ₁, a topological invariant independent of graph shape and scale.

The three terms are not an ad-hoc feature set. They are the atoms of the three basic space concepts of modern mathematics: ΔEPC is *metric* (distances, edit costs), ΔH is *measure* (probability mass), Δβ₁ is *topology* (connectivity, cycles). Each existing method in Figure 1's comparison fails precisely because it retains only one or two of the three; geDIG keeps all three, in one dimensionless scalar. We state the modest version of this claim: the three terms cover the three basic space concepts *currently known* to describe structure — whether three are also *necessary* is an open problem we place in §11, and the independence of the three terms is demonstrated by example (Figure 1), with its rejection condition registered in §9.

On the time axis, geDIG's control is event-driven by construction. A two-stage gate turns the scalar into discrete decisions: **AG** (Attention Gate, 0-hop) fires on local ambiguity — "something here does not fit"; **DG** (Decision Gate, multi-hop) fires when a genuine structural shortcut is confirmed — "commit it." Firing times are kept as discrete events, not averaged away: the moment of commitment is data.

### §1.3 Contributions

1. **A single gauge, generalized to topology.** F = ΔEPC − λ(ΔH + γΔβ₁), with β₁ replacing v6's shortest-path term; one equation drives exploration, integration, backtracking, and memory eviction across three domains (maze navigation, RAG, Transformer analysis) with no domain rules.
2. **A unified core implementation.** All three experiment streams share one F-evaluation core (`src/gedig/`, 71 unit tests), replacing the per-experiment implementations of earlier versions.
3. **Maze proof-of-concept: emergent control and a memory write-gate.** On 15×15 mazes (n=100) the agent reaches 98% success — on par with, not above, a greedy-DFS baseline (99%) — while pruning ~98% of candidate edges: it keeps only the topological skeleton. The claim is the *control mechanism and write-gate behavior*, not raw success superiority. Stage-2 (Wake–Sleep–Wake graph persistence) shows large second-episode improvements as a package effect; the sleep-only ablation is pre-registered and in progress.
4. **AGHT, an analytical graph transformer for zero-shot retrieval.** QKV attention over a unified sentence–token graph; BRIGHT biology nDCG@10 = 0.439 (50 queries, single seed; full 3-domain ≈ 0.19, well below SOTA ≈ 0.63 — an early proof-of-concept, reported with its limits), HotpotQA R@2 = 0.405 (+170% over an internal PageRank baseline).
5. **Transformer F-trajectory observation and a registered intervention line.** Layer-wise F decomposition across 8 models shows structured, scale-sensitive behavior; F-regularization interventions are reported as preliminary (single-run), with the sign question explicitly unresolved and its deciding experiment pre-registered.
6. **Falsifiability as method.** Every major claim carries a rejection condition (§9); the project registry records defeats as well as wins (e.g., the routing-signal prediction P1/P2, registered and failed in 2026-06). We consider this discipline itself a contribution to how insight-metrics research is reported.

### §1.4 Organization

§2 defines the gauge and its two-stage gating, and locates them against existing criteria (including the Figure 1 comparison). §3 describes the unified core implementation. §4–§8 report the experiments (maze; BRIGHT; HotpotQA paragraph selection; Transformer F-regularization; HotpotQA dual-process). §9 collects ablations and rejection conditions. §10 reviews related work, §11 states limitations and open problems, §12 concludes.

---

## 差分サマリ（v6.1 §1 → 本ドラフト）

| | v6.1 | 本ドラフト |
|---|---|---|
| 起点 | 動的 KG の受容規範の欠如 | 確率への押し込め（空間・時間の 2 軸）→ その帰結として受容規範の欠如 |
| F の構造項 | ΔSP | Δβ₁（v6 との関係を 1 文で明示） |
| 3 項の根拠 | （なし） | Bourbaki 3 基本空間 + 控えめな十分性主張 + 反証条件参照 |
| 貢献 | 4 項目 | 6 項目（統一コア、AGHT、F-trajectory、反証規律を追加） |
| 神経科学 | hippocampal replay 引用が第 1 段落 | §1.2 の AG/DG で operational metaphor として軽く（詳細は §2） |
| 主張の強度 | — | 監査後の誠実版（on par not above、preliminary 明示、SOTA 未達明示） |
