# v7 §2.4 Helmholtz Correspondence — 改称・改稿ドラフト

> **作成**: 2026-07-02（Tier 1 作業 C、[tier1_action_plan.md](../tier1_action_plan.md) §2.C）
> **状態**: DRAFT — 著者レビュー待ち。
> **v6.1 からの変更**: 節名を「FEP–MDL Bridge」→「Helmholtz Correspondence」に改称し、v6 §9 の operational proposition と Appendix C の熱力学ノートをここに統合。構造項の対応を SP 版（U := ΔEPC_norm − λγΔSP_rel）から β₁ 版（U := ΔEPC − λγΔβ₁）へ更新。
> **素材**: [gedig_formula_three_readings_20260306.md](../../../research/thinking/gedig_formula_three_readings_20260306.md) §3–§4（3 つの読み方、読み 2 が正しい対応）、[gedig_core_theory_unified.md](../../../research/gedig_core_theory_unified.md) §6、v6.1 EN §9 + Appendix C
> **書き方の原則**（three_readings §6.5 に従う）: 本文の制御・実験は**読み 1（canonical）のみ**で書く。読み 2/3 は本節に閉じ込め、spec には混ぜない。

---

## 編集メモ（日本語 — 本文には入れない）

1. **v6.1 の FEP–MDL Bridge（§9、Lemma 1/2、仮定 B1–B4）をどうするか**: 本ドラフトは「Helmholtz 対応を §2.4 の主役に、FEP–MDL は §2.4 内の 1 段落に縮約」する案。Lemma 1/2 を残したい場合は Appendix へ移す（削除はしない — v6 読者への連続性）。→ **著者判断待ち**
2. 係数の扱いを正直に: 理念形 (EPC − B) − H は係数吸収後の形。本文では F = (ΔEPC − λγΔβ₁) − λΔH という恒等変形で示し、「λγ を構造側に吸収した」と明記する。
3. λ の温度制御（annealing、仮説 H_temp）は §11 Future Work 送り。ここでは 1 文の示唆に留める。

---

## §2.4 Helmholtz Correspondence (English draft)

The gauge F admits three algebraically equivalent groupings, each with a distinct physical reading. We state all three once, fix one as canonical, and derive the thermodynamic correspondence from the second. Nothing in the control algorithm (§2.1) depends on this section; it is interpretive.

**Reading 1 (canonical — economic).**
$$\mathcal{F} = \Delta\mathrm{EPC} - \lambda(\Delta H + \gamma\Delta\beta_1) \;=\; \text{cost} - \lambda \cdot \text{gain}.$$
Structural cost against information gain; F < 0 fires the commit. This is the reading used throughout the paper: it maps directly onto the AG/DG gates and onto every experiment. All control claims are stated and tested in this form.

**Reading 2 (Helmholtz-like — physical).** Regrouping the same expression,
$$\mathcal{F} = \underbrace{(\Delta\mathrm{EPC} - \lambda\gamma\,\Delta\beta_1)}_{U} \;-\; \lambda \underbrace{\Delta H}_{S},$$
which is formally F = U − TS with

| Helmholtz | geDIG | reading |
|---|---|---|
| U (internal energy) | ΔEPC − λγΔβ₁ | essential structural change: edit cost minus the part that creates/destroys redundant cycles — the "skeleton cost" |
| T (temperature) | λ | information temperature: how much gain one unit of structural cost must buy |
| S (entropy) | ΔH | residual disorder on the probability side |
| spontaneous process (F < 0) | commit fires (F < 0) | the direction in which the system "wants" to move |

The subtraction inside U is the substantive point: of the total edit cost, the part that merely changes cycle count (Δβ₁) is a change of *topological order*, not of skeletal structure; removing it leaves a natural discrete analogue of internal energy. λ then behaves as a temperature — at high λ (low temperature) the system demands large information gain per unit cost and consolidates; at low λ (high temperature) it tolerates structural churn and explores. This suggests treating λ as a *schedule* rather than a constant (annealing across the Wake–Sleep–Wake cycle); we leave that as a registered hypothesis for future work (§11).

**Reading 3 (order-parameter — speculative).** Grouping as (ΔEPC − λΔH) − λγΔβ₁ reads β₁ as an order parameter separating an exploratory (high-entropy) phase from a consolidated (structure-locked) phase, in the spirit of Landau theory. We flag this reading as the most distant from current evidence and use it only when discussing phase-transition-like phenomena (§10.3).

**Relation to FEP and MDL.** The correspondence above — like the FEP–MDL bridge of v6 (§9 there; assumptions and code-length lemmas reproduced in Appendix C) — is an *operational correspondence, not a claim of physical identity*. FEP describes convergence of large ensembles of predictions; MDL describes asymptotic code length. geDIG operates at a granularity both leave open: the single discrete edit, with its timestamp. The mapping (0-hop AG ↔ local prediction-error side; multi-hop DG ↔ compression side) is retained from v6 as intuition, with residual terms O(1/N) under assumptions B1–B4 (Appendix C).

**Why one canonical reading.** The equation is one; the readings are many; the canon is one. Specifications and experiments use Reading 1 only. Readings 2 and 3 live in this section and in the discussion — they generate hypotheses (λ-annealing, phase-transition probes) but carry no load in any result of this paper.

---

## 差分サマリ（v6.1 → 本ドラフト)

| | v6.1 | 本ドラフト |
|---|---|---|
| 節名・位置 | §9 FEP–MDL Bridge + Appendix C（熱力学は「note」扱い） | §2.4 Helmholtz Correspondence（理論部に昇格、FEP–MDL は 1 段落に縮約） |
| U の定義 | ΔEPC_norm − λγΔSP_rel | ΔEPC − λγΔβ₁（β₁ 移行に整合） |
| 読み方 | 熱力学 1 種（heuristic 注記付き） | 3 読み方を明示、canonical を 1 つに固定 |
| 留保 | "does not claim physical identity" | 同留保を維持 + 「どの結果もこの節に依存しない」を冒頭に明示 |
| λ | information temperature (kT) と言及 | 同 + annealing 仮説を §11 送りで登録 |
