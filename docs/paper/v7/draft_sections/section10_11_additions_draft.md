# v7 §10.3 / §11.5–11.6 追加セクション — ドラフト

> **作成**: 2026-07-02（Tier 1 作業 D + E、[tier1_action_plan.md](../tier1_action_plan.md) §2.D–E）
> **状態**: DRAFT — 著者レビュー待ち。
> **素材**: [insight_transformer_phase_transition_landscape.md](../../../research/thinking/insight_transformer_phase_transition_landscape.md)（§10.3 の対比表）、[gedig_core_theory_unified.md](../../../research/gedig_core_theory_unified.md) §9.6 + 付録 D、[experiment_grokking_curl.md](../../../research/thinking/experiment_grokking_curl.md)、[insight_three_terms_orthogonality.md](../../../research/thinking/insight_three_terms_orthogonality.md)
> **注意**: 引用 arXiv 番号は思考メモからの転記。**投稿前に全件を原典照合すること**（/ars-citation-check 対象）。

---

## §10.3 Transformer, Phase Transitions, and Topology (NEW — English draft)

A rapidly growing 2025–2026 literature connects Transformer internals to statistical-physics order parameters and topological invariants. geDIG's position in this landscape is specific: it treats the topological quantity not as a diagnostic *proxy* but as a term of a free-energy-like *control gauge*.

- **Özönder (2025), "Attention to Order" (arXiv:2510.07401)** detects 2D-Ising-like phase transitions in attention matrices via attention entropy. This validates the *measure* axis of our decomposition — entropy sees the transition — but an entropy-only gauge cannot separate Figure 1's Case A from Case B; geDIG adds the β₁ term precisely for that distinction.
- **Sun & Haghighat (2025) (arXiv:2501.16241)** reformulate the Transformer as a continuous O(N) field model. geDIG is complementary on the discrete side: it operates on the token/sentence graph itself, where edits and cycles are integer events.
- **TAG-DS (2025)** reports Betti–Fiedler partitions tracking grokking — β₁ as a *proxy* for the generalization transition. geDIG integrates β₁ as a free-energy term with a cost side (ΔEPC) and a temperature (λ), so the same quantity that diagnoses the transition also *drives* accept/reject control; this is the step from measurement to actuation.
- **Grokking as dimensional phase transition (arXiv:2604.04655)** ties grokking to an effective-dimension jump; consistent with our d_eff = β₁/V + 1 reading (Part 1 §5.7.2), and with the temporal-squeeze argument of §1.1 — the event is discrete.
- **T3former / TopoFormer (2025)** inject TDA features into Transformers as engineered inputs. geDIG differs in role: topology enters the objective/gauge, not the feature vector.

*(編集メモ: 上記対比は landscape メモの表を散文化したもの。投稿版では表形式（手法 × 測る量 × proxy か gauge か）に戻す選択肢もある。)*

---

## §11.5 Open Theoretical Problems (NEW — English draft)

We list what this paper does *not* establish, as claims we would like others to attack.

1. **Structure ≡ probability.** Whether the structural information kept by (ΔEPC, ΔH, Δβ₁) can be losslessly re-encoded in a probabilistic representation is open. Our Figure 1 argument shows current practice drops it; it does not show it *must* be dropped. We leave the rigorous statement to information geometry / MDL / MaxEnt theorists (Appendix D sketches the question).
2. **Necessity of exactly three terms.** The three terms cover the three basic space concepts currently used to describe structure (metric, measure, topology). Sufficiency-by-example is demonstrated (Figure 1); *necessity* — that no smaller or different atom set suffices — is open. Our independence claim is exemplary, not statistical; its promotion path (exemplary → statistical independence → information-geometric orthogonality) is future work.
3. **Discrete curl and Hodge decomposition.** The AG/DG ledger defines a discrete flow on the knowledge graph; whether its rotational component (curl) connects to a Hodge decomposition — and whether "prediction" corresponds to the curl term — is unformalized (Part 2 §3.3).
4. **Scaling law for β₁.** How β₁-based control behaves as graphs grow by orders of magnitude (beyond 51×51 mazes and ~10³-node retrieval graphs) is unmeasured; the rejection condition table (§9) registers the scale-invariance claim and its kill criterion.

## §11.6 High-Priority Experiments (registered, not yet run — English draft)

- **H_grokking-curl**: reproduce Nanda et al. (2023) grokking, instrument β₁ *and* curl(attention) per epoch. Extends the TAG-DS β₁-proxy finding to a gauge-driven account; first engineering test of the curl hierarchy (Part 2 §3.3). ~10 weeks, GPU required.
- **H_ising-bkt**: reproduce Özönder (2025) and augment attention-entropy with β₁; tests whether topology adds detection power for BKT-type (topological) transitions where entropy alone is blind.
- **Concept-Reuse Asymmetry (nominalization) test**: whether committed (DG-passed) subgraphs are preferentially reused as units in later reasoning — the reuse asymmetry predicted by the write-gate account of §4.
- **Maze sleep-only ablation** (pre-registered 2026-07-02, [docs/prereg/maze_sleep_ablation.md](../../../prereg/maze_sleep_ablation.md)): isolates the sleep-propagation contribution inside the stage-2 package effect; its falsification condition is registered.

*(編集メモ: §11.6 に sleep ablation を追加したのは tier1_action_plan にない今回の判断 — 実験が完了したら §4 本文に昇格させ、ここからは削除する。)*

---

## Tier 1 作業 F（略称統一）の適用状況

- 本ドラフト群（§1 / §2.4 / §10–11）はすべて **geDIG = "graph edit Distance and Information Gain"** で統一済み（v5/v6 表記と整合）。
- v7 の tex 起稿時に、"Generalized Differential Information Gain" の全文検索を 1 回実行して混入を防ぐこと。
