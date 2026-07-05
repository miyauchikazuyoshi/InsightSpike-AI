# InsightSpike-AI

**A structural fitness score for knowledge graphs — can one equation decide when to restructure?**

$$\mathcal{F} = \underbrace{\Delta \text{EPC}}_{\text{Metric}} \;-\; \lambda \left( \underbrace{\Delta H}_{\text{Measure}} \;+\; \gamma\, \underbrace{\Delta \beta_1}_{\text{Topology}} \right)$$

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://miyauchikazuyoshi.github.io/InsightSpike-AI)

---

## The Equation

$$\mathcal{F} = \underbrace{\Delta \text{EPC}}_{\text{Metric}} \;-\; \lambda \left( \underbrace{\Delta H}_{\text{Measure}} \;+\; \gamma\, \underbrace{\Delta \beta_1}_{\text{Topology}} \right)$$

| Term | Mathematical Structure | What It Captures | Grounding |
|------|----------------------|------------------|-----------|
| $\Delta\text{EPC}$ | **Metric** (distance) | Cost of restructuring the knowledge graph | Graph Edit Distance |
| $\Delta H$ | **Measure** (probability) | Change in entropy / uncertainty | Shannon (1948); Entropy-Lens (Ali et al., 2025) |
| $\Delta\beta_1$ | **Topology** (loops) | Change in the number of independent cycles | Algebraic Topology; Betti numbers |

Three independent mathematical structures. One dimensionless, scale-invariant scalar.

**$\mathcal{F} < 0$** means information gain exceeds structural cost — the system should commit the change.

> *Note: The v6 paper uses $\Delta\text{SP}$ (shortest-path shortening) as the structural term. The ongoing research generalizes this to $\Delta\beta_1$ (first Betti number), which is a topological invariant independent of graph shape or scale.*

---

## Origin

> *"How do we build an AI that thinks like Einstein?"*

From this question came geDIG: the hypothesis that **insight is the topological reconstruction of memory**, and that it can be measured as a thermodynamic quantity — isomorphic to Helmholtz free energy $F = E - TS$.

**[Read the full origin story](docs/research/gedig_origin_story.md)** (EN / JA)

**[Visual intuition: The Matchstick Figure](https://miyauchikazuyoshi.github.io/InsightSpike-AI/research/thinking/matchstick_figure_v2_en.html)** — interactive HTML illustrating the independence of metric, measure, and topology

---

> **Navigating this repo** (agents & collaborators): start with [docs/MAP.md](docs/MAP.md) —
> directory map, terminology traps (the four meanings of "Phase"), the claims ledger, and known debt.
> **Which F / implementation / tests / claims are authoritative:** [docs/CANONICAL.md](docs/CANONICAL.md).

## Project Status

This is an **active research project** by an individual researcher with AI-assisted implementation.
It is not a production library.

| Component | Status | Location |
|-----------|--------|----------|
| **Unified geDIG Core** | 71 unit tests, run in CI on every push (60 in the torch-less light CI, all 71 locally); F-eval gives equivalent results across 3 backends (maze / RAG / transformer) | [`src/gedig/`](src/gedig/) |
| geDIG theory (v6 paper) | Pre-print — position paper + proof-of-concept | [`docs/paper/`](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf) |
| BRIGHT reasoning-intensive retrieval | nDCG@10 = 0.439 on **biology, 50 queries, single seed** (preliminary; full 3-domain ≈ 0.19; SOTA ≈ 0.63) | [`experiments/hotpotqa_v2/`](experiments/hotpotqa_v2/) |
| AGHT (Graph Transformer) | HotpotQA R@2 = 0.405, **+170% over an internal PageRank baseline** (zero-shot, 100q, single seed) | [`experiments/hotpotqa_v2/src/unified_graph.py`](experiments/hotpotqa_v2/src/unified_graph.py) |
| Transformer F-regularization (Exp4) | **Preliminary, single seed**: F-maximize > baseline under SP-based F only — *not* confirmed under β₁ or against a random-regularization control | [`experiments/transformer/`](experiments/transformer/) |
| Maze PoC (single episode) | 98% goal-reach (15x15, n=100) — on par with greedy-DFS / oracle (99%); distinctive result is ~98% map compression via emergent AG/DG control | [`experiments/maze/`](experiments/maze/) |
| Maze stage-2 (Wake-Sleep-Wake) | Five pre-registered ablations (2026-07): v1 — original undirected propagation contributed **zero** (headline 71.9%→95.3% re-attributed to curriculum+10D). v2 — **trajectory-Q replay propagation, self-navigation: eval steps −39%** (p=4.0e-05, n=23 paired). v3 — **replicated on 18 fresh seeds, −51%** (115 vs 234, p=2.9e-04); lift-of-failed-warmups inconclusive (underpowered, recorded as such). v4 — dead-end carving: **defeat recorded** (seed-52 effect idiosyncratic). v5 — unconditional budget-split warmup: **regression confirmed in the succeeded stratum** (+122.5 steps, p=1.8e-05) → withheld; failed-stratum rescue directionally positive but underpowered (n=6); adaptive splitting registered as the v6 candidate | [`experiments/maze/graph_persistent_dg/`](experiments/maze/graph_persistent_dg/) |
| HotpotQA dual-process (v3) | GPT-4o EM 51.2% vs IRCoT 47.6% (500q) — **p=0.086, not significant**; reverses on GPT-4o-mini | [`experiments/hotpotqa_v2/REPORT_v3_dual_process.md`](experiments/hotpotqa_v2/REPORT_v3_dual_process.md) |
| Visual proof (matchstick figure) | Interactive HTML | [EN](https://miyauchikazuyoshi.github.io/InsightSpike-AI/research/thinking/matchstick_figure_v2_en.html) / [JA](https://miyauchikazuyoshi.github.io/InsightSpike-AI/research/thinking/matchstick_figure_v2.html) |

---

## Unified Core

Three independent experiment streams share a single F-eval implementation:

```
src/gedig/core/f_eval.py  →  F = ΔEPC - λ(ΔH + γΔB)
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Maze  RAG  Transformer
```

See [`docs/architecture/unified_core_architecture.md`](docs/architecture/unified_core_architecture.md) for full architecture.

---

## Experiments

### Maze Navigation

A partial-observation maze agent that builds a persistent knowledge graph and uses geDIG to decide when to explore vs. exploit. This is a **proof-of-concept that every decision — what to accept into memory, what to discard, and when to backtrack — can emerge from operating a single gauge F + quantile gates (AG/DG), with no domain rules**. It is *not* a claim that geDIG is the best maze solver.

- **Architecture**: Wake-Sleep-Wake cycle with three-layer search (L0: O(1) hash, L1: O(degree) attention walk, L2: O(N log N) full sort)
- **15x15 maze (n=100)**: geDIG reaches **98%** goal-reach (Wilson 95% CI [0.93, 0.99]). For reference, a plain greedy-DFS baseline reaches **99%** and an oracle BFS **99%** — so geDIG does *not* beat the simplest complete baseline on success rate. (DFS is a maze-specific algorithm; geDIG knows nothing about mazes — success rate is a sanity check, not the claim.)
- **What is distinctive**: geDIG attains this via *emergent* AG/DG control (no hard-coded "if dead-end" rule) while compressing the explored map by **~98%** (retaining only the topological skeleton). The PoC's contribution is the control mechanism and the **memory write-gate behavior** (selecting what is worth remembering), not raw success superiority.
- **25x25 maze**: preliminary — success rate varies by configuration (≈ 0.42–0.75 across runs) and statistics are not yet finalized.

**Stage-2 (Wake-Sleep-Wake / graph persistence)** asks a different question: does the memory graph built in one episode, consolidated during a sleep phase (Q-style reward propagation + isolate cleanup), make the *next* episode better?
- Package-level evidence (25x25, v6_perseed): success 71.9% → 95.3%, mean steps −41%, edges −34% (10D vectors + sleep propagation + curriculum changed together).
- **Pre-registered sleep-only ablation v1 — completed 2026-07-02, defeat recorded**: with 10D + curriculum held fixed (n=30 paired seeds), the original undirected max-propagation contributed *nothing* — 29/30 pairs produced identical eval trajectories. The headline improvement was **re-attributed to graph carryover + curriculum + dictionary guidance** ([prereg + defeat record](docs/prereg/maze_sleep_ablation.md)); a design audit identified four gaps (saturating values, max-propagation killing negative examples, dictionary guidance dominating action selection, no gradient source on failed warmups) and found the 71.9→95.3 delta was in fact a *Wake1* effect ([audit](docs/audits/sleep_ablation_design_audit.md)).
- **Redesigned ablation v2 — completed same day, claim established**: propagation replaced by *trajectory-based Q backup* (bounded, directed, negatives survive) and evaluation switched to self-navigation (no dictionary guidance), on the 23 warmup-success seeds. Result: **eval steps 124.2 vs 202.3 (paired −39%, p=4.0e-05); dead-end encounters 0.17 vs 4.74 per run (4 vs 109 total)** — gradient navigation matches the BFS-plan dictionary's optimality (mean ≈ theoretical shortest 124.0) with no plan at all ([prereg v2](docs/prereg/maze_sleep_ablation_v2.md)). This is the first isolated demonstration of the sleep value-consolidation contribution; it remains F-independent (F-driven sleep is the next registered line).
- Stage-2 metrics are **steps / edge count / second-episode improvement**, not success rate (which is near ceiling at 15x15 and insensitive).

```bash
# Reproduce (requires .venv with networkx, numpy, etc.)
.venv/bin/python3 experiments/maze/run_experiment_query.py \
  --maze-size 15 --max-steps 250 --seeds 12 \
  --search-mode threelayer --vector-mode extended
```

See [`experiments/maze/README.md`](experiments/maze/README.md) for full CLI reference.

### Transformer Inference F-Trajectory

Layer-by-layer measurement of $\Delta\text{EPC}$, $\Delta H$, and $\Delta\beta_1$ across Transformer hidden states, testing whether $\mathcal{F}$ tracks model quality.

- **8 token-level models tested**: BERT, DistilBERT, GPT-2, GPT-2 Medium, DistilGPT2, TinyLlama (2 checkpoints)
- **Key finding**: GPT series shows monotonic improvement in $\Delta R^2_{\text{struct}}$ with model scale — $\mathcal{F}$ has sensitivity to model quality
- **Connection to prior work**:
  - Extends [Oyama et al. (ACL 2025 Outstanding)](https://aclanthology.org/2025.acl-long.1584/) model mapping from $H$-only to full $\mathcal{F}$ decomposition
  - [Gao et al. (2025)](https://arxiv.org/abs/2511.13653) sparse-circuit interpretability corresponds to $\beta_1$ reduction in $\mathcal{F}$
  - [Hewitt & Manning (2019)](https://aclanthology.org/N19-1419/) structural probes provide the measurement basis for EPC

**Experiment 4 — F-Regularization (Training Intervention)** — *preliminary, single run (SST-2 / DistilBERT, 2k train, β=0.1); not yet replicated across seeds*:

| Condition | Accuracy (SP-based F) | Accuracy (β₁-based F) |
|-----------|:---:|:---:|
| Baseline (CE only) | 88.1%¹ | 88.5% |
| Positive (CE + F minimize) | 87.2% | 83.5% |
| **Negative (CE + F maximize)** | **89.4%** | 85.5% |

¹ The baseline reached 89.4% at epochs 1–2 and dropped to 88.1% by epoch 3 (overfitting); the negative condition's 89.4% equals the baseline's own earlier peak.

**Conclusion (preliminary, not established):** Under SP-based F, the negative (F-maximize) condition edges out the epoch-3 baseline — consistent with "preserving DG topology helps." But this is a **single run**, and the effect does **not** hold up under scrutiny: (i) under β₁-based F the negative condition (85.5%) is *below* baseline (88.5%); (ii) a separate negative-control experiment found geDIG-F regularization (66.5%) did **not** beat random-value regularization (69.5%), suggesting any gain may come from regularization in general rather than geDIG-F specifically. Multi-seed replication is required before claiming `negative_better`.

See [`experiments/transformer/`](experiments/transformer/) for experiment design and results.

### BRIGHT Reasoning-Intensive Retrieval

AGHT (Analytical Heterogeneous Graph Transformer) applied to BRIGHT benchmark (ICLR 2025) — document retrieval requiring multi-hop reasoning.

- **Architecture**: Unified Sentence-Token heterogeneous graph with QKV attention (10 analytical parameters, zero-shot)
- **BRIGHT biology, 50 queries, single seed**: nDCG@10 = 0.439 (Spec X: Enhanced Graph + Early Token Graph). This is a single-config screening result on **one domain**; the 50-query CI is ≈ ±13pt. On the full 3-domain set (323 queries) the best configuration reaches overall nDCG@10 ≈ **0.19**.
- **Context — not competitive with SOTA**: on BRIGHT, BM25 ≈ 0.145, BM25 + GPT-4 reasoning + rerank ≈ 0.30, and current SOTA (INF-X-Retriever) ≈ 0.63. This zero-shot result is an early proof-of-concept, well below these references.
- **HotpotQA paragraph selection (100q, zero-shot, single seed)**: R@2 = 0.405, **+170% over an internal PageRank baseline** (not a supervised SOTA). SF F1 = 0.334 ≈ 41% of supervised DFGN (81.1).

| Benchmark | Metric | AGHT (ours, zero-shot) | Internal baseline (PageRank) | Delta |
|-----------|--------|-------------|--------|-------|
| HotpotQA (comparison/AG) | R@2 | **0.429** | 0.143 | +200% |
| HotpotQA (bridge/DG) | R@2 | **0.256** | 0.151 | +70% |
| HotpotQA (all) | SF F1 | 0.334 | — | zero-shot |

*Deltas are vs an internal PageRank baseline, single seed; statistical tests are pending (planned for v7).*

See [`experiments/hotpotqa_v2/`](experiments/hotpotqa_v2/) and [`docs/architecture/unified_core_architecture.md`](docs/architecture/unified_core_architecture.md).

### Multi-Hop QA

Two independent experiment lines targeting multi-hop question answering:

#### v10: Entity-Graph Paragraph Reordering (MuSiQue) — Latest

Entity-graph からの推論チェーンでパラグラフを並べ替え、LLM の注意力を暗黙的に誘導する試み。
**500q フルランで統計的に有意な改善は確認できず** (+1.2pt, p>0.05)。
ただし「guided テキストは GPT-4o に害」「暗黙的誘導 > 明示的指示」等の再利用可能な知見を確立。

| 条件 | EM (500q) | 備考 |
|------|-----------|------|
| Baseline A (全20パラ + CoT) | 47.4% | ref |
| v10d reorder_only | 48.6% | +1.2pt (有意でない) |

根本原因: 20パラ (~2,500 tokens) は GPT-4o ウィンドウの 2% で、"Lost in the Middle" が発生しない。
エラーの 45% は推論の誤り (distractor entity 選択) であり、パラ位置の問題ではない。

See [`experiments/hotpotqa_v2/docs/report_v10_entity_graph.md`](experiments/hotpotqa_v2/docs/report_v10_entity_graph.md) for the full report.

#### v11: Pre-computed Topology Routing (MuSiQue) — Current

v10 と v2/v3 の知見を統合: コンテキストを 50 パラ (~6,000 tok) に拡大し、
事前構築グラフからのサブグラフ抽出 + F 値ルーティング (System 1/2) で性能劣化を回復する試み。

- **Offline**: 全パラから sentence-level 三層グラフを事前構築
- **Online**: クエリからサブグラフ抽出 -> F 値 -> System 1 (サブグラフのみ) / System 2 (全パラ)

See [`experiments/hotpotqa_v2/docs/experiment_design_v11.md`](experiments/hotpotqa_v2/docs/experiment_design_v11.md) for the experiment design.

#### v2/v3: geDIG Dual-Process Architecture (HotpotQA)

geDIG applied to multi-hop question answering on HotpotQA (distractor setting). The v3 **dual-process architecture** uses Betti numbers as a cognitive routing signal — the gauge value F decides when to answer instantly (System 1) vs. reason step-by-step (System 2), inspired by Kahneman's dual-process theory.

- **Model-dependent interaction**: On GPT-4o, Hybrid-E1 reaches EM 51.2% vs IRCoT 47.6% (+3.6pt) at far fewer LLM calls — but the gap is **not statistically significant (McNemar p=0.086)**. On GPT-4o-mini the ordering reverses and IRCoT wins significantly (p=0.008).
- **Tentative observation (two model sizes only)**: topology-guided routing *appears* to improve with model capability. This is an observation from two data points, not an established scaling law.

**500-question evaluation (primary reference):**

| Model | Hybrid-E1 EM | IRCoT EM | LLM Calls² | McNemar p |
|:-----:|:---:|:---:|:---------:|:-------:|
| **GPT-4o** | **51.2%** | 47.6% | ~2.2 vs ≤8 | 0.086 (n.s.) |
| GPT-4o-mini | 45.2% | **50.4%** | ~2.1 vs ≤8 | **0.008 (sig.)** |

² Call counts are not directly logged. Hybrid's "~2.2" is derived as 1 + mean(CoT steps) (4o: 1.24, mini: 1.14). IRCoT's "≤8" is its `max_steps` budget (a ceiling, not a measured value; early-exit makes the actual count lower but it is unrecorded). So the "fewer calls" advantage is real in direction but the exact ratio is approximate.

See [`experiments/hotpotqa_v2/`](experiments/hotpotqa_v2/) for full experiment code and [`REPORT_v3_dual_process.md`](experiments/hotpotqa_v2/REPORT_v3_dual_process.md) for the detailed report.

### Earlier Experiments

Cross-domain analogy experiments were conducted in earlier phases and informed the theory. These have not been reproduced under the current codebase and are archived.

---

## Quick Start

This example uses **Flash-geDIG** (`insightspike.gedig`), the torch-native fast
path for attention matrices. The reference implementation is the unified core
[`src/gedig/`](src/gedig/); the two are tied by equivalence tests — see
[docs/CANONICAL.md](docs/CANONICAL.md).

```python
import torch
from insightspike.gedig import compute_f_score

# Attention matrix: (Batch, Heads, Seq, Seq)
attn = torch.rand(1, 12, 64, 64)
f_values, metrics = compute_f_score(attn, lambda_param=1.0, gamma=0.5)

# f_values: (Batch, Heads) — lower is better
print(f"Mean F: {f_values.mean():.4f}")
print(f"EPC={metrics['delta_epc'].mean():.4f}, "
      f"H={metrics['delta_h'].mean():.4f}, "
      f"SP={metrics['delta_sp'].mean():.4f}")
```

```bash
pip install -e .          # Install from source
make test                 # Run unit tests
```

---

## Theoretical Background

geDIG (Generalized Differential Information Gain) is built on the hypothesis that intelligence acts to minimize $\mathcal{F}$: balancing the cost of restructuring knowledge against the information gained.

**Thermodynamic correspondence**:

$$\mathcal{F} = E - TS \quad\longleftrightarrow\quad \mathcal{F} = \Delta\text{EPC} - \lambda\,\Delta\text{IG}$$

**Two-stage gating**:
- **AG (Attention Gate)**: Detects ambiguity/novelty (0-hop). Analogous to noradrenaline.
- **DG (Decision Gate)**: Confirms valid restructuring (multi-hop). Analogous to dopamine.

> *The neurotransmitter correspondence is a computational analogy, not a physiological claim.*

For formal definitions, see [`docs/gedig_spec.md`](docs/gedig_spec.md). For the full paper, see the [v6 pre-print (PDF)](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf).

---

## Applied Research: TSD-OCR

geDIG's principle — *"don't make the network learn what can be solved"* — is validated in visual cognition through [TSD-OCR](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr): a character recognition pipeline that replaces Conv1 with differential geometry.

- **19K parameters** outperform **11M-parameter** pixel CNNs in cross-domain transfer
- Hand-designed V1 (curvature κ, orientation θ) + learned V2-V4 (CNN) mirrors the brain's evolutionary solution
- The Splatting-Attention duality discovered in TSD-OCR maps directly to geDIG's AG/DG gates

**→ [TSD-OCR Origin Story](https://github.com/miyauchikazuyoshi/vector-based-cnn-ocr/blob/main/docs/research/origin_story.md)** (JA/EN)

---

## Open Questions

Specific, actionable research questions where external collaboration would be valuable:

1. **Independent F-trajectory reproduction** — Does $\mathcal{F}$ show structured layer-wise behavior on models we haven't tested? Scripts and data are provided in [`experiments/transformer/inference_gedig_v2/`](experiments/transformer/inference_gedig_v2/).

2. **$\beta_1$ vs SP as structural term** — Under what conditions does topological $\Delta\beta_1$ outperform metric $\Delta\text{SP}$? Expertise in topological data analysis (TDA) welcome.

3. **Scaling to 70B+ models** — The hypothesis predicts that $(\lambda, \gamma)$ converge across model families at sufficient scale. Verification requires GPU resources beyond the current individual setup.

4. **F-regularization robustness** — The v6 paper shows weak F-regularization improves downstream performance (+0.33pt on SST-2). Is this robust across tasks and model families?

---

## References

**Core theory**:
- geDIG v6 paper: [`docs/paper/arxiv_v6_en/`](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf)
- Canonical spec: [`docs/gedig_spec.md`](docs/gedig_spec.md)

**Each term of $\mathcal{F}$ is grounded in independent prior work**:

| $\mathcal{F}$ term | Prior work |
|---------------------|-----------|
| $\Delta\text{EPC}$ (metric) | Hewitt & Manning (2019), [A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/) |
| $\Delta H$ (measure) | Ali et al. (2025), [Entropy-Lens: The Information Signature of Transformer Computations](https://arxiv.org/abs/2502.16570) |
| $\Delta\beta_1$ (topology) | Kushnareva et al. (2021), [Artificial Text Detection via Examining the Topology of Attention Maps](https://aclanthology.org/2021.emnlp-main.50/) — TDA/Betti features on BERT attention graphs (EMNLP 2021) |

**Related work on structure and interpretability**:
- Gao et al. (2025), [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653) — sparsification as $\beta_1$ reduction
- Oyama et al. (2025), [Mapping 1,000+ Language Models via the Log-Likelihood Vector](https://aclanthology.org/2025.acl-long.1584/) — ACL Outstanding Paper; model-map methodology related to our $H$-based comparisons (not a topology grounding)

---

## Citation, License, and Patent

**Paper**: [geDIG v6 (pre-print)](docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf)

**Patent**: JP 2025-082988, JP 2025-082989

**License**: Apache 2.0

**Contact**: miyauchikazuyoshi@gmail.com

> *All theoretical contributions and experimental design are by the author. Implementation is AI-assisted (Claude, GitHub Copilot).*
