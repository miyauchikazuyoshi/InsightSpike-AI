---
layout: default
title: geDIG — A Structural Fitness Score for Knowledge Graphs
---

<div align="center">

<br>

# $\mathcal{F} = \Delta\text{EPC} - \lambda(\Delta H + \gamma \Delta \beta_1)$

<h2 style="font-weight: normal; color: #666;">
A structural fitness score for knowledge graphs —<br>
can one equation decide when to restructure?
</h2>

<br>

| Term | Mathematical Structure | What It Captures |
|:-----|:----------------------|:-----------------|
| **ΔEPC** | **Metric** (distance) | Cost of restructuring |
| **ΔH** | **Measure** (probability) | Change in entropy |
| **Δβ₁** | **Topology** (loops) | Change in independent cycles |

<p style="color: #666; font-size: 0.95em; margin-top: 1em;">
Three independent mathematical structures. One dimensionless, scale-invariant scalar.<br>
<strong>F &lt; 0</strong> means information gain exceeds structural cost — the system should commit the change.
</p>

<br>

[Paper (PDF)](paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf) ・ [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) ・ [Demo](https://huggingface.co/spaces/miyaukaz/gedig-demo) ・ [Interactive Visualizer](demo.html)

---

</div>

<br>

## Visual Intuition: The Matchstick Figure

Why are these three terms independent? The [Matchstick Figure](research/thinking/matchstick_figure_v2_en.html) provides an interactive visual explanation.

For the same edit cost (EPC = 1), topology (Δβ₁) and information (ΔH) vary independently — one move can create a loop without changing entropy, or reduce entropy without changing topology. This independence is why F needs all three terms.

[**View the Matchstick Figure (EN)**](research/thinking/matchstick_figure_v2_en.html) ・ [**(JA)**](research/thinking/matchstick_figure_v2.html)

<br>

---

## Origin

> *"How do we build an AI that thinks like Einstein?"*

From this question came geDIG: the hypothesis that **insight is the topological reconstruction of memory**, and that it can be measured as a thermodynamic quantity — isomorphic to Helmholtz free energy F = E − TS.

The equation evolved through research:

```
ΔIG (undifferentiated gain)
  ↓  decompose
ΔH + γΔSP (SP = average shortest path — graph-dependent)
  ↓  abstract
ΔH + γΔβ₁ (β₁ = first Betti number — topological invariant)
```

The SP → β₁ abstraction was the decisive leap: from an operational definition that depends on graph shape, to a structural definition that exists wherever space exists.

[**Read the full origin story**](research/gedig_origin_story.md) (EN / JA)

> *Note: The v6 paper uses ΔSP (shortest-path shortening) as the structural term. The ongoing research generalizes this to Δβ₁, which is a topological invariant independent of graph shape or scale.*

<br>

---

## The Mechanism: AG/DG Gating

Two gates govern intelligent processing:

<div align="center">

```
Input
  │
  ▼
┌─────────────────────────┐
│  AG (Attention Gate)    │
│  "Is this surprising?"  │
│                         │
│  Prediction ≠ Input     │
│  → Fire                 │
└───────────┬─────────────┘
            │ Yes
            ▼
      [ Process ]
            │
            ▼
┌─────────────────────────┐
│  DG (Decision Gate)     │
│  "Does F decrease?"     │
│                         │
│  ΔF < 0 → Update        │
│  ΔF ≥ 0 → Reject        │
└───────────┬─────────────┘
            │ Yes
            ▼
   [ Structure Update ]
```

</div>

**AG** (Attention Gate): Detects ambiguity/novelty (0-hop). Analogous to noradrenaline.

**DG** (Decision Gate): Confirms valid restructuring (multi-hop). Analogous to dopamine.

> *The neurotransmitter correspondence is a computational analogy, not a physiological claim.*

<br>

---

## Experiments

### Maze Navigation

A partial-observation maze agent that builds a persistent knowledge graph and uses geDIG to decide when to explore vs. exploit.

| Experiment | Result |
|:-----------|:-------|
| **15×15 Maze** (Dynamic Gating) | 60% → **98%** goal-reach rate |
| **25×25 Maze** (Graph-Persistent DG) | Active experimentation with 10D vector extension |

Architecture: Wake-Sleep-Wake cycle with three-layer search (L0: O(1) hash, L1: O(degree) attention walk, L2: O(N log N) full sort).

```bash
# Reproduce (requires .venv with networkx, numpy, etc.)
.venv/bin/python3 experiments/maze/run_experiment_query.py \
  --maze-size 15 --max-steps 250 --seeds 12 \
  --search-mode threelayer --vector-mode extended
```

### Transformer Inference F-Trajectory

Layer-by-layer measurement of ΔEPC, ΔH, and Δβ₁ across Transformer hidden states, testing whether F tracks model quality.

- **8 token-level models tested**: BERT, DistilBERT, GPT-2, GPT-2 Medium, DistilGPT2, TinyLlama (2 checkpoints)
- **Key finding**: GPT series shows monotonic improvement in ΔR²_struct with model scale

```
distilgpt2:    ΔR²_struct = -0.777
gpt2:          ΔR²_struct = -0.086
gpt2-medium:   ΔR²_struct = -0.065
```

F has sensitivity to model quality — better models produce more structured F-trajectories.

**Status**: Preliminary. ΔR²_struct is still negative for all models (random-init outperforms baseline), but trending toward zero with scale. Large-scale model verification (70B+) is future work.

### Earlier Experiments

HotpotQA (multi-hop QA) and cross-domain analogy experiments were conducted in earlier phases and informed the theory. These have not been reproduced under the current codebase and are archived.

<br>

---

## Theoretical Connections

<div align="center">

| Framework | Mapping to geDIG |
|:----------|:-----------------|
| **Helmholtz Free Energy** | F = E − TS ↔ F = ΔEPC − λ(ΔH + γΔβ₁) |
| **FEP** (Free Energy Principle) | Model complexity ↔ EPC, Prediction error ↔ IG |
| **MDL** (Minimum Description Length) | Code length ↔ EPC, Compression ↔ IG |
| **Active Inference** | Precision weighting ↔ AG/DG gating |

</div>

Each term of F is grounded in independent prior work:

| F term | Prior work |
|:-------|:-----------|
| **ΔEPC** (metric) | Hewitt & Manning (2019), [A Structural Probe for Finding Syntax](https://aclanthology.org/N19-1419/) |
| **ΔH** (measure) | Ali et al. (2025), [Entropy-Lens](https://arxiv.org/abs/2502.16570) |
| **Δβ₁** (topology) | Oyama et al. (2025), [Mapping 1,000+ Language Models](https://aclanthology.org/2025.acl-long.1584/) — ACL Outstanding Paper |

Related: Gao et al. (2025), [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653) — sparsification as β₁ reduction in F.

<br>

---

## Open Questions

We are exploring specific, actionable research questions:

1. **Independent F-trajectory reproduction** — Does F show structured layer-wise behavior on models we haven't tested?

2. **β₁ vs SP as structural term** — Under what conditions does topological Δβ₁ outperform metric ΔSP? Expertise in topological data analysis (TDA) welcome.

3. **Scaling to 70B+ models** — We hypothesize that (λ, γ) converge across model families at sufficient scale. Verification requires GPU resources beyond our current individual setup.

4. **F-regularization robustness** — The v6 paper shows weak F-regularization improves downstream performance (+0.33pt on SST-2). Is this robust across tasks and model families?

These are **open research questions**, not claims. We welcome critical feedback.

<br>

---

## Resources

| Type | Link |
|:-----|:-----|
| **Paper (English)** | [geDIG v6 (PDF)](paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf) |
| **Paper (Japanese)** | [geDIG v6 (PDF)](paper/geDIG_onegauge_improved_v6.pdf) |
| **GitHub** | [InsightSpike-AI](https://github.com/miyauchikazuyoshi/InsightSpike-AI) |
| **Live Demo** | [HuggingFace Space](https://huggingface.co/spaces/miyaukaz/gedig-demo) |
| **Interactive Visualizer** | [geDIG Graph Demo](demo.html) |
| **Matchstick Figure** | [EN](research/thinking/matchstick_figure_v2_en.html) / [JA](research/thinking/matchstick_figure_v2.html) |
| **Canonical Spec** | [gedig_spec.md](gedig_spec.md) |

<br>

---

<div align="center">

<br>

## Contact

**Email**: miyauchikazuyoshi@gmail.com

**X (Twitter)**: [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)

<br>

---

<br>

**Apache-2.0** | Patent: JP 2025-082988, JP 2025-082989

> *All theoretical contributions and experimental design are by the author. Implementation is AI-assisted (Claude, GitHub Copilot).*

<br>

</div>
