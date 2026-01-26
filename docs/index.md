---
layout: default
title: geDIG - Structure IS Probability
---

<div align="center">

<br>

# Structure IS Probability

<h2 style="font-weight: normal; color: #666;">
What if structural optimization and probabilistic inference<br>
are the same operation?
</h2>

<br>

<div style="font-size: 1.4em; font-family: 'Times New Roman', serif; letter-spacing: 0.05em;">

**F = ΔEPC − λΔIG**

</div>

<p style="color: #888; font-size: 0.9em;">
One equation. From mazes to multi-hop QA.
</p>

<br>

[Paper](paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf) ・ [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) ・ [Demo](https://huggingface.co/spaces/miyaukaz/gedig-demo)

---

</div>

<br>

## The Insight

<div align="center">

```
Selecting by STRUCTURE          Selecting by PROBABILITY
        │                               │
   Graph Topology               Distribution over States
   Nodes & Edges                    P(x)
   Edge Cost (EPC)                  Entropy (H)
        │                               │
        └───────────┬───────────────────┘
                    │
            F-minimization
            governs both
                    │
                    ▼
        Selection by structure
                 =
        Selection by probability
```

</div>

<br>

**This equivalence explains:**

| Phenomenon | Structure View | Probability View |
|:-----------|:---------------|:-----------------|
| Learning | Graph rewiring | Distribution update |
| Insight | Structural isomorphism | Sudden entropy collapse |
| Forgetting | Edge pruning | Probability decay |
| Intuition | Consolidated path (FFN) | High-confidence shortcut |

<br>

---

## Why This Matters

<br>

<div align="center">

| Domain | Application | Status |
|:-------|:------------|:-------|
| **AI** | Adaptive RAG, Dynamic Gating | ✅ Demonstrated |
| **Cognition** | Episodic memory, Insight | ✅ Modeled |

</div>

<br>

We believe this framework may extend to other domains (neuroscience, economics), but these are **hypotheses to be tested**, not claims.

<br>

---

## The Mechanism: AG/DG Gating

Two gates govern all intelligent processing:

<div align="center">

```
Input
  │
  ▼
┌─────────────────────────┐
│  AG (Ambiguity Gate)    │
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

<br>

**AG** = What to attend to (Precision weighting)

**DG** = What to learn (Model selection)

<br>

---

## Proof: It Works

<br>

<div align="center">

| Experiment | Domain | Result |
|:-----------|:-------|:-------|
| **Dynamic Gating** | 15×15 Maze | 60% → **98%** success |
| **Adaptive RAG** | HotpotQA | **+3.5pt** Exact Match |
| **Analogy Discovery** | Cross-domain KG | **+60%** F1 |

</div>

<br>

All three experiments use **the same F-gauge**. No task-specific tuning.

<br>

---

## Theoretical Connections

<br>

<div align="center">

| Framework | Mapping to geDIG |
|:----------|:-----------------|
| **FEP** (Free Energy Principle) | Model complexity ↔ EPC, Prediction error ↔ IG |
| **MDL** (Minimum Description Length) | Code length ↔ EPC, Compression ↔ IG |
| **Helmholtz Free Energy** | F = U − TS ↔ F = ΔEPC − λΔIG |
| **Active Inference** | Precision weighting ↔ AG/DG gating |

</div>

<br>

geDIG provides an **operational bridge** — turning abstract principles into computable graph algorithms.

<br>

---

## Get Started

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
poetry install

# Run the maze experiment
make reproduce-maze15

# Run HotpotQA benchmark
make reproduce-hotpotqa
```

<br>

---

## Resources

<br>

| Type | Link |
|:-----|:-----|
| 📄 **Paper (English)** | [geDIG v6 (PDF)](paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf) |
| 📄 **Paper (日本語)** | [geDIG v6 (PDF)](paper/geDIG_onegauge_improved_v6.pdf) |
| 💻 **GitHub** | [InsightSpike-AI](https://github.com/miyauchikazuyoshi/InsightSpike-AI) |
| 🎮 **Demo** | [HuggingFace Space](https://huggingface.co/spaces/miyaukaz/gedig-demo) |
| 💬 **Discussion** | [Active Inference Institute Discord](https://discord.gg/activeInference) |

<br>

---

## Open Questions

We are exploring whether this framework extends beyond AI:

| Domain | Hypothesis | Status |
|:-------|:-----------|:-------|
| Cognition | Episodic memory as AG/DG graph construction | Testing |
| Neuroscience | Synaptic plasticity follows F-gradient | Hypothesis |
| Economics | Market equilibria as F-minima | Hypothesis |

These are **open research questions**, not claims. We welcome critical feedback.

<br>

---

## Collaboration

We are looking for collaborators to test and challenge the **"Structure = Probability"** hypothesis:

| Role | Contribution |
|:-----|:-------------|
| 🧠 **Cognitive Scientists** | Human insight experiments |
| 🔬 **Neuroscientists** | Synaptic plasticity validation |
| 💻 **ML Engineers** | Large-scale experiments |
| 🔍 **Critics** | Find where this breaks down |

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

> *"Structure and probability are two views of the same reality.*
> *F-minimization is the lens that unifies them."*

<br>

**Apache-2.0** | Patent Pending (JP 2025-082988, 2025-082989)

<br>

---

<br>

**geDIG** — *Where Structure Meets Probability.*

<br>

</div>
