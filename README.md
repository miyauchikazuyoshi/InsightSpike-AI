# InsightSpike-AI
**The Thermodynamic Engine for Intelligence**

> **"Chaos cannot be intelligent. Only structure can."**
> 
> *Are you training models to decrease loss, or to create structure?*

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://miyauchikazuyoshi.github.io/InsightSpike-AI)

## ⚡ What is InsightSpike?

InsightSpike is a high-velocity, GPU-native library that computes the **"Structural Fitness" (geDIG F-score)** of neural networks and knowledge graphs.

It turns the abstract concept of **"Free Energy Minimization"** into a differentiable, actionable engineering metric.

### 🚀 Core Product: Flash-geDIG

A PyTorch-native library to compute structural metrics in milliseconds.

*   **Zero-Copy / All-Tensor**: Runs entirely on GPU without CPU sync.
*   **End-to-End Differentiable**: Use F-score as a loss function.
*   **Metrics**:
    *   **Entropy ($H$)**: Concentration of attention.
    *   **Edge Processing Cost ($EPC$)**: Sparsity of connections.
    *   **Shortest Path ($SP$)**: Global information integration efficiency.
    *   **Clustering ($C$)**: Semantic coherence (Triangle count).

```python
from insightspike.gedig import compute_structural_fitness

# Input: Attention Matrix (Batch, Heads, Seq, Seq)
# Output: Structural Fitness Score (F)
fitness, metrics = compute_structural_fitness(attention_map)
```

---

## 🏆 Verified Benchmarks (Proven Results)

We have rigorously verified the impact of geDIG in the following domains:

| Experiment | Domain | Metric | Result | Reproduction |
|---|---|---|---|---|
| **Dynamic Gating** | 15x15 Maze Navigation | Success Rate | **98%** (vs Baseline 60%) | `make reproduce-maze15` |
| **Adaptive RAG** | HotpotQA (Multi-hop) | Exact Match | **+3.5pt** | `make reproduce-hotpotqa` |
| **Analogy Finding** | Cross-Domain KG | F1 Score | **+60%** | `make reproduce-analogy` |

> *See [Experiments Overview](docs/experiments/index.md) for detailed reports and logs.*

---

## 🧪 Current Research (Beta)

We are actively exploring new applications of the thermodynamic engine:

*   **Structure-Guided RAG**: Re-ranking retrieved documents by their internal structural coherence to reduce hallucinations. (See `experiments/rag_reranking/`)
*   **Neuro-Pruning**: Identifying and removing "thermodynamically entropic" attention heads in Transformers. (See `experiments/neuro_pruning/`)

---

## 📜 Theory: The Universal Principle

At its core, geDIG is based on the hypothesis that Intelligence acts to balance **Information Gain** (Novelty) and **Structural Cost** (Energy).

$$F = \underbrace{\Delta \text{Structure Cost}}_{\text{Energy}} - \lambda \cdot \underbrace{\Delta \text{Information Gain}}_{\text{Entropy}}$$

*   **Low F (Low Free Energy)** = Stable, Efficient, Low Entropy (Understanding).
*   **High F (High Free Energy)** = Unstable, Costly, High Entropy (Confusion).

---

## 🤝 Join the Revolution

We are looking for collaborators to test the **"Thermodynamics of Intelligence"**.

*   **Transformers**: Can we train a model *only* using F-score?
*   **Neuroscience**: Does the brain maximize Clustering like geDIG does?
*   **Biology**: Do protein folding pathways follow the geDIG gradient?

**Contact**: miyauchikazuyoshi@gmail.com

---
*Licensed under Apache 2.0*
