# InsightSpike-AI

**A structural fitness score for knowledge graphs — can one equation decide when to restructure?**

$$\mathcal{F} = \underbrace{\Delta \text{EPC}}_{\text{Metric}} \;-\; \lambda \left( \underbrace{\Delta H}_{\text{Measure}} \;+\; \gamma\, \underbrace{\Delta \beta_1}_{\text{Topology}} \right)$$

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
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

## Project Status

This is an **active research project** by an individual researcher with AI-assisted implementation.
It is not a production library.

| Component | Status | Location |
|-----------|--------|----------|
| geDIG theory (v6 paper) | Pre-print available | [`docs/paper/`](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf) |
| HotpotQA multi-hop QA (v3) | **11-method benchmark complete** | [`experiments/hotpotqa_v2/`](experiments/hotpotqa_v2/) |
| MuSiQue entity-graph reorder (v10) | **500q 完了 — 有意差なし、知見確立** | [`experiments/hotpotqa_v2/docs/report_v10_entity_graph.md`](experiments/hotpotqa_v2/docs/report_v10_entity_graph.md) |
| MuSiQue Pre-computed Topology Routing (v11) | **Phase 1 実験準備中** | [`experiments/hotpotqa_v2/docs/experiment_design_v11.md`](experiments/hotpotqa_v2/docs/experiment_design_v11.md) |
| Maze navigation (Phase 2) | Prototype complete | [`experiments/maze/`](experiments/maze/) |
| Transformer F decomposition | Exploratory (8+ models) | [`experiments/transformer/`](experiments/transformer/inference_gedig_v2/) |
| Flash-geDIG (attention scorer) | Functional | [`src/insightspike/gedig/`](src/insightspike/gedig/) |
| Visual proof (matchstick figure) | Interactive HTML | [EN](https://miyauchikazuyoshi.github.io/InsightSpike-AI/research/thinking/matchstick_figure_v2_en.html) / [JA](https://miyauchikazuyoshi.github.io/InsightSpike-AI/research/thinking/matchstick_figure_v2.html) |

---

## Experiments

### Maze Navigation

A partial-observation maze agent that builds a persistent knowledge graph and uses geDIG to decide when to explore vs. exploit.

- **Architecture**: Wake-Sleep-Wake cycle with three-layer search (L0: O(1) hash, L1: O(degree) attention walk, L2: O(N log N) full sort)
- **15x15 maze**: 98% goal-reach rate (baseline ~60%)
- **25x25 maze**: Active experimentation with graph-persistent DG and 10D vector extension

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

**Status**: Preliminary. $\Delta R^2_{\text{struct}}$ is still negative for all models (random-init outperforms baseline), but trending toward zero with model quality. Large-scale model verification (70B+) is future work.

See [`experiments/transformer/inference_gedig_v2/`](experiments/transformer/inference_gedig_v2/) for experiment design and results.

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

- **Model-dependent interaction**: On GPT-4o, Hybrid-E1 **leads IRCoT (EM 51.2% vs 47.6%) at 3.6x fewer LLM calls**
- **Scaling property**: Topology-guided routing improves with model capability (+6pt from mini→4o vs -3pt for IRCoT)

**500-question evaluation (primary reference):**

| Model | Hybrid-E1 EM | IRCoT EM | LLM Calls | p-value |
|:-----:|:---:|:---:|:---------:|:-------:|
| **GPT-4o** | **51.2%** | 47.6% | **2.2 vs 8** | 0.086 |
| GPT-4o-mini | 45.2% | **50.4%** | 2.2 vs 8 | **0.008** |

See [`experiments/hotpotqa_v2/`](experiments/hotpotqa_v2/) for full experiment code and [`REPORT_v3_dual_process.md`](experiments/hotpotqa_v2/REPORT_v3_dual_process.md) for the detailed report.

### Earlier Experiments

Cross-domain analogy experiments were conducted in earlier phases and informed the theory. These have not been reproduced under the current codebase and are archived.

---

## Quick Start

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

For formal definitions, see [`docs/gedig_spec.md`](docs/gedig_spec.md). For the full paper, see the [v6 pre-print (PDF)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf).

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
- geDIG v6 paper: [`docs/paper/arxiv_v6_en/`](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
- Canonical spec: [`docs/gedig_spec.md`](docs/gedig_spec.md)

**Each term of $\mathcal{F}$ is grounded in independent prior work**:

| $\mathcal{F}$ term | Prior work |
|---------------------|-----------|
| $\Delta\text{EPC}$ (metric) | Hewitt & Manning (2019), [A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/) |
| $\Delta H$ (measure) | Ali et al. (2025), [Entropy-Lens: The Information Signature of Transformer Computations](https://arxiv.org/abs/2502.16570) |
| $\Delta\beta_1$ (topology) | Oyama et al. (2025), [Mapping 1,000+ Language Models via the Log-Likelihood Vector](https://aclanthology.org/2025.acl-long.1584/) — ACL Outstanding Paper |

**Related work on structure and interpretability**:
- Gao et al. (2025), [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653) — sparsification as $\beta_1$ reduction

---

## Citation, License, and Patent

**Paper**: [geDIG v6 (pre-print)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)

**Patent**: JP 2025-082988, JP 2025-082989

**License**: Apache 2.0

**Contact**: miyauchikazuyoshi@gmail.com

> *All theoretical contributions and experimental design are by the author. Implementation is AI-assisted (Claude, GitHub Copilot).*
