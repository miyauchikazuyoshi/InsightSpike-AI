# InsightSpike-AI — geDIG: Giving AI the Power to Have Insights

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![CI (Unit)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://miyauchikazuyoshi.github.io/InsightSpike-AI)

---

## How did Einstein, an amateur independent researcher, discover relativity?

In 1905, Einstein was a patent clerk—an outsider to academic physics. Yet he unified electromagnetism and classical mechanics, two contradictory theories.

**What did he do?**

Our hypothesis: He recognized **structural patterns** across different theories and discovered the **minimal transformation** (Lorentz transformation) that made them isomorphic. This is the computational essence of "insight."

```
Electromagnetism ─┬─→ Contradiction ←─┬─ Classical Mechanics
                  │                   │
                  └────→ T* ←────────┘
                         │
                         ▼
                 T* = Lorentz Transformation
                 (minimal edit to make structures isomorphic)
```

**geDIG** is a computational model that formalizes this "insight."

---

## The Core Equation

```
T* = argmin_T GED(T(G₁), G₂)
```

> Find the minimal transformation T* that makes two knowledge structures isomorphic.
> That transformation IS the insight.

### Hypothesis About Human Cognition

The human brain treats **structural similarity** and **probabilistic relevance** as equivalent:

```
"A and B have similar structure"
    ↓ brain converts this to
"A and B are probably related"
```

geDIG quantifies this transformation:

```
F = ΔEPC_norm − λ · ΔIG
```

| Term | Meaning | Cognitive Correspondence |
|------|---------|-------------------------|
| ΔEPC_norm | Cost of structural change | "How much to change your thinking" |
| ΔIG | Information gain | "How much understanding improves" |
| F | Unified gauge | "Value of insight" (smaller = better) |

---

## Demo: The Moment AI Has an Insight

### Automatic Discovery of Novel Analogies

From 16 different domains (physics, biology, psychology, arts...), the algorithm automatically discovered surprising analogies:

| Discovery | Structural Meaning |
|-----------|-------------------|
| **Revolution ≈ Emotion** | Social revolution as "collective emotional response" |
| **Compiler ≈ Rock Cycle** | Code transformation and geological metamorphism as "staged transformation" |
| **Immune System ≈ Information Diffusion** | Virus infection and viral marketing share the same structure |
| **Gene Expression ≈ Learning** | Same structure at molecular and cognitive levels |

```bash
# Try it yourself
poetry run python experiments/isomorphism_discovery/novel_analogy_discovery.py
```

**Nobody taught these. The algorithm discovered them from structure alone.**

---

## Three Levels of "Insight"

```
Level 3: Isomorphism Discovery ─────────────────────
         T* = argmin_T GED(T(G₁), G₂)
         "Finding the transformation that resolves contradiction"
         Example: Einstein's relativity
                    │
Level 2: Analogy Detection ─────────────────────────
         SS(G₁, G₂) > θ
         "Structural correspondence across domains"
         Example: Bohr's atomic model (solar system ≈ atom)
                    │
Level 1: Pattern Matching ──────────────────────────
         sim(a, b) = cos(φ(a), φ(b))
         "Element-level similarity"
         Example: Standard RAG search
```

geDIG is a unified framework covering all three levels.

---

## Validation Results

### Reproducing Scientific History

| Discovery | Year | Structural Similarity | Detected by geDIG |
|-----------|------|----------------------|-------------------|
| Bohr's Atomic Model | 1913 | 0.995 | ✓ |
| Kekulé's Benzene Ring | 1865 | 0.967 | ✓ |
| Darwin's Natural Selection | 1859 | 0.985 | ✓ |

### Cross-Domain QA

| Condition | F1 Score |
|-----------|----------|
| Without structural similarity | 0.062 |
| With structural similarity | **0.660** |
| **Improvement** | **+60%** |

### Scalability

| Nodes | Processing Time |
|-------|-----------------|
| 100 | 32ms |
| 500 | 1.6s |
| 1000 | 5.5s |

---

## Connection to Molecular Design AI

Drug discovery AI uses molecular graph edit distance to find "molecules with the same efficacy but different structure" (Scaffold Hopping).

geDIG uses the same mathematics to find "theories with the same explanatory power but different structure."

```
Molecular Design AI              geDIG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Molecular graph       ←→        Knowledge graph
Molecular edit dist.  ←→        GED
Same efficacy, diff.  ←→        Same explanatory power,
  structure                       different theory
Scaffold Hopping      ←→        Theory Unification
```

**The algorithm that discovers new drugs can discover new theories.**

---

## Quick Start

```bash
# Setup
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
poetry install

# Novel analogy discovery
poetry run python experiments/isomorphism_discovery/novel_analogy_discovery.py

# Basic isomorphism discovery test
poetry run python -c "
from src.insightspike.algorithms.isomorphism_discovery import discover_insight, create_test_graphs

solar, atom, _ = create_test_graphs()
transform = discover_insight(solar, atom)
print(transform)
print(transform.to_insight_description())
"
```

---

## Applications

### Currently Working

- **RAG Optimization**: Autonomous "when to search" decisions (EM +3.5pt on HotpotQA)
- **Maze Exploration**: Efficient search in partially observable environments (98% success on 15×15)
- **Analogy Detection**: Cross-domain structural similarity discovery (F1 +60%)
- **Isomorphism Discovery**: Minimal transformation between knowledge structures (1000 nodes)

### Future Vision

- **Scientific Discovery Support**: Finding unexpected cross-domain connections from papers
- **Education**: Auto-generating "X is like Y" explanations
- **Creative Support**: Cross-domain transfer of narrative structures
- **Drug Discovery Integration**: Sharing technology with molecular design AI

---

## Technical Details

### Unified Gauge

```
F = ΔEPC_norm − λ · ΔIG

ΔEPC_norm: Normalized edit path cost (structural change)
ΔIG = ΔH_norm + γ · ΔSP_rel (information gain)
  - ΔH_norm: Entropy change
  - ΔSP_rel: Shortest path improvement
```

### Two-Stage Gates (AG/DG)

```
AG (Attention Gate): 0-hop ambiguity detection → exploration trigger
DG (Decision Gate): Multi-hop stability check → decision trigger
```

### Theoretical Background

- **Free Energy Principle (FEP)**: The brain minimizes "surprise"
- **Minimum Description Length (MDL)**: The best hypothesis is the most compressible
- **Graph Edit Distance (GED)**: Minimal cost of structural transformation

geDIG operationally bridges these frameworks.

---

## Papers

### Main Paper
- [geDIG: Unified Gauge Control for Dynamic Knowledge Graphs](docs/paper/geDIG_onegauge_improved_v6.pdf)

### Planned Submissions
- JSAI 2026: HotpotQA benchmark + computational model of insight
- Independent paper: "Graph Edit Distance as a Computational Model of Scientific Insight"

---

## Collaboration Wanted

For the ambitious goal of "computational model of insight":

| Role | Contribution |
|------|-------------|
| **Cognitive Scientist** | Validating correspondence with human insight |
| **Molecular Design AI Researcher** | Technical integration with Scaffold Hopping |
| **Theoretical Physicist** | Mathematical rigor for FEP-MDL bridge |
| **ML Engineer** | Validation on large-scale knowledge graphs |

**Contact**: miyauchikazuyoshi@gmail.com / X: @kazuyoshim5436

---

## License

Apache-2.0

## Patents

- JP 2025-082988, 2025-082989 (pending)

---

> "Just as drug discovery AI searches for molecular isomorphs,
>  geDIG searches for theoretical isomorphs.
>  The edit operations that achieve this ARE the essence of insight."
