# geDIG in 5 Minutes

> **One sentence**: geDIG is a unified gauge that tells a system **when** to accept new information, by balancing structural cost against information gain.

---

## The Problem (30 sec)

AI systems are good at finding **what** to retrieve. But they're terrible at deciding **when** to update their knowledge.

```
Current AI:
  "Here are 10 relevant documents!"
  → But 6 of them are noise
  → And the knowledge graph becomes a mess
```

**Result**: Knowledge pollution, redundant searches, unstable behavior.

---

## The Intuition (1 min)

Humans do this naturally. When you hear new information, you ask:

1. **"Does this fit what I know?"** → Structure cost
2. **"Does this help me understand?"** → Information gain

Example: Someone says "The Earth is flat"
- Information gain: Small (explains horizon... kind of)
- Structure cost: HUGE (contradicts everything)
- **Decision: Reject**

Example: Someone says "Water boils at 100°C at sea level"
- Information gain: Reduces uncertainty about cooking
- Structure cost: Low (fits existing physics knowledge)
- **Decision: Accept**

**geDIG makes this explicit and computable.**

---

## The Formula (1 min)

```
F = Structure Cost − λ × Information Gain
```

Or more precisely:

```
F = ΔEPC_norm − λ·(ΔH_norm + γ·ΔSP_rel)
```

| Term | Meaning | Intuition |
|------|---------|-----------|
| ΔEPC | Edit-path cost | "How much do I have to change?" |
| ΔH | Entropy difference | "How much uncertainty is reduced?" |
| ΔSP | Path shortening | "Does this create a useful shortcut?" |
| λ | Temperature | Balance between caution and curiosity |

**Rule: Smaller F = Better update**

---

## The Mechanism (1 min)

Two gates control the decision:

### AG (Ambiguity Gate) — "Should I explore more?"
- Fires when local structure looks wrong
- Triggers: more retrieval, deeper search

### DG (Decision Gate) — "Is this a good connection?"
- Fires when multi-hop evaluation confirms a shortcut
- Triggers: commit the update, prune bad branches

```
New Info → [AG: Explore?] → [DG: Accept?] → Update Graph
              ↓                   ↓
           "Hmm, uncertain"    "Yes, this helps!"
```

---

## The Results (1 min)

### Maze (Proof of Concept)
| Method | Success Rate | Steps | Compression |
|--------|--------------|-------|-------------|
| Random | 45% | 210 | 0% |
| Greedy | 92% | 85 | 0% |
| **geDIG** | **98%** | **69** | **95%** |

geDIG finds the goal efficiently AND builds a minimal map.

### RAG (HotPotQA)
| Method | EM | F1 | Latency |
|--------|-----|-----|---------|
| BM25 | 36.6% | 52.3% | 820ms |
| **geDIG** | **37.5%** | **53.8%** | 873ms |

geDIG improves accuracy with minimal overhead.

---

## Why This Matters (30 sec)

geDIG is not just an algorithm. It's a **design principle**:

> **Balance structure and information to decide when to change.**

This principle appears everywhere:
- 🧠 Brains (learning vs. stability)
- 🌱 Cells (growth vs. integrity)
- 🏢 Organizations (innovation vs. process)
- 🤖 AI systems (exploration vs. exploitation)

If this principle is fundamental, then:
- We can build **self-updating AI** that knows when to learn
- We can design **dynamic Transformers** that adapt during inference
- We can create systems with **intrinsic motivation** (no external reward needed)

---

## Try It

```python
from insightspike import geDIG

controller = geDIG(lambda_weight=1.0)

decision = controller.evaluate(
    current_graph=my_graph,
    candidate=new_node
)

if decision.accept:
    my_graph.add(new_node)
```

---

## Learn More

- [Intuitive Guide (No Math)](intuition.md)
- [Full Paper (arXiv)](../paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
- [Interactive Playground](../../examples/playground.py)
- [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI)

---

*geDIG: Graph Edit Distance + Information Gain*
*A unified gauge for dynamic knowledge graphs*
