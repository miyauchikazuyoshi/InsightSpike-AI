# Solving RAG's "When to Update" Problem with geDIG

*A unified gauge for deciding when to accept new information*

---

RAG (Retrieval-Augmented Generation) has made massive progress on the **"what to retrieve"** problem. BM25, DPR, Contriever, ColBERT — retrieval accuracy improves every year.

But have you ever had this experience?

```
User: "What is the population of Tokyo?"

RAG: "Retrieved 5 relevant documents!"
  → Doc 1: Tokyo's population is 14 million (2023)  ← Correct
  → Doc 2: Tokyo's area is 2,194 km²               ← Noise
  → Doc 3: Tokyo is Japan's capital                ← Noise
  → Doc 4: Tokyo population trends                 ← Redundant
  → Doc 5: Tokyo Olympics 2020                     ← Noise

Result: The answer is in there, but so is a lot of noise
→ LLM context gets polluted
→ Answer quality degrades
```

The problem? There's no principled way to decide **"when"** to accept new information.

## What is geDIG?

geDIG (Graph Edit Distance + Information Gain) is a **unified gauge** for making this "when" decision.

The core formula:

```
F = Structural Cost − λ × Information Gain
```

- **Small F** → Good update (low cost, high gain)
- **Large F** → Bad update (high cost, low gain)

## Intuitive Understanding

Think about what you do when you hear new information.

**Example 1**: Someone says "The Earth is flat"
- Information gain: Small (kind of explains the horizon...)
- Structural cost: **Huge** (contradicts everything you know)
- Decision: **Reject** 🙅

**Example 2**: Someone says "Tokyo has 14 million people"
- Information gain: Large (directly answers the question)
- Structural cost: Small (fits existing knowledge)
- Decision: **Accept** 🙆

geDIG makes this judgment computable.

## Two-Stage Gating

geDIG has two gates:

### AG (Ambiguity Gate) — "Should I search more?"

Fires when local structure looks uncertain.

```python
if g0 > theta_AG:
    # Things look ambiguous, let's search more
    expand_search()
```

### DG (Decision Gate) — "Is this a good connection?"

Confirms via multi-hop evaluation whether this really creates a shortcut.

```python
if g_min < theta_DG:
    # Confident this helps, accept it
    commit_to_graph()
```

## Real Results

### HotPotQA (Multi-hop QA)

| Method | EM | F1 | Latency (ms) |
|--------|-----|-----|-------------|
| BM25 | 36.6% | 52.3% | 820 |
| **geDIG** | **37.5%** | **53.8%** | 873 |
| Diff | +2.4% | +2.9% | +6.5% |

On 7,405 questions, geDIG beats BM25 with only 53ms additional latency.

### Maze Navigation (Proof of Concept)

A toy environment where "knowledge graph updates" become "maze exploration":

| Method | Success | Steps | Map Compression |
|--------|---------|-------|-----------------|
| Random | 45% | 210 | 0% |
| Greedy | 92% | 85 | 0% |
| **geDIG** | **98%** | **69** | **95%** |

geDIG finds the goal efficiently AND builds a minimal map by **discarding unnecessary information**.

## Try It Yourself

### Installation

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
pip install -e .
```

### Minimal Example

```python
from insightspike import create_agent

# Create agent (uses mock LLM by default)
agent = create_agent(provider="mock")

# Process a question
result = agent.process_question("What is geDIG?")
print(result.response)
```

### See geDIG's Decision Making

```python
from insightspike.algorithms.gedig_core import GedigCore

# Initialize geDIG core
core = GedigCore(
    lambda_weight=1.0,  # Balance structure vs information
    gamma=1.0,          # Weight for path shortening
)

# Evaluate an update
result = core.calculate(
    graph_before=current_graph,
    graph_after=graph_with_new_node,
    linkset_info=linkset,
)

print(f"F-score: {result.f_score}")
print(f"Structural cost: {result.epc_norm}")
print(f"Information gain: {result.ig_norm}")

if result.f_score < threshold:
    print("→ Accept this update")
else:
    print("→ Reject this update")
```

## Why This Matters

geDIG isn't just an algorithm — it's a **design principle**:

> Balance structure and information to decide when to change.

This principle might apply to:
- 🧠 How brains decide what to remember
- 🌱 How cells decide when to divide
- 🏢 How organizations decide when to change

If this principle is fundamental, we might be able to design AI that **knows when it should learn** — without external reward signals.

## Try the Demo

**Try it in your browser right now!**

👉 **[geDIG Demo on Hugging Face](https://huggingface.co/spaces/miyauchikazuyoshi/gedig-demo)**

- Experience geDIG's decision-making on sample questions
- Visualize BM25 vs geDIG comparison
- See F-score, AG/DG states in real-time

## Learn More

- **[GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI)** — Code and documentation
- **[geDIG in 5 Minutes](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes.md)** — Quick overview
- **[Paper (arXiv)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)** — Full theory
- **[Interactive Playground](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/examples/playground.py)** — Run locally

## Summary

| Problem | geDIG's Solution |
|---------|-----------------|
| When to search? | AG (Ambiguity Gate) decides |
| When to accept? | DG (Decision Gate) decides |
| How to filter noise? | Reject high F-score information |
| What's the criterion? | Structural Cost − λ × Information Gain |

From **"what to retrieve"** to **"when to update"**.

geDIG provides one answer.

---

*Questions or feedback? Open an issue on [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) or reach out on Twitter [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)!*
