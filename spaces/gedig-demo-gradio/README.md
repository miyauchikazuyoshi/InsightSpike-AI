---
title: geDIG Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
---

# geDIG Demo

**geDIG** (Graph Edit Distance + Information Gain) is a unified gauge for deciding **when** to accept new information in RAG systems.

## The Core Idea

```
F = Structural Cost − λ × Information Gain
```

- **Small F** → Good update (accept)
- **Large F** → Bad update (reject)

## Features

- Interactive demo with sample HotPotQA questions
- Visualize geDIG's decision-making process
- See how AG (Ambiguity Gate) and DG (Decision Gate) work

## Results (HotPotQA, 7,405 questions)

| Method | EM | F1 |
|--------|-----|-----|
| BM25 | 36.6% | 52.3% |
| **geDIG** | **37.5%** | **53.8%** |

## Learn More

- [GitHub Repository](https://github.com/miyauchikazuyoshi/InsightSpike-AI)
- [Paper (arXiv)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/)
- [5-Minute Guide](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes.md)

## Contact

- Twitter: [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)
- Email: miyauchikazuyoshi@gmail.com
