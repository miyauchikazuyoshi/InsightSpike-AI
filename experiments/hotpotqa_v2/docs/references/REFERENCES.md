# External references (cited papers)

The PDF copies previously vendored in this directory were removed to keep the
repository lean and to avoid redistributing third-party papers. Retrieve them
directly from arXiv:

| Local name | arXiv ID | Link |
|---|---|---|
| IRCoT — Interleaving Retrieval with Chain-of-Thought Reasoning | 2212.10509 | https://arxiv.org/abs/2212.10509 |
| FLARE — Active Retrieval Augmented Generation | 2305.06983 | https://arxiv.org/abs/2305.06983 |
| ITER-RETGEN — Iterative Retrieval-Generation Synergy | 2305.15294 | https://arxiv.org/abs/2305.15294 |
| Self-RAG — Retrieve, Generate, and Critique through Self-Reflection | 2310.11511 | https://arxiv.org/abs/2310.11511 |
| Topologies of Reasoning (chains / trees / graphs of thought) | 2401.14295 | https://arxiv.org/abs/2401.14295 |
| Topo-RAG (topology-aware RAG) | 2405.17602 | https://arxiv.org/abs/2405.17602 |
| BRIGHT — Reasoning-Intensive Retrieval benchmark | 2407.12883 | https://arxiv.org/abs/2407.12883 |
| DIVER (reasoning-intensive retrieval) | 2508.07995 | https://arxiv.org/abs/2508.07995 |
| Lattice (retrieval) | 2510.13217 | https://arxiv.org/abs/2510.13217 |
| TDA-in-NLP survey | 2411.10298 | https://arxiv.org/abs/2411.10298 |

To fetch all of them locally (not tracked by git):

```bash
cd "$(dirname "$0")" 2>/dev/null || cd experiments/hotpotqa_v2/docs/references
for id in 2212.10509 2305.06983 2305.15294 2310.11511 2401.14295 \
          2405.17602 2407.12883 2508.07995 2510.13217 2411.10298; do
  curl -sL -o "${id}.pdf" "https://arxiv.org/pdf/${id}"
done
```
