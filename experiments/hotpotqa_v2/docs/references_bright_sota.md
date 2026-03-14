# BRIGHT Benchmark — SOTA Reference (as of 2026-03)

## Benchmark Overview

- **BRIGHT**: Reasoning-Intensive Document Retrieval (ICLR 2025, XLang NLP Lab)
- 1,398 queries across 12 domains (economics, psychology, math, coding, etc.)
- Metric: nDCG@10
- Paper: https://arxiv.org/abs/2407.12883
- Leaderboard: https://brightbenchmark.github.io/
- GitHub: https://github.com/xlang-ai/BRIGHT

## Leaderboard Top Methods (short documents, nDCG@10)

| Rank | Method | nDCG@10 | Training | Organization |
|------|--------|:--:|:--:|------|
| 1 | INF-X-Retriever | 63.4 | ? | (paper unpublished?) |
| 2 | DIVER-v3-GroupRank | 46.8 | Yes | Ant Group, Sun Yat-sen Univ |
| 3 | BGE-Reasoner-0928 | 46.4 | Yes | USTC, BUPT, BAAI |
| 4 | Lattice Hierarchical | 42.1 | **No** | UT Austin, UCLA, Google |
| 5 | ReasonRank | 40.8 | Yes | RUC, Baidu, CMU |
| 6 | XRR2 | 40.3 | Yes | |
| 7 | RaDeR + Qwen reranking | 39.2 | Yes | |
| 8 | ReasonIR + Rank-R1 | 38.8 | Yes | |
| 9 | ReasonIR + TongSearch | 38.3 | Yes | |
| 10 | BGE-Reasoner-Embed | 38.1 | Yes | |
| ~80 | BM25 baseline | 14.5 | No | |

Note: All top-10 entries use a second-stage reranker.

## Per-Domain nDCG@10 (Biology / Economics / StackOverflow)

| Method | Biology | Economics | StackOverflow |
|--------|---------|-----------|---------------|
| DIVER v2 | 68.0 | 42.0 | 44.3 |
| LATTICE | 51.6 | 62.4 | 47.6 |
| Rank1-7B (rerank BM25 top-100) | 48.8 | 20.8 | 18.7 |
| Rank1-32B (rerank BM25 top-100) | 49.7 | 22.0 | 21.7 |
| BM25 baseline (LATTICE paper) | 34.8 | 54.1 | 18.9 |
| ReasonIR-8B | 33.1 | 42.9 | 20.9 |
| MonoT5-3B | 16.0 | 17.7 | 10.5 |
| RankLLaMA-13B | 21.6 | 16.3 | 7.7 |

**重要**: 標準的な cross-encoder (MonoT5, RankLLaMA) は BRIGHT で非常に悪い。
Reasoning-aware reranker (Rank1) は 2-3x 良い。

## Key Methods Detail

### DIVER (2nd, nDCG@10 = 46.8)

- Paper: https://arxiv.org/abs/2508.07995
- 4-stage pipeline:
  1. **Document Preprocessing**: noise cleaning + segmentation
  2. **Query Expansion**: LLM iterative refinement (2 rounds, top-5 docs/round)
  3. **Retrieval**: fine-tuned Qwen3-Embedding-4B + BM25 (50/50 weight)
  4. **Reranking**: pointwise (Qwen-2.5-32B, 0-10 scale) + listwise (Deepseek-R1), combined 0.6/0.4
- Without query expansion: nDCG@10 = 31.9 (vs 46.8 with)
- Key: query expansion is critical (+47%)

### LATTICE (4th, nDCG@10 = 42.1) — Training-Free

- Paper: https://arxiv.org/abs/2510.13217
- **Training-free / zero-shot** — most relevant comparison for our approach
- Method:
  1. **Offline**: semantic tree construction (agglomerative clustering + multi-level summaries)
  2. **Online**: LLM navigates tree with logarithmic search complexity (~250 docs evaluated/query)
  3. **Calibration**: latent relevance score estimation from noisy LLM judgments
- Achieves +9% Recall@100, +5% nDCG@10 over next best zero-shot baseline

### BGE-Reasoner (3rd, nDCG@10 = 46.4)

- Three-stage: Rewrite, Embed, Rerank
- Rewriter: LLM-based query rewriting for reasoning
- Retriever: BGE-Reasoner-Embed (fine-tuned from Qwen3-8B) + BM25, top-2000
- BM25 + Embed fusion weights: 0.75 / 0.25
- Reranker: BGE-Reasoner-Reranker for fine-grained scoring

### Rank1 (used in leaderboard ranks 8-9)

- Paper: https://arxiv.org/abs/2502.18418
- First reranker trained on reasoning traces (600k+ R1 traces from MS-MARCO)
- Reranks BM25 top-100 candidates
- Key: test-time compute via reasoning traces
- Rank1-7B on biology: 48.8 (vs RankLLaMA-13B: 21.6, MonoT5-3B: 16.0)

### ReasonIR-8B

- Paper: https://arxiv.org/abs/2504.20595
- Dense retriever trained for reasoning tasks (LLaMA3.1-8B base)
- Without reranker: 29.9, with reranker: 36.9

## Our Position (InsightSpike-AI)

| Method | nDCG@10 | Training | Eval | Note |
|--------|:--:|:--:|------|------|
| DIVER-v3 (SOTA-2) | 46.8 | Yes | 12 domains | Query exp + fine-tuned + rerank |
| LATTICE (best zero-shot) | 42.1 | No | 12 domains | LLM + semantic tree |
| Rank1-7B (rerank only) | 48.8 | Yes | biology only | Reasoning-trace reranker |
| **Our Spec H (best)** | **0.2496** | **No** | **50q biology** | geDIG refine + CoT + graph |
| Our Spec A (baseline) | 0.2438 | No | 50q biology | BM25 + CoT + entity graph |
| Our 323q (full) | 0.1520 | No | 323q (3 dom) | Classic scoring |

### Gap Analysis

- vs LATTICE (zero-shot): 0.25 vs 0.42 = 60% of target
- vs Rank1 (biology): 0.25 vs 0.49 = 51% of target
- vs BM25 baseline (LATTICE paper): 0.25 vs 0.35 (biology)

Note: BM25 baseline varies by implementation (我々の 50q subset vs LATTICE の full biology set)

### Key Differences from Top Methods

1. **No neural reranker**: Cross-encoder は BRIGHT で ineffective (MonoT5: 16.0)
   だが reasoning reranker (Rank1: 48.8) は非常に effective
2. **No query expansion**: DIVER の nDCG が expansion なしで 31.9→46.8 (+47%)
3. **Small candidate pool**: top-100 vs BGE-Reasoner の top-2000
4. **LLM**: gpt-4o-mini vs stronger models (Qwen-2.5-32B, Deepseek-R1)

### Actionable Insights (Spec J/K 設計用)

1. **LLM Pointwise Scoring** (DIVER style)
   - 0-10 scale で各文書をスコアリング
   - CoT context を prompt に含める (我々独自の強み)
   - gpt-4o-mini で十分 (reasoning capability がある)
   - 期待: biology 0.35-0.45

2. **Query Decomposition** (all top methods で使用)
   - 複雑なクエリを sub-questions に分解
   - 各 sub-question で独立に BM25 検索
   - Recall 改善 (70% missing gold の一部を回収)
   - 期待: +20-30% recall

3. **Larger Candidate Pool**
   - BM25 top-100 → top-500
   - CoT re-retrieval top-50 → top-200
   - 期待: recall 改善
