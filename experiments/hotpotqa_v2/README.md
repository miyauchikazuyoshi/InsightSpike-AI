# HotpotQA v2/v3: geDIG with Betti Numbers and Dual-Process Architecture

## Key Result

**Hybrid-E1 v3.1** — topology-guided System 1/System 2 switching with **model-dependent scaling**:

**500-question evaluation (primary reference):**

| Model | Hybrid-E1 EM | IRCoT EM | Δ | p-value | LLM Calls |
|:-----:|:---:|:---:|:-:|:------:|:---------:|
| **GPT-4o** | **51.2%** | 47.6% | **+3.6pt** | 0.086 | **2.2 vs 8** |
| GPT-4o-mini | 45.2% | **50.4%** | -5.2pt | **0.008** | 2.2 vs 8 |

Key findings:
- On GPT-4o, Hybrid-E1 **leads IRCoT at 3.6x fewer LLM calls**
- On GPT-4o-mini, IRCoT is significantly better — but Hybrid-E1 achieves 90% quality at 27% cost
- **Model scaling favors topology**: +6pt improvement (mini→4o) vs -3pt for IRCoT
- The gauge value F decides when to think fast (System 1) vs. slow (System 2) — **zero-cost routing**

> See [REPORT_v3_dual_process.md](REPORT_v3_dual_process.md) for the full experiment report.

---

## Overview

This experiment tests two hypotheses:

1. **(v2)** Adding **Betti number** (topological) terms to the geDIG gauge improves multi-hop QA performance
2. **(v3)** Using the gauge value as a **cognitive routing signal** (System 1/System 2 dual-process) improves quality-cost trade-off

### Extended Gauge Formula

```
F = ΔEPC_norm − λ·(ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀)
```

- **ΔEPC** (metric): Cost of restructuring the knowledge graph
- **ΔH** (measure): Change in entropy / uncertainty
- **Δβ₁** (topology): Penalizes redundant cycle formation
- **Δβ₀** (topology): Rewards island merging (bridge question signal)

### v3 Dual-Process Architecture

```
Question → BM25 Retrieval → Build Knowledge Graph → Compute F (topology)
                                                         |
                                              ┌──────────┴──────────┐
                                         F < θ_dg              F >= θ_dg
                                        (confident)            (uncertain)
                                              |                      |
                                       ┌──────┴──────┐      ┌───────┴───────┐
                                       │  System 1    │      │   System 2     │
                                       │  Direct (1x) │      │   CoT (2-3x)  │
                                       └──────────────┘      └───────────────┘
```

---

## Full Results (GPT-4o-mini, 100 questions)

| Rank | Method | Category | EM | F1 | LLM Calls |
|:----:|--------|----------|:---:|:---:|:---------:|
| 1 | **Hybrid-E1 v3.1** | **geDIG+CoT** | **48.0%** | **0.622** | **~2.2** |
| 2 | IRCoT | Dynamic RAG | 46.0% | 0.637 | ~8 |
| 3 | GraphRAG | Static RAG | 43.0% | 0.589 | 1 |
| 4 | Hybrid-E1 v3.0 | geDIG+CoT | 40.0% | 0.600 | ~2.2 |
| 5 | geDIG-B | geDIG | 40.0% | 0.570 | 1 |
| 6 | Hybrid(B) | geDIG+CoT | 39.0% | 0.572 | ~1.5 |
| 7 | ReAct | Dynamic RAG | 39.0% | 0.536 | ~7 |
| 8 | E1-tuned | geDIG | 38.0% | 0.553 | 1 |
| 9 | geDIG-C | geDIG | 38.0% | 0.553 | 1 |
| 10 | geDIG-A | geDIG | 37.0% | 0.545 | 1 |
| 11 | geDIG-D | geDIG | 37.0% | 0.544 | 1 |
| 12 | BM25 | Baseline | 37.0% | 0.536 | 1 |
| 11 | ReAct | Dynamic RAG | 39.0% | 0.536 | ~7 |

---

## Experimental Conditions

### v2 Conditions (Betti number ablation)

| Condition | Config | structural_mode | gamma_0 | gamma_1 | Description |
|-----------|--------|-----------------|:-------:|:-------:|-------------|
| A | condition_a_sp.yaml | sp | 0 | 0 | v1 reproduction (no Betti) |
| B | condition_b_beta1.yaml | betti | 0 | 1.0 | beta_1 only (best v2) |
| C | condition_c_beta0.yaml | betti_full | 1.0 | 0 | beta_0 only |
| D | condition_d_betti_full.yaml | betti_full | 1.0 | 1.0 | Full Betti |

### v3 Conditions (Dual-process + tuning)

| Condition | Config | gamma_0 | gamma_1 | theta_dg | hybrid | Description |
|-----------|--------|:-------:|:-------:|:--------:|:------:|-------------|
| E1-tuned | condition_e1_tuned.yaml | 0.3 | 0.5 | -0.5 | no | Tuned params only |
| Hybrid(B) | condition_hybrid.yaml | 0 | 1.0 | 0.0 | yes | Untuned + CoT |
| **Hybrid-E1** | **condition_hybrid_e1.yaml** | **0.3** | **0.5** | **-0.5** | **yes** | **Tuned + CoT (best)** |

### Baselines

| Method | Implementation | Description |
|--------|---------------|-------------|
| BM25 | baselines/bm25_gpt.py | BM25 retrieval + GPT-4o-mini |
| GraphRAG | baselines/static_graphrag.py | Entity-overlap graph + centrality ranking |
| IRCoT | baselines/ircot.py | Interleaving Retrieval with CoT (Trivedi+ 2023) |
| ReAct | baselines/react_baseline.py | Reason + Act loop (Yao+ 2023) |

---

## Quick Start

```bash
# 1. Download data
PYTHONPATH=src .venv/bin/python3 experiments/hotpotqa_v2/scripts/download_data.py

# 2. Run smoke test (mock LLM, 10 examples)
LLM_PROVIDER=mock PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_experiment.py \
    --config experiments/hotpotqa_v2/configs/condition_hybrid_e1.yaml \
    --limit 10

# 3. Run full experiment (requires OPENAI_API_KEY)
set -a && source .env && set +a && \
PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_experiment.py \
    --config experiments/hotpotqa_v2/configs/condition_hybrid_e1.yaml \
    --data experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl \
    --output experiments/hotpotqa_v2/results/real_condition_hybrid_e1

# 4. Run baselines
PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_baseline.py --baseline ircot \
    --data experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl \
    --output experiments/hotpotqa_v2/results/real_baseline_ircot

# 5. Compare conditions
python experiments/hotpotqa_v2/tools/compare_conditions.py \
    experiments/hotpotqa_v2/results/*/summary.json
```

## Tests

```bash
.venv/bin/python3 -m pytest experiments/hotpotqa_v2/test/ -v
```

## Directory Structure

```
hotpotqa_v2/
├── SPEC.md                        # Formal experiment specification
├── README.md                      # This file
├── DESIGN_v3_improvements.md      # v3 improvement design document
├── REPORT_v3_dual_process.md      # v3 experiment report
├── configs/                       # YAML configs for all conditions
├── src/                           # Core modules
│   ├── adapter.py                 # geDIG v2/v3 adapter (extended F + hybrid mode)
│   ├── answerer.py                # Shared LLM handler (mock/GPT-4o-mini)
│   ├── config.py                  # YAML config loader
│   ├── data_loader.py             # HotpotQA data loading
│   ├── evaluator.py               # EM/F1/SF-F1 metrics (type-stratified)
│   ├── graph_builder.py           # beta_0-sensitive knowledge graph construction
│   └── retriever.py               # BM25 retrieval module
├── baselines/                     # BM25, GraphRAG, IRCoT, ReAct baselines
│   ├── base.py                    # BaseRAG interface
│   ├── bm25_gpt.py                # BM25 + GPT baseline
│   ├── static_graphrag.py         # Static GraphRAG baseline
│   ├── ircot.py                   # IRCoT baseline (Trivedi+ 2023)
│   └── react_baseline.py          # ReAct baseline (Yao+ 2023)
├── scripts/                       # Experiment runners
│   ├── run_experiment.py          # geDIG condition runner
│   ├── run_baseline.py            # Baseline runner
│   ├── analyze_results.py         # Results analysis
│   └── tune_gamma.py              # Parameter tuning
├── tools/                         # Post-hoc analysis tools
├── test/                          # Unit tests (35 tests)
├── data/                          # Dataset files (.gitignored)
└── results/                       # Experiment outputs (.gitignored)
```

## Documents

| Document | Description |
|----------|-------------|
| [SPEC.md](SPEC.md) | Formal experiment specification (v2) |
| [DESIGN_v3_improvements.md](DESIGN_v3_improvements.md) | v3 improvement design and System 1/2 architecture |
| [REPORT_v3_dual_process.md](REPORT_v3_dual_process.md) | Full v3 experiment report with 11-method comparison |
