# Multi-hop QA Experiments

Two independent experiment lines in this directory:

1. **v2/v3 (HotpotQA)**: geDIG + Betti numbers + dual-process architecture
2. **v10 (MuSiQue)**: Entity-graph guided paragraph reordering

---

## v10: Entity-Graph Guided Paragraph Reordering (MuSiQue)

### Result

**統計的に有意な改善は確認できず。** ただし複数の再利用可能な知見を得た。

**500q フルラン (GPT-4o, MuSiQue distractor setting):**

| 条件 | EM | F1 | vs Baseline |
|------|----|----|-------------|
| Baseline A (全20パラ + CoT) | 47.4% | 0.614 | ref |
| v10d reorder_only | 48.6% | 0.620 | +1.2pt (有意でない) |

ホップ別では 4-hop に +5.3pt の傾向があるが N=76 で有意に達せず。

### 何をやったか

MuSiQue (20パラ/問題、2-4 hop) において、エンティティグラフから推論チェーンを
特定し、関連パラグラフを先頭に配置して LLM の注意力を暗黙的に誘導する試み。

**アプローチの進化 (v9 → v10d):**

| Version | アプローチ | 結果 (50q) |
|---------|-----------|-----------|
| v9 | タイトル相互参照 + ヒントテキスト | EM=60% (-2pt) |
| v10a | パラレベルグラフ + shortest path | EM=58-60% |
| v10b | 3シグナル混合エッジ | EM=56-60% |
| v10c | 文レベル Three-Tier エッジ分離 | EM=52-62% |
| v10d | v10c + チェーン拡張 | EM=68% (+6pt) → **500q で +1.2pt に縮小** |

### 確立された知見

| 知見 | 確実度 | 根拠 |
|------|--------|------|
| **Guided テキストは GPT-4o に害** | ★★★★★ | 全バージョン全条件で悪化 |
| **暗黙的誘導 > 明示的指示** | ★★★★★ | 一貫した結果 |
| **reorder_only は唯一 "損をしない" 介入** | ★★★★☆ | 500q で最悪 -0.4pt |
| **弱いモデルには reorder が逆効果** | ★★★★☆ | GPT-4o-mini で -12pt |
| 20パラ (~2,500 tokens) では "Lost in the Middle" が発生しない | ★★★★★ | Gold パラ位置とエラーに相関なし |
| エラーの 45% は推論の誤り (distractor entity 選択) | ★★★★★ | 500q エラー分析 |

### 主な教訓

1. **50q スモークテストは方向性のスクリーニングにしかならない** — 95% CI ≈ ±13pt
2. **GPT-4o の非決定性で同じ50問が ±8pt 揺れる** — N=50 での数値は無意味
3. **GPT-4o + 2,500 tokens はパラ位置に無関心** — コンテキストウィンドウの 2% では注意力の偏りが生じない
4. **Baseline のエラーは「情報が見えない」ではなく「推論を間違える」** — reorder で直る問題ではない

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/report_v10_entity_graph.md](docs/report_v10_entity_graph.md) | 完全な実験レポート (12セクション) |
| [docs/experiment_design_v10.md](docs/experiment_design_v10.md) | 実験設計書 |
| [src/entity_graph.py](src/entity_graph.py) | エンティティグラフ + チェーン抽出コア |
| [scripts/run_allcontext.py](scripts/run_allcontext.py) | 実験ランナー (baseline_a / v10_reorder_only / v10_pruned) |

### 次の打ち手 (未実施)

| # | 打ち手 | コスト | 期待値 |
|---|--------|-------|--------|
| 1 | フォーマット修正 (プロンプト改良) | ~$8 | +5~16pt (80問のフォーマット不一致を回収) |
| 2 | Oracle reorder テスト | ~$8 | reorder 仮説を正式に棄却/確認 |
| 3 | ~~ディストラクタ増量 (50-100パラ)~~ | — | **v11 で実施** |
| 4 | 推論改善にピボット | ~$8-20 | 質問分解、self-consistency 等 |

---

## v11: Pre-computed Topology Routing (MuSiQue)

### 仮説

コンテキストを 20 -> 50 パラに拡大して "Lost in the Middle" が発生する条件を作り、
事前構築グラフ + F 値ルーティングで性能劣化を回復できるか検証。

### アーキテクチャ

- **Offline**: 全パラから sentence-level 三層グラフを事前構築 (entity_graph.py 再利用)
- **Online**: クエリからサブグラフ抽出 -> F 値計算 -> System 1/System 2 ルーティング

```
Offline (問題ごとに 1 回):
  全 50 パラ -> sentence-level グラフ構築 -> beta_0, beta_1, centrality 事前計算

Online (クエリごと):
  質問 -> エンティティ抽出 -> グラフノードマッチ -> k-hop サブグラフ抽出
       -> F 値計算 -> System 1 (サブグラフのみ) / System 2 (全パラ、サブグラフ先頭)
```

### 実験条件

| ID | 条件 | パラ数 | 手法 |
|----|------|--------|------|
| B_20 | baseline_a | 20 | Plain CoT (既存結果) |
| B_50 | baseline_a | 50 | Plain CoT |
| V11_S | v11_subgraph | 50 | サブグラフのみ |
| V11_R | v11_routing | 50 | F 値ルーティング |

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/experiment_design_v11.md](docs/experiment_design_v11.md) | 実験設計書 |
| [src/corpus_graph.py](src/corpus_graph.py) | 事前グラフ + サブグラフ抽出 + F 値ルーティング |
| [scripts/build_scaled_data.py](scripts/build_scaled_data.py) | 50 パラデータ生成 |
| [scripts/run_allcontext.py](scripts/run_allcontext.py) | 実験ランナー (v11 モード追加) |

---

## v12: Open-World Topology-Guided Retrieval (FRAMES / BRIGHT)

### 仮説

geDIG の β₀-driven 反復検索で、Wikipedia (FRAMES) や大規模コーパス (BRIGHT) から
マルチホップ推論に必要な記事/文書を自動的に発見できるか検証。

v11 (コンテキストエンジニアリング) から真の RAG への移行:
- **β₀ > 1** → 情報ギャップを検出 → Wikipedia API で橋渡し記事を検索
- **F 値収束** → 検索の自然な停止条件
- **Component Gap Query (v8)** → ブリッジ検索クエリの自動生成

### アーキテクチャ

```
Question → Entity Extraction → Wikipedia Search (Initial)
                                       ↓
                              Entity Graph Construction
                                       ↓
                                   β₀ check
                                    ↓    ↓
                              β₀ = 1  β₀ > 1
                              (done)  (gap detected)
                                        ↓
                              Component Gap Query (LLM)
                                        ↓
                              Bridge Wikipedia Search
                                        ↓
                              Graph Reconstruction → β₀ check (repeat)
                                       ↓
                              Subgraph-first Context → LLM Answer
```

### FRAMES ベンチマーク

- 824 問: 2-11 Wikipedia 記事を要するマルチホップ推論
- 推論タイプ: Multiple constraints, Numerical, Temporal, Tabular

### BRIGHT ベンチマーク

- 1,384 クエリ: 推論集約型文書検索 (12 ドメイン, 1.33M 文書)
- BM25 = 14.5, SOTA = 63.4 nDCG@10

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/experiment_design_v12.md](docs/experiment_design_v12.md) | 実験設計書 |
| [src/wiki_retriever.py](src/wiki_retriever.py) | Wikipedia API 検索 + テキスト取得 |
| [src/open_world_pipeline.py](src/open_world_pipeline.py) | β₀-driven 反復検索パイプライン |
| [scripts/run_frames.py](scripts/run_frames.py) | FRAMES 実験ランナー |

---

## v2/v3: geDIG with Betti Numbers and Dual-Process Architecture (HotpotQA)

### Key Result

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

| Document | Experiment | Description |
|----------|-----------|-------------|
| [docs/experiment_design_v11.md](docs/experiment_design_v11.md) | v11 (MuSiQue) | Pre-computed Topology Routing 実験設計書 |
| [docs/report_v10_entity_graph.md](docs/report_v10_entity_graph.md) | v10 (MuSiQue) | Entity-graph 実験レポート (仮説の生死判定、根本原因診断含む) |
| [docs/experiment_design_v10.md](docs/experiment_design_v10.md) | v10 (MuSiQue) | v10 実験設計書 |
| [SPEC.md](SPEC.md) | v2 (HotpotQA) | Formal experiment specification |
| [DESIGN_v3_improvements.md](DESIGN_v3_improvements.md) | v3 (HotpotQA) | v3 improvement design and System 1/2 architecture |
| [REPORT_v3_dual_process.md](REPORT_v3_dual_process.md) | v3 (HotpotQA) | Full v3 experiment report with 11-method comparison |
