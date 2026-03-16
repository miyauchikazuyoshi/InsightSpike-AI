# Multi-hop QA Experiments

Four experiment lines in this directory:

1. **v2/v3 (HotpotQA)**: geDIG + Betti numbers + dual-process architecture
2. **v10 (MuSiQue)**: Entity-graph guided paragraph reordering
3. **v11 (MuSiQue)**: Pre-computed topology routing (50-paragraph)
4. **v12 (FRAMES / BRIGHT)**: Open-world topology-guided retrieval ← **active**

---

## v12 BRIGHT: Reasoning-Intensive Document Retrieval (Active)

### 目標

nDCG@10 = **0.45** (現在ベスト: bio50q=0.3181, full323q=0.1898)

BRIGHT ベンチマーク (ICLR 2025) の 3 ドメイン (biology, economics, stackoverflow) で、
geDIG ベースのグラフ re-ranking パイプラインの性能を検証・改善する。

### ベンチマーク概要

- **BRIGHT**: 1,384 クエリ, 12 ドメイン, 1.33M 文書
- 推論集約型 — 標準的な検索モデルは大幅に性能低下
- BM25 baseline = 14.5, SOTA (INF-X-Retriever) = 63.4 nDCG@10
- Leaderboard: https://brightbenchmark.github.io/

### パイプライン アーキテクチャ

```
Phase 0:  [Optional] Query decomposition (LLM → 3-5 sub-questions) [Spec K]
Phase 1:  BM25 initial retrieval (top-100)
Phase 1a: [Optional] Sub-query BM25 retrieval (50/sub-query → ~250 new) [Spec K]
Phase 1b: [Optional] Dense pool expansion (E5-base-v2 + FAISS) [Spec I]
Phase 2:  LLM CoT reasoning → entity extraction
Phase 2.5: CoT re-retrieval (BM25 + optional Dense)
Phase 2.6: [Optional] RIA iterative expansion (β₀-gated, max 3 rounds) [Spec M]
Phase 3:  Entity graph construction (sentence-level, three-tier edges)
Phase 4:  CoT node injection (virtual nodes → bridge edges)
Phase 4.5: [Optional] Per-document token graph scoring (spaCy dep parse + DG/AG walk) [Spec N/N.1]
Phase 5:  Scoring
  ├─ scoring_mode="classic"      → 5-component (PageRank+Entity+Token+Degree+CoT bridge)
  ├─ scoring_mode="gedig"        → pure geDIG (MessagePassing + EdgeReevaluation)
  └─ scoring_mode="gedig_refine" → geDIG graph refinement + classic scoring ★best
Phase 5.5: [Optional] Token graph blend (external) [Spec N]
Phase 6:  Combined ranking: α·BM25 + (1-α)·graph_score
Phase 7:  [Optional] LLM listwise rerank
Phase 7b: [Optional] LLM pointwise reasoning rerank (gpt-4o-mini) [Spec J]
Phase 7c: [Optional] LLM reasoning rerank (gpt-4o, 1doc/call) [Spec L]
```

### 実験結果サマリー (Biology 50q, α=0.1)

| Spec | Configuration | nDCG@10 | R@10 | MRR | Note |
|------|---------------|---------|------|-----|------|
| A | Classic (CoT re-retrieval) | 0.2438 | 0.2173 | 0.3740 | 5成分 baseline |
| H | geDIG refine | **0.2496** | **0.2419** | **0.4183** | ★ ベスト |
| H | geDIG (pure) | 0.1130 | 0.1629 | 0.1436 | CoT bridge なし |
| I | Classic + Dense (full) | 0.1797 | 0.1950 | 0.2342 | -26% 劣化 |
| I | geDIG refine + Dense (full) | 0.1509 | 0.1995 | 0.1969 | -38% 劣化 |
| I | Classic + Dense (pool-only) | 0.2341 | 0.1969 | 0.4125 | ≈ baseline |
| I | geDIG refine + Dense (pool-only) | 0.2297 | 0.2276 | 0.3648 | ≈ baseline |
| J | PW rerank blend=0.2 | 0.2410 | 0.2299 | 0.3867 | -3.4% |
| J | PW rerank blend=0.4 | 0.1971 | 0.1847 | 0.3189 | -21.0% |
| J | PW rerank blend=0.6 | 0.2313 | 0.2430 | 0.3499 | -7.3% |
| K | Query decomposition | 0.1978 | 0.1680 | 0.3457 | -1.3% (68 new gold 発見) |
| L | gpt-4o reasoning rerank (w=0.7) | 0.2342 | 0.2159 | 0.3756 | -6.2% (pointwise の限界) |
| M | RIA iterative expansion | 0.2564 | 0.2661 | 0.3815 | +2.7% (38 new gold via RIA) |
| N | Token graph (Spec N only) | 0.2544 | 0.2486 | 0.3917 | +1.9% (independent ranking signal) |
| M+N | RIA + Token Graph | 0.2707 | 0.2643 | 0.3972 | +8.5% (相乗効果) |
| N.1 | Walk Score (dg=2.0, のみ) | 0.2238 | 0.2158 | 0.3489 | -10.3% (RIA なしでは逆効果) |
| **M+N.1** | **RIA + Walk Score (dg=2.0)** | **0.3181** | **0.3139** | **0.4424** | **★ ベスト +27.4% (Wake-Sleep-Wake)** |
| O | Entity F-eval | — | — | — | entity-feval 単体テスト未実施 |
| P | Multi-CoT Ensemble (N=3) | 0.0829 | 0.1005 | 0.1211 | **-32% 劣化** (DG信号が発生しない) |
| P+O | Multi-CoT Ensemble + Entity F-eval | 0.0788 | 0.0960 | 0.1183 | **-35% 劣化** |

#### Spec P 詳細結果 (v19, 3ドメイン×50q = 150q)

| # | 構成 | nDCG@10 | Bio | Econ | SO | 備考 |
|---|------|---------|-----|------|----|------|
| C1a | baseline N=1 run1 | **0.1184** | 0.165 | 0.100 | 0.090 | ベースライン |
| C1b | baseline N=1 run2 | **0.1258** | 0.191 | 0.091 | 0.095 | CoT非決定性で6.3%ぶれ |
| C2 | ensemble N=3 | 0.0829 | 0.133 | 0.055 | 0.061 | **-32%** (平均化で劣化) |
| C3 | ensemble N=3 + feval | 0.0788 | 0.120 | 0.055 | 0.062 | **-35%** |

**Spec P の教訓**:
- temperature=0.7 の3本CoTでは agreement=0.97~1.00 → **DG信号が一切発生しない**
- Ensemble 平均化がスコアを平坦化し、正解文書のランクが下がる
- LLM は同じクエリに対して似た推論をするため、multi-CoT で多様性を得にくい
- **結論: Spec P は保留。他のアプローチ（Retrieval Recall 改善）を優先**

### Full 323q 結果 (3 domains, α=0.1)

| Configuration | Biology | Economics | StackOverflow | Overall |
|---------------|---------|-----------|---------------|---------|
| Spec A (classic) | 0.1879 | 0.1240 | 0.1470 | 0.1520 |
| Spec H (geDIG refine) | 0.2069 | 0.1187 | 0.1296 | 0.1508 |
| **M+N.1 (RIA + Walk Score)** | **0.2574** | **0.1402** | **0.1739** | **0.1898** |

**M+N.1 vs Spec H**: Biology +24%, Economics +18%, StackOverflow +34%, **Overall +26%** 🔥

### Spec 進行状況

| Spec | 内容 | 状態 | 結果 |
|------|------|------|------|
| A | CoT re-retrieval + entity graph | ✅ 完了 | nDCG=0.152 (323q) |
| B-D | Adaptive routing, LLM rerank | ✅ 完了 | 微改善 (0.152→0.160) |
| E | geDIG routing (tier selection) | ✅ 完了 | 効果なし |
| F-G | Episode graph, Hybrid graph | ✅ 完了 | 効果なし |
| H | geDIG scoring (graph refinement) | ✅ 完了 | ★ベスト biology +10% (0.2496) |
| I | Dense retrieval integration | ✅ 完了 | 改善なし (-4%～-38%, 構造的限界) |
| J | Pointwise LLM reranking | ✅ 完了 | 改善なし (-3%～-21%, gpt-4o-mini 限界) |
| K | Query decomposition | ✅ 完了 | ≈中立 (-1.3%, recall改善/ranking未活用) |
| L | Stronger LLM Reranking (gpt-4o) | ✅ 完了 | 改善なし (-6.2%, pointwise の限界) |
| M | RIA Iterative Expansion | ✅ 完了 | ★ **初の正改善** +2.7% (nDCG=0.2564, R@10+10%) |
| N | Token-level Graph Scoring | ✅ 完了 | M+Nで +8.5% (nDCG=0.2707, 独立ranking信号) |
| **N.1** | **geDIG Walk Score** | ✅ 完了 | **★ M+N.1で最高 +27.4%** (nDCG=0.3181, Wake-Sleep-Wake) |
| O | Entity F-eval scoring | ✅ 完了 | P との組合せで検証 |
| P | Multi-CoT Ensemble (DG/AG) | ✅ 完了 | **-32% 劣化** (保留: CoT間の多様性不足) |

### 学んだこと (Spec I-N.1 の統一的教訓)

Spec I-L の 4 つの negative result → Spec M+N+N.1 で劇的改善:

- **Pointwise reranking は BRIGHT に不適** — gpt-4o-mini (J) も gpt-4o (L) も改善せず
- **候補プール拡張 (I, K)** は gold recovery に成功するが、scoring で活かせない
- **反復的クエリ拡張 (M, RIA)** が Recall 改善 (+2.7% nDCG, +10% Recall)
  - 38 個の新 gold 文書を発見 (19/50 クエリ)
- **Token-level graph (N)** が Ranking 改善 (+1.9% 単体)
  - BM25 と完全に独立したランキング信号 (Spearman ρ ≈ -0.38)
- **geDIG Walk Score (N.1)** — DG/AG エッジ分類で構造的接続品質を評価
  - 単体では逆効果 (-10.3%) — RIA なしでは gold doc がプールにない
  - RIA 併用で **+27.4%** (0.2496 → 0.3181) — **Wake-Sleep-Wake アーキテクチャ**
- **M+N.1 相乗効果**: **+27.4% nDCG** (0.2496 → 0.3181)
  - RIA (Wake/探索) → Walk Score (Sleep/DG/AG分類) → proximity (Wake/確認)
  - 迷路実験と同じ原理: 先に探索しないとループ検知は無意味

### ボトルネック分析

2 つの独立したボトルネックを特定:

1. **Retrieval Recall**: Gold docs の 70-80% が BM25 top-100 に不在
   - Dense retrieval (Spec I, E5-base-v2) は追加 gold ≈ 0 (biology)
   - Query decomposition (Spec K) は 68 new gold / 50q — **recall 改善に成功**
   - 理論上限 nDCG ≈ 0.60 (現在のプールで完璧にランキングした場合)
2. **Ranking Quality**: プール内 gold の配置が理論上限の 35%
   - 現在 0.21 / 理論上限 0.60 = 35%
   - gpt-4o-mini pointwise reranking (Spec J) は改善なし
   - gpt-4o reasoning reranking (Spec L) も改善なし — **pointwise アプローチ自体が BRIGHT に不適**
   - 56% のクエリで gold が BM25 top-100 に 0 件 → reranking では解決不能

**結論**: Pointwise reranking はモデル強度に関わらず BRIGHT に不適 (J, L で確認)。
RIA (Spec M) は recall を改善 (+10%) し、初の正の改善を達成 (+2.7%)。
Spec N.1 (geDIG Walk Score) で **ranking quality を大幅改善** — nDCG 0.2496 → **0.3181** (+27.4%)。

### Quick Start (BRIGHT)

**注意**: `answerer.py` が `.env` を自動読み込み (`python-dotenv`) するため、
`export OPENAI_API_KEY=...` は不要。プロジェクトルートの `.env` に `OPENAI_API_KEY=sk-...` があれば OK。

```bash
# 1. データ準備 (初回のみ)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/prepare_bright.py

# 2. Dense index 構築 (初回のみ, ~10分)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/build_dense_index.py \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output-dir experiments/hotpotqa_v2/data/bright/dense_index/

# 3. Smoke test (10q, biology, geDIG refine)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/smoke_test \
    --limit 10 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine

# 4. 50q biology (geDIG refine)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_gedig_refine_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine

# 5. 50q biology with Query Decomposition (Spec K)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_qd_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --query-decomp --query-decomp-top-k 50 --query-decomp-max-sub 5

# 6. 50q biology with Reasoning Reranking (Spec L, gpt-4o)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_reasoning_rerank_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --reasoning-rerank --rerank-model gpt-4o \
    --reasoning-rerank-top-k 20 --reasoning-rerank-blend-weight 0.7

# 7. 50q biology with RIA iterative expansion (Spec M)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v13_bright_ria_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --ria-loop --ria-max-rounds 3

# 8. 50q biology with Token Graph scoring (Spec N)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v14_tg_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --token-graph

# 9. 50q biology with RIA + Token Graph (Spec M+N)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v14_tg_ria_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --token-graph --ria-loop --ria-max-rounds 3

# 10. 50q biology with RIA + Walk Score (★ best, Spec M+N.1, nDCG=0.3181)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v15_walk_ria_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score --ria-loop --ria-max-rounds 3

# 11. Full 323q (3 domains)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval \
    --domains biology,economics,stackoverflow \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_full \
    --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine
```

### CLI オプション一覧

| Option | Default | Description |
|--------|---------|-------------|
| **基本** | | |
| `--mode` | — | `cot_retrieval` (graph+CoT), `unified` (dense+LLM) |
| `--scoring-mode` | `classic` | `classic`, `gedig`, `gedig_refine` |
| `--graph-top-k` | 50 | グラフ構築に使う文書数 |
| `--rerank-alpha` | 0.1 | BM25 weight (低い = graph 重視) |
| `--limit` | — | クエリ数制限 (smoke test 用) |
| **geDIG Scoring (Spec H)** | | |
| `--gedig-lambda` | 1.0 | geDIG: GED vs IG balance |
| `--gedig-sp-beta` | 0.5 | geDIG: shortest-path weight |
| `--gedig-k-hop` | 2 | Local subgraph k-hop radius |
| `--gedig-mp-iterations` | 2 | Message passing iterations |
| `--gedig-mp-alpha` | 0.3 | Query influence weight |
| **Dense Retrieval (Spec I)** | | |
| `--dense-index-dir` | — | Dense retrieval index directory |
| **Pointwise Reranking (Spec J)** | | |
| `--pointwise-rerank` | false | Enable pointwise LLM reranking |
| `--pointwise-blend-weight` | 0.4 | PW weight in blend (0=ignore, 1=replace) |
| **Query Decomposition (Spec K)** | | |
| `--query-decomp` | false | Enable query decomposition |
| `--query-decomp-top-k` | 50 | BM25 top-k per sub-query |
| `--query-decomp-max-sub` | 5 | Max sub-questions |
| **Reasoning Reranking (Spec L)** | | |
| `--reasoning-rerank` | false | Enable gpt-4o reasoning reranking |
| `--rerank-model` | (same as main) | Reranking LLM model (e.g. gpt-4o) |
| `--reasoning-rerank-top-k` | 20 | Top-k candidates to rerank |
| `--reasoning-rerank-doc-chars` | 4000 | Max chars per document |
| `--reasoning-rerank-blend-weight` | 0.7 | Blend weight (0=ignore, 1=replace) |
| **RIA Iterative Expansion (Spec M)** | | |
| `--ria-loop` | false | Enable RIA iterative query expansion |
| `--ria-max-rounds` | 3 | Maximum RIA iteration rounds |
| `--ria-docs-per-round` | 50 | New docs to retrieve per RIA round |
| `--ria-feedback-top-k` | 5 | Top-k docs to feed back to LLM per round |
| `--ria-beta0-target` | 1 | Target β₀ for RIA convergence |
| **Token Graph (Spec N/N.1)** | | |
| `--token-graph` | false | Token graph scoring 有効化 (Spec N) |
| `--token-graph-weight` | 0.15 | Graph scores とのブレンド比率 |
| `--token-graph-max-tokens` | 500 | spaCy パース対象のトークン上限 |
| `--token-graph-walk-score` | false | DG/AG 重み付き最短経路 (geDIG Walk Score, Spec N.1) |
| `--token-graph-dg-penalty` | 2.0 | Bridge (DG) エッジのコストペナルティ |
| **Entity F-eval (Spec O)** | | |
| `--entity-feval` | false | Entity F-eval scoring 有効化 |
| `--entity-feval-weight` | 0.1 | F-eval スコアのブレンド比率 |
| `--entity-feval-lambda` | 1.0 | F-eval 内部 DG/AG バランス |
| **Multi-CoT Ensemble (Spec P)** | | |
| `--n-cot-ensemble` | 1 | CoT 生成本数 (1=従来, 3=ensemble) |
| `--cot-cache-dir` | — | CoT キャッシュディレクトリ (再現性用) |
| `--cot-temperature` | 0.7 | Ensemble CoT 生成 temperature |

### BRIGHT 関連ファイル

| ファイル | 説明 |
|---------|------|
| **Pipeline** | |
| [src/bright_cot_pipeline.py](src/bright_cot_pipeline.py) | CoT × Graph re-ranking pipeline (main) |
| [src/bright_pipeline.py](src/bright_pipeline.py) | BM25 baseline pipeline |
| [src/entity_graph.py](src/entity_graph.py) | Three-tier entity graph + TF-IDF features |
| [src/gedig_scoring.py](src/gedig_scoring.py) | MessagePassingNX + EdgeReevaluatorNX + GeDIGDocScorer |
| [src/gedig_router.py](src/gedig_router.py) | geDIG routing (tier selection) |
| [src/dense_retriever.py](src/dense_retriever.py) | E5-base-v2 + FAISS dense retrieval |
| [src/episode_graph.py](src/episode_graph.py) | Episode-based graph construction |
| [src/answerer.py](src/answerer.py) | LLM API handler (OpenAI) |
| **Scripts** | |
| [scripts/run_bright.py](scripts/run_bright.py) | BRIGHT 実験ランナー (全モード対応) |
| [scripts/build_dense_index.py](scripts/build_dense_index.py) | Dense index 構築 |
| [scripts/prepare_bright.py](scripts/prepare_bright.py) | BRIGHT データ準備 |
| **Reports** | |
| [results/REPORT_SPEC_H_geDIG_scoring.md](results/REPORT_SPEC_H_geDIG_scoring.md) | Spec H: geDIG scoring 実験レポート |
| [results/REPORT_SPEC_I_dense_retrieval.md](results/REPORT_SPEC_I_dense_retrieval.md) | Spec I: Dense retrieval 実験レポート |
| [results/REPORT_SPEC_J_pointwise_reranking.md](results/REPORT_SPEC_J_pointwise_reranking.md) | Spec J: Pointwise LLM reranking 実験レポート |
| [results/REPORT_SPEC_K_query_decomposition.md](results/REPORT_SPEC_K_query_decomposition.md) | Spec K: Query decomposition 実験レポート |
| [results/REPORT_SPEC_L_reasoning_reranking.md](results/REPORT_SPEC_L_reasoning_reranking.md) | Spec L: gpt-4o reasoning reranking 実験レポート |
| [docs/spec_m_ria_report.md](docs/spec_m_ria_report.md) | Spec M: RIA iterative expansion 実験レポート |
| [docs/experiment_design_spec_m_ria.md](docs/experiment_design_spec_m_ria.md) | Spec M: RIA iterative expansion 設計メモ |
| [docs/spec_n_token_graph_report.md](docs/spec_n_token_graph_report.md) | Spec N: Token-level Graph Scoring 実験レポート |
| [docs/experiment_design_spec_n_unified_graph.md](docs/experiment_design_spec_n_unified_graph.md) | Spec N: Unified Token-level Graph 設計メモ |
| [docs/references_bright_sota.md](docs/references_bright_sota.md) | BRIGHT SOTA 手法リファレンス |
| [docs/report_bright_cot_retrieval.md](docs/report_bright_cot_retrieval.md) | CoT retrieval 初期レポート |
| [docs/report_bright_unified.md](docs/report_bright_unified.md) | Unified mode レポート |

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
