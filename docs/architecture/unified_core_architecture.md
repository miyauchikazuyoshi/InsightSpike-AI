# Unified geDIG Core Architecture

## Overview

geDIG (generalized Differential Information Gain) は**1つの原理**を
3つの異なるドメインに適用する統一フレームワーク。

```
                        src/gedig/core/
                    ┌─────────────────────┐
                    │  f_eval.py          │  F composition
                    │  protocols.py       │  snapshots / result types
                    │  edge_partition.py  │  score-band partition
                    │  message_passing.py │  Attention-weighted MP
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    adapters/maze.py  adapters/rag.py  adapters/transformer.py
              │              │              │
              ▼              ▼              ▼
    experiments/maze  experiments/      experiments/
                      hotpotqa_v2       transformer
```

## F-Eval: 統一された評価関数

```
F = ΔEPC - λ(ΔH + γΔB)

  EPC  = Edit Path Cost (before/afterの構造変更コスト)
  H    = Shannon Entropy of edge weights (情報の均一さ)
  B    = pluggableなstructure benefit
         既定・確立済み: ΔSP (relative shortest-path gain)
         明示的な研究モード: Δβ₁ (Betti number)
  λ, γ = 重みパラメータ
```

無修飾のFはSP形式を指す。β₁形式は`use_betti=True`のように明示し、
進行中のtopological generalizationとして区別する。

### ドメイン間の対応表

```
                Maze              RAG (AGHT)          Transformer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graph          空間グラフ         文書ヘテログラフ     attention グラフ
Node           セル              Sentence/Token       トークン
Edge           通路              entity/similarity    attention weight
Q              ゴール方向        query 情報ニーズ      W_Q · x
K              探索状態          ノード情報量          W_K · x
V              通路コスト        エッジ特徴量          W_V · x
Event gate     AG→探索 / DG→確定  domain policy          experiment-specific
Edge partition n/a               low/high edge F        low/high profile score
F-eval         ΔGED-λ(ΔH+γΔB)   cost-λ·dot(Q,K)      ΔEPC-λ(ΔH+γΔB)
Wake           探索              検索/RIA              forward pass
Sleep          報酬伝搬          グラフ分析/F-eval     attention 構造評価
Wake₂          活用              partition-guided検索  次の forward
```

## Wake-Sleep-Wake Cycle

全ドメインに共通するサイクル:

```
    ┌──── Wake₁ ────┐
    │  情報収集      │
    │  (探索/検索)   │
    └───────┬───────┘
            ▼
    ┌──── Sleep ────┐
    │  構造分析      │
    │  F-eval        │
    │ gate/partition │
    │ policy         │
    └───────┬───────┘
            ▼
    ┌──── Wake₂ ────┐
    │ policy-guided │
    │  次のアクション │
    └───────┬───────┘
            │
            └──→ (繰り返し)
```

## AG/DG Event Gate とEdge Score Partition

AG/DGは二段のイベントゲートである。

```python
ag_fire = g0 > theta_ag
# AG (Attention Gate): hop-0の曖昧性を検出し、追加探索を開く

dg_fire = best_hop >= 1 and gmin_multihop < theta_dg
# DG (Decision Gate): multi-hopで改善した低F候補を確認し、commitする
```

一方、RAG/Transformerにはscalar edge scoreをpercentileで分ける処理もある。

```python
f = cost(edge) - lambda_param * dot(Q, K) / sqrt(d_k)
partition = partition_edges(edge_scores)
# partition.low_score_edges / partition.high_score_edges
```

これはedge集合の分割であってAG/DGゲート発火ではない。旧APIの
`AGDGResult.ag_edges/dg_edges`は再現性のため残す互換ラベルであり、新規コードは
`EdgePartitionResult`を使用する。

## AGHT (Analytical Heterogeneous Graph Transformer)

RAG ドメインでの具体実装:

```
Unified Heterogeneous Graph
┌──────────────────────────────────────┐
│  Sentence nodes ──Tier1/2/3──> Sent  │
│       │                              │
│   contains (cross-level edge)        │
│       ▼                              │
│  Token nodes ──dep/lemma──> Token    │
│                                      │
│  cross-doc same_lemma_x edges (NEW)  │
└──────────────────────────────────────┘

QKV Features (10 parameters, zero-shot):
  Q = [w_q1·match, w_q2·density, w_q3·cot]
  K = [w_k1·importance, w_k2·discrim, w_k3·struct]
  V = [w_v1·cost, w_v2·bridge, w_v3·cross_level]
```

## Experiment 4 Preliminary Observation

```
single-state Flash profile maximization > Baseline > profile minimization
（単一seed・SP条件の予備結果）

この実験固有のprofile最大化目的は、canonical delta Fの
「lower is better」という判断方向を変更しない。β₁条件やrandom regularization
controlでは優位性が確認されておらず、DG保存の確立済み証拠とは扱わない。
```

## Test Structure

```
src/gedig/tests/
├── test_f_eval.py              # F composition, backends, partitions
├── test_adapters.py            # Maze/RAG/Transformer adapter contracts
├── test_maze_equiv.py          # MazeFEval contracts (not active equivalence)
├── test_maze_active_trace.py   # Active legacy evaluator golden trace
├── test_rag_equiv.py           # Independent RAG formula comparison
└── test_transformer_equiv.py   # Frozen-oracle value/gradient comparison

tests/unit/test_flash_gedig_api.py
└── public Flash profile/delta, value/gradient, loss-direction contracts
```

`test_transformer_equiv.py`のold armは
`experiments/refactor_transformer/thermodynamic_gedig_legacy.py`を直接使い、
unified adapterを生成していないこともassertする。SP/β₁、mask、非既定parameter、
全component、gradientを比較する。T4/Exp4 E2Eはこの検証では再実行していない。

RAG/AGHTのactive pathは常に`RAGFEval`へ委譲済みで、残る
`use_unified*`引数はdeprecated no-opである。Maze active evaluatorは
experiment固有のCmax/linkset/scoped-SP意味論を持つため未移行であり、
`MazeFEval`とのdrop-in等価性は主張しない。詳細は
`src/gedig/docs/MIGRATION_PROGRESS.md`を参照。

## File Map

```
src/gedig/
├── __init__.py
├── core/
│   ├── f_eval.py              # F composition; SP default, β₁ explicit
│   ├── protocols.py           # snapshots, F/gate/partition results
│   ├── edge_partition.py      # low/high scalar edge-score partition
│   ├── ag_dg.py               # historical edge-label compatibility shims
│   ├── message_passing.py     # Attention-weighted propagation
├── adapters/
│   ├── transformer.py         # Attention matrix → TorchGraphSnapshot
│   ├── rag.py                 # Document graph adapter
│   └── maze.py                # Spatial graph adapter
└── backends/
    ├── networkx_backend.py
    └── torch_backend.py
```
