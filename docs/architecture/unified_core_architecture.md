# Unified geDIG Core Architecture

## Overview

geDIG (generalized Differential Information Gain) は**1つの原理**を
3つの異なるドメインに適用する統一フレームワーク。

```
                        src/gedig/core/
                    ┌─────────────────────┐
                    │  f_eval.py          │  F = ΔEPC - λ(ΔH + γΔB)
                    │  protocols.py       │  GedigGraph / FResult
                    │  message_passing.py │  Attention-weighted MP
                    │  betti.py           │  β₁ = E - V + C
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

  EPC  = Euler Poincaré Characteristic = V - E (グラフの複雑さ)
  H    = Shannon Entropy of edge weights (情報の均一さ)
  B    = β₁ Betti number = E - V + C (位相的穴の数)
  λ, γ = 重みパラメータ
```

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
AG             探索済みパス      表層一致エッジ        high attention
DG             未探索パス        推論ギャップ          low attention
F-eval         ΔGED-λ(ΔH+γΔB)   cost-λ·dot(Q,K)      ΔEPC-λ(ΔH+γΔB)
Wake           探索              検索/RIA              forward pass
Sleep          報酬伝搬          グラフ分析/F-eval     attention 構造評価
Wake₂          活用              DG-guided 再検索      次の forward
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
    │  AG/DG 分類    │
    └───────┬───────┘
            ▼
    ┌──── Wake₂ ────┐
    │  DG-guided     │
    │  次のアクション │
    └───────┬───────┘
            │
            └──→ (繰り返し)
```

## AG/DG Edge Classification

F-eval による自動分類:

```python
α = dot(Q, K) / √d_k          # attention score
f = cost(e) - λ · α            # F-eval value

f < θ  →  AG (Assertion Graph)   # 確認済み情報
f ≥ θ  →  DG (Derivation Graph)  # 推論ギャップ
```

**核心的発見**: AG = high attention edge, DG = low attention edge
→ Transformer の attention が「どの token に注目するか」を学ぶように、
   geDIG は「どのエッジが情報ギャップか」を検出する。

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

## Experiment 4 Key Finding

```
F maximization (DG preservation) > Baseline > F minimization

→ Transformer は「知らないこと」(DG) を明示的に保持した方が学習が良い
→ CE Loss だけでは「知ること」に最適化するが、
   「何を知らないか」の構造は保存しない
→ F 最大化 = 「無知の構造」を保存する正則化
```

## Test Structure

```
tests/
├── test_gedig_core.py          # 25 tests: F-eval, protocols, betti
├── test_adapters.py            # 21 tests: transformer, RAG, maze equiv
├── test_migration.py           # 25 tests: legacy ↔ unified equivalence
└── E2E reproduction:
    ├── R4-R5: HotpotQA 100q    # diff = 0.0000 (exact match)
    ├── T4: Exp4 conclusion     # negative_better (matches legacy)
    └── R6: BRIGHT 50q          # diff = 0.0500 (RIA non-determinism)
```

## File Map

```
src/gedig/
├── __init__.py
├── core/
│   ├── f_eval.py              # F = ΔEPC - λ(ΔH + γΔB)
│   ├── protocols.py           # GedigGraph, GedigNode, GedigEdge, FResult
│   ├── message_passing.py     # Attention-weighted propagation
│   └── betti.py               # β₁ computation (exact + differentiable)
├── adapters/
│   ├── transformer_adapter.py # Attention matrix → GedigGraph
│   ├── rag_adapter.py         # Document graph → GedigGraph
│   └── maze_adapter.py        # Spatial graph → GedigGraph
└── backends/
    └── networkx_backend.py    # NetworkX implementation
```
