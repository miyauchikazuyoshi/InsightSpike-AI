# geDIG v4+ Adaptive Architecture Design

> Evolving geDIG from static RAG to fully adaptive, F-value-driven reasoning.

## 1. Architecture Evolution

```
v1: Static geDIG (Conditions A-D)
    → Fixed gate thresholds, fixed retrieval

v2: Tuned geDIG (Condition E1)
    → Tuned parameters (γ₀=0.3, γ₁=0.5, θ_dg=-0.5)

v3: Hybrid Dual Process (Condition Hybrid-E1)
    → System 1 (DG fires → 0 CoT) / System 2 (fixed 2-step CoT)
    → Zero-cost routing via topology

v4: Adaptive Depth (Condition Hybrid-E2)  ← CURRENT
    → cot_depth = f(F) — harder questions get deeper reasoning
    → Formula: cot_depth = clamp(⌈(F - θ_dg) / α⌉, 1, max_depth)

v5: Adaptive Graph Density (proposed)
    → Graph structure adapts to F value
    → Two-edge architecture: context attention + similarity attention

v6: Unified Adaptive RAG (vision)
    → F value controls ALL dimensions:
       - Reasoning depth (v4)
       - Graph density / retrieval strategy (v5)
       - Chunk granularity
       - Model selection (weak/strong)
```

## 2. v4: Adaptive Depth CoT (Implemented)

### Core Idea

F value = information gap magnitude → directly maps to needed reasoning depth.

```
F << θ_dg       → System 1:   0 steps (direct answer, high confidence)
F ≈ θ_dg        → System 1.5: 1 step  (light confirmation)
F > θ_dg        → System 2:   2 steps (standard CoT)
F >> θ_dg + α   → System 3:   3-4 steps (deep reasoning)
```

### Formula

```
cot_depth = clamp(⌈(F - θ_dg) / α⌉, 1, max_depth)

Parameters:
  θ_dg = -0.5  (Decision Gate threshold)
  α = 0.5      (depth sensitivity)
  max_depth = 4 (upper bound)
```

### Expected Depth Distribution (500q, GPT-4o-mini)

| Depth | Count | % | Description |
|:-----:|:-----:|:-:|:-----------|
| 0 | ~37% | System 1 | DG fires → skip CoT entirely |
| 1 | ~11% | Minimal | F just above threshold |
| 2 | ~20% | Standard | Same as fixed E1 |
| 3 | ~5% | Extended | Moderate information gap |
| 4 | ~28% | Deep | Large information gap |

Average LLM calls: ~2.36/question (vs E1's 2.2, IRCoT's 8)

### Key Insight: Model-Dependent Interaction

- **GPT-4o**: Strong model finds answers early → early termination → depth 3-4 effectively becomes depth 1-2
- **GPT-4o-mini**: Weak model exhausts all planned steps → full depth used → more API calls but potentially better accuracy

This creates a natural "self-regulation" — strong models automatically use fewer resources while weak models get the help they need.


## 3. v4 Experiment Results (500 questions)

### E2 vs E1 Comparison

| Model | E1 Fixed EM | E2 Adaptive EM | Diff | p-value |
|:------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | **45.1%** | 44.1% | -1.0pt | 0.47 (NS) |
| GPT-4o | **51.2%** | 50.6% | -0.6pt | 0.74 (NS) |

**Result: Adaptive Depth does NOT improve over fixed depth=2.**

### Depth-Stratified Analysis (Key Finding)

| Depth | GPT-4o-mini EM | GPT-4o EM | n | LLM Calls |
|:-----:|:-:|:-:|:---:|:---:|
| 0 (System 1) | 34.6% | 47.6% | 191 | 1 |
| 1 | 48.2% | 50.0% | 56 | 2 |
| **2** | **57.0%** | **61.1%** | ~108 | 3 |
| 3 | 45.5% | 50.0% | 22 | 4 |
| 4 | 45.5% | 46.3% | ~122 | 5 |

**Critical Insight: Depth 2 is optimal for BOTH models. Depth 3-4 hurts.**

### Interpretation

1. **F value correctly identifies hard questions** — high-F questions assigned depth 3-4
2. **But deeper CoT doesn't solve them** — extra steps introduce noise/hallucination
3. **The bottleneck is retrieval quality, not reasoning depth**
4. **System 1 (depth 0) underperforms** — DG gate routes ~37% to System 1,
   but not all are truly "easy" (EM=34.6% for mini vs 57.0% at depth 2)

### Implications for v5

- F value should control **retrieval strategy**, not reasoning depth
- Next experiment: use F to adjust graph density / retrieval scope
- The "2-edge architecture" (context + similarity attention) should improve
  the quality of what depth-2 CoT receives, rather than adding more CoT steps


## 4. v5: Adaptive Retrieval (Proposed — Next Priority)

### Problem

Fixed graph construction parameters (entity_overlap_threshold, q_link_top_k) don't adapt to question difficulty.

### Design: F-Driven Graph Construction

```python
def adaptive_graph_build(F, base_threshold=0.3):
    if F < theta_dg:  # Information sufficient
        # Sparse graph: only strong connections
        threshold = base_threshold * 1.5
        link_top_k = 2
    elif F < theta_dg + alpha:  # Moderate gap
        # Standard graph
        threshold = base_threshold
        link_top_k = 3
    else:  # Large information gap
        # Dense graph: more connections for exploration
        threshold = base_threshold * 0.7
        link_top_k = 5
```

### Two-Edge Architecture

Inspired by the maze experiment's Three-Layer Search Architecture:

**Edge Type 1: Context (Temporal) Attention**
- Connects chunks within the same document
- Preserves document structure even with fine-grained chunking
- Analogous to maze's spatial adjacency edges
- Weight: inversely proportional to paragraph distance within document

```
w_context(chunk_i, chunk_j) = {
    1.0   if same paragraph (adjacent sentences)
    0.7   if adjacent paragraphs (same article)
    0.3   if same article but distant
    0.0   if different articles
}
```

**Edge Type 2: Similarity (Semantic) Attention**
- Connects semantically related chunks across documents
- Enables multi-hop reasoning bridges
- Analogous to maze's "teleport" or shortcut edges
- Weight: TF-IDF cosine similarity + entity overlap

```
w_similarity(chunk_i, chunk_j) = α·cos_sim(tfidf_i, tfidf_j) + β·entity_overlap
```

### Integration with geDIG Gauge

The Extended F formula can weight these edge types differently:

```
F = ΔEPC_norm − λ·(ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀)

Where EPC now considers:
  EPC = Σ edges (w_context · context_info + w_similarity · similarity_info)
```

This gives the gauge a richer signal:
- High context-edge information → document structure is being leveraged
- High similarity-edge information → cross-document bridges are forming
- Balance between them → healthy multi-hop reasoning


## 5. Maze → QA Transfer

### Maze Experiment Architecture (Accurate Summary)

**ノード構造（迷路）:**
- **Query node** `(position_hash, 0, 0)`: 迷路内の位置（状態）を表す
- **Direction node** `(x, y, action_index)`: 各位置から取りうるアクション候補を表す
- **10D vector**: [x, y, action, dx, dy, revisit, success, unknown, reward, tanh(propagated)]

**2種類のエッジ特徴量（迷路）:**
1. **ag_attention**: エッジ接続時のコサイン類似度。候補スコアリング時に記録。
   - 「この接続はどれくらい既知の構造と類似しているか」を示す
2. **dg_attention**: エッジ選択時のgeDIG g値（情報利得）。コミット時に記録。
   - 「この接続がどれくらいの新規情報をもたらしたか」を示す（負 = 改善）

**Three-Layer Search（迷路）:**
- **L0 (VectorHashIndex)**: O(1) ハッシュベース再訪検出。位置ベクトルを量子化してバケットに格納。
  再訪ノードが見つかれば L1 に渡す。
- **L1 (AttentionGraphWalker)**: O(degree) 注意重み付きグラフ歩行。
  再訪ノードの隣接エッジで `attention > θ` のものだけ辿る。
  スコアリング: `ag_attention × σ(-dg_attention/τ_dg) × σ(propagated/τ_reward)`
  → 過去に有用だった（ag高い）× 情報利得が高かった（dg負）× 将来の報酬が高い（propagated正）
- **L2 (Full Memory Sort)**: O(N log N) 全メモリからの距離ソート。L0/L1で候補不足時のフォールバック。

**Attention Lifecycle（迷路）:**
- `on_new_edge()`: attention = 1.0 で初期化
- `on_step()`: 毎ステップ `attention *= decay_rate` (0.95) で減衰
- `on_traverse()`: 走行時 `attention = min(1.0, attention + boost)` (boost=0.1) で強化
- `on_ag_fire()`: 閾値以下のエッジを再活性化

**Sleep Propagation（迷路）:**
- Q-learning式バックワード伝播: `propagated(n) = reward(n) + γ·max(propagated(neighbor))`
- ゴール→スタート方向に報酬を逆伝播。上流ノードに「この先に何があるか」を書き込む。
- 結果をベクトルの dim8 (reward) と dim9 (tanh(propagated)) に格納。
- 次エピソードの inherited_graph として引き継ぎ → 累積知識で探索をガイド。

### QAへの転移：正確な対応関係

| 迷路の概念 | 具体的な動作 | QA/RAGでの対応候補 |
|:----------|:-----------|:-----------------|
| Query node | 位置（状態） | 質問 or 検索クエリの状態 |
| Direction node | アクション候補 | 検索候補パラグラフ |
| ag_attention | 接続時の構造類似度 | 文脈的近接度（同一文書内距離） |
| dg_attention | geDIG情報利得 | 情報利得（F値からのエッジ重み） |
| L0 Hash lookup | 再訪検出 | キャッシュヒット / BM25完全一致 |
| L1 Graph walk | 注意フィルタ付き隣接探索 | グラフ上のリンク辿り（entity overlap等） |
| L2 Full sort | 全候補の距離ソート | Dense retrieval / reranking |
| Attention decay | エッジ利用頻度の指数減衰 | 情報の鮮度減衰（古い検索結果の重み低下） |
| on_traverse boost | 走行したエッジの強化 | 実際に使った検索結果の重み増加 |
| Sleep propagation | 報酬の逆伝播（Q-learning） | オフライン学習：過去のQA結果から知識グラフを強化 |
| inherited_graph | 累積知識の引き継ぎ | エピソード間の知識グラフ永続化 |

**注意**: 上記の「QA/RAGでの対応候補」は設計上の仮説であり、実装済みではない。
迷路とQAではドメインが大きく異なるため、直接移植ではなく概念的インスピレーションとして扱う。

### ユーザーの提案: 2種類のエッジ特徴量（QA版）

迷路の `ag_attention` + `dg_attention` に対応する QA 版:

1. **文脈アテンション** (← ag_attention のアナロジー)
   - 同一文書内のチャンク間距離に基づく重み
   - チャンクが細かくても文書構造を保持
   - 「この2つのチャンクは元々近い位置にあった」

2. **類似度アテンション** (← dg_attention のアナロジー)
   - TF-IDF / エンティティ重複 / 意味類似度に基づく重み
   - 異なる文書間のブリッジを形成
   - 「この2つのチャンクは意味的に関連する」

迷路では L1 スコアリングで両方を掛け合わせている:
```
score = ag_attention × σ(-dg_attention/τ_dg) × σ(propagated/τ_reward)
```

QA版でも同様に、チャンク選択時に:
```
score = context_attention × similarity_attention × F_gate
```
とすることで、文書構造と意味的関連性の両方を考慮した検索が可能になる。


## 6. Unified Adaptive RAG (Vision)

The ultimate architecture uses F as a universal control signal:

```
                    F = geDIG gauge
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │ Adaptive │    │ Adaptive  │   │ Adaptive  │
    │  Depth   │    │  Graph    │   │ Retrieval │
    │ (v4)     │    │ Density   │   │ Strategy  │
    │          │    │ (v5)      │   │ (v5)      │
    │ 0-4 CoT  │    │ sparse↔   │   │ L0↔L1↔L2 │
    │ steps    │    │ dense     │   │           │
    └────┬────┘    └─────┬─────┘   └─────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                    Final Answer
```

**F as the universal gauge:**
- F < θ_dg: Everything minimal (0 CoT, sparse graph, L0 retrieval)
- F ≈ θ_dg: Standard processing (2 CoT, normal graph, L0+L1)
- F >> θ_dg: Maximum effort (4 CoT, dense graph, L0+L1+L2)

**Cost model:**
```
Cost(F) = base_cost + depth_cost(F) + retrieval_cost(F) + graph_cost(F)

Easy questions (F < θ_dg): Cost ≈ 1 LLM call
Medium questions:           Cost ≈ 2-3 LLM calls
Hard questions (F >> θ_dg): Cost ≈ 4-6 LLM calls
```

This achieves the ultimate goal: **spend compute proportional to difficulty**, measured objectively by the information-theoretic gauge.


## 7. Implementation Priority (Updated after v4 results)

1. ~~**v4 Adaptive Depth**~~ — DONE. Result: depth 2 is optimal, deeper hurts.
2. ~~**v4 Results Analysis**~~ — DONE. Key finding: bottleneck is retrieval, not reasoning.
3. **v5 Two-Edge Architecture** — **NEXT PRIORITY**
   - Implement context attention + similarity attention edges
   - Expected: better retrieval quality → better input to depth-2 CoT
4. **v5 Adaptive Graph Density** — use F to control edge density
5. **Unified Controller** — integrate all adaptive dimensions
6. **System 1 Gate Improvement** — depth-0 EM is too low (34.6%); refine DG threshold


---

*Created: 2026-03-08*
*Updated: 2026-03-08 (v4 experiment results added)*
*Status: v4 implemented and evaluated (negative result), v5 proposed as next priority*
