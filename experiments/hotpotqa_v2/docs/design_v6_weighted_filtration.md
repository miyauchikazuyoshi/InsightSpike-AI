# v6 Design: Weighted Filtration & Unified Edge Features

**Date**: 2026-03-09
**Status**: Design
**Motivation**: v5 Two-Edge Architecture caused topology collapse (EM -4.9pt)

---

## 1. Problem Statement

### 1.1 v5 Topology Collapse

v5 added context attention edges (intra-title, distance-decay) and similarity
attention edges (cross-title, TF-IDF cosine + entity overlap). Result:

| Metric | E1 (baseline) | E3 (v5) | Problem |
|--------|:---:|:---:|:--------|
| Avg edges | 7.7 | 8.2 | +6.5% more edges |
| beta_0 | 2.76 | 1.39 | Graph always connected |
| beta_1 | 2.26 | 3.12 | Too many cycles |
| DG fire | 38% | 77% | DG fires on everything |
| System 2 | 62% | 23% | CoT suppressed |
| EM | 45.2% | 40.3% | -4.9pt degradation |

**Root cause**: The gauge computes Betti numbers from **binary edge presence**.
Any edge addition — regardless of weight — changes topology and can destroy
the routing signal.

### 1.2 The Gauge-Topology Coupling

The extended gauge formula:

```
F = base_gedig - lambda * (gamma_1 * delta_beta_1 - gamma_0 * delta_beta_0)
```

Where:
- `beta_0 = number_connected_components(g)` — ignores weights
- `beta_1 = E - V + C` — counts edges, ignores weights

Both Betti computations treat the graph as **unweighted**. Edge weights
(`w_ctx`, `w_sim`) exist in the graph but are invisible to the gauge.

### 1.3 Connection to Maze Experiment

The maze experiment's graph-persistent DG already solves a similar problem
with multi-dimensional edge features:

| Mechanism | Maze Implementation | HotpotQA Analog |
|-----------|:-------------------|:----------------|
| propagation_weight | `max(prop[u], prop[v])` on edges | Edge strength for filtration |
| dg_attention | geDIG gauge value stored on edges | Gauge feedback to edge weights |
| attention lifecycle | decay + boost on traversal | Strengthen/weaken based on outcomes |
| sleep_edge_weights | Action-level reward accumulation | Positive/negative example learning |

Key pattern: edges carry **multi-dimensional features** that evolve, and
these features affect downstream computation.

---

## 2. Solution Architecture

### 2.1 Unified Edge Feature Vector

Replace separate edge types (context, similarity) with a single edge
per node pair carrying a feature vector:

```python
@dataclass
class EdgeFeature:
    w_ctx: float = 0.0      # Context attention (intra-title distance decay)
    w_sim: float = 0.0      # Similarity attention (TF-IDF + entity overlap)
    # Future extensions:
    # w_gauge: float = 0.0  # DG feedback signal
    # w_reward: float = 0.0 # Positive/negative example signal

    @property
    def strength(self) -> float:
        """Unified edge strength for filtration threshold."""
        return max(self.w_ctx, self.w_sim)
```

Benefits:
- Adding new signal = adding new field (no edge type explosion)
- `strength` provides a single scalar for filtration
- Same structure works for future positive/negative feedback

### 2.2 Weighted Filtration for Betti Computation

Instead of computing Betti on all edges (binary), filter by `strength >= theta`:

```python
def compute_betti_filtered(g: nx.Graph, threshold: float) -> tuple[int, int]:
    """Compute (beta_0, beta_1) on subgraph with strong edges only."""
    sub = nx.Graph()
    sub.add_nodes_from(g.nodes(data=True))
    for u, v, d in g.edges(data=True):
        if d.get('strength', d.get('weight', 1.0)) >= threshold:
            sub.add_edge(u, v, **d)
    b0 = nx.number_connected_components(sub)
    b1 = sub.number_of_edges() - sub.number_of_nodes() + b0
    return b0, b1
```

**Effect**: Weak edges (low w_sim, distant w_ctx) are invisible to the gauge
but still available for re-ranking. This decouples graph enrichment from
topological signal.

### 2.3 Two-Threshold Architecture

The key insight: use **different thresholds** for different purposes:

```
theta_betti = 0.5   # Only strong edges count for Betti numbers
theta_rerank = 0.0  # All edges (including weak) used for re-ranking
```

This means:
1. **Graph construction**: Build all edges (context + similarity) with weights
2. **Gauge computation**: Filter to strong edges -> compute Betti -> compute F
3. **Re-ranking**: Use all edges (including weak) for context selection
4. **Routing**: Use filtered-Betti F for System 1/2 decision

---

## 3. Experiment Design

### Experiment A: Threshold Increase (Quick Validation)

**Hypothesis**: Reducing edge count by raising `sim_edge_threshold` will
restore Betti signals and improve EM toward E1 baseline.

**Changes**: Config-only. `sim_edge_threshold: 0.25 -> 0.45`

**Expected outcome**:
- Fewer similarity edges (avg ~4 -> ~1-2)
- beta_0 and beta_1 closer to E1 values
- System 2 rate increases (toward 60%)
- EM improves from 40.3% toward 45%

**Purpose**: Validate that edge reduction restores the gauge. If yes,
weighted filtration is the right direction.

### Experiment B: Weighted Filtration (Core Fix)

**Hypothesis**: Computing Betti on strong-edge subgraph preserves topological
signal while allowing graph enrichment for re-ranking.

**Changes**:
1. `graph_utils.py`: Add `compute_betti_0_filtered()`, `compute_betti_1_filtered()`
2. `adapter.py`: Add `betti_threshold` parameter; use filtered Betti in gauge
3. Keep all edges for re-ranking (unchanged from v5)

**Key parameter**: `betti_threshold`
- Too low (0.1): same as v5, Betti sees all edges, topology collapses
- Too high (0.9): only adjacent same-title edges, same as E1 legacy
- Sweet spot (~0.5): context dist<=1 (0.9) + strong similarity (>0.5) count;
  weak context (0.3) and marginal similarity don't affect Betti

**Expected outcome**:
- Betti signals preserved (close to E1)
- Re-ranking still benefits from all edges
- EM should match or exceed E1 baseline

### Experiment Matrix

| Experiment | Config | Key Change | LLM Cost |
|:----------:|--------|:-----------|:--------:|
| E3-A | threshold=0.45 | Config change only | ~70q (mock) + 139q (resume) |
| E3-B | betti_threshold=0.5 | Code + config | ~70q (mock) + 139q |

Both experiments use the same 500q sample and GPT-4o-mini for comparison
with E1 (EM=45.2%) and E3 (EM=40.3%).

---

## 4. Implementation Plan

### 4.1 Experiment A (Config-only)

1. Create `condition_hybrid_e3a_high_threshold.yaml`:
   - Copy E3 config, change `sim_edge_threshold: 0.45`
2. Mock smoke test (10q)
3. Run 139q with GPT-4o-mini (match E3 sample size for comparison)

### 4.2 Experiment B (Code Change)

#### Step 1: `graph_utils.py` — Filtered Betti functions

```python
def compute_betti_0_filtered(g: nx.Graph, threshold: float = 0.0) -> int:
    """beta_0 on subgraph with edge strength >= threshold."""
    if threshold <= 0.0:
        return compute_betti_0(g)
    sub = nx.Graph()
    sub.add_nodes_from(g.nodes())
    for u, v, d in g.edges(data=True):
        if d.get('strength', d.get('weight', 1.0)) >= threshold:
            sub.add_edge(u, v)
    return nx.number_connected_components(sub) if sub.number_of_nodes() > 0 else 0

def compute_betti_1_filtered(g: nx.Graph, threshold: float = 0.0) -> int:
    """beta_1 on subgraph with edge strength >= threshold."""
    if threshold <= 0.0:
        return compute_betti_1(g)
    sub = nx.Graph()
    sub.add_nodes_from(g.nodes())
    for u, v, d in g.edges(data=True):
        if d.get('strength', d.get('weight', 1.0)) >= threshold:
            sub.add_edge(u, v)
    V = sub.number_of_nodes()
    if V == 0:
        return 0
    E = sub.number_of_edges()
    C = nx.number_connected_components(sub)
    return E - V + C
```

#### Step 2: `graph_builder.py` — Add `strength` attribute

In both context and similarity edge creation, add `strength`:

```python
# Context edge
strength = w_ctx  # or max(w_ctx, w_sim)
g.add_edge(..., strength=strength, ...)

# Similarity edge
strength = w_sim
g.add_edge(..., strength=strength, ...)
```

#### Step 3: `adapter.py` — Use filtered Betti

Add `betti_threshold` parameter. In `_calculate_gedig()`:

```python
# Replace direct Betti calls with filtered versions
if self.betti_threshold > 0.0:
    b0_before = compute_betti_0_filtered(g_prev, self.betti_threshold)
    b0_after = compute_betti_0_filtered(g_now, self.betti_threshold)
    b1_before = compute_betti_1_filtered(g_prev, self.betti_threshold)
    b1_after = compute_betti_1_filtered(g_now, self.betti_threshold)
else:
    # Legacy: use GeDIGCore's unfiltered Betti
    b0_before = core_result.betti_0_before
    ...
```

#### Step 4: Config + Test

Create `condition_hybrid_e3b_filtered.yaml`:
```yaml
two_edge_mode: true
betti_threshold: 0.5
sim_edge_threshold: 0.25  # Keep low — filtration handles it
```

---

## 5. Success Criteria

| Metric | E3 (failed) | Target | E1 (baseline) |
|--------|:-----------:|:------:|:-------------:|
| EM | 40.3% | >= 45% | 45.2% |
| System 2 rate | 23% | >= 50% | 62% |
| DG fire rate | 77% | <= 50% | 38% |
| beta_0 | 1.39 | >= 2.0 | 2.76 |
| beta_1 | 3.12 | <= 2.5 | 2.26 |

If Experiment B matches or exceeds E1 while maintaining richer graph
structure (more edges, re-ranking active), it validates the weighted
filtration approach and opens the path to:
- v6.1: DG feedback (write gauge value back to edge features)
- v6.2: Positive/negative example learning (cross-question edge persistence)
- v6.3: Persistent homology (multi-threshold Betti diagrams)

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|:-----------|
| `betti_threshold` too sensitive | Test 3 values: 0.3, 0.5, 0.7 |
| Filtered graph has isolated nodes | Include Q-link edges (always strong) |
| Re-ranking still ineffective | Compare with/without re-ranking separately |
| Computation overhead from subgraph | O(V+E) — negligible at N~50-100 |

---

## 7. Experiment Results

### 7.1 E3A: Threshold Increase (Config-only)

| Metric | E3 (failed) | E3A | E1 (baseline) | Target |
|--------|:-----------:|:---:|:-------------:|:------:|
| EM | 40.3% | **48.9%** | 45.2% | >= 45% |
| S2 rate | 23% | **74%** | 62% | >= 50% |
| DG fire | 77% | **26%** | 38% | <= 50% |
| AG fire | 0% | **38%** | 4% | - |
| β₀ | 1.39 | **3.30** | 2.76 | >= 2.0 |
| β₁ | 3.12 | **2.18** | 2.26 | <= 2.5 |
| sim_edges | 3.9 | **1.6** | 0.0 | - |

**E3A achieves best GPT-4o-mini EM to date (+3.7pt over E1).**
Head-to-head on 51 common questions: E3A wins 4, loses 0, ties 47 vs E1.

### 7.2 E3B: Weighted Filtration

| Metric | E3 (failed) | E3B | E1 (baseline) | Target |
|--------|:-----------:|:---:|:-------------:|:------:|
| EM | 40.3% | **44.8%** | 45.2% | >= 45% |
| S2 rate | 23% | **57%** | 62% | >= 50% |
| DG fire | 77% | **43%** | 38% | <= 50% |
| AG fire | 0% | **17%** | 4% | - |
| sim_edges | 3.9 | **6.3** | 0.0 | - |

**E3B recovers from v5 collapse but does not exceed E1 baseline.**

### 7.3 Why E3A > E3B

E3B's design assumed Betti filtration alone would suffice. However:
1. **Re-ranking pollution**: E3B keeps all weak edges (sim_edges=6.3) for
   re-ranking. Weak edges promote irrelevant facts in LLM context.
2. **AG suppression**: Many edges → fewer knowledge gaps → AG fires only 17%
   (vs E3A's 38%). Less productive graph expansion.
3. **Lesson**: Filtering must be **holistic** (both gauge AND re-ranking),
   not just applied to Betti computation. E3A achieves this by simply not
   creating weak edges in the first place.

---

## 8. Next: 3-Attention Edge Architecture (from Maze Experiment)

### 8.1 Maze Experiment's Proven Architecture

The maze experiment implements a **3-attention scoring system** on edges
that solves exactly the problem E3B exposed — edge quality assessment
through multi-dimensional features with multiplicative gating:

```
# Three attention types per edge (maze implementation)
edge = {
    "attention":    0.85,   # Temporal: decay × 0.95/step, boost +0.1 on use
    "ag_attention": 0.72,   # Relevance: similarity at connection time
    "dg_attention": -0.3,   # Structural: geDIG F value (negative = good)
}

# Multiplicative scoring (all must be good → high score)
score_3att = ag_att * σ(-dg_att / τ_dg) * σ(propagated / τ_r)
```

| Factor | Formula | What it answers |
|--------|---------|-----------------|
| **Relevance** | `ag_attention` | このエッジは類似度が高いか？ |
| **Confidence** | `σ(-dg_attention / τ)` | geDIGが構造的に有用と判断したか？ |
| **Value** | `σ(propagated / τ)` | 報酬が伝播されているか？ |

**Key property**: 乗算なので、どれか一つでも低ければスコアが潰れる。
これにより「類似度は高いが構造的にノイズ」なエッジが自然に除外される。

### 8.2 Attention Lifecycle (Maze Implementation)

```
[Edge Creation]
├─ attention = 1.0 (new edge starts at maximum)
├─ ag_attention = similarity_score (固定)
└─ dg_attention = 0.0 (未評価)

[Every Step]
├─ attention *= decay_rate (0.95)  → 使わないエッジは減衰
└─ use_count unchanged

[On Traversal]
├─ attention += use_boost (0.1)    → 使ったエッジは強化
├─ attention = min(1.0, attention) → 上限1.0
└─ use_count += 1

[On geDIG Evaluation]
├─ dg_attention = g0 (hop0 edges)
└─ dg_attention = gmin_mh (DG shortcut edges)

[Sleep Phase]
└─ propagated = reward + γ * max(propagated[neighbors])
```

### 8.3 HotpotQAへの適用設計

HotpotQAでは1質問＝1グラフ（質問間で永続化しない）なので、
迷路の「時間的減衰」は不要。代わりに以下のマッピング：

```python
# HotpotQA edge feature vector
edge = {
    "w_ctx":     0.90,   # ← maze.attention相当（context distance decay）
    "w_sim":     0.72,   # ← maze.ag_attention相当（TF-IDF cosine）
    "dg_score":  0.0,    # ← maze.dg_attention相当（ゲージF値フィードバック）
    "w_reward":  0.0,    # ← maze.propagated相当（正解/誤答フィードバック）
}

# 統合スコア（乗算 → 全次元が良いエッジだけ残る）
strength = w_sim * σ(-dg_score / τ_dg) * σ(w_reward / τ_r)
```

**E3Bとの違い**: E3Bは `strength = max(w_ctx, w_sim)` で1次元のthreshold。
3-attention は**乗算ゲーティング**で多次元フィルタ。弱い類似度でも
構造的に重要（dg_score良好）なら残り、強い類似度でも構造的にノイズ
（dg_score悪化）なら除外される。

### 8.4 Implementation Path

| Phase | What | Complexity | Dependency |
|:-----:|------|:----------:|:----------:|
| Phase 1 | `dg_score`記録: geDIG計算後、各エッジにΔF値を書き戻す | Low | E3Aベース |
| Phase 2 | 乗算スコア: `strength = w_sim * σ(-dg/τ)` でBetti+re-rank | Medium | Phase 1 |
| Phase 3 | `w_reward`追加: 正解→強化、誤答→減衰（質問間永続化が必要） | High | Phase 2 |

Phase 1-2 は**1質問内で完結**するため、永続化不要で実験可能。
Phase 3 は質問間でグラフを持ち越す必要があり、アーキテクチャ変更が大きい。

### 8.5 Why This Works (E3Bの問題を解決する理由)

E3Bの失敗原因と3-attentionの対処:

| E3Bの問題 | 3-attention の解決 |
|-----------|-------------------|
| Re-rankingが弱いエッジで汚染 | 乗算スコアで弱エッジのgraph_scoreが自然に低下 |
| Bettiフィルタだけでは不十分 | strength が Betti AND re-ranking の両方に適用される |
| AGが活性化しない (17%) | dg_score低いエッジは除外 → 隙間が生まれ → AG発火 |
| 閾値が固定的 (threshold=0.5) | sigmoid温度τで連続的に制御、離散的な閾値不要 |

**核心**: 迷路実験の3-attentionは「離散的な構造体（エッジの有無）を
確率的な特徴量で操作する」アーキテクチャ。これはまさにgeDIGの
「離散構造 + 確率操作」パラダイムの実装。

---

## 9. Future: From 3-Attention to Persistent Homology

Weighted filtration at a single threshold is a stepping stone. The full
persistent homology approach computes Betti numbers at **all thresholds**
and produces a **persistence diagram** (birth-death pairs):

```
For threshold t in [0.0, 0.1, ..., 1.0]:
    (beta_0(t), beta_1(t)) = compute_betti_filtered(g, t)
```

Long-lived features (born early, die late) represent robust topological
structure. Short-lived features are noise. This provides a richer signal
than single-threshold filtration and is the mathematical framework for
the maze experiment's multi-feature edge architecture.

Implementation complexity is moderate (nested loop over thresholds) but
requires redesigning the gauge formula to consume persistence diagrams
rather than scalar Betti numbers.
