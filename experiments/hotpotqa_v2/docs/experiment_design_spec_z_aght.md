# Spec Z: Analytical Heterogeneous Graph Transformer (AGHT)

## 概要

Sentence Graph と Token Graph を統合した **Heterogeneous Graph** 上で、
**QKV ベースのエッジ評価** (Analytical Graph Attention) を行う。

HGT (Hu et al., WWW 2020) の構造を geDIG 原理で分析的に構築し、
学習不要（教師データ不要）で AG/DG エッジ分類を実現する。

## 動機

### 現状の問題

```
現状: 2つの独立グラフ → スコア後混ぜ

Sentence Graph          Token Graph
(文書間構造)            (文書内構造)
  score_s                 score_t
     \                   /
      (1-w)·s + w·t         ← 構造的情報が失われる
           ↓
      final_score
```

失われている情報:
1. Token Graph の DG ギャップが Sentence Graph のエッジ重みに反映されない
2. Sentence Graph の文書間接続が Token Graph のクエリ関連度計算に使えない
3. クロスレベル推論パス（文書A のトークンが文書B の文に橋渡し）が不可能

### 解決策: 統合ヘテログラフ + QKV エッジ評価

```
統合グラフ (Unified Heterogeneous Graph)
┌──────────────────────────────────────────┐
│ Sentence nodes ──Tier1/2/3──> Sentence   │
│      │↕                                  │
│  contains / represents (cross-level)     │
│      │↕                                  │
│ Token nodes ──dep/same_lemma──> Token    │
│                                          │
│ + cross-doc same_lemma エッジ (新規)      │
└──────────────────────────────────────────┘
```

## アーキテクチャ

### Phase 配置

```
Phase 1:   BM25 retrieval (100 docs)
Phase 1.8: Early Token Graph → AG/DG routing (Spec W)
Phase 2:   CoT (DG-Deep のみ)
Phase 2.6: RIA (DG-Deep のみ)
Phase 3':  Unified Heterogeneous Graph 構築  ← NEW (Phase 3+4.5 統合)
Phase 4':  QKV Edge Evaluation              ← NEW
Phase 5':  Graph Attention Scoring          ← NEW (Phase 5+5.25 統合)
Phase 6:   BM25 blend → final ranking
```

### ノードタイプ

| Type | ID format | Attributes | 説明 |
|------|-----------|------------|------|
| S (Sentence) | `s_{doc}_{sent}` | para_idx, sent_idx, text, title, entities | 文ノード |
| T (Token) | `t_{doc}_{token_i}` | lemma, pos, sent_idx, doc_idx | トークンノード |

### エッジタイプ (8 種)

#### Intra-level edges (既存の拡張)

| # | Type | Level | Direction | Base Cost | 説明 |
|---|------|-------|-----------|-----------|------|
| 1 | `context` | S↔S (同文書) | bidir | 0.05-0.10 | 隣接文 |
| 2 | `entity_overlap` | S↔S (異文書) | bidir | 0.20-0.50 | 判別的エンティティ共有 |
| 3 | `similarity` | S↔S (異文書) | bidir | 0.50-0.80 | TF-IDF cosine |
| 4 | `dep` | T→T (同文) | directed | 0.10-0.20 | spaCy 依存構文 |
| 5 | `same_lemma` | T↔T (同文書) | bidir | 0.30-0.70 | 文書内語彙結束 |

#### Cross-level edges (新規)

| # | Type | Level | Direction | Base Cost | 説明 |
|---|------|-------|-----------|-----------|------|
| 6 | `contains` | S→T | directed | 0.05 | 文がトークンを含む |
| 7 | `same_lemma_x` | T↔T (異文書) | bidir | 0.30-0.70 | 文書横断の語彙結束 |

#### Virtual edges

| # | Type | Level | Direction | Base Cost | 説明 |
|---|------|-------|-----------|-----------|------|
| 8 | `cot_link` | S↔S | bidir | 0.10-0.30 | CoT 概念による仮想接続 |

### エッジ数の見積もり

```
50 docs × 6 sents/doc = 300 S nodes
50 docs × 200 tokens/doc = 10,000 T nodes (content words only)
Total nodes: ~10,300

Edges:
  context (S↔S):       ~250  (隣接文)
  entity_overlap (S↔S): ~500  (判別的エンティティ)
  similarity (S↔S):    ~300  (TF-IDF > 0.3)
  dep (T→T):           ~8,000 (依存木)
  same_lemma (T↔T):    ~3,000 (文書内)
  contains (S→T):      ~10,000 (全 content token)
  same_lemma_x (T↔T):  ~2,000 (文書横断、頻出語除外)
  cot_link (S↔S):      ~100  (CoT 概念)
Total edges: ~24,000

NetworkX DiGraph with ~10K nodes, ~24K edges → メモリ ~50MB, 構築 ~2s
```

## QKV Edge Evaluation

### 理論的基盤

Standard Transformer:
```
α_ij = softmax(Q_i · K_j / √d_k) · V_j
```

Graph Transformer (HGT):
```
Q_i = W_Q^(τ(i)) · H[i]    (ノードタイプ別の射影)
K_j = W_K^(τ(j)) · H[j]
V_j = W_V^(φ(e)) · H[j]    (エッジタイプ別の射影)
```

Analytical HGT (我々):
```
Q(u) = [w_q · handcrafted_features(u, query)]   (3次元)
K(v) = [w_k · handcrafted_features(v)]           (3次元)
V(e) = [w_v · edge_features(e)]                  (3次元)

α = dot(Q, K) / √3                               (scaled dot-product)
f = cost(e) - λ · α                               (F-eval)
flow = α · norm(V)                                (message passing)
```

### Q 特徴量 (Query-dependent, per source node)

ノードタイプ別に異なる特徴量を使用 (HGT の type-specific projection に対応):

**Token ノード:**
```python
Q_T = [
    w_q1 * direct_match,      # lemma ∈ query_lemmas ? 1.0 : 0.0
    w_q2 * nbr_density,       # 1-hop 近傍の query match 率
    w_q3 * 0.0,               # (Token には CoT 概念なし)
]
```

**Sentence ノード:**
```python
Q_S = [
    w_q1 * query_entity_frac,  # query entities 含有率
    w_q2 * nbr_query_density,  # 隣接文の query match 率
    w_q3 * cot_concept_overlap, # CoT concept overlap
]
```

### K 特徴量 (Node-intrinsic, query-independent)

**Token ノード:**
```python
K_T = [
    w_k1 * pos_weight,        # POS importance (NOUN=1.0, PROPN=1.0, VERB=0.8, ADJ=0.6)
    w_k2 * idf_norm,          # IDF (文書横断出現頻度の逆数), normalized [0,1]
    w_k3 * dep_centrality,    # 依存木での centrality (head=1.0, leaf=0.3)
]
```

**Sentence ノード:**
```python
K_S = [
    w_k1 * entity_density,    # エンティティ数 / 文長, normalized
    w_k2 * tfidf_norm,        # TF-IDF 特徴量ノルム, normalized
    w_k3 * position_score,    # 文書内位置 (先頭=1.0, 末尾=0.5)
]
```

### V 特徴量 (Edge-intrinsic)

```python
V = [
    w_v1 * base_cost,         # エッジタイプ別基本コスト [0.05-0.80]
    w_v2 * bridge_flag,       # Tarjan bridge = 1.0, cycle edge = 0.0
    w_v3 * level_crossing,    # cross-level = 1.0, same-level = 0.0
]
```

### Attention 計算

```python
def compute_edge_attention(u, v, e, query_lemmas, cot_concepts):
    """QKV-based edge evaluation."""
    # Step 1: Feature vectors
    Q = compute_Q(u, query_lemmas, cot_concepts)  # [3,]
    K = compute_K(v)                                # [3,]
    V = compute_V(e)                                # [3,]

    # Step 2: Scaled dot-product attention
    alpha = np.dot(Q, K) / np.sqrt(3)  # d_k = 3

    # Step 3: F-eval (AG/DG classification)
    f_value = base_cost(e) - lambda_ * alpha
    if f_value < theta:
        f_class = "AG"
        edge_cost = 1.0                    # confirmed, low traversal cost
    else:
        f_class = "DG"
        edge_cost = 1.0 + (f_value - theta)  # uncertain, scaled cost

    # Step 4: Flow weight for message passing
    flow = alpha * np.linalg.norm(V)

    return alpha, f_class, edge_cost, flow
```

### AG/DG 分類の意味

```
α = dot(Q, K) / √3

Q high × K high → α large  → f = cost - λα < θ → AG
  "クエリが求めている" かつ "相手が持っている" = 確認済み情報

Q high × K low  → α small  → f = cost - λα ≥ θ → DG
  "クエリが求めている" が "相手が持っていない" = 推論ギャップ

Q low  × K high → α small  → f ≥ θ → DG
  "クエリが求めていない" が "相手は情報豊富" = 未利用情報

Q low  × K low  → α small  → f ≥ θ → DG
  両方弱い = ノイズ
```

## Message Passing (Graph Attention Propagation)

F-eval 後のグラフ上で message passing を実行:

```python
def graph_attention_propagation(graph, n_iterations=2, mp_alpha=0.3):
    """Attention-weighted message passing on unified graph."""
    # Initialize: node relevance = Q-score (query-dependent)
    for node in graph.nodes:
        graph.nodes[node]["relevance"] = graph.nodes[node]["q_score"]

    for _ in range(n_iterations):
        new_relevance = {}
        for node in graph.nodes:
            # Aggregate: attention-weighted neighbor relevance
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                new_relevance[node] = graph.nodes[node]["relevance"]
                continue

            agg = sum(
                graph[node][nbr]["flow"] * graph.nodes[nbr]["relevance"]
                for nbr in neighbors
            )
            agg /= max(len(neighbors), 1)

            # Update: interpolate with self
            new_relevance[node] = (
                (1 - mp_alpha) * graph.nodes[node]["relevance"]
                + mp_alpha * agg
            )

        for node, rel in new_relevance.items():
            graph.nodes[node]["relevance"] = rel
```

## Document Scoring

Message passing 後、文書ごとのスコアを算出:

```python
def score_documents(graph, doc_id_map):
    """Aggregate node relevance to document scores."""
    doc_scores = defaultdict(list)

    for node, data in graph.nodes(data=True):
        doc_idx = data.get("para_idx") or data.get("doc_idx")
        if doc_idx is not None:
            doc_key = doc_id_map.get(f"doc_{doc_idx}")
            if doc_key:
                doc_scores[doc_key].append(data["relevance"])

    # Aggregate: mean relevance of all nodes in document
    return {
        doc_id: np.mean(scores)
        for doc_id, scores in doc_scores.items()
        if scores
    }
```

## パラメータ

### 調整可能パラメータ (10個)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `w_q1` | 1.0 | [0.5, 2.0] | Q: direct match weight |
| `w_q2` | 0.6 | [0.2, 1.0] | Q: neighborhood density weight |
| `w_q3` | 0.8 | [0.3, 1.5] | Q: CoT relevance weight |
| `w_k1` | 1.0 | [0.5, 2.0] | K: importance weight |
| `w_k2` | 0.8 | [0.3, 1.5] | K: discriminativeness weight |
| `w_k3` | 0.5 | [0.2, 1.0] | K: structural role weight |
| `w_v1` | 1.0 | fixed | V: base cost (= edge type cost) |
| `w_v2` | 0.3 | [0.1, 0.5] | V: bridge bonus |
| `w_v3` | 0.2 | [0.1, 0.5] | V: level crossing penalty |
| `lambda` | 1.0 | [0.5, 2.0] | F-eval lambda |

### 固定パラメータ

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_k` | 3 | Attention dimension |
| `theta` | 30th percentile | F-eval threshold (data-driven) |
| `mp_iterations` | 2 | Message passing iterations |
| `mp_alpha` | 0.3 | Message passing interpolation |
| `max_tokens_per_doc` | 200 | Content words only |
| `same_lemma_x_max_freq` | 10 | Cross-doc lemma 最大出現文書数 |
| `contains_max_per_sent` | 20 | Contains edges per sentence cap |

## CLI Interface

```bash
python scripts/run_bright.py \
    --mode cot_retrieval \
    --bm25-engine pyserini \
    --unified-graph \               # Enable Spec Z
    --aght-lambda 1.0 \             # F-eval lambda
    --aght-mp-iterations 2 \        # Message passing iterations
    --aght-mp-alpha 0.3 \           # MP interpolation weight
    --aght-cross-doc-lemma \        # Enable cross-doc same_lemma_x edges
    --domains biology \
    --limit 10
```

`--unified-graph` 有効時:
- Phase 3 (entity graph) + Phase 4.5 (token graph) → Phase 3' (unified graph) に統合
- Phase 5 (geDIG scoring) + Phase 5.25 (entity F-eval) → Phase 5' (graph attention scoring) に統合
- 既存の `--token-graph`, `--entity-feval` は `--unified-graph` と排他

## 後方互換性

- `--unified-graph` なし → 既存パイプライン (Phase 3/4.5/5/5.25 分離) がそのまま動作
- `--unified-graph` あり → Phase 3'/4'/5' に切り替え
- 結果の JSONL には `unified_graph: true` フラグと AGHT 固有の diagnostics を追加

## 評価計画

### Smoke test (10q biology)
```bash
python scripts/run_bright.py --mode cot_retrieval --bm25-engine pyserini \
    --unified-graph --domains biology --limit 10 \
    --output-dir results/v31_specz_10q
```

### 50q comparison
```bash
python scripts/run_bright.py --mode cot_retrieval --bm25-engine pyserini \
    --unified-graph --early-token-graph --enhanced-graph \
    --domains biology --limit 50 \
    --output-dir results/v31_specz_50q
```

### Baseline comparison targets

| Config | nDCG@10 | Description |
|--------|---------|-------------|
| Spec X (v27) | 0.439 | Enhanced Graph (current best) |
| Spec W (v26) | 0.417 | Early TG |
| Pyserini+RIA (v25) | 0.410 | Baseline |

### 成功基準

- nDCG@10 ≥ 0.45 (Spec X 対比 +2.5% 以上)
- レイテンシ ≤ 50s/query (Spec X の 35s + 統合グラフ構築のオーバーヘッド)
- AG/DG 分類精度: DG クエリでの改善幅 > AG クエリでの改善幅

## 論文フレーミング

```
Title: "Analytical Heterogeneous Graph Transformer for
        Reasoning-Intensive Document Retrieval"

Key contributions:
1. geDIG 原理に基づく分析的 Graph Attention (学習不要)
2. Sentence-Token 統合ヘテログラフによるクロスレベル推論パス
3. QKV 分離による AG/DG エッジ分類の理論的基盤
4. BRIGHT ベンチマークでの SOTA
```

## 実装ファイル

| File | Description |
|------|-------------|
| `src/unified_graph.py` | 統合ヘテログラフ構築 + QKV エッジ評価 (NEW) |
| `src/bright_cot_pipeline.py` | Phase 3'/4'/5' 統合 (MODIFY) |
| `scripts/run_bright.py` | CLI flags (MODIFY) |
