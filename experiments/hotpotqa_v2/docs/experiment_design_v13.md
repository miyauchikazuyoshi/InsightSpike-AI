# Experiment Design v13 — Spec G: Hybrid Sentence+Episode Entity Graph

## 背景

### これまでの実験結果

| Spec | Pipeline | nDCG@10 | 変更点 |
|------|----------|---------|--------|
| BM25 | baseline | 0.068 | — |
| A | CoT Re-retrieval (sentence) | **0.152** | entity_graph + CoT injection |
| B+C | Adaptive retrieval | 0.160 | early stopping + threshold |
| E | geDIG routing (episode_graph) | 0.098 | episode独自グラフ構造 |
| F | Episode → entity_graph | 0.127 | sentence→episodeに置換 |

### 核心的な知見

1. **entity_graph のスコアリング (5-component) は優秀** — Spec E (episode_graph独自構造 0.098) < Spec F (episode→entity_graph 0.127) < Spec A (sentence→entity_graph 0.152)
2. **sentence split は entity_graph と相性が良い** — Tier 2 (entity overlap) エッジは単語レベルの共有に依存。文分割は粒度が細かく、エンティティ密度が高い
3. **episode は意味的に完結した知識単位** — だが entity_graph では粒度が粗すぎ (nodes: ~85 vs ~240)、グラフが疎になる
4. **episode のテキスト自体は質が高い** — LLM が抽出した原子的知識 (S-V-O 構造)

### 仮説

**Sentence と Episode を別レイヤーのノードとして同一グラフに入れれば、sentence の細粒度エンティティ接続と、episode の意味的ブリッジを同時に活用できるのではないか？**

## Spec G: ハイブリッドグラフ設計

### Single Variable

`build_sentence_graph()` に渡す `sentences_list` に、各ドキュメントの sentence split **に加えて** episode texts も含める。

```
現行 (Spec A): sentences_list[i] = _split_sentences(doc.content)
提案 (Spec G): sentences_list[i] = _split_sentences(doc.content) + [ep.text for ep in episodes]
```

### グラフ構造

```
Doc 0:
  node 0: sentence "Photosynthesis converts light energy..."
  node 1: sentence "Chlorophyll absorbs red and blue light..."
  node 2: sentence "The Calvin cycle fixes CO2..."
  node 3: episode  "Photosynthesis converts light energy into chemical energy through chlorophyll"
  node 4: episode  "The Calvin cycle fixes carbon dioxide into organic molecules"

Doc 1:
  node 5: sentence "Plants evolved chloroplasts from..."
  node 6: episode  "Chloroplasts evolved from cyanobacteria through endosymbiosis"
```

エッジは既存の Tier 1/2/3 ルールがそのまま適用される:
- Tier 1: 同一ドキュメント内の隣接ノード (sentence-sentence + sentence-episode)
- Tier 2: エンティティ共有 (cross-document, **sentence-episode 間も自然に接続**)
- Tier 3: TF-IDF cosine similarity

### 期待されるメリット

1. **Episode がクロスドキュメントのブリッジとなる** — Episode は意味的に完結しているため、異なるドキュメントの sentence と entity overlap しやすい
2. **Sentence の細粒度は維持** — 既存の sentence ノードはそのまま残る
3. **PageRank の伝播パス増加** — episode を介した新しい経路が PageRank を高関連ドキュメントに集中させる

### リスク

1. **PageRank 分散** — 同一ドキュメントのノード数が増える (~5 sentences + ~3 episodes → ~8 nodes/doc)。ただしスコアリングはドキュメントレベルで集約するので影響は限定的
2. **同一ドキュメント内の自明なエッジ** — sentence と episode は同じ文書由来なので Tier 2 overlap は当然高い → Tier 1 扱い (cost 0.05-0.10) になるのでグラフの重み構造には影響少ない
3. **ノード数増加** — ~240 (sentence) + ~85 (episode) ≈ ~325 nodes。エッジ計算 O(N²) は 325² = 105K ペアで、十分高速

### スケール見積もり

| 項目 | Spec A (sentence) | Spec G (hybrid) | 増加率 |
|------|-------------------|-----------------|--------|
| Nodes/query | ~240 | ~325 | +35% |
| Edge pairs | ~57K | ~106K | +83% |
| 推定latency | ~10s | ~13s | +30% |

## 実装

### 変更箇所

**`bright_cot_pipeline.py` Phase 3 のみ** (lines ~412-444):

```python
# graph_mode == "hybrid"
if self.graph_mode == "hybrid" and self.episode_index is not None:
    # Base: sentence split (proven effective)
    sents = _split_sentences(content, max_sentences=30)
    if not sents:
        sents = [content[:500]]

    # Augment: add episode texts
    doc_eps_list = self.episode_index.get_doc_episodes(
        self.dense_domain, [doc["id"]]
    )
    ep_texts = [ep.text for ep in doc_eps_list[0].episodes if ep.text]

    # Cap total nodes per doc to avoid explosion
    max_total = 30
    if len(sents) + len(ep_texts) > max_total:
        ep_texts = ep_texts[:max(max_total - len(sents), 3)]

    sents = sents + ep_texts
```

**`run_bright.py`**:
- `--graph-mode` choices に `"hybrid"` を追加

### 変更しないもの

- `entity_graph.py` — `build_sentence_graph()` はテキスト非依存 (list[list[str]])。変更不要
- CoT injection — 既存のまま
- 5-component scoring — 既存のまま
- BM25 blending — 既存のまま

## 実験計画

### Step 1: スモークテスト (10q, biology)

```bash
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v13_smoke_hybrid \
    --graph-top-k 50 --rerank-alpha 0.1 \
    --graph-mode hybrid \
    --episode-index-dir experiments/hotpotqa_v2/data/bright/episodes/ \
    --max-queries 10
```

確認事項:
- ノード数が sentence mode (~240) より多い (~300+) か
- nDCG が 0 ではないか (グラフ構造が壊れていないか)
- latency が許容範囲内か (< 20s/query)

### Step 2: フルラン (323q, 3-domain)

```bash
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology,economics,stackoverflow \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v13_bright_hybrid_graph \
    --graph-top-k 50 --rerank-alpha 0.1 \
    --graph-mode hybrid \
    --episode-index-dir experiments/hotpotqa_v2/data/bright/episodes/
```

### 成功基準

| 基準 | 閾値 |
|------|------|
| Spec A 超え | nDCG@10 > 0.152 |
| 有意な改善 | nDCG@10 > 0.165 (Spec A +8.5%) |
| latency許容 | < 20s/query (Spec A比 +100%) |

## 代替案: Episode → CoT 拡張 (Spec G')

ハイブリッドグラフが不発の場合の代替アプローチ:

**グラフ構造は sentence のまま (Spec A)。Episode のエンティティを CoT entities に追加して、Phase 4 の CoT bridge bonus を強化する。**

```python
# Phase 2.5: Extract entities from episodes
episode_entities = set()
for doc_idx, _ in graph_candidates:
    doc_eps = self.episode_index.get_doc_episodes(domain, [doc["id"]])
    for ep in doc_eps[0].episodes:
        episode_entities |= extract_entities(ep.text)

# Phase 4: Inject CoT with augmented concepts
augmented_concepts = cot_concepts | episode_entities
self._inject_cot_nodes(graph, cot_text, augmented_concepts, ...)
```

メリット: Spec A の構造を完全に保持しつつ、episode の知識を CoT 経由で注入
リスク: CoT bridge が過剰接続になり、スコアが平滑化される可能性
