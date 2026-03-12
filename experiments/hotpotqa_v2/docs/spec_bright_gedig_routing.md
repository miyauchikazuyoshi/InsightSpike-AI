# BRIGHT Spec E: geDIG-driven Adaptive Routing 実験仕様

**日付**: 2026-03-12
**実験ID**: v12 BRIGHT Spec E
**担当**: InsightSpike-AI

---

## 1. 動機

### 1.1 Spec D の教訓

Spec D (Unified) は全対策を同時投入したが、nDCG@10 = 0.1564 と
Adaptive (0.1596) を下回った。最大の原因は **ルーティング精度の不足**:

| ルーティング | 効果 |
|------------|------|
| β₀閾値 (Spec B+C) | Biology 0.2367（最高性能） |
| 全クエリ tier 3 (Spec D) | Biology 0.1989（-16%後退） |

**結論**: パイプライン複雑化よりもルーティング精度が最大レバレッジ。

### 1.2 現行ルーティングの問題

現行の `_compute_pre_beta0()` はクエリをグラフに注入せず、
BM25 top-50 文書のみで entity graph を構築し、connected components 数を返す。

```python
# 現行 (bright_cot_pipeline.py L455-480)
def _compute_pre_beta0(self, top_candidates, docs):
    # BM25 top-k → entity graph → β₀ = len(connected_components)
    pre_graph = build_sentence_graph(titles, sentences_list)
    return len(list(nx.connected_components(pre_graph)))
```

これは **クエリの情報を一切使わない** ルーティング:
- β₀ が高い（多数の孤立コンポーネント）→ 候補プール内の話題が分散 → CoT 必要と判定
- β₀ が低い（少数の大コンポーネント）→ 候補が密に接続 → CoT 不要と判定

問題:
1. クエリの推論的複雑さを無視 — 簡単なクエリでも β₀ が高ければ tier 3 に
2. クエリがグラフにどう「効く」かの情報がない — geDIG の本質的な強みを使っていない
3. 閾値の恣意性 — β₀=5 の閾値に理論的根拠がない

### 1.3 geDIG の本来の設計

geDIG (Generalized Differential Information Gain) はクエリ（観測）を
知識グラフに注入し、トポロジー変化を測定する理論的フレームワーク:

```
geDIG = Δ_GED_norm - λ · (Δ_H_norm + β_sp · Δ_SP_rel)
```

- **Δ_GED_norm**: 構造コスト（グラフ編集距離）— クエリ注入による構造変化量
- **Δ_H_norm**: 情報利得（エントロピー変化）— クエリが局所的多様性をどう変えるか
- **Δ_SP_rel**: 最短路利得 — クエリがグラフの「ショートカット」を作るか

**geDIG < 0**: 情報利得が構造コストを上回る → DG モード（探索不要、グラフで統合済み）
**geDIG > 0**: 構造コストが情報利得を上回る → AG モード（追加探索＝CoT が必要）

→ **geDIG 値そのものがルーティングの最良指標**。

---

## 2. 提案手法

### 2.1 概要

BM25 top-k 文書から entity graph を構築し、
クエリテキストをチャンク化してグラフに注入。
注入前後の geDIG 値でルーティングを決定:

```
[geDIG Routing Pipeline]
Phase 0:  BM25(query) → top-100
Phase 0.5: Entity graph 構築 (top-50) → G_before
Phase 0.6: クエリチャンク化 → 注入 → G_after     ★新規
Phase 0.7: geDIG(G_before, G_after) → routing_decision ★新規
Phase 1+:  routing_decision に応じたパイプライン実行
```

### 2.2 ルーティング判定

| geDIG 値 | 解釈 | パイプライン | 処理 |
|----------|------|------------|------|
| geDIG ≪ 0 | 高情報統合 | DG (tier 1) | BM25 + Graph のみ |
| geDIG ≈ 0 | 中間 | Moderate (tier 2) | + CoT injection (re-retrieve なし) |
| geDIG ≫ 0 | 低統合 | AG (tier 3) | + CoT + Re-retrieval (aggressive) |

閾値: τ_dg (DG判定), τ_ag (AG判定) — データドリブンで決定。

---

## 3. エピソード化設計 ★核心

### 3.1 メイズとの構造的対応

geDIG のメイズ実験では、知識グラフの単位は**エピソード**:
- 1エピソード = 1つの行動 + その結果（原子的な経験単位）
- エピソード間は temporal edge で時系列接続
- geDIG はエピソードの注入前後でトポロジー変化を測定

BRIGHT に正しく写像するなら、文書もクエリも**エピソード単位**で
グラフに表現すべき。生の文をノードにするのは、メイズで言えば
「座標の1ピクセルごとにノードを作る」に相当し、粒度が不適切。

### 3.2 エピソードの定義

| | メイズ | BRIGHT |
|---|------|--------|
| エピソード | 1移動 + 結果 | 1推論ステップ |
| 原子性 | これ以上分割不可 | 1つの概念・主張・問い |
| 時系列 | step_t → step_{t+1} | 推論の論理的順序 |
| 接続先 | 空間的隣接セル | 関連する文書エピソード |

### 3.3 LLM によるエピソード分解

文単位や固定長ではなく、**LLM がエピソード境界と論理的接続を判定**する。

#### 3.3.1 クエリのエピソード化

```
Decompose into reasoning episodes. Each episode is ONE atomic
reasoning step. Output JSON: [{text, type, connects_to}]

Query: "Insects exhibit phototaxis, moving toward light.
However, recent studies suggest this is a navigational error
using celestial cues. What neurological mechanisms cause
this confusion with artificial light?"

→ [
    {"id": 0, "text": "Insects exhibit phototaxis toward light",
     "type": "premise", "connects_to": []},
    {"id": 1, "text": "Recent studies suggest navigational error
     using celestial cues",
     "type": "hypothesis", "connects_to": [0]},
    {"id": 2, "text": "What neurological mechanisms cause confusion
     with artificial light",
     "type": "question", "connects_to": [0, 1]}
  ]
```

エピソード type の分類:
- `premise`: 前提・背景の提示
- `hypothesis`: 仮説・主張
- `question`: 問い
- `evidence`: 根拠・データ
- `constraint`: 制約条件

#### 3.3.2 文書のエピソード化

クエリだけでなく**文書側も**エピソード化する。
片方だけでは episode ↔ sentence の非対称グラフになり geDIG の測定が不均一。

```
Decompose into knowledge episodes. Each episode is ONE atomic
fact, definition, or claim. Output JSON: [{text, type, connects_to}]

Document: "Phototaxis is the movement of organisms toward or away
from light. Positive phototaxis in insects involves compound eye
photoreceptors. Studies show dorsal light response maintains
flight orientation."

→ [
    {"id": 0, "text": "Phototaxis is movement toward or away from light",
     "type": "definition", "connects_to": []},
    {"id": 1, "text": "Positive phototaxis involves compound eye photoreceptors",
     "type": "mechanism", "connects_to": [0]},
    {"id": 2, "text": "Dorsal light response maintains flight orientation",
     "type": "evidence", "connects_to": [1]}
  ]
```

→ **episode ↔ episode のグラフ**: メイズと同型の構造が実現。

### 3.4 コスト管理: Biology パイロット

全 214K 文書のエピソード化はコストが大きいため、**biology ドメインのみ**で先行検証。

| 対象 | 件数 | LLM コスト | タイミング |
|------|------|----------|----------|
| Biology 文書 | 57,359 | ~$3-5 (Batch API) | オフライン (1回) |
| Biology クエリ | 103 | ~$0.02 | オンライン |

**コスト削減策**:
1. **Batch API** (50%オフ)
2. **短い文書 (≤3文) はスキップ**: そのまま1エピソード扱い → LLM不要
3. **パラグラフ境界を事前ヒント**: 改行がある文書は候補として渡す → 出力短縮

```python
def episodify_document(doc_text: str) -> list[dict]:
    sentences = split_sentences(doc_text)
    if len(sentences) <= 3:
        # 短い文書: ヒューリスティック (LLM不要)
        return [{"id": 0, "text": doc_text, "type": "single",
                 "connects_to": []}]
    else:
        # LLM でエピソード分解
        return llm_decompose(doc_text)
```

推定: 57K 文書のうち ~40% が 3文以下 → LLM 呼び出し ~35K 件。

---

## 4. エッジ設計

### 4.1 設計原則: シナプスモデル

geDIG は**エッジを二値（存在/非存在）で扱う**:
- GED: エッジ本数の変化のみ
- IG: エッジ存在で neighborhood 定義 → ノード feature で計算
- SP: ホップ数ベースの非加重最短路

シナプスが発火するかしないかの二値であるように、
エッジの属性（cost, type）ではなく**接続パターン（トポロジー）**が情報を担う。

→ エッジ設計 = **「どのノード間にエッジを張るか」の密度制御**。

### 4.2 エッジの種類と生成条件

#### A) エピソード内エッジ（同一文書/クエリ内）

| エッジ | 接続 | 生成条件 |
|--------|------|---------|
| **Sequential** | ep_i ↔ ep_{i+1} | 常時（時系列順序） |
| **connects_to** | ep_i ↔ ep_j | LLM が出力した論理的接続 |

Sequential = メイズの temporal edge と同型。
connects_to = LLM が判定した推論の論理構造。

#### B) エピソード間エッジ（文書↔文書、クエリ↔文書）

生成条件は**複数の類似度シグナル**を使うが、geDIG 的にはすべて同一の二値エッジ。
エッジ存否の判定に複数シグナルを使い、密度を制御する。

```python
def should_create_edge(ep_a, ep_b, k_budget_remaining):
    """複数シグナルのスコアで上位 k_target に入るか判定。"""
    score = 0.0
    # 1. Entity overlap
    overlap = entity_jaccard(ep_a.text, ep_b.text)
    if overlap > 0:
        score += 3.0 * overlap  # entity は強いシグナル
    # 2. Dense embedding similarity
    cos_sim = ep_a.embedding @ ep_b.embedding
    score += cos_sim
    # 3. TF-IDF cosine
    tfidf = tfidf_cosine(ep_a.text, ep_b.text)
    score += 0.5 * tfidf
    return score, k_budget_remaining > 0
```

#### C) 密度制御: k_target

各エピソードノードの接続数をターゲット値で制御:

| ノード種別 | k_target | 根拠 |
|-----------|----------|------|
| クエリエピソード | 4 | 少なすぎると geDIG 不感、多すぎるとハブ化 |
| 文書エピソード | 自然接続数 | 既存の entity/context edge で決まる |

クエリ→文書エピソードへのエッジは、複合スコア上位 k_target 件のみ生成。
全種類の類似度をまとめて1つのスコアにし、top-k で切る。

### 4.3 結果としてのグラフ構造

```
[クエリ側]                              [文書側]
q_ep0 (premise)                         d_ep0 (definition) ─── d_ep1 (mechanism)
  │  ╲                                     │  connects_to         │  sequential
  │seq ╲ connects_to                       │                      │
  │     ╲                                  d_ep2 (evidence) ──── d_ep3 ...
q_ep1 (hypothesis) ─── cross-edge ──→  d_ep1 (mechanism)
  │  ╲
  │seq ╲ connects_to
  │     ╲
q_ep2 (question) ───── cross-edge ──→  d_ep4 (related claim)
```

geDIG が測定するもの:
- q_ep0→d_ep0, q_ep1→d_ep1, q_ep2→d_ep4 が**異なるコンポーネントを橋渡し**するか → Δβ₀
- クエリチェーンがグラフに**ショートカットを作る**か → SP 利得
- クエリ注入で局所的多様性が**変化する**か → ΔH

---

## 5. ノード特徴量設計

### 5.1 メイズの設計思想

メイズの 10D ノード特徴量は**異なる性質のシグナルを1つのベクトルに同居**させる:

```
dim 0-1:  位置 (空間)        → 「どこか」
dim 2-3:  方向 (行動)        → 「何をしたか」
dim 4:    通過可能性 (構造)   → 「通れるか」
dim 5:    訪問回数 (履歴)     → 「経験量」
dim 6-7:  成功/ゴール (結果)  → 「報酬シグナル」
dim 8-9:  reward/propagated  → 「学習済み価値」
```

geDIG の IG はこのベクトル間の cosine similarity で局所エントロピーを計算。
→ **多面的な類似性**が IG に反映される。

### 5.2 BRIGHT エピソードの特徴量

同じ設計思想で、エピソードノードの特徴量を多次元化:

```python
def compute_episode_feature(episode, query_text=None):
    """Multi-signal episode feature vector."""
    # 意味的内容 (← メイズの position に相当)
    dense_emb = e5_embed(episode.text)           # 384d

    # 語彙一致度 (← passability に相当)
    tfidf_sim = tfidf_cosine(episode.text, query_text) if query_text else 0.0  # 1d

    # エンティティ共有度 (← direction に相当)
    entity_score = entity_jaccard(episode.text, query_text) if query_text else 0.0  # 1d

    # BM25 関連度 (← success に相当)
    bm25_score = normalized_bm25(episode.text, query_text) if query_text else 0.0  # 1d

    # 文書内位置 (← visit count に相当)
    position = episode.id / max(episode.total_in_doc, 1)  # 1d

    return np.concatenate([dense_emb, [tfidf_sim, entity_score, bm25_score, position]])
    # 388d total
```

`feature_weights` でシグナル種別ごとの重みを制御（既存の geDIG 機能を再利用）:

```python
feature_weights = np.ones(388)
feature_weights[0:384] = 0.5    # dense embedding
feature_weights[384] = 2.0      # TF-IDF (語彙一致を重視)
feature_weights[385] = 3.0      # entity (構造一致をさらに重視)
feature_weights[386] = 1.0      # BM25
feature_weights[387] = 0.3      # position (低め)
```

---

## 6. 実装計画

### 6.1 修正・作成ファイル

| ファイル | 操作 | 概要 |
|---------|------|------|
| `scripts/episodify_corpus.py` | **新規** | 文書エピソード化 (Batch API) |
| `scripts/episodify_queries.py` | **新規** | クエリエピソード化 |
| `src/episode_graph.py` | **新規** | エピソードグラフ構築 |
| `src/gedig_router.py` | **新規** | geDIG ベースルーター |
| `src/bright_cot_pipeline.py` | **修正** | geDIG routing 統合 |
| `scripts/run_bright.py` | **修正** | `gedig_routing` モード追加 |
| `scripts/analyze_gedig_routing.py` | **新規** | geDIG 値分布分析 |

### 6.2 `episodify_corpus.py` (新規)

Biology 57K 文書を Batch API でエピソード化。

```bash
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/episodify_corpus.py \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --domain biology \
    --output experiments/hotpotqa_v2/data/bright/episodes/biology_episodes.jsonl \
    --model gpt-4o-mini --batch-api
```

出力: `biology_episodes.jsonl`
```json
{"doc_id": "bio_123", "episodes": [{"id": 0, "text": "...", "type": "definition", "connects_to": []}, ...]}
```

### 6.3 `episode_graph.py` (新規, ~250行)

エピソード単位のグラフを構築。entity_graph の代替。

```python
class EpisodeGraph:
    """Episode-level knowledge graph for geDIG routing."""

    def __init__(self, dense_retriever=None, k_target=4,
                 feature_weights=None):
        ...

    def build_graph(
        self,
        doc_episodes: list[DocEpisodes],   # 文書エピソード群
        query_episodes: list[Episode],      # クエリエピソード群
        query_text: str,
    ) -> tuple[nx.Graph, nx.Graph, np.ndarray, np.ndarray]:
        """
        Returns:
            g_before: 文書エピソードのみのグラフ
            g_after:  g_before + クエリエピソード注入後
            features_before: g_before のノード特徴量
            features_after:  g_after のノード特徴量
        """
        g = nx.Graph()

        # 1. 文書エピソードをノードとして追加
        for doc in doc_episodes:
            for ep in doc.episodes:
                node_id = f"{doc.doc_id}_ep{ep.id}"
                g.add_node(node_id, text=ep.text, type=ep.type,
                           is_query=False)
                # Sequential edges (同一文書内)
                if ep.id > 0:
                    prev = f"{doc.doc_id}_ep{ep.id - 1}"
                    g.add_edge(prev, node_id)
                # connects_to edges (同一文書内)
                for ref in ep.connects_to:
                    ref_node = f"{doc.doc_id}_ep{ref}"
                    if ref_node in g:
                        g.add_edge(ref_node, node_id)

        # 2. 文書間エッジ (entity/dense top-k)
        self._add_cross_doc_edges(g, doc_episodes)

        g_before = g.copy()
        features_before = self._compute_features(g_before, query_text)

        # 3. クエリエピソード注入
        query_nodes = set()
        for ep in query_episodes:
            node_id = f"query_ep{ep.id}"
            g.add_node(node_id, text=ep.text, type=ep.type,
                       is_query=True)
            query_nodes.add(node_id)
            # Sequential
            if ep.id > 0:
                g.add_edge(f"query_ep{ep.id - 1}", node_id)
            # connects_to (クエリ内)
            for ref in ep.connects_to:
                g.add_edge(f"query_ep{ref}", node_id)
            # Cross-edges to doc episodes (top-k_target)
            self._add_query_edges(g, node_id, ep, doc_episodes)

        features_after = self._compute_features(g, query_text)

        return g_before, g, features_before, features_after, query_nodes
```

### 6.4 `gedig_router.py` (新規, ~150行)

```python
class GeDIGRouter:
    """geDIG-based routing using episode graphs."""

    def __init__(self, lambda_weight=1.0, max_hops=2, sp_beta=0.2,
                 tau_dg=-0.3, tau_ag=0.1, feature_weights=None):
        self.gedig_core = GeDIGCore(
            lambda_weight=lambda_weight,
            enable_multihop=True,
            max_hops=max_hops,
            adaptive_hops=True,
            sp_beta=sp_beta,
            ig_source_mode='graph',
            feature_weights=feature_weights,
        )
        ...

    def compute_routing(self, episode_graph_result) -> RoutingDecision:
        g_before, g_after, feat_before, feat_after, focal = episode_graph_result

        result = self.gedig_core.calculate(
            g_prev=g_before, g_now=g_after,
            features_prev=feat_before, features_after=feat_after,
            focal_nodes=focal,
        )

        tier = self._decide_tier(result.gedig_value, result.delta_betti_0)
        return RoutingDecision(
            tier=tier,
            gedig_value=result.gedig_value,
            delta_betti_0=result.delta_betti_0,
            ig_value=result.ig_value,
            ged_value=result.ged_value,
        )

    def _decide_tier(self, gedig_value, delta_betti_0):
        if gedig_value < self.tau_dg:
            return 1  # DG: 高統合 → CoT 不要
        elif gedig_value > self.tau_ag:
            return 3  # AG: 低統合 → aggressive CoT
        elif delta_betti_0 < -2:
            return 1  # component merge → 統合済み
        return 2      # moderate CoT
```

### 6.5 `bright_cot_pipeline.py` 修正

```python
# geDIG routing モード時
if self.gedig_router is not None and self.episode_index is not None:
    # 文書エピソード取得
    doc_episodes = self.episode_index.get_episodes(
        [docs[i]["id"] for i, _ in graph_candidates]
    )
    # クエリエピソード化 (LLM 1call)
    query_episodes = episodify_query(query, self.llm)
    # エピソードグラフ構築 + geDIG routing
    eg_result = self.episode_graph.build_graph(
        doc_episodes, query_episodes, query
    )
    routing = self.gedig_router.compute_routing(eg_result)
    routing_tier = routing.tier
```

### 6.6 `run_bright.py` 修正

```python
# 新モード
--mode gedig_routing

# 新パラメータ
--episode-index-dir       エピソードインデックスディレクトリ
--gedig-lambda 1.0        geDIG λ weight
--gedig-max-hops 2        Multi-hop depth
--gedig-tau-dg -0.3       DG threshold
--gedig-tau-ag 0.1        AG threshold
--gedig-k-target 4        クエリエピソードの接続数ターゲット
```

---

## 7. 実験計画: Biology パイロット

### 7.1 Phase 0: 文書エピソード化 (オフライン, 1回)

```bash
# Batch API で biology 57K 文書をエピソード化
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/episodify_corpus.py \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --domain biology \
    --output experiments/hotpotqa_v2/data/bright/episodes/ \
    --model gpt-4o-mini --batch-api

# エピソード embedding + FAISS
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/build_dense_index.py \
    --data-dir experiments/hotpotqa_v2/data/bright/episodes/ \
    --index-dir experiments/hotpotqa_v2/data/bright/episodes/dense_index/ \
    --domains biology --episode-mode
```

推定: Batch API 24時間以内 + embedding 15分

### 7.2 Phase 1: geDIG 値の探索的分析

Biology 103q のみ。エピソードグラフ + geDIG 値を全クエリで記録（ルーティングなし）。

```bash
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/analyze_gedig_routing.py \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --domains biology \
    --episode-index-dir experiments/hotpotqa_v2/data/bright/episodes/ \
    --dense-index-dir experiments/hotpotqa_v2/data/bright/dense_index \
    --output experiments/hotpotqa_v2/results/v12_gedig_episode_analysis
```

**分析項目**:
1. geDIG 値の分布（ヒストグラム）
2. geDIG 値 vs nDCG@10 の相関（Spec A の biology 結果と照合）
3. Δβ₀, ΔH, GED の各成分分布
4. エピソード数分布（文書平均、クエリ平均）
5. connects_to エッジの密度
6. **geDIG < 0 のクエリは BM25+Graph で nDCG > 0 か？** ← 最重要検証

### 7.3 Phase 2: 閾値最適化

Phase 1 の geDIG 値と既存 nDCG データから τ_dg, τ_ag を決定。
Biology 103q のドメイン内 LOO で過学習チェック。

### 7.4 Phase 3: Biology フルラン (103q)

```bash
export $(grep -v '^#' .env | xargs)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode gedig_routing \
    --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_gedig_biology \
    --graph-top-k 50 --rerank-alpha 0.1 \
    --episode-index-dir experiments/hotpotqa_v2/data/bright/episodes/ \
    --dense-index-dir experiments/hotpotqa_v2/data/bright/dense_index \
    --gedig-lambda 1.0 --gedig-max-hops 2 \
    --gedig-tau-dg -0.3 --gedig-tau-ag 0.1 \
    --gedig-k-target 4
```

### 7.5 Phase 4: 全ドメイン展開判定

Biology パイロットの結果に基づき:
- 成功 (nDCG 改善 + geDIG 判別力あり) → economics, SO に展開 (~$10 追加)
- 部分成功 (geDIG 判別力あるがルーティング閾値が難しい) → 閾値調整
- 失敗 (geDIG と nDCG 無相関) → エピソード粒度 or 特徴量設計を再検討

---

## 8. 成功基準

### 8.1 Biology パイロット

| 指標 | 目標 | 根拠 |
|------|------|------|
| Biology nDCG@10 | ≥ 0.24 | Adaptive の 0.237 超え |
| Biology nDCG>0 率 | ≥ 50% | Adaptive 45/103 超え |
| geDIG-nDCG 相関 | r > 0.3 | 弱〜中相関以上 |
| DG群平均nDCG > AG群平均nDCG | | geDIG が特性を捕捉 |
| CoT スキップ率 | 30-50% | コスト効率 |

### 8.2 全ドメイン展開時

| 指標 | 目標 |
|------|------|
| 全体 nDCG@10 | ≥ 0.18 (BM25比 +164%) |
| 全体 nDCG>0 率 | ≥ 40% |
| 全ドメインで Adaptive 超え | |

---

## 9. コスト見積もり

### 9.1 Biology パイロット

| 項目 | コスト |
|------|-------|
| 文書エピソード化 (57K docs, Batch API) | ~$3-5 |
| エピソード embedding + FAISS | CPU 30分 / $0 |
| クエリエピソード化 (103q) | ~$0.02 |
| Phase 1: geDIG 分析 (103q) | CPU ~10分 / $0 |
| Phase 3: フルラン (103q, CoT) | ~$2 |
| **Biology パイロット合計** | **~$5-7** |

### 9.2 全ドメイン展開時 (追加)

| 項目 | 追加コスト |
|------|----------|
| Economics 文書エピソード化 (50K) | ~$3-4 |
| SO 文書エピソード化 (107K) | ~$6-8 |
| 220q 追加フルラン | ~$4 |
| **全ドメイン追加合計** | **~$13-16** |

---

## 10. リスクと緩和策

| リスク | 影響 | 緩和策 |
|-------|------|--------|
| LLM エピソード分解の品質ばらつき | グラフ構造が不安定 | 短文書スキップ + 出力バリデーション |
| geDIG 値の分布が偏る | ルーティングが一方に集中 | λ, sp_beta, feature_weights 調整 |
| エピソード数が多すぎる | グラフ肥大化 | 文書あたり max 10 ep に制限 |
| Batch API のレイテンシ (24h) | 開発サイクルが遅い | Phase 0 は1回だけ、以降はキャッシュ |
| geDIG 値が nDCG と無相関 | ルーティング改善なし | Phase 1 で早期検証、失敗なら中止 |
| Biology 固有の結果が他ドメインで汎化しない | 展開失敗 | Phase 4 でドメイン別閾値も検討 |

---

## 11. 理論的位置づけ

### 11.1 メイズからの写像

本実験は geDIG フレームワークの **RAG 応用** としての初の本格的検証である。

```
                    メイズ              BRIGHT (本実験)
知識グラフ:          位置グラフ          episode graph
グラフ単位:          移動エピソード       知識/推論エピソード ★LLM生成
時系列接続:          temporal edge       sequential + connects_to
観測 (クエリ):       隣接セル状態         クエリエピソード群
AG (探索):          新しい方向へ移動      CoT + re-retrieval
DG (統合):          既知経路の活用        BM25 + Graph rerank
geDIG 値:           移動の意思決定        パイプライン選択
ノード特徴:          10D (位置,行動,報酬)  388D (semantic,lexical,entity,BM25,pos)
エッジ:              二値 (シナプス型)     二値 + 密度制御 (k_target)
```

### 11.2 理論的拡張

1. **エピソード単位の一般化**: メイズの「1ステップ=1エピソード」を
   「LLM が判定する原子的推論/知識ステップ=1エピソード」に拡張。
2. **connects_to による論理構造**: メイズの temporal edge（線形）に加えて、
   LLM が判定する論理的接続（DAG構造）をグラフに反映。
3. **多次元ノード特徴量**: メイズの 10D 設計思想を 388D に拡張し、
   意味的・語彙的・構造的類似性を同時に IG 計算に反映。
4. **知識側のエピソード化**: メイズの固定空間グラフに対し、BRIGHT では
   知識自体を LLM でエピソード化 → 環境のセマンティクスを抽出。
