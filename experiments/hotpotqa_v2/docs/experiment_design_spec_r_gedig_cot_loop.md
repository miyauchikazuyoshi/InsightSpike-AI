# Spec R: geDIG CoT Loop (Query-Induced Cycle Bridging)

## Series Name: QICB — Query-Induced Cycle Bridging

## 核心思想

> 「0hop評価(AG)が閾値を上回り、DGを探しにマルチホップ展開するが、DGが見つからない時にCoTを入れる。
> クエリに関するLLMの1次回答を再クエリとして注入して、クエリ-LLM1次回答を同時にグラフに注入。
> それで再度AG-DG評価でgoldランクのナレッジと繋がるのを期待する。
> 既存知識とちゃんと繋がるまで繰り返す。」

### 既存手法との決定的な違い

| 手法 | CoT の役割 | 停止条件 | 構造的根拠 |
|------|-----------|---------|-----------|
| IRCoT | 検索クエリの生成器 | 回数制限 | なし |
| FLARE | 不確実時の再検索トリガー | 確信度 | なし |
| DIVER | クエリ拡張 | 回数制限 | なし |
| **geDIG CoT** | **グラフ上のブリッジノード** | **Δβ₁ > 0（サイクル形成）** | **位相的接続確認** |

## 背景

### 現状の問題

| 指標 | 現在値 | 目標 |
|------|--------|------|
| Gold 接続率 (gold in top-10) | ~44% | **70%** |
| nDCG@10 (bio 50q) | 0.294 | 0.40+ |
| Zero-zero queries (both runs miss) | 34% (17/50) | <15% |

### Q.1 の教訓

- Δβ₁ はランキングシグナルとしては中立（+0.3%）
- 原因: **gold docs がプールに入っていない** (recall bottleneck)
- Δβ₁ が効くのは gold docs がプールにある時のみ（76% win rate）
- **CoT でプールを拡張し、Δβ₁ で構造的に接続確認する** のが正しい順序

## 設計

### アーキテクチャ: geDIG CoT Loop

```
Phase 0-2.6: 既存パイプライン（BM25 → CoT → RIA → Graph構築）
                    ↓
Phase 7: ★NEW★ geDIG CoT Loop
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  7a: AG 評価 (0-hop)                                       │
  │      query → グラフノードとの類似度計算                      │
  │      → 閾値以上のノード数 = n_ag                            │
  │                                                             │
  │  7b: DG 探索 (multi-hop Δβ₁)                               │
  │      query をグラフに注入 → Δβ₁ 計算                        │
  │      → Δβ₁ > 0 なら DG 確立 → ★Phase 8 へ (完了)           │
  │      → Δβ₁ = 0 なら DG 未発見 → 7c へ                      │
  │                                                             │
  │  7c: CoT ブリッジ生成                                       │
  │      query を LLM に投げて1次回答を得る                      │
  │      prompt: 「このクエリに答えるために必要な知識は？」       │
  │      → LLM回答テキスト + 抽出概念                           │
  │                                                             │
  │  7d: ブリッジノード注入                                     │
  │      query + LLM回答 をグラフにノードとして同時注入           │
  │      → query ↔ LLM回答 ↔ 既存ノード のエッジ形成            │
  │                                                             │
  │  7e: 再検索 (LLM回答由来のキーワードで)                     │
  │      LLM回答から抽出した概念で BM25 再検索                   │
  │      → 新文書をプールに追加 → グラフ再構築                   │
  │                                                             │
  │  7f: グラフ更新 + 再評価                                    │
  │      新文書を含むグラフで再度 AG-DG 評価                     │
  │      → Δβ₁ > 0? → サイクル形成 → グラフ更新して完了 ✅      │
  │      → Δβ₁ = 0? → t < max_rounds? → 7c へ (別CoTで再試行)  │
  │                                                             │
  │  停止条件:                                                  │
  │    1. Δβ₁ > threshold (サイクル形成 = 構造的接続確認)        │
  │    2. max_rounds 到達 (default: 3)                          │
  │    3. 新文書ゼロ (検索で新規が見つからない)                  │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
Phase 8: 最終スコアリング（Δβ₁ ボーナス込み）
```

### 7c: CoT ブリッジ生成プロンプト

```
You are a domain expert reasoning about a complex query.

Query: {query}

{previous_context}

What specific knowledge, concepts, or intermediate reasoning steps are needed
to connect this query to relevant documents? Think about:

1. What domain-specific terms or theories relate to this query?
2. What are the "bridge concepts" that connect the query topic to the answer?
3. What would a knowledgeable person think of when reading this query?

Provide your reasoning in 3-5 sentences with specific terms and concepts.
Focus on concepts that BRIDGE the gap between the query and potential answers.
```

ラウンド2以降:
```
Previous search found these documents but couldn't establish structural
connections (no cycles detected in the knowledge graph):

Top documents found so far:
{top_5_doc_summaries}

What DIFFERENT concepts or reasoning paths might connect the query to
relevant knowledge? Think of alternative angles, related fields, or
indirect connections.
```

### 7d: ブリッジノード注入の詳細

```python
def _inject_bridge_nodes(graph, query, llm_answer, cot_concepts):
    """
    query と LLM回答を同時にグラフに注入する。

    ノード構造:
      __query__          : クエリノード
      __bridge_r{t}__    : LLM回答ノード (ラウンド t)

    エッジ:
      __query__     ↔ 既存ノード  (類似度 > AG_THRESHOLD)
      __bridge_r{t}__ ↔ 既存ノード  (概念オーバーラップ)
      __query__     ↔ __bridge_r{t}__  (常に接続、cost=0.1)

    期待されるサイクル形成:
      query → bridge → doc_sentence_A → ... → doc_sentence_B → query
      ↑ このサイクルが Δβ₁ として検出される
    """
```

### Phase 7 の位置づけ

```
既存パイプライン:
  Phase 0-1:   BM25 検索
  Phase 2:     CoT 生成 (既存: 検索概念抽出用)
  Phase 2.5:   CoT 再検索
  Phase 2.6:   RIA ループ (β₀-gated)
  Phase 3:     エンティティグラフ構築
  Phase 4:     CoT ノード注入 (既存)
  Phase 4.5:   トークングラフスコアリング
  Phase 5:     グラフスコアリング
  Phase 5.25:  Entity F-eval (Δβ₁)  ← 現在ここで1回だけ評価

  ★NEW★
  Phase 7:     geDIG CoT Loop
    7a: AG 評価
    7b: DG 評価 (Δβ₁)
    7c-f: CoT ブリッジ → 再検索 → グラフ更新 → 再評価
  Phase 8:     最終スコアリング (Δβ₁ ボーナス込み)
```

**重要**: Phase 7 は Phase 5.25 の結果を見て「DG が見つからなかった」場合にのみ発動。
DG が既に見つかっている場合はスキップ（コスト節約）。

## 実装計画

### 変更ファイル

#### 1. `bright_cot_pipeline.py`

```python
# BrightCoTResult に追加
gedig_loop_applied: bool = False
gedig_loop_rounds: int = 0
gedig_loop_delta_beta1_history: list[int] = []  # 各ラウンドの Δβ₁
gedig_loop_n_bridge_nodes: int = 0
gedig_loop_n_new_docs: int = 0
gedig_loop_n_new_gold: int = 0
gedig_loop_converged: bool = False  # Δβ₁ > 0 で停止したか
gedig_loop_ms: float = 0.0

# コンストラクタに追加
gedig_loop: bool = False
gedig_loop_max_rounds: int = 3
gedig_loop_delta_beta1_target: int = 1  # Δβ₁ ≥ この値で停止
```

新メソッド:
```python
def _gedig_cot_loop(self, query, graph, candidates, ...) -> tuple[nx.Graph, dict]:
    """
    geDIG CoT Loop: DG未発見時にCoTブリッジで接続を試みる。

    Returns:
        updated_graph: ブリッジノード注入済みグラフ
        loop_diagnostics: ループ診断情報
    """

def _generate_bridge_cot(self, query, round_t, top_docs, prev_cot=None) -> str:
    """ラウンドごとのブリッジCoT生成。"""

def _inject_bridge_and_evaluate(self, graph, query, bridge_text, ...) -> int:
    """ブリッジ注入 → Δβ₁ 計算。返値: delta_beta1。"""

def _bridge_re_retrieve(self, bridge_text, existing_ids) -> list:
    """ブリッジCoTから抽出した概念で再検索。"""
```

#### 2. `gedig_scoring.py`

```python
def compute_delta_beta1_for_bridge(graph, query_node, bridge_nodes) -> int:
    """
    query + bridge ノード注入後の Δβ₁ を計算。
    既存の _entity_graph_feval_v2 のロジックを再利用。
    """
```

#### 3. `run_bright.py`

```python
# CLI 引数追加
parser.add_argument("--gedig-loop", action="store_true",
                    help="Enable geDIG CoT Loop (Spec R)")
parser.add_argument("--gedig-loop-max-rounds", type=int, default=3)
parser.add_argument("--gedig-loop-delta-beta1-target", type=int, default=1)
```

### 統合フロー（擬似コード）

```python
# Phase 5.25 の後
if self.gedig_loop and entity_feval_applied:
    # Δβ₁ が 0 のクエリ（DG 未発見）にのみ適用
    if ef_diag.get("delta_beta1_global", 0) == 0:
        graph, loop_diag = self._gedig_cot_loop(
            query=query,
            graph=graph,
            candidates=merged_candidates,
            cot_text=cot_text,
            existing_gold_ids=gold_ids,
        )
        # ループ結果でスコアを再計算
        if loop_diag["converged"]:
            # グラフが更新されたので再スコアリング
            entity_feval_scores, ef_diag = entity_graph_feval_scores(
                graph, query, cot_text, ...)
```

## 評価計画

### Smoke Test (10q)

```bash
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --output experiments/hotpotqa_v2/results/v22_specr_smoke/results.jsonl \
    --limit 10 --scoring-mode gedig_refine \
    --rerank-alpha 0.1 --graph-top-k 50 \
    --token-graph --token-graph-walk-score \
    --ria-loop --ria-max-rounds 3 \
    --entity-feval --entity-feval-version v2 \
    --gedig-loop --gedig-loop-max-rounds 3
```

### 50q A/B Test

| Config | 説明 |
|--------|------|
| Baseline | M+N.1 (RIA + Walk Score, feval なし) |
| Q.1 | M+N.1 + entity-feval v2 (Δβ₁ only) |
| **R.1** | **M+N.1 + entity-feval v2 + geDIG CoT Loop** |

### 成功基準

| 指標 | Baseline | Q.1 | R.1 目標 | Stretch |
|------|----------|-----|---------|---------|
| nDCG@10 (bio 50q) | 0.294 | 0.295 | **0.38+** | 0.42+ |
| Gold 接続率 | ~44% | ~44% | **70%** | 80% |
| Recall@10 | 0.288 | 0.288 | **0.35+** | 0.40+ |
| Zero-zero queries | 34% | 34% | **<15%** | <10% |

### 診断指標

- `gedig_loop_applied`: ループが発動したクエリ数（DG未発見率）
- `gedig_loop_rounds`: 平均ラウンド数（少ないほど良い）
- `gedig_loop_converged`: サイクル形成で停止した割合
- `gedig_loop_n_new_gold`: ループで新たに発見した gold 文書数
- `gedig_loop_delta_beta1_history`: Δβ₁ の推移（0 → >0 になるか）

## コスト見積もり

| Item | Per query (ループ発動時) | 50q total |
|------|--------------------------|-----------|
| LLM calls (bridge CoT) | 1-3 calls × gpt-4o-mini | ~$0.30 |
| BM25 再検索 | +50 docs × 1-3 rounds | ~0 (ローカル) |
| グラフ再構築 | ~2-5s per round | ~250s total |
| **追加コスト** | **~$0.01/query** | **~$0.30** |

ループ発動率を ~50% と仮定 → 全体コスト +15% 程度。

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| ブリッジ CoT が的外れ | Δβ₁=0 のまま、ラウンド浪費 | max_rounds=3 でコスト上限 |
| 再検索で gold 未到達 | recall 改善なし | ラウンドごとに異なるプロンプト（多様性確保） |
| グラフ再構築コスト | レイテンシ増大 | 差分更新（新ノード追加のみ）で最適化 |
| Δβ₁ 閾値が甘すぎる | 偽の収束 | per-document Δβ₁ で gold 文書周辺のサイクルを確認 |

## 理論的根拠

### geDIG における CoT の意味

```
従来の理解:
  CoT = 「検索クエリを改善するための推論テキスト」

geDIG の理解:
  CoT = 「グラフ上のブリッジノード」

  query が直接 gold doc に接続できない (AG 不足)
  → query から multi-hop で gold doc に到達する経路を探す (DG 探索)
  → 経路がない = グラフに「穴」がある
  → CoT が「穴を埋めるブリッジ」として機能
  → ブリッジ経由で新サイクル (Δβ₁) が形成
  → サイクル = 「複数の独立な経路で確認された接続」= 信頼性の高い DG
```

### Wake-Sleep-Wake との対応

```
Wake 1:  BM25 検索 + グラフ構築 (探索)
Sleep:   AG-DG 評価 + Δβ₁ 計算 (構造分析)
Wake 2:  geDIG CoT Loop (ブリッジ構築 + 再探索)
  ↓ (Δβ₁ > 0 まで繰り返し)
Sleep 2: 最終 Δβ₁ 評価 (構造確認)
Wake 3:  最終ランキング出力
```

## 参考論文

| 論文 | 参考箇所 | geDIG との違い |
|------|---------|--------------|
| IRCoT (ACL 2023) | CoT ステップごとの再検索 | geDIG: Δβ₁ で停止判定 |
| FLARE (EMNLP 2023) | 必要時のみ検索 | geDIG: AG-DG 評価で必要性判定 |
| DIVER (BRIGHT SOTA-2) | Iterative query expansion | geDIG: グラフ注入 + サイクル検出 |
| LATTICE (zero-shot SOTA) | 階層ナビゲーション | geDIG: サイクルベースの確認 |
| Topo-RAG (CIKM 2024) | グラフ構造活用 | geDIG: Δβ₁ による動的評価 |

論文 PDF: `experiments/hotpotqa_v2/docs/references/` に格納済み。
