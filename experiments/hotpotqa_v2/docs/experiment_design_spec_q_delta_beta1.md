# Spec Q: True geDIG Δβ₁ Scoring

## シリーズ名: **Query-Induced Cycle Detection (QICD)**

## 背景

### 問題
Spec O/P の AG/DG 実装は geDIG の本質を捉えていなかった：
- v1: f_val の 30 パーセンタイルで AG/DG を分類（**相対閾値、常に 30/70 固定**）
- Spec P: CoT 間の score variance で DG を検出（**LLM の出力多様性に依存、機能せず**）

### geDIG の本質（InsightSpike の核心価値）
```
AG = 0-hop で確実な接続。見た瞬間にわかる関連性。
DG = AG では不十分なとき、multi-hop 探索して閉路を観測（β₁ の穴を検出）。
     複数の独立パスで到達できることが構造的な確認になる。
```

**Δβ₁ こそが geDIG の価値**：
- クエリをグラフに投入する前後で β₁ がどう変わるか
- Δβ₁ > 0 = クエリが新しいサイクルを作った = 構造的に確認された関連性
- 密なグラフでも、**クエリが作る新しいサイクル** は文書ごとに異なる

## アーキテクチャ

### Q.1: Δβ₁ Entity F-eval (実装済み)

```
Entity Graph (query 投入前):
  SentA ─── SentB ─── SentC
  (doc1)    (doc2)    (doc3)

β₁_before = E - V + C

Query node を CoT bridge で接続:
  SentA ─── SentB ─── SentC
  (doc1)    (doc2)    (doc3)
    \                  /
     └─── Query ──────┘

β₁_after = E' - V' + C'
Δβ₁ = β₁_after - β₁_before  ← これが DG シグナル

Per-document Δβ₁:
  doc_d の 2-hop 局所サブグラフで同じ計算
  → 文書ごとに「クエリが何個の新サイクルを作ったか」
```

**スコアリング**:
```python
proximity = 1.0 / (1.0 + shortest_path_from_query)
beta1_bonus = min(doc_delta_beta1 / 3.0, 2.0)
score = proximity * (1.0 + 0.3 * beta1_bonus)
```

### Q.2: CoT-Enhanced Δβ₁ (次のステップ)

CoT が Δβ₁ をさらに強化する理論的根拠：

```
BRIGHT の課題:
  Query: 「虫はなぜ光に集まるか？」
  Gold doc: 「近接原因 vs 究極原因」(proximate cause)
  キーワード一致率: 9%

BM25 だけの場合:
  Query → (keyword match) → Doc_A, Doc_B, Doc_C
  → 直接マッチしない gold doc はプールに入らない
  → Δβ₁ を計算しても gold doc がグラフにない

CoT を使う場合:
  Query → CoT推論 → "proximate cause", "ultimate cause" 等の概念を生成
  → CoT概念で re-retrieval → gold doc がプールに入る
  → Query node から CoT 概念ノード経由で gold doc に接続
  → 新しいサイクル発生 = Δβ₁ > 0 = DG confirmed!
```

**CoT = Δβ₁ の「橋渡し概念」を生成する装置**

現在の実装でも CoT re-retrieval は行われているが、
CoT 概念を**明示的にグラフノードとして注入**し、
それ経由の Δβ₁ を測定することで、
CoT の推論品質を構造的に評価できる。

### Q.3: CoT Concept Node Injection (将来)

```python
# CoT から抽出した概念をグラフに明示的ノードとして追加
for concept in cot_concepts:
    graph.add_node(f"cot_{concept}")
    # 概念を含むセンテンスに接続
    for sent_node in find_sentences_containing(concept):
        graph.add_edge(f"cot_{concept}", sent_node)
    # クエリノードにも接続
    graph.add_edge(Q_NODE, f"cot_{concept}")

# これで Query → CoT_concept → Sentence → Document のパスが生まれる
# 複数の CoT 概念が同じ文書にリンクすれば Δβ₁ >> 0
```

## 実験計画

### Phase 1: Q.1 検証 (現在)

| テスト | 構成 | 目的 |
|--------|------|------|
| Baseline | M+N.1 (RIA + Walk Score) | Bio 50q, entity-feval なし |
| Q.1 | M+N.1 + entity-feval v2 (Δβ₁) | Bio 50q, Δβ₁ スコアリング |

**期待**:
- Δβ₁ が文書間で分散を持つこと（スモークテストで確認済み: 0-17 docs の差）
- nDCG 微改善 or 中立（entity-feval の weight=0.20 はまだ保守的）
- 仮に改善がなくても、Δβ₁ の分布データが Q.2/Q.3 設計の根拠になる

### Phase 2: Q.2 CoT 概念注入 (次)

- CoT 概念をグラフノードとして明示的に注入
- Query → CoT概念 → Document の multi-hop パスを構築
- Δβ₁ で CoT の推論品質を構造的に評価
- **期待**: CoT が良い推論をしたクエリでは Δβ₁ が高く、nDCG も高い相関

### Phase 3: Q.3 Multi-CoT Δβ₁ (将来)

- Spec P の失敗を Δβ₁ で再設計
- 3本の CoT から異なる概念ノードを注入
- 各 CoT が独立に新サイクルを作るか（Δβ₁ の多様性）で DG を測定
- score 平均ではなく、**Δβ₁ に基づく構造的 ensemble**

## 見込み

### なぜ Q.1 は控えめな改善が予想されるか
- 現在の entity-feval weight=0.20 は全スコアの 20% しか影響しない
- Δβ₁ 自体は正しいシグナルだが、他のスコア成分（BM25, graph, walk score）が既に強い
- 改善幅: **+0〜5% 程度**

### なぜ Q.2/Q.3 で大きな改善が期待できるか
- BRIGHTの核心問題 = クエリと gold doc のキーワード乖離（一致率 14-18%）
- CoT 概念が「橋渡し」を提供 → Δβ₁ がその橋渡しの品質を測定
- **良い橋渡し = 複数サイクル = 高 Δβ₁ = 高スコア**
- 改善幅: **+10〜20% 目標** (0.19 → 0.21-0.23)

### 0.45 到達への道筋
```
現在:     0.1898 (M+N.1, 323q)
Q.1:      ~0.19-0.20 (Δβ₁ 微改善)
Q.2:      ~0.21-0.23 (CoT 概念注入)
Recall:   ~0.30-0.35 (HyDE / stronger retriever)
Q.3:      ~0.35-0.40 (Multi-CoT Δβ₁ ensemble)
Full opt: ~0.40-0.45 (ハイパラ最適化 + 全ドメイン調整)
```

## ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/gedig_scoring.py` | `_entity_graph_feval_v2()`: Δβ₁ 実装, `_compute_beta1()` ヘルパー |
| `src/bright_cot_pipeline.py` | `entity_feval_version` パラメータ追加 |
| `scripts/run_bright.py` | `--entity-feval-version v1\|v2` CLI |

## 実行コマンド

```bash
# Q.1 Baseline (M+N.1, no feval)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
  experiments/hotpotqa_v2/scripts/run_bright.py \
  --mode cot_retrieval --domains biology \
  --output experiments/hotpotqa_v2/results/v21_specq1_baseline/results.jsonl \
  --limit 50 --scoring-mode gedig_refine --rerank-alpha 0.1 --graph-top-k 50 \
  --token-graph --token-graph-walk-score --ria-loop --ria-max-rounds 3

# Q.1 Δβ₁ (M+N.1 + entity-feval v2)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
  experiments/hotpotqa_v2/scripts/run_bright.py \
  --mode cot_retrieval --domains biology \
  --output experiments/hotpotqa_v2/results/v21_specq1_delta_beta1/results.jsonl \
  --limit 50 --scoring-mode gedig_refine --rerank-alpha 0.1 --graph-top-k 50 \
  --token-graph --token-graph-walk-score --ria-loop --ria-max-rounds 3 \
  --entity-feval --entity-feval-version v2
```
