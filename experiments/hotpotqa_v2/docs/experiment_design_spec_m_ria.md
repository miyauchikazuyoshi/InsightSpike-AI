# Spec M: RIA (Recursive Insight Architecture) on BRIGHT

## 概要

論文 §Future Work の RIA（再帰的洞察ベクトルによる段階的強化）を BRIGHT ベンチマークで実装・検証する。
現在の 1-shot CoT re-retrieval (Phase 2.5) を **β₀-gated ループ**に拡張し、
反復的にクエリを強化して retrieval recall を改善する。

## 背景と動機

### ボトルネック（Spec I/J/K/L の統一的教訓）

- **56% のクエリで gold が BM25 top-100 に 0 件** — reranking では解決不能
- Pointwise reranking は gpt-4o-mini (J) も gpt-4o (L) も改善なし
- Query decomposition (K) は recall 改善するが scoring が追いつかない
- **根本原因**: 1 回の検索では推論的関連文書に到達できない

### DIVER (BRIGHT SOTA-2, nDCG=46.8) の知見

- 2 ラウンドの iterative query expansion が +47% のゲインの主因
- 各ラウンド: top-5 docs を LLM に渡して query refinement
- Query expansion なしだと 46.8 → 31.9 (-32%)

### 論文 RIA との対応

```
論文 RIA:   q^(t) → Retriever → KG → h^(t) → augment(q, h) → q^(t+1) → ...
            停止: g₀/g_min 改善 + PSZ 制約

Spec M:     query → BM25 → CoT → Graph(β₀) → expand_query(CoT, top_docs) → BM25 → ...
            停止: β₀ 改善 or max_rounds
```

## 設計

### アーキテクチャ

```
Phase 0-1:   Query → BM25 top-100 (既存)
Phase 2:     CoT reasoning (既存)
Phase 2.5:   CoT re-retrieval (既存, 1回目)
                    ↓
Phase 2.6:   ★NEW★ RIA Loop (max_rounds=3)
  ┌─────────────────────────────────────────────┐
  │  2.6a: mini-graph 構築 → β₀ 計算            │
  │  2.6b: β₀ ≤ 1? → 収束、ループ脱出           │
  │  2.6c: top-k docs の内容を LLM に投入        │
  │  2.6d: LLM が新しい推論 + 検索キーワード生成  │
  │  2.6e: 新キーワードで BM25 再検索             │
  │  2.6f: 新文書をプールに追加                   │
  │  2.6g: t ← t+1, ループ先頭へ                 │
  └─────────────────────────────────────────────┘
Phase 3-7:   Graph construction → Scoring → Ranking (既存)
```

### 2.6c-d: Query Expansion Prompt

```
You are a search expert analyzing retrieval results for a complex query.

Query: {query}

Previous reasoning: {cot_text}

Top retrieved documents (round {t}):
{top_5_docs_summaries}

Based on these documents, identify:
1. What information gaps remain to fully answer the query?
2. What new search terms, concepts, or entities should we look for?
3. What related topics or domains might contain relevant documents?

Output 5-10 new search keywords/phrases, one per line.
```

### 停止条件（β₀-gated）

1. **β₀ 収束**: `β₀^(t) ≤ β₀_target` (default: 1, 完全連結)
2. **β₀ 非改善**: `β₀^(t) ≥ β₀^(t-1)` (新文書が接続を増やさない)
3. **最大ラウンド**: `max_rounds` (default: 3, コスト上限)
4. **新文書ゼロ**: 再検索で新規文書が見つからない

### パラメータ

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ria-loop` | false | Enable RIA iterative expansion |
| `--ria-max-rounds` | 3 | Maximum iteration rounds |
| `--ria-docs-per-round` | 50 | New docs to retrieve per round |
| `--ria-feedback-top-k` | 5 | Top-k docs to feed back to LLM |
| `--ria-beta0-target` | 1 | Target β₀ for convergence |

### コスト見積もり

| Item | Per query | 50q total |
|------|-----------|-----------|
| LLM calls (expansion) | +1-3 calls × gpt-4o-mini | ~$0.50 |
| BM25 re-retrieval | +50-150 docs × 2-3 rounds | ~0 (ローカル) |
| Latency | +5-15s per round | +250-750s total |
| **Total追加コスト** | **~$0.01/query** | **~$0.50** |

現在の baseline: ~10s/query, ~$0.50/50q → RIA: ~25-40s/query, ~$1.00/50q

## 実装計画

### 変更ファイル

1. **`bright_cot_pipeline.py`**
   - `BrightCoTResult` に RIA フィールド追加 (rounds, β₀_history, new_docs_per_round)
   - コンストラクタに RIA パラメータ追加
   - Phase 2.5 の後に Phase 2.6 RIA ループ追加
   - `_ria_expand_query()` メソッド追加 (LLM call)
   - `_ria_compute_beta0()` メソッド追加 (mini-graph β₀)

2. **`run_bright.py`**
   - CLI 引数追加 (5 個)
   - Pipeline 構築の 3 箇所更新
   - 診断出力追加

### 既存コードの再利用

- `_cot_retrieval()`: 各ラウンドの BM25 再検索に再利用
- `_compute_pre_beta0()`: mini-graph β₀ 計算に再利用
- `extract_entities()` + `_extract_lowercase_concepts()`: 新概念抽出に再利用

## 評価計画

### Smoke test (10q)

```bash
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v12_bright_ria_smoke \
    --limit 10 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --ria-loop --ria-max-rounds 3
```

### 成功基準

| Metric | Baseline (Spec H) | Target | Stretch |
|--------|-------------------|--------|---------|
| nDCG@10 (50q) | 0.2496 | 0.30+ | 0.35+ |
| Recall@10 | 0.2419 | 0.30+ | 0.35+ |
| Pool に gold ある query 比率 | 44% | 60%+ | 70%+ |

### 診断指標

- `ria_rounds`: 実際のラウンド数 (平均, 分布)
- `ria_beta0_history`: 各ラウンドの β₀ 推移
- `ria_new_docs`: 各ラウンドの新規文書数
- `ria_new_gold`: 各ラウンドで新たに発見した gold 文書数

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| LLM が無関係なキーワード生成 | pool 汚染 → ranking 悪化 | Spec K と同じ傾向 → graph_top_k 内のスロット配分で制御 |
| β₀ が下がらない | ループが max_rounds まで回る | 2 ラウンド連続非改善で早期終了 |
| BM25 では推論的文書に到達不能 | DIVER は fine-tuned embedder 使用 | まず BM25 で検証、必要なら Dense 追加 |
| コスト増大 | gpt-4o-mini expansion は安価 | max_rounds=3 で上限 |

## Spec K との差異

| | Spec K (Query Decomposition) | Spec M (RIA) |
|--|------------------------------|-------------|
| 拡張方式 | 事前分解 (クエリのみ) | 反復的 (検索結果をフィードバック) |
| 検索結果利用 | なし | top-5 docs の内容を LLM に渡す |
| ラウンド数 | 1回 | 1-3回 (β₀-gated) |
| 停止条件 | なし (固定) | β₀ 収束 |
| 理論的根拠 | なし | 論文 RIA + DIVER |
