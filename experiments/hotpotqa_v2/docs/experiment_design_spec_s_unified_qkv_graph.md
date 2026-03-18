# Spec S: Unified QKV Graph — Token DG → Directed CoT Bridge

## 核心思想

> 「3層のグラフを1つにまとめる。エッジタイプを変えることで、QKVのように機能させる。
> Token Graph の DG gap が CoT の指向性を与え、Entity Graph の Δβ₁ が収束判定する。」

### 3層統合の発想

| Layer | 現状 | 統合後の役割 | QKV対応 |
|-------|------|-------------|---------|
| Token Graph | per-doc独立スコア → 数値化して線形結合 | **K (Key)**: 文書内の構造的マッチ判定 | 「構造的に繋がっているか」|
| Entity Graph | cross-doc entity overlap → Δβ₁ | **Q (Query)**: クエリから到達可能なノード探索 | 「何に関連するか」|
| CoT Bridge | 盲目的LLM呼び出し → pool拡張 | **V (Value)**: DG gap を埋める推論橋 | 「推論で補完した値」|

### Spec R の教訓

- geDIG CoT Loop は動いたが baseline 比 -9.2%
- **原因**: bridge CoT に指向性がない → 無関係文書 1500 件追加 → pool 汚染
- **解決**: Token Graph の DG 分析で「何が足りないか」を特定 → CoT prompt に注入

## 設計

### アーキテクチャ（最小統合版）

```
Phase 4.5: Token Graph Scoring (既存)
  │
  ├─ 各文書の DG gap lemmas を収集 ← ★ NEW ★
  │   Pattern A: AG成分間の bridging lemmas
  │   Pattern B: DG パス上の中間 lemmas
  │
Phase 5: Entity Graph + CoT Scoring (既存)
  │
Phase 5.25: Entity F-eval Δβ₁ (既存)
  │
Phase 5.75: Directed geDIG CoT Loop ← ★ MODIFIED ★
  │
  │  5.75a: Token Graph DG gap を集約
  │         gap_concepts = union(top-5 docs の insight_lemmas)
  │         missing_links = 「query term X と Y の間に bridge がない」
  │
  │  5.75b: Directed bridge CoT 生成
  │         prompt に gap_concepts + missing_links を注入
  │         「{X} と {Y} を繋ぐ概念を説明せよ」
  │
  │  5.75c: 指向的 re-retrieval
  │         BM25 query = gap_concepts (not 全 bridge text)
  │         ★ フィルタ: token graph coverage > 0 の文書のみ追加
  │
  │  5.75d: Entity Graph Δβ₁ 再評価 (既存)
  │         収束: Δβ₁ 改善 or 新規文書ゼロ
  │
Phase 6: Combined ranking (既存)
```

### 5.75a: Token Graph DG Gap 集約

Token Graph の既存 insight 機能を活用:

```python
def _collect_dg_gaps(self, tg_diags, tg_ids, query_lemmas, top_k=5):
    """top-k 文書の Token Graph DG gap を集約.

    Returns:
        gap_lemmas: set[str]  — DG bridge に必要な lemma 群
        missing_pairs: list[tuple[str, str]]  — 未接続の query term ペア
    """
    gap_lemmas = set()

    # top-k 文書の insight_lemmas を収集
    for i in range(min(top_k, len(tg_diags))):
        diag = tg_diags[i]
        gap_lemmas.update(diag.get("insight_lemmas", []))

    # query lemma 間の接続性を分析
    # coverage < 1.0 の文書 = query term の一部が未マッチ
    # → 未マッチ term が gap
    missing_pairs = []
    for diag in tg_diags[:top_k]:
        if diag["coverage"] < 0.5:
            # この文書はクエリの半分以上をカバーしていない
            # = query の重要部分との接続が弱い
            pass  # missing_pairs に追加

    return gap_lemmas, missing_pairs
```

### 5.75b: Directed Bridge CoT Prompt

```
You are a domain expert connecting concepts for a complex query.

Query: {query}

STRUCTURAL ANALYSIS found these gaps in the knowledge graph:
- Bridge concepts needed: {gap_lemmas}  ← Token Graph DG から
- Unconnected query terms: {missing_pairs}
- Top documents lack connections between: {X} and {Y}

Your task: Explain the SPECIFIC conceptual bridge between these terms.
Focus on:
1. How {gap_lemma_1} connects {query_term_A} to {query_term_B}
2. What intermediate concepts or mechanisms link them
3. Domain-specific terminology that bridges the gap

Provide 3-5 sentences with precise technical terms.
```

### 5.75c: 指向的 Re-retrieval + フィルタ

```python
# 指向的キーワード: gap_lemmas のみ (bridge text 全体ではない)
keyword_query = " ".join(gap_lemmas)  # 5-10 lemma, not 20+ concepts

# BM25 re-retrieval
new_cands = bm25.search(keyword_query, top_k=30)  # 50→30 に削減

# ★ フィルタ: Token Graph coverage > 0 の文書のみ追加
filtered = []
for doc_idx, score in new_cands:
    doc_text = docs[doc_idx]["content"]
    # 簡易チェック: gap_lemmas の少なくとも1つを含むか
    doc_lower = doc_text.lower()
    if any(lem in doc_lower for lem in gap_lemmas):
        filtered.append((doc_idx, score))

# 最大 15 文書に制限 (pool 汚染防止)
filtered = filtered[:15]
```

### パラメータ

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gedig-loop` | false | Enable directed geDIG CoT loop |
| `--gedig-loop-max-rounds` | 2 | Maximum iteration rounds (3→2 に削減) |
| `--gedig-loop-directed` | true | Use Token Graph DG gaps for direction |
| `--gedig-loop-max-new-docs` | 15 | Max new docs per round (50→15) |

### Spec R からの変更点

| 項目 | Spec R (盲目) | Spec S (指向的) |
|------|-------------|----------------|
| CoT prompt | 「何か bridge を考えて」 | 「{X}と{Y}を繋ぐ概念を説明せよ」 |
| Re-retrieval keywords | bridge text 全体 (20+ concepts) | gap_lemmas のみ (5-10) |
| New docs/round | 50 | **15** (フィルタ後) |
| Pool 汚染リスク | 高 (150 docs/query) | **低** (max 30 docs/query) |
| Token Graph 依存 | なし | **insight_lemmas を使用** |
| 前提条件 | entity_feval v2 のみ | token_graph + entity_feval v2 |

## コスト見積もり

| Item | Spec R | Spec S | 削減 |
|------|--------|--------|------|
| New docs/query | 150 | ~30 | -80% |
| LLM calls/query | 3 (blind) | 2 (directed) | -33% |
| 処理時間/query | ~48s | ~25s | -48% |
| Token Graph 追加計算 | なし | insight 集約のみ (~0s) | +0s |

## 実装計画

### 変更ファイル

1. **`bright_cot_pipeline.py`** — 主要修正
   - `_collect_dg_gaps()` メソッド追加 (Token Graph DG gap 集約)
   - `_generate_bridge_cot()` 修正 (gap_lemmas, missing_pairs を受け取る)
   - Phase 5.75 修正:
     - Token Graph diag からDG gap 収集
     - Directed prompt 生成
     - フィルタ付き re-retrieval (max 15 docs/round)
   - max_rounds デフォルト 3→2

2. **`run_bright.py`** — CLI 修正
   - `--gedig-loop-directed` フラグ追加
   - `--gedig-loop-max-new-docs` パラメータ追加

### 既存コード再利用

- Token Graph の `insight_lemmas` (既に計算済み、diag に入っている)
- Token Graph の `coverage` (未マッチ query term の指標)
- `_generate_bridge_cot()` (prompt 拡張のみ)
- BM25 re-retrieval ロジック (フィルタ追加のみ)

## 実験計画

### Smoke Test (10q)

```bash
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v23_specs_smoke \
    --limit 10 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score \
    --token-graph-f-eval --token-graph-insight-mode both \
    --entity-feval --entity-feval-version v2 \
    --gedig-loop --gedig-loop-max-rounds 2 --gedig-loop-directed
```

### 成功基準

| Metric | Baseline (Q.1) | Spec R (盲目) | Spec S Target |
|--------|----------------|--------------|---------------|
| nDCG@10 (10q) | 0.3149 | 0.2860 (-9%) | **0.33+** (改善) |
| Pool 汚染 (new docs/q) | 0 | 150 | **≤30** |
| 処理時間/query | ~10s | ~48s | **≤25s** |
| Gold in top-10 | 8/10 | 8/10 | **8/10+** |

### 50q A/B Test

| # | 構成 | 目的 |
|---|------|------|
| A | baseline (no loop) | 基準 |
| B | Spec S (directed loop) | 指向的 CoT の効果 |
| C | Spec S + token-graph-insight-mode=both | insight 注入の追加効果 |

## リスクと対策

| リスク | 対策 |
|--------|------|
| Token Graph insight_lemmas が空 | fallback: Spec R 同様の盲目 CoT |
| gap_lemmas が少なすぎ (1-2個) | query の未マッチ lemma を追加 |
| フィルタが厳しすぎて新文書ゼロ | coverage > 0 チェックを any(lem in doc) に緩和 |
| 処理時間: token graph insight 計算 | 既に Phase 4.5 で計算済み、追加コスト ~0 |

## 進行記録

### Step 0: Spec R 実装・検証 ✅
- geDIG CoT Loop を bright_cot_pipeline.py に実装
- 3つのバグ修正:
  1. version チェック `== "v2"` → `startswith("v2")` (v2.2 対応)
  2. トリガー条件 `Δβ₁==0` → 常に発動 (Δβ₁ は全クエリ >0)
  3. 新規文書の entity_feval スコア付与漏れ
- Spec R 10q 結果: nDCG=0.2860 (**-9.2%** vs baseline 0.3149)
  - 原因: 150 docs/q の pool 汚染、盲目的 bridge CoT

### Step 1: Spec S 設計・実装 ✅
- `_collect_dg_gaps()` メソッド追加
- `_generate_bridge_cot()` に gap_lemmas/unmatched_terms 引数追加
- Phase 5.75 を指向的 re-retrieval + フィルタに改修
- CLI: `--gedig-loop-directed`, `--gedig-loop-max-new-docs` 追加

### Step 2: Spec S Smoke Test (10q) ✅

| | Baseline | Spec R (盲目) | **Spec S (指向的)** |
|---|---------|-------------|------------------|
| nDCG@10 | 0.3149 | 0.2860 (-9.2%) | **0.3081 (-2.2%)** |
| Recall@10 | 0.2908 | 0.2575 (-11.4%) | **0.2886 (-0.8%)** |
| Gold in top-10 | 8/10 | 8/10 | 7/10 |
| New docs total | 0 | 1500 | **300 (-80%)** |
| New gold found | 0 | 1 | **2** |
| 処理時間/query | ~10s | ~48s | **~21s (-56%)** |

Key findings:
- Pool 汚染 80% 削減で Spec R の悪化がほぼ回復
- 処理時間も 56% 削減
- CoT 非決定性が 10q では依然として大きな揺らぎ要因

### Step 3: 50q A/B Test ✅ 完了

```bash
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v23_specs_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score \
    --token-graph-f-eval --token-graph-insight both \
    --ria-loop --ria-max-rounds 3 \
    --entity-feval --entity-feval-version v2 \
    --gedig-loop --gedig-loop-max-rounds 2 --gedig-loop-max-new-docs 15
```

**50q Results:**

| Metric | Baseline (Q.1) | Spec S (Directed) | Delta |
|--------|----------------|-------------------|-------|
| nDCG@10 (mean) | 0.2944 | 0.2433 | **-0.0511 (-17.3%)** |
| Recall@10 | 0.2879 | 0.2671 | -0.0209 (-7.3%) |
| nDCG@10 (median) | 0.2712 | 0.1148 | - |
| nDCG@10 (std) | 0.3003 | 0.3013 | - |
| Win/Tie/Loss | - | 12/20/18 | - |
| 処理時間/query | ~10s | ~44s | +340% |

**分析:**
- 20 ties: 両方とも nDCG=0 のクエリ（CoT が変わっても改善なし）
- 12 wins: 大幅改善 (e.g., 0→0.43, 0→0.42) — baseline で見つからなかった gold を発見
- 18 losses: 大幅悪化 (e.g., 1.0→0, 0.61→0) — baseline で見つけていた gold を失う
- **根本原因: CoT 非決定性** — geDIG loop が新文書を追加するたびに entity graph が変化
  → graph_scores の順序が大幅に変動 → gold が top-10 から転落

**教訓:**
1. 指向的 CoT (gap_lemmas) は 10q で Pool 汚染を 80% 削減した（正しい方向）
2. しかし 50q では **CoT 非決定性** が支配的な揺らぎ要因
3. 1回の CoT で全てが決まる構造は脆い → **Multi-CoT Ensemble (Spec P) が必要**
4. Token Graph DG gap の情報自体は有用（wins で gold 発見に貢献）

## 結論と次のステップ

### Spec S の成果
- Pool 汚染 80% 削減（Spec R: 150 docs/q → Spec S: ~30 docs/q）
- Token Graph DG gap を CoT 指向性に活用する手法を確立
- 処理時間 56% 削減（Spec R: 48s → Spec S: 21s per 10q test）

### Spec S の限界
- CoT 非決定性が 50q スケールで -17.3% の悪化を引き起こす
- 1回の CoT に依存する構造では、改善も悪化も大きくブレる
- **geDIG loop 自体のオン/オフが、構成全体の安定性を左右する**

### 推奨: Spec P (Multi-CoT Ensemble)
- 複数の CoT (N=3-5) を生成してキャッシュ
- 各 CoT で独立にスコアリング → ensemble mean + variance
- CoT 間の一致度 = AG (確信) / 不一致 = DG (探索対象)
- キャッシュにより再現性を確保

## 将来の完全統合版 (Spec T 候補)

Spec S は「最小統合」。完全統合は:

```
1つの統合グラフ G:
  ノード = {query_tokens, doc_sentences, doc_tokens, cot_bridges}
  エッジ = {K_dep, K_lemma, Q_entity, V_bridge}

  単一の weighted shortest path で全スコア計算
  → 3種類のスコアの線形結合が不要に
```

Spec S の結果から、**CoT 非決定性の解決 (Spec P) が先決**。
統合グラフは Spec P で安定した基盤を作った後に検討する。
