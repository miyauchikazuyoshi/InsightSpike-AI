# v10: Entity-Graph Guided Multi-hop QA 実験計画

## Context

MuSiQue ベンチマーク（random 500q）での結果比較：
- **Baseline A** (全20パラ丸投げ): EM=62.0% (50q)
- **v8 Full** (BM25 top-5 + GapQ): EM=31.6% (500q)
- **26ptの差はBM25 Recall@5=51.7%が原因** — 金パラの半分を取りこぼしている

データ分析から判明した事実：
- 20パラ = ~2,500トークン（GPT-4oの2%、Lost in the Middleの問題なし）
- 中間回答（bridge entity）が2+の金パラに出現: **92%**
- タイトル相互参照のみ: 推論チェーン接続率 **33%**
- エンティティ重複含む: 推論チェーン接続率 **97%**
- v9（ナイーブなタイトルヒント）は Baseline A と同等（60% vs 62%）→ 改善されず

**目標**: グラフ構造による推論ガイドで **Baseline A (62%) を上回る**

---

## 結果サマリー（50q スモークテスト）

| Version | Mode | EM | F1 | chain_found率 | chain_found=True EM | vs Baseline A |
|---------|------|----|----|--------------|---------------------|---------------|
| Baseline A | — | 62.0% | 0.756 | — | — | ref |
| v9 | guided | 60.0% | 0.729 | — | — | -2.0pt |
| **v10a** | guided | 58.0% | 0.739 | 86% | 61.9% | -4.0pt |
| **v10a** | reordered | 60.0% | 0.736 | 86% | **66.7%** | -2.0pt |
| **v10b** | guided | 56.0% | 0.740 | 90% | 59.1% | **-6.0pt** |
| **v10b** | reordered | 60.0% | 0.748 | 90% | 62.2% | -2.0pt |
| **v10c** | guided | 52.0% | 0.691 | 88% | 54.5% | **-10.0pt** |
| **v10c** | reordered (old) | 58.0% | 0.712 | 88% | 59.1% | -4.0pt |
| **v10c** | reordered (improved) | **62.0%** | 0.757 | **98%** | **63.3%** | ±0pt |
| **v10c** | reorder_only | **62.0%** | 0.751 | **98%** | **63.3%** | ±0pt |

### v10c 主要な発見

1. **Guided mode は常に性能を悪化させる**: 全バージョンで Baseline A より低い EM
2. **Reordering のみが有効**: パラグラフ並べ替えは一貫して guided より良い結果
3. **chain_found の改善が効果的**: 87.5% → 98% に改善したことで v10c reordered が 58% → 62% に回復
4. **Baseline A と同等が天井 (v10c)**: reorder_only も reordered も EM=62.0%

---

## v10d: Chain Expansion + Pruning 実験

### 改良点

v10c のチェーンは平均 2.1 パラで、gold平均 2.4 パラをカバーしきれない（recall=55.5%）。
`_expand_chain()` を追加し、Q + チェーンパラのエンティティ重複が高いパラを最大3個追加。

| 指標 | v10c | v10d | 改善 |
|------|------|------|------|
| Avg chain size | 2.1 | 3.8 | +1.7 |
| Chain Recall | 55.5% | 68.1% | +12.6pt |
| Full Recall | 30% | 40% | +10pt |
| chain_found | 98% | 98% | ±0 |

### v10d 結果

| Model | Mode | EM | F1 | vs Baseline |
|-------|------|----|----|-------------|
| **GPT-4o** | Baseline A | 62.0% | 0.756 | ref |
| **GPT-4o** | v10c reorder_only | 62.0% | 0.751 | ±0pt |
| **GPT-4o** | **v10d reorder_only** | **68.0%** | **0.804** | **+6.0pt** |
| **GPT-4o** | v10d pruned | 48.0% | 0.621 | -14.0pt |
| GPT-4o-mini | Baseline A | 42.0% | 0.646 | ref |
| GPT-4o-mini | v10d reorder_only | 30.0% | 0.553 | -12.0pt |

### v10d 主要な発見

**1. GPT-4o + reorder_only: EM=68.0% (+6pt)** 🎉
- chain_found=True: EM=**69.4%** (+7.4pt vs Baseline A)
- Win/Loss: +6 wins / -3 losses = **net +3**
- 全ホップで改善: 2-hop 56→62%, 3-hop 83→83%, 4-hop 50→**75%**
- **チェーン拡張が 3-hop/4-hop の足引っ張りを解消**

**2. Pruning は時期尚早: EM=48.0% (-14pt)**
- Chain Recall 68% では gold パラが欠落しすぎる
- Recall 90%+ が必要 → 現状では全パラ供給 + reorder が最適

**3. GPT-4o-mini は reorder で悪化: 30.0% (-12pt)**
- 弱いモデルは reorder に混乱（正しいパラ順序に依存している可能性）
- rate limit エラーも 3件あり結果が歪んでいる可能性
- **graph guidance は弱モデルでは別のアプローチが必要**

### ホップ別分析（GPT-4o）

| Hop | Baseline A | v10c reorder | v10d reorder | 改善幅 |
|-----|-----------|-------------|-------------|--------|
| 2-hop (34q) | 55.9% | 61.8% | **61.8%** | +5.9pt |
| 3-hop (12q) | 83.3% | 66.7% | **83.3%** | ±0pt |
| 4-hop (4q) | 50.0% | 50.0% | **75.0%** | **+25pt** |

v10c では 3-hop が -16.6pt 悪化していたが、v10d のチェーン拡張で解消。

---

## v10a: Entity Overlap Count のみ（最初の実装）

- エッジ重み = 共有 discriminative エンティティ数（整数カウント）
- unweighted BFS による最短パス

**重要な発見:**
- chain_found=True: guided EM=**61.9%**, reordered EM=**66.7%** → BaseA同等〜超え
- chain_found=False (7q, 14%): EM=42.9% / 28.6% → 足を引っ張っている
- **チェーンの品質が精度を左右する** → エッジ重みの改善が鍵

---

## v10b: フラットミキシング（失敗）

### 問題点
v10a のエンティティカウントのみでは distractor 経由の弱いパスと真の bridge が区別できない。

### 改良（v10b）: 3シグナルのフラット合成

```
strength = 0.4 * ent_ratio + 0.3 * cos_sim + 0.3 * title_ref
cost = 1.0 - strength
```

### v10b 結果と根本原因分析

**結果**: v10a より悪化（guided EM: 58→56%、chain_found=True EM: 61.9→59.1%）

**根本原因**: graph_builder.py v5 の Two-Edge Architecture と比較して以下の問題：

1. **フラットミキシング**: Context attention（確実、同タイトル内）と Similarity attention（推定、クロスタイトル）を1つの重みに混合 → 優先度が失われる
2. **TF-IDF false positives**: distractor パラ間の偶然の高コサイン類似度がショートカットを作成
3. **beta_1 倍増**: v10a beta_1=2.0 → v10b beta_1=5.4 → ノイジーなサイクルエッジが増加
4. **同タイトル内 context attention がない**: graph_builder.py では intra-title 距離減衰（0.9/0.6/0.3）があるが、v10b にはない

**教訓**: **エッジタイプは分離して、明確な優先度階層を保つべき**

---

## v10c: 文レベル Three-Tier アーキテクチャ（現在の実装）

### 設計思想

graph_builder.py v5 の Two-Edge Architecture を拡張し、文レベルの粒度で
3階層のエッジタイプを持つグラフを構築する。

**核心**: エッジタイプを分離し、コストレンジを重複させない
→ Dijkstra が自然に Tier 1 > Tier 2 > Tier 3 の優先度で経路選択

### ノード設計

- **文レベル**: 各文が1ノード（パラグラフレベルではない）
- 20パラ × ~3-5文 = ~60-100 ノード
- ノード属性: `para_idx`, `sent_idx`, `title`, `text`

### Three-Tier エッジ階層

| Tier | 名称 | コストレンジ | 対象 | 説明 |
|------|------|-------------|------|------|
| **Tier 1** | Context | 0.05 - 0.10 | 同パラグラフ内 + タイトル相互参照 | 確実な構造的接続 |
| **Tier 2** | Entity | 0.20 - 0.50 | クロスパラグラフ | Discriminative エンティティ重複 |
| **Tier 3** | Similarity | 0.50 - 0.80 | クロスパラグラフ | TF-IDF コサイン類似度 (≥0.30) |

#### Tier 1: Context Attention（確実、cost 0.05-0.10）

graph_builder.py v5 の距離減衰パターンを踏襲：

```python
# 同パラグラフ内の隣接文
if sent_dist <= 1: cost = 0.05  # adjacent (strength=0.95)
elif sent_dist <= 3: cost = 0.08  # nearby (strength=0.92)
else:              cost = 0.10  # distant 4-6 (strength=0.90)

# タイトル相互参照（文が他パラのタイトルを言及）
title_cross_ref: cost = 0.08  # (strength=0.92)
```

#### Tier 2: Entity Overlap（推定、cost 0.20-0.50）

- クロスパラグラフのみ（同パラグラフは Tier 1 でカバー）
- Discriminative entity filtering (freq <= max_para_freq)
- `cost = 0.20 + 0.30 * (1 - overlap_ratio)`

#### Tier 3: TF-IDF Cosine（弱い、cost 0.50-0.80）

- クロスパラグラフのみ
- 閾値: cos_sim >= 0.30
- `cost = 0.80 - 0.30 * cos_sim`
- 上位 Tier のエッジが既存の場合は上書きしない

### コストレンジの設計根拠

```
Tier 1:  [0.05 ---- 0.10]                          ← 最優先
                            gap (0.10)
Tier 2:                     [0.20 ---- 0.50]        ← 中優先
                                              gap (0.00)
Tier 3:                                      [0.50 ---- 0.80]  ← 低優先
```

- Tier 1 max (0.10) < Tier 2 min (0.20): 2× のギャップ → context は常に entity より優先
- Tier 2 max (0.50) ≤ Tier 3 min (0.50): 境界で接触 → 弱い entity overlap ≈ 強い cosine similarity

### v10b との違い

| 項目 | v10b（フラット） | v10c（Three-Tier） |
|------|-----------------|-------------------|
| ノード粒度 | パラグラフ（20） | 文（60-100） |
| エッジタイプ | 1種類（混合） | 3種類（分離） |
| 同パラグラフ内接続 | なし | Tier 1 context（距離減衰） |
| コスト設計 | `1 - (混合重み)` | Tier別の非重複レンジ |
| 優先度制御 | なし（全フラット） | Tier 1 > Tier 2 > Tier 3 |
| beta_1 問題 | TF-IDF false positive → 倍増 | Tier 3 は弱コストで抑制 |

---

## 修正対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `experiments/hotpotqa_v2/src/entity_graph.py` | **全面改良**: 文レベルノード、Three-Tier エッジ |
| `experiments/hotpotqa_v2/scripts/run_allcontext.py` | Tier診断フィールド追加 |

---

## 実験計画

### モード一覧

| Mode | パラグラフ順序 | 推論ガイド |
|------|-------------|-----------|
| `baseline_a` | 原文のまま | なし（既存） |
| `v9_guided` | 原文のまま | タイトル相互参照（既存） |
| `v10_guided` | 原文のまま | Three-Tier チェーン |
| `v10_reordered` | チェーン先頭に並べ替え | Three-Tier チェーン |

### v10プロンプトテンプレート（変更なし）

```
Read ALL of the following paragraphs carefully, then answer the question.

{paragraphs}

=== REASONING GUIDE ===
REASONING CHAIN (follow this path):
  Step 1: "[Title A]" connects to "[Title B]" via [bridge entity]
  Step 2: "[Title B]" connects to "[Title C]" via [bridge entity 2]

KEY PARAGRAPHS (in reasoning order):
  1. [Title A]
  2. [Title B]
  3. [Title C]

BRIDGE ENTITIES: entity1, entity2
=== END GUIDE ===

Question: {question}

Follow the reasoning guide above to trace the multi-hop chain.
Your final answer must be a short phrase. Write it after "Answer: ".
```

### 出力レコードフィールド

```python
{
    ...,  # 既存のem, f1, etc.
    "beta_0": int,          # グラフ連結成分数
    "beta_1": int,          # サイクル数
    "n_bridges": int,       # bridge エッジ数
    "chain_length": int,    # 推論チェーン長
    "chain_found": bool,    # チェーンが見つかったか
    # v10c 追加
    "n_sent_nodes": int,    # 文ノード数
    "n_tier1": int,         # Tier 1 (context) エッジ数
    "n_tier2": int,         # Tier 2 (entity) エッジ数
    "n_tier3": int,         # Tier 3 (similarity) エッジ数
}
```

---

## 実験実行

### Phase 1: v10c 50q スモークテスト

```bash
# v10c_guided (50q)
python run_allcontext.py --mode v10_guided --data musique_random_500.jsonl \
    --output results/musique_v10c_guided --limit 50

# v10c_reordered (50q)
python run_allcontext.py --mode v10_reordered --data musique_random_500.jsonl \
    --output results/musique_v10c_reordered --limit 50
```

Baseline A (62.0%), v10a (58.0%/60.0%), v10b (56.0%/60.0%) と比較。

### Phase 2: 500q フルラン（v10c が Baseline A を上回った場合）

---

## 改善メカニズム（v10c が v10b を上回る理由）

1. **エッジタイプの分離**: Context / Entity / Similarity を別タイプとして管理 → ノイジーなショートカットを防止
2. **コストレンジの非重複**: Tier 1 (0.05-0.10) < Tier 2 (0.20-0.50) < Tier 3 (0.50-0.80) → 確実な接続が常に優先
3. **文レベル粒度**: パラグラフ内の個別文がノード → bridge entity を含む特定の文を正確に特定
4. **同パラグラフ Context attention**: graph_builder.py v5 の距離減衰を採用 → 同タイトル内は最小コストで確実に接続
5. **Title cross-ref の適切な位置付け**: Tier 1 の一部として最優先 → v9 の 33% 接続率も確実に活用
6. **弱いシグナルの抑制**: Tier 3 は高コスト（0.50-0.80）→ 他に経路がある場合は使われない

## リスクと対策

| リスク | 対策 |
|--------|------|
| 文レベルで O(n^2) ペア数増加 | n=60-100 → 4950ペア、十分高速 |
| 短い文の TF-IDF ノイズ | Tier 3（最低優先）で隔離、閾値 ≥0.30 |
| Tier 1 エッジが多すぎて beta_1 増加 | 同パラ内 context は構造的に正しい → beta_1 は許容 |
| entity 抽出が攻撃的すぎる | `max_para_freq` を 2/3/5 でテスト |
| LLM がガイドを無視 | v10 は具体的アクション指示（v9 の構造観察とは質的に異なる） |

## 検証方法

1. **50q スモークテスト**: Baseline A / v10a / v10b / v10c を同一 50q で比較
2. **chain_found 分析**: chain_found=True/False 別の EM 比較
3. **Tier 別エッジ統計**: n_tier1/n_tier2/n_tier3 の分布と EM との相関
4. **ホップ別分析**: 2-hop / 3-hop / 4-hop 別に EM 比較
5. **beta_1 変化**: v10b (5.4) → v10c で改善するか（Tier 分離の効果）
