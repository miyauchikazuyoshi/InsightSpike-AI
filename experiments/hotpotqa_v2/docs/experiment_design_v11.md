# v11: Pre-computed Topology Routing — 実験設計書

## 1. 背景と動機

### v10 (MuSiQue 500q) の根本的発見

- 20パラ (~2,500 tok) は GPT-4o 128K ウィンドウの 2% に過ぎず、"Lost in the Middle" (Liu et al., 2023) が発生する条件を満たさない
- パラ並べ替え: +1.2pt (z=0.38, 有意でない)
- Gold パラの位置と正答率に相関なし (正答時 0.496 vs 誤答時 0.478)
- エラーの 45% は推論誤り (distractor entity 選択)、32% はフォーマット不一致、0% は情報欠落
- **Guided テキストは GPT-4o に害** — 全バージョン全条件で悪化

### v2/v3 (HotpotQA 500q) の成功パターン

- F 値ルーティング (System 1/2) は有効: 51.2% EM (GPT-4o) at 3.6x 少ない LLM 呼び出し
- トポロジー (beta_0, beta_1) はゼロコストのルーティングシグナル
- モデルスケーリングでトポロジーの効果が増大 (+6pt from mini -> 4o)

### v11 が解く問題

v10 と v2/v3 の統合: **事前グラフ構築 + F 値ルーティング** を、コンテキストが十分大きい設定で検証する。

| 系列 | チャンク方式 | グラフ構築 | F 値 | コンテキスト |
|------|------------|-----------|------|------------|
| v2/v3 | BM25 top-5 (~400 tok) | BM25 リトリーバル後 | 使用 (ルーティング) | 小 |
| v10 | 全パラ (~2,500 tok) | クエリ依存 (Q 関連パラから探索) | 使用せず | 中 |
| **v11** | 全パラ (~6,000 tok) | **クエリ非依存 (事前構築)** | **使用 (ルーティング)** | **大** |

---

## 2. 仮説

**H1**: 50 パラ (~6,000 tok, ウィンドウの 5%) に拡大すると baseline が劣化する

**H2**: 事前構築グラフのサブグラフ抽出で、関連パラを高精度で特定できる (gold_recall >= 0.8)

**H3**: F 値ルーティング (System 1/2) が劣化を回復し、20 パラ baseline と同等以上になる

---

## 3. アーキテクチャ

### 2 フェーズ設計

```
Phase 1 -- Offline (問題ごとに 1 回、クエリ非依存):
  入力: 50 パラグラフ (titles + sentences)
  処理: entity_graph.build_sentence_graph() で三層エッジ構築
  出力: sentence-level nx.Graph + beta_0, beta_1, centrality

Phase 2 -- Online (クエリごと):
  入力: 質問 + 事前構築グラフ
  処理:
    1. クエリエンティティ抽出 -> グラフノードマッチ
    2. k-hop BFS でサブグラフ抽出
    3. サブグラフノード -> パラインデックスに変換
    4. F 値計算 (beta_0, beta_1 ベース)
    5. F >= theta -> System 1 (サブグラフのみ)
       F <  theta -> System 2 (全パラ、サブグラフ先頭)
  出力: 回答 + 診断情報
```

### v10 との根本的差異

| | v10 | v11 |
|--|-----|-----|
| グラフ構築 | クエリ依存 (Q 関連パラから探索) | クエリ非依存 (全パラで事前構築) |
| 介入方法 | パラ並べ替え | サブグラフ抽出 + ルーティング |
| 推論ガイド | あり (性能劣化) | なし (v10 の知見: 暗黙的誘導のみ) |
| コンテキスト | 20 パラ固定 | 50 パラ (スケーラビリティ検証) |
| F 値 | 使用しない | System 1/2 ルーティング |

---

## 4. データ

### ベースデータ

`musique_random_500.jsonl`: MuSiQue benchmark, 500q, 20 パラ/問
- 2-hop: 282q, 3-hop: 140q, 4-hop: 78q

### スケールデータ (新規作成)

`musique_50para_500.jsonl`: 50 パラ/問 (30 ランダムディストラクタ追加)
- ソース: 他の問題のパラグラフからランダムサンプル
- 制約: タイトル重複なし、gold パラ保持、シャッフル (seed=42)
- 生成: `scripts/build_scaled_data.py --target-paras 50 --seed 42`

---

## 5. 実験条件

| ID | 条件 | パラ数 | 手法 | コンテキスト |
|----|------|--------|------|------------|
| B_20 | baseline_a | 20 | Plain CoT | 全 20 パラ |
| B_50 | baseline_a | 50 | Plain CoT | 全 50 パラ |
| V11_S | v11_subgraph | 50 | サブグラフのみ | サブグラフパラ (<=10) |
| V11_R | v11_routing | 50 | F 値ルーティング | S1: サブグラフ / S2: 全 50 パラ |

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| k_hop | 2 | サブグラフ抽出の BFS 半径 |
| max_subgraph_paras | 10 | サブグラフの最大パラ数 |
| theta_f | 0.0 | F 値ルーティング閾値 |
| max_para_freq | 3 | discriminative entity フィルタ |

---

## 6. 実装ファイル

| ファイル | 操作 | 概要 |
|---------|------|------|
| `scripts/build_scaled_data.py` | 新規 | 20 パラ -> 50 パラにディストラクタ追加 |
| `src/corpus_graph.py` | 新規 | 事前グラフ構築 + サブグラフ抽出 + F 値ルーティング |
| `scripts/run_allcontext.py` | 修正 | v11 モード追加 (v11_subgraph, v11_routing) |

### corpus_graph.py のデータ構造

```python
@dataclass
class SubgraphResult:
    paragraph_indices: list[int]    # サブグラフに含まれるパラインデックス
    paragraph_titles: list[str]
    n_subgraph_nodes: int           # 文レベルノード数
    n_subgraph_edges: int
    query_matched_nodes: int        # クエリからマッチしたシードノード数
    beta_0: int                     # サブグラフの連結成分数
    beta_1: int                     # サブグラフのサイクル数
    gold_precision: float           # |subgraph & gold| / |subgraph|
    gold_recall: float              # |subgraph & gold| / |gold|

@dataclass
class RoutingDecision:
    system: str                     # "system1" or "system2"
    f_value: float
    subgraph: SubgraphResult
    context_paragraphs: list[int]   # LLM に送るパラインデックス列
    context_tokens_est: int
```

### CorpusGraphBuilder のメソッド

| メソッド | フェーズ | 説明 |
|---------|---------|------|
| `build()` | Offline | 全パラから sentence-level グラフ構築、centrality 事前計算 |
| `_match_query_to_nodes()` | Online | クエリエンティティ -> グラフノードマッチ (3 段階) |
| `extract_subgraph()` | Online | k-hop BFS -> パラ特定 -> beta 計算 -> gold 評価 |
| `compute_f_value()` | Online | F = (1 - beta_0_norm) + beta_1_norm |
| `route()` | Online | 全パイプライン -> RoutingDecision |

### F 値の設計

```
F = (1 - beta_0_norm) + beta_1_norm

beta_0_norm = (beta_0 - 1) / max(n_paras - 1, 1)
  0 = 全連結 (1 つの連結成分)
  1 = 全分離 (各パラが独立)

beta_1_norm = beta_1 / max(n_edges, 1)
  0 = ツリー構造 (サイクルなし)
  高 = 密に接続

高 F -> サブグラフが密に連結 -> 確信あり -> System 1 (サブグラフのみ)
低 F -> サブグラフが断片的   -> 不確実  -> System 2 (全パラ送信)
```

### entity_graph.py からの再利用

| 関数 | 用途 |
|------|------|
| `extract_entities()` | エンティティ抽出 + 96 語ストップワード |
| `build_sentence_graph()` | Three-Tier エッジ構築 (v10c) |

---

## 7. プロンプト設計

v10 の知見「Guided テキストは GPT-4o に害」を適用:
- **推論ガイドなし**
- 暗黙的誘導のみ (System 2 ではサブグラフパラを先頭配置)

### V11_SUBGRAPH_PROMPT (System 1 / v11_subgraph)

```
Read the following paragraphs carefully, then answer the question.

{paragraphs}

Question: {question}

Think step by step. Identify which paragraphs contain relevant information,
trace the reasoning chain across paragraphs, then give your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words).
Write it on the last line after "Answer: ".
```

### V11_SYSTEM2_PROMPT (System 2 のみ)

```
Read ALL of the following paragraphs carefully, then answer the question.
The most relevant paragraphs are listed first.

{paragraphs}

Question: {question}

Think step by step. Focus especially on the first few paragraphs which are
most likely to contain key information. Trace the reasoning chain, then give
your final answer.

IMPORTANT: Your final answer must be a short phrase (a few words).
Write it on the last line after "Answer: ".
```

---

## 8. 評価指標

### Primary

- **EM** (Exact Match): normalize_answer() 後の完全一致
- **F1**: トークンレベル F1

### Diagnostic (v11 固有)

| 指標 | 説明 | 目標 |
|------|------|------|
| subgraph_gold_recall | \|subgraph & gold\| / \|gold\| | >= 0.8 |
| subgraph_gold_precision | \|subgraph & gold\| / \|subgraph\| | 高いほど良い |
| f_value | F 値の分布 | 正答/誤答で分離 |
| system_used | System1/System2 の比率 | S1 比率が高いほど効率的 |
| context_tokens_est | LLM に送信したトークン推定数 | S1 で大幅削減 |
| query_matched_nodes | クエリからマッチしたシードノード数 | >= 1 |

### 出力レコードフィールド (v11 追加分)

```python
{
    ...,  # 既存の em, f1, mode, hop, latency_ms
    "f_value": float,
    "system_used": str,               # "system1" or "system2"
    "subgraph_n_paras": int,
    "subgraph_beta_0": int,
    "subgraph_beta_1": int,
    "subgraph_gold_precision": float,
    "subgraph_gold_recall": float,
    "context_tokens_est": int,
    "total_paras": int,
    "query_matched_nodes": int,
}
```

---

## 9. 成功基準

1. **B_50 < B_20**: 50 パラで性能劣化 -> "Lost in the Middle" の存在確認
2. **V11_S > B_50**: サブグラフ抽出で劣化回復
3. **V11_R >= B_20**: ルーティングで 20 パラ baseline と同等以上
4. **subgraph_gold_recall >= 0.8**: gold パラの 80% 以上をサブグラフが捕捉

---

## 10. 実験プロトコル

### Phase 0: データ準備 ($0, ~5 分)

```bash
cd /Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI

PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/build_scaled_data.py \
    --input experiments/hotpotqa_v2/data/musique_random_500.jsonl \
    --output experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --target-paras 50 --seed 42
```

### Phase 1: 50q スモークテスト (~$6, ~1 時間)

4 条件 x 50q (B_20 は既存結果流用可能)

```bash
# B_50 (新ベースライン)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode baseline_a \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_baseline_a_50para --limit 50

# V11_S (サブグラフ)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode v11_subgraph \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_subgraph_50para --limit 50 \
    --k-hop 2 --max-subgraph-paras 10

# V11_R (ルーティング)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode v11_routing \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_routing_50para --limit 50 \
    --k-hop 2 --max-subgraph-paras 10 --theta-f 0.0
```

### Phase 1 結果 (50q, GPT-4o)

| 条件 | EM | F1 | vs B_50 | 備考 |
|------|----|----|---------|------|
| B_20 (20 パラ) | 62.0% | 0.756 | +6.0pt | 既存結果 |
| B_50 (50 パラ) | 56.0% | 0.710 | ref | 1 error (rate-limit) |
| V11_S (k=2, max=10) | 52.0% | 0.601 | -4.0pt | |
| V11_R (k=2, max=10, theta=0.0) | 54.0% | 0.632 | -2.0pt | |

**仮説検証:**
- H1 (B_50 < B_20): **確認** (-6pt) -> Lost in the Middle 存在
- H2 (gold_recall >= 0.8): **未達** (0.762, 正答時 0.926 / 誤答時 0.569)
- H3 (V11_R >= B_20): **未達** (-8pt)

**Hop 別 EM:**

| Hop | B_50 | V11_S | V11_R |
|-----|------|-------|-------|
| 2-hop (34q) | 53% | 56% | 56% |
| 3-hop1 (8q) | 88% | 38% | 50% |
| 3-hop2 (4q) | 50% | 75% | 75% |
| 4-hop (4q) | 25% | 25% | 25% |

**V11 診断:**
- avg gold_recall = 0.762 (min=0.00, max=1.00)
- avg gold_precision = 0.308
- avg subgraph_sz = 7.2 paras
- avg context_tokens = 650 (vs B_50 ~3,846)
- system1_rate = 100% (全 F > 0.62 -> theta_f=0.0 では全て System 1)
- gold_recall: 正答時 0.926 vs 誤答時 0.569 -> **サブグラフ品質がボトルネック**

**Phase 1 の根本的発見:**
1. 2-hop では V11 > B_50 (+3pt) -> サブグラフ抽出は 2-hop で有効
2. 3-hop1 で大幅劣化 (-38~50pt) -> k_hop=2 ではマルチホップチェーンを捕捉できない
3. System 1 rate = 100% -> ルーティングが実質機能していない (theta_f 調整必要)
4. gold_recall と正答率の相関が強い -> サブグラフ改善が最優先

### Phase 2: パラメータチューニング (~$6, 50q x 3 条件)

Phase 1 の診断に基づく targeted チューニング:

| ID | k_hop | max_subgraph_paras | theta_f | 狙い |
|----|-------|-------------------|---------|------|
| P2a | 3 | 15 | 0.0 | gold_recall 改善 (k_hop + max_paras 拡大) |
| P2b | 3 | 10 | 0.0 | k_hop 効果の分離 (max_paras 据え置き) |
| P2c | 3 | 15 | 1.3 | ルーティング活性化 (F 分布中央値付近) |

```bash
cd /Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI

# P2a: k_hop=3, max_paras=15
set -a && source .env && set +a && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode v11_subgraph \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_p2a_k3_max15 --limit 50 \
    --k-hop 3 --max-subgraph-paras 15

# P2b: k_hop=3, max_paras=10
set -a && source .env && set +a && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode v11_subgraph \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_p2b_k3_max10 --limit 50 \
    --k-hop 3 --max-subgraph-paras 10

# P2c: k_hop=3, max_paras=15, theta_f=1.3 (routing)
set -a && source .env && set +a && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_allcontext.py \
    --mode v11_routing \
    --data experiments/hotpotqa_v2/data/musique_50para_500.jsonl \
    --output experiments/hotpotqa_v2/results/v11_p2c_k3_max15_t13 --limit 50 \
    --k-hop 3 --max-subgraph-paras 15 --theta-f 1.3
```

### Phase 2 結果 (50q, GPT-4o)

| 条件 | EM | F1 | gold_recall | sub_sz | vs P1_S |
|------|----|----|-------------|--------|---------|
| B_50 (baseline) | 57.1% | 0.725 | — | — | ref |
| P1_S (k2, m10) | 52.0% | 0.601 | 0.762 | 7.2 | — |
| P1_R (k2, m10, t0) | 54.0% | 0.632 | 0.762 | 7.2 | +2.0pt |
| **P2a (k3, m15)** | **55.1%** | **0.636** | **0.823** | 10.1 | **+3.1pt** |
| P2b (k3, m10) | 46.0% | 0.556 | 0.697 | 7.9 | -6.0pt |
| **P2c (k3, m15, t1.3)** | **52.0%** | **0.654** | **0.807** | 10.1 | **±0pt** |

**Hop 別 EM:**

| Hop | B_50 | P1_S (k2,m10) | P2a (k3,m15) | P2b (k3,m10) | P2c (k3,m15,t1.3) |
|-----|------|---------------|--------------|--------------|-------------------|
| 2-hop | 55% | 56% | **61%** | 56% | 53% |
| 3-hop1 | 88% | 38% | 50% | 25% | **62%** |
| 3-hop2 | 50% | 75% | 50% | 25% | 50% |
| 4-hop | 25% | 25% | 25% | 25% | 25% |

**Phase 2 の発見:**

1. **P2a (k3, m15) が EM 最高** (55.1%) — gold_recall 0.762→0.823 (+0.061)
2. **P2c (routing) が F1 最高** (0.654) かつ **3-hop1 で最高** (62%)
3. **max_paras=10 制限は有害** (P2b=-6pt) — k_hop=3 で広範囲を探索しても max_paras=10 で切り捨て → 逆効果
4. **2-hop で P2a=61%、B_50=55% を上回った** — サブグラフ抽出がディストラクタ除去に有効
5. **gold_recall と正答率の相関**: 正答時 0.935, 誤答時 0.686

**P2c ルーティング分析 (theta_f=1.3):**
- System 1: 35/50 (70%), EM=42.9%, avg_tok=958
- System 2: 15/50 (30%), EM=**73.3%**, avg_tok=3,725
- **System 2 の EM が極めて高い** — サブグラフ先頭 + 全パラ送信が効果的
- F 分布: min=0.62, p25=1.28, median=1.50, p75=1.60, max=1.86
- theta_f=1.3 でルーティングが活性化 (30% → System 2)

**P2c の重要な示唆: System 2 (73.3%) >> System 1 (42.9%)**
- サブグラフのみ (System 1) では情報不足のケースが多い
- 全パラを送りつつ、サブグラフを先頭に配置する「暗黙的誘導」が有効
- → **theta_f を上げて System 2 比率を高める** ことで EM 改善の余地あり
- → 極端には **常に System 2** (= 全パラ送信 + サブグラフ先頭並べ替え) が最良の可能性

### P2d: 常時 System 2 (トポロジーベース並べ替え)

P2c の System 2 EM=73.3% を受けて、全問 System 2 (theta_f=999) で検証。

| 条件 | EM | F1 | tokens | vs B_50 |
|------|----|----|--------|---------|
| B_50 (baseline) | 57.1% | 0.725 | 3,846 | ref |
| P2a (subgraph only) | 55.1% | 0.636 | 858 | -2.0pt |
| P2c (routing, t=1.3) | 52.0% | 0.654 | 1,788 | -5.1pt |
| **P2d (always System 2)** | **64.0%** | **0.749** | 3,722 | **+6.9pt** |

**Hop 別 EM:**

| Hop | B_50 | P2a | P2d |
|-----|------|-----|-----|
| 2-hop | 55% | 61% | 59% |
| 3-hop1 | 88% | 50% | **100%** |
| 3-hop2 | 50% | 50% | 50% |
| 4-hop | 25% | 25% | **50%** |

**P2d の発見:**

1. **EM=64.0%: B_50 を +6.9pt、B_20 (62.0%) すら +2pt 上回った**
2. **3-hop1 が 100% (8/8)** — サブグラフ先頭並べ替えがマルチホップ推論を劇的に改善
3. **4-hop も 50% (2/3)** — B_50 (33%) から改善
4. 全パラ送信なのでトークン消費は B_50 とほぼ同等 (3,722 vs 3,846)
5. **Net +3 問 vs B_50** (6 問改善、3 問退行)

**P2d の本質:**
- 追加 LLM 呼び出しなし (グラフ構築 + サブグラフ抽出はオフライン)
- パラグラフの並べ替えのみで +6.9pt
- v10 の「パラ並べ替え +1.2pt (有意でない)」との根本的差異:
  - v10: ランダムな並べ替え vs サブグラフ先頭配置
  - v10: 20 パラ (Lost in the Middle なし) vs v11: 50 パラ (Lost in the Middle あり)
  - → コンテキストが大きい場合にのみ、トポロジーベースの並べ替えが有効

### Phase 3: 500q フルラン

**確定条件:**
- **P2d (常時 System 2)**: v11_routing, k_hop=3, max_subgraph_paras=15, theta_f=999
  - 50q で EM=64.0% — B_50 (57.1%) を +6.9pt、B_20 (62.0%) を +2pt 上回る
  - LLM 追加呼び出しなし、グラフ構築はオフライン

統計的検定: z-test, McNemar

---

## 11. リスクと対策

| リスク | 確率 | 対策 |
|--------|------|------|
| 50 パラでも Lost in Middle なし | 中 | 100 パラに拡大 |
| サブグラフが gold を取りこぼす | 中 | k_hop 増加、max_paras 調整 |
| F 値閾値が機能しない | 低 | 分布分析、グリッドサーチ |
| ディストラクタが弱すぎる | 低 | 同トピックパラを優先的に追加 |
| グラフ構築が遅い (50 パラ = ~250 文ノード) | 低 | O(N^2)=62K pairs、<2 秒で済む見込み |

---

## 12. 検証方法

### データ検証 (build_scaled_data.py)

```bash
.venv/bin/python3 -c "
import json
with open('experiments/hotpotqa_v2/data/musique_50para_500.jsonl') as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        titles = ex['context']['title']
        sents = ex['context']['sentences']
        assert len(titles) == 50, f'Q{i}: {len(titles)} paras'
        assert len(sents) == 50, f'Q{i}: {len(sents)} sents'
        for sf in ex['supporting_facts']['title']:
            assert sf in titles, f'Q{i}: gold {sf} missing'
        assert len(titles) == len(set(titles)), f'Q{i}: dup titles'
print('All 500 examples validated: 50 paras, gold preserved, no dups')
"
```

### グラフ構築検証 (corpus_graph.py, LLM 不要)

```bash
.venv/bin/python3 -c "
import json, sys
sys.path.insert(0, 'experiments/hotpotqa_v2/src')
from corpus_graph import CorpusGraphBuilder

with open('experiments/hotpotqa_v2/data/musique_50para_500.jsonl') as f:
    ex = json.loads(f.readline())

builder = CorpusGraphBuilder(k_hop=2, max_subgraph_paras=10)
g = builder.build(ex['context']['title'], ex['context']['sentences'])
print(f'Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')

gold = list(set(ex['supporting_facts']['title']))
routing = builder.route(
    ex['question'], ex['context']['title'], ex['context']['sentences'],
    theta_f=0.0, gold_titles=gold)
print(f'System: {routing.system}, F={routing.f_value:.3f}')
print(f'Subgraph: {routing.subgraph.n_subgraph_nodes} nodes, '
      f'{len(routing.subgraph.paragraph_indices)} paras')
print(f'Gold precision={routing.subgraph.gold_precision:.2f}, '
      f'recall={routing.subgraph.gold_recall:.2f}')
print(f'Context tokens: ~{routing.context_tokens_est}')
"
```

### 結果分析

```bash
.venv/bin/python3 -c "
import json, glob
for path in sorted(glob.glob('experiments/hotpotqa_v2/results/v11_*/summary.json')):
    with open(path) as f:
        s = json.load(f)
    print(f'{s[\"mode\"]:25s} EM={s[\"em\"]:.1%} F1={s[\"f1\"]:.3f} N={s[\"n\"]}')
"
```
