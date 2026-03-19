# Multi-hop QA Experiments

Four experiment lines in this directory:

1. **v2/v3 (HotpotQA)**: geDIG + Betti numbers + dual-process architecture
2. **v10 (MuSiQue)**: Entity-graph guided paragraph reordering
3. **v11 (MuSiQue)**: Pre-computed topology routing (50-paragraph)
4. **v12 (FRAMES / BRIGHT)**: Open-world topology-guided retrieval ← **active**

---

## v12 BRIGHT: Reasoning-Intensive Document Retrieval (Active)

### 目標

nDCG@10 = **0.45** (現在ベスト: bio50q=**0.4390** Pyserini+Enhanced Graph+Early TG+RIA [Spec X])

BRIGHT ベンチマーク (ICLR 2025) の 3 ドメイン (biology, economics, stackoverflow) で、
geDIG ベースのグラフ re-ranking パイプラインの性能を検証・改善する。

### ベンチマーク概要

- **BRIGHT**: 1,384 クエリ, 12 ドメイン, 1.33M 文書
- 推論集約型 — 標準的な検索モデルは大幅に性能低下
- BM25 baseline = 14.5, SOTA (INF-X-Retriever) = 63.4 nDCG@10
- Leaderboard: https://brightbenchmark.github.io/

---

### パイプライン アーキテクチャ (v26 現行, Spec W)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BM25 Engine Selection                                                      │
│  --bm25-engine rank_bm25   → Python rank_bm25 (Porter stemmer + stopwords)│
│  --bm25-engine pyserini    → Lucene BM25 via Pyserini (Java 21)     ★推奨 │
│                              k1=0.9, b=0.4 (BRIGHT論文設定)               │
│                              DefaultEnglishAnalyzer (内部トークナイズ)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: BM25 Initial Retrieval (top-100)                                  │
│  → bm25_tokenize(query) でトークナイズ                                     │
│    - rank_bm25 モード: NLTK stopwords + Porter stemmer                     │
│    - pyserini モード: raw split (Lucene が内部でトークナイズ)              │
│  → BM25 scoring → top-100 候補を取得                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1.8: Early Token Graph (DG Gap Detection) [Spec W] ★NEW             │
│  → BM25 top-20 文書に対して軽量 Token Graph を構築                        │
│  → AG/DG エッジ分類 + Insight Pattern A/B で gap_lemmas を抽出            │
│    - Pattern A: AG サブグラフの非連結クラスタを橋渡しする lemma            │
│    - Pattern B: DG エッジを含む最短経路上の中間 lemma                      │
│  → gap_lemmas = 「クエリと文書を繋ぐのに不足している概念」                 │
│  → Phase 2 の CoT prompt と Phase 2.6 の RIA keywords に注入              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: LLM CoT Reasoning (gpt-4o-mini)                                  │
│  → クエリに対する推論チェーンを生成                                        │
│  → ★ gap_lemmas がある場合、CoT prompt に bridging concepts として注入    │
│     「構造分析で特定された橋渡し概念: {gap_lemmas}」                       │
│  → 推論エンティティ抽出 (NER + 大文字語)                                  │
│  → [Optional] Multi-CoT Ensemble: N本生成 (temp=0.7) [Spec P]            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2.5: CoT Re-retrieval                                                │
│  → 抽出エンティティを BM25 クエリとして再検索 (top-50 新規文書)            │
│  → Phase 1 候補とマージ → 拡張候補プール                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2.6: RIA Iterative Expansion [Spec M, --ria-loop]                   │
│  → エンティティグラフから β₀ (連結成分数) を計算                           │
│  → β₀ > target なら追加検索 (最大3ラウンド, 50docs/round)                 │
│  → ★ 初回ラウンドで gap_lemmas を RIA keywords に追加 [Spec W]            │
│  → 候補プールを段階的に拡大 → gold doc の recall 改善                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Entity Graph Construction                                         │
│  → top-50 文書から文レベルの三層エンティティグラフを構築                   │
│    Tier 1 (Context):    隣接文 (cost 0.05-0.10)                           │
│    Tier 2 (Entity):     判別的エンティティ重複 (cost 0.20-0.50)            │
│    Tier 3 (Similarity): TF-IDF cosine sim ≥ 0.30 (cost 0.50-0.80)        │
│  → β₀ (連結成分), β₁ (サイクルランク) を計算                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: CoT Virtual Node Injection                                        │
│  → クエリを仮想ノードとしてグラフに追加                                    │
│  → 関連ノードへのブリッジエッジを生成 → 推論ギャップを橋渡し              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4.5: Per-Document Token Graph [Spec N, --token-graph]               │
│  → 各文書を spaCy 依存構文解析 → トークンレベルグラフ構築                  │
│  → エッジ分類: AG (Anchor-Generation) vs DG (Development)                 │
│  → F-evaluation: f_theta = cost - λ·relevance                            │
│  → Walk Score: DG ペナルティ付き重み付き最短経路 [Spec N.1]               │
│  → Insight Vector 注入 (graph_agg / path_bridge / both)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Scoring (--scoring-mode)                                          │
│                                                                            │
│  scoring_mode="gedig_refine" ★推奨                                        │
│    1. MessagePassing: クエリ関連度をグラフ上で伝播                         │
│    2. EdgeReevaluation: 更新された特徴量でエッジを再評価                   │
│       → 類似度の低いエッジを除去、新しい接続を発見                         │
│    3. 精錬されたグラフ上で classic 5-component scoring:                    │
│       score = 0.4·PageRank + 0.3·entity_overlap                           │
│             + 0.2·token_overlap + 0.1·degree + CoT_bridge                 │
│                                                                            │
│  [Optional] Entity F-eval blend [Spec O, --entity-feval]                  │
│    → 文書間 AG/DG エッジ分類 + Δβ₁ 構造スコア                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 5.5: Token Graph Blend [Spec N]                                      │
│  → Walk Score / F-eval を最終ランキングにブレンド (weight=0.15)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 6: Combined BM25 + Graph Ranking                                     │
│  → final_score = α·BM25_norm + (1-α)·graph_score                          │
│  → α = rerank_alpha (default 0.1 → graph 重視)                            │
│  → Top-10 を選択 → nDCG@10, Recall@10, MRR を算出                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Spec W: Early Token Graph の設計思想

```
問題: Token Graph (Phase 4.5) の AG/DG 分類は CoT (Phase 2) より後
      → DG gap 情報が CoT 生成時に利用できない (循環依存)

解決: 2-pass アプローチ
      Pass 1: BM25 top-20 で軽量 TG → gap_lemmas 抽出 (Phase 1.8)
      Pass 2: gap_lemmas で指向的 CoT + RIA → 本格 TG (Phase 4.5)

教訓: Spec S の geDIG loop (Phase 5.75 → re-CoT → re-search) は
      CoT 非決定性で -17.3% 劣化 → ループせず前方注入が安全
```

### BM25 エンジン比較

| エンジン | トークナイズ | BM25 パラメータ | nDCG@10 (bio50q) |
|----------|-------------|----------------|-------------------|
| `rank_bm25` (旧) | NLTK stopwords + Porter stemmer (Python) | k1=1.5, b=0.75 | 0.175 |
| **`pyserini` (新)** | **Lucene DefaultEnglishAnalyzer (Java)** | **k1=0.9, b=0.4** | **0.175** |

BM25 only の nDCG は同等だが、Pyserini は BRIGHT 論文と同一エンジン・同一パラメータ。
グラフ re-ranking と組み合わせた際に大幅な改善を実現。

### 主要コンポーネント

#### geDIG Graph Refinement (Spec H)
```
geDIG_local(doc) = Δ_GED − λ·(Δ_H + β·Δ_SP)
```
- **MessagePassingNX**: クエリ関連度をグラフ上で K 回反復伝播
- **EdgeReevaluatorNX**: 更新特徴量に基づくエッジの動的追加/除去
- **GeDIGDocScorer**: 文書ごとの局所 geDIG スコア算出
- `gedig_refine` モード: geDIG でグラフを精錬 → classic scoring を適用 (★ベスト)

#### Token Graph (Spec N / N.1)
- spaCy 依存構文解析 → トークンレベル有向グラフ
- エッジタイプ: `dep` (依存関係), `root_chain` (文間接続), `same_lemma` (語彙結束)
- **F-evaluation**: f_theta < 0 → AG (関連), f_theta > 0 → DG (発見)
- **Walk Score**: AG エッジを優遇、DG エッジにペナルティ → 構造的接続品質を評価

#### RIA Iterative Expansion (Spec M)
- エンティティグラフの β₀ (連結成分数) をゲート信号として使用
- β₀ > target → CoT クエリで追加検索 → 候補プール拡大
- 最大3ラウンド、各50文書 → gold doc の recall を +10% 改善

#### Early Token Graph (Spec W)
- BM25 top-20 文書に対して軽量 Token Graph を Phase 2 の前に実行
- AG/DG 分類 → Insight Pattern A (非連結橋渡し) + Pattern B (DG 経路中間) で gap_lemmas 抽出
- gap_lemmas を CoT prompt に注入 → 推論の指向性を改善
- gap_lemmas を RIA 初回 keywords に追加 → 情報ギャップに対する直接的な検索
- Spec S (geDIG loop) の教訓: ループ方式は CoT 非決定性で不安定 → 前方注入が安全

#### Enhanced Graph Construction (Spec X)
- spaCy sentencizer (regex → NLP ベースの文分割)
- spaCy NER + noun chunks (大文字語 regex → 構造的エンティティ抽出)
- Lemma matching (表層一致 → lemma 一致 + overlap 比例コスト)
- AG クエリで +5.7%, DG クエリで ≈中立 → グラフの AG エッジ品質を改善

#### AGHT: Analytical Heterogeneous Graph Transformer (Spec Z)
```
統合グラフ: Sentence ノード + Token ノード + クロスレベルエッジ

QKV Edge Evaluation (内積ベース) → src/gedig/core/f_eval.py に統一:
  Q(u) = [w_q1·direct_match, w_q2·nbr_density, w_q3·cot_rel]
  K(v) = [w_k1·importance,   w_k2·discriminativeness, w_k3·struct]
  V(e) = [w_v1·base_cost,    w_v2·bridge_flag,  w_v3·level_cross]

  α = dot(Q, K) / √d_k          ← attention weight
  f = cost(e) - λ·α              ← AG/DG classification (= F-eval)
  flow = α · |V|                  ← message passing information flow

AG (f < θ): high attention = confirmed edge
DG (f ≥ θ): low attention = reasoning gap
```
- HGT (Heterogeneous Graph Transformer) と同構造だが学習不要 (10 パラメータ)
- Sentence-Token 間に `contains` エッジ + 文書横断 `same_lemma_x` エッジ
- Grid search で最適化: mp_alpha=0.1 (平滑化抑制) が文レベル精度の鍵

#### Unified geDIG Core (`src/gedig/`)
```
3実験 (maze, RAG, transformer) の F-eval を統一:

  F = ΔEPC - λ(ΔH + γΔB)

  src/gedig/
  ├── core/          protocols.py, f_eval.py, ag_dg.py, message_passing.py
  ├── backends/      networkx_backend.py (maze+RAG), torch_backend.py (transformer)
  ├── adapters/      maze.py, rag.py, transformer.py
  └── tests/         71 tests pass

Transformer Exp4 で実証: F最大化(DG保持) > Baseline > F最小化(DG破壊)
→ DG構造は「罰する」のではなく「保存する」= attention の多様性が学習に必要
```

#### Wake-Sleep-Wake アーキテクチャ (M + N.1)
```
Wake (RIA)        → 候補プールを反復的に拡大 (β₀-gated)
Sleep (Walk Score) → DG/AG エッジ分類で構造品質を評価
Wake (Confirmation) → 精錬されたスコアで最終ランキング
```
迷路実験と同じ原理: **先に探索しないとループ検知は無意味**

---

### 実験結果サマリー

#### Pyserini BM25 + geDIG パイプライン (Biology 50q)

| 構成 | nDCG@10 | 備考 |
|------|---------|------|
| Pyserini BM25 only | 0.175 | ベースライン |
| Pyserini + graph rerank (CoT なし) | 0.238 | +36% |
| Pyserini + CoT + TG + EF (RIA なし) | 0.376 | RIA なしベスト |
| Pyserini + CoT + TG + EF + RIA (v25) | 0.410 | +135% |
| Pyserini + Early TG + CoT + TG + EF + RIA (v26, Spec W) | 0.417 | +1.6% |
| **Pyserini + Enhanced Graph + Early TG + RIA (v27, Spec X)** | **0.439** | **★ BRIGHT ベスト (+7.1%)** |

#### 旧 BM25 (rank_bm25) 結果 (Biology 50q, α=0.1)

| Spec | Configuration | nDCG@10 | R@10 | MRR | Note |
|------|---------------|---------|------|-----|------|
| A | Classic (CoT re-retrieval) | 0.2438 | 0.2173 | 0.3740 | 5成分 baseline |
| H | geDIG refine | 0.2496 | 0.2419 | 0.4183 | graph refinement |
| M | RIA iterative expansion | 0.2564 | 0.2661 | 0.3815 | +2.7% |
| N | Token graph (Spec N only) | 0.2544 | 0.2486 | 0.3917 | +1.9% |
| M+N | RIA + Token Graph | 0.2707 | 0.2643 | 0.3972 | +8.5% |
| **M+N.1** | **RIA + Walk Score** | **0.3181** | **0.3139** | **0.4424** | **+27.4% (WSW)** |

#### BM25 エンジン × パイプライン構成の比較

| BM25 Engine | Pipeline | nDCG@10 | 旧比 |
|-------------|----------|---------|------|
| rank_bm25 | CoT + noRIA (ベスト旧構成) | 0.325 | — |
| rank_bm25 | CoT + RIA + Walk Score | 0.318 | — |
| pyserini | CoT + noRIA | 0.376 | +15.7% |
| pyserini | CoT + RIA (v25) | 0.410 | +26.2% |
| pyserini | Early TG + CoT + RIA (v26, Spec W) | 0.417 | +28.3% |
| **pyserini** | **Enhanced Graph + Early TG + RIA (v27, Spec X)** | **0.439** | **★ +35.1%** |

**Pyserini BM25 への切替で全構成が大幅改善。** 論文同一エンジン・パラメータが鍵。

#### Negative Results (効果なし/劣化した Spec)

| Spec | 内容 | nDCG@10 | 変化 |
|------|------|---------|------|
| I | Dense retrieval (E5-base-v2) | 0.150-0.234 | -4%～-38% |
| J | Pointwise LLM rerank (gpt-4o-mini) | 0.197-0.241 | -3%～-21% |
| K | Query decomposition | 0.198 | -1.3% |
| L | gpt-4o reasoning rerank | 0.234 | -6.2% |
| P | Multi-CoT Ensemble (N=3) | 0.083 | -32% |

### Full 323q 結果 (3 domains, rank_bm25, α=0.1)

| Configuration | Biology | Economics | StackOverflow | Overall |
|---------------|---------|-----------|---------------|---------|
| Spec A (classic) | 0.1879 | 0.1240 | 0.1470 | 0.1520 |
| Spec H (geDIG refine) | 0.2069 | 0.1187 | 0.1296 | 0.1508 |
| **M+N.1 (RIA + Walk Score)** | **0.2574** | **0.1402** | **0.1739** | **0.1898** |

---

### Spec 進行状況

| Spec | 内容 | 状態 | 結果 |
|------|------|------|------|
| A | CoT re-retrieval + entity graph | ✅ 完了 | nDCG=0.152 (323q) |
| B-D | Adaptive routing, LLM rerank | ✅ 完了 | 微改善 (0.152→0.160) |
| E | geDIG routing (tier selection) | ✅ 完了 | 効果なし |
| F-G | Episode graph, Hybrid graph | ✅ 完了 | 効果なし |
| H | geDIG scoring (graph refinement) | ✅ 完了 | gedig_refine が最良 scoring mode |
| I | Dense retrieval integration | ✅ 完了 | 改善なし (-4%～-38%, 構造的限界) |
| J | Pointwise LLM reranking | ✅ 完了 | 改善なし (-3%～-21%) |
| K | Query decomposition | ✅ 完了 | ≈中立 (-1.3%) |
| L | Stronger LLM Reranking (gpt-4o) | ✅ 完了 | 改善なし (-6.2%) |
| M | RIA Iterative Expansion | ✅ 完了 | +2.7% (初の正改善, R@10+10%) |
| N | Token-level Graph Scoring | ✅ 完了 | M+Nで +8.5% |
| **N.1** | **geDIG Walk Score** | ✅ 完了 | **M+N.1で +27.4% (Wake-Sleep-Wake)** |
| O | Entity F-eval scoring | ✅ 完了 | P との組合せで検証 |
| P | Multi-CoT Ensemble | ✅ 完了 | -32% 劣化 (保留) |
| S | geDIG CoT Loop (gap_lemmas → re-CoT) | ✅ 完了 | -17.3% 劣化 (CoT 非決定性) |
| **V** | **Pyserini BM25 統合** | ✅ 完了 | **★ 全構成で +15-26% 改善** |
| **W** | **Early Token Graph → DG-Guided CoT/RIA** | ✅ 完了 | **★ +1.6% (0.410→0.417, 安定改善)** |
| **X** | **Enhanced Graph (spaCy NER + lemma match)** | ✅ 完了 | **★ +5.3% (0.417→0.439, R@10 +13%)** |
| Y | Progressive DG Escalation | ✅ 完了 | 閾値調整中 (BRIGHT は全 DG のため要検証) |
| **Z** | **AGHT: Unified Heterogeneous Graph Transformer** | ✅ 完了 | **★ HotpotQA R@2=0.405 (+170% vs Legacy)** |
| **Core** | **Unified geDIG Core (src/gedig/)** | ✅ 完了 | **3実験統合, 71 tests, 旧コード refactor_ にアーカイブ** |

#### Spec Z: AGHT — HotpotQA Paragraph Selection (100q)

| | AGHT (ours) | Legacy (PageRank) | 改善 |
|---|---|---|---|
| **R@2** (段落) | **0.405** | 0.150 | **+170%** |
| **MRR** | **0.659** | 0.346 | **+90%** |
| **SF F1** (文) | **0.334** | — | ゼロショット |

Bridge (DG) vs Comparison (AG) 分析:

| | Bridge (n=86) | Comparison (n=14) |
|---|---|---|
| R@2 | 0.401 | 0.429 |
| SF F1 | 0.335 | 0.327 |

最適パラメータ (grid search 144 configs):
`mp_alpha=0.1, mp_iterations=1, w_q1=0.5, f_lambda=0.5`

**ゼロショット (10 パラメータ) で SF F1=33.4** — 教師ありモデル (DFGN: 81.1) の 41% を学習データなしで達成。

#### Transformer Experiment 4: F-Regularized Training

| Condition | SP版 | β₁版 | 結論 |
|-----------|------|------|------|
| Baseline (CE only) | 88.1% | 88.5% | 対照 |
| Positive (CE + F_min) | 87.2% | 83.5% | **F最小化で劣化** |
| Negative (CE + F_max) | **89.4%** | 85.5% | **F最大化で改善** |

**結論**: `negative_better` — DG 構造 (attention の多様性) を保持した方が学習が良い。
事前学習で獲得した attention トポロジーを F 最小化が破壊 → 表現力が失われる。

---

### 学んだこと

#### BM25 エンジンの影響 (Spec V)
- **Pyserini (Lucene) BM25** は BRIGHT 論文と同一実装 (k1=0.9, b=0.4)
- BM25 only では rank_bm25 と同等だが、**グラフパイプラインとの相乗効果が大きい**
- Lucene の DefaultEnglishAnalyzer が Python 側の NLTK tokenizer より高品質
- **全構成で +15-26% の改善** — BM25 基盤品質がパイプライン全体に波及

#### Early Token Graph の教訓 (Spec W vs Spec S)
- **ループ方式 (Spec S)** は CoT 非決定性により -17.3% 劣化 — 毎回異なる CoT → 異なるグラフ → 不安定
- **前方注入方式 (Spec W)** は +1.6% の安定改善 — BM25 top-20 で軽量 TG → gap_lemmas を CoT/RIA に注入
- **2-pass で循環依存を解消**: Token Graph は graph 構築後 (Phase 4.5) に必要だが、CoT は Phase 2 → 軽量版を先行実行
- 追加コストは ~700ms/query — 全体 23.6s の ~3% で再現性のある改善

#### グラフ re-ranking の教訓 (Spec I-N.1)
- **Pointwise reranking は BRIGHT に不適** — gpt-4o-mini (J) も gpt-4o (L) も改善せず
- **候補プール拡張 (I, K)** は gold recovery に成功するが、scoring で活かせない
- **RIA 反復拡張 (M)** が Recall 改善 (+10%) — 初の正改善
- **Token Graph Walk Score (N.1)** が Ranking 改善 — DG/AG 構造品質評価
- **M+N.1 相乗効果** = Wake-Sleep-Wake: 先に探索しないとループ検知は無意味

---

### Quick Start (BRIGHT)

**注意**: `answerer.py` が `.env` を自動読み込み (`python-dotenv`) するため、
プロジェクトルートの `.env` に `OPENAI_API_KEY=sk-...` があれば OK。

**前提条件**:
- Python 3.11 (`.venv`)
- Java 21 (Pyserini 用): `brew install openjdk@21`
- Pyserini: `.venv/bin/pip install pyserini==0.25.0`
- spaCy モデル: `.venv/bin/python3 -m spacy download en_core_web_sm`

```bash
# 1. データ準備 (初回のみ)
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/prepare_bright.py

# 2. Lucene index 構築 (初回のみ, Pyserini 使用時)
#    → build_bm25_index() が自動で構築するため手動不要

# 3. Smoke test (10q, biology, Pyserini + Spec W)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/smoke_test \
    --limit 10 --graph-top-k 50 --rerank-alpha 0.1 \
    --bm25-engine pyserini \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score \
    --token-graph-f-eval --token-graph-insight both \
    --entity-feval --entity-feval-version v2 \
    --early-token-graph

# 4. ★推奨構成: Pyserini + Early TG + CoT + TG + EF + RIA (50q biology)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v26_best \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --bm25-engine pyserini \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score \
    --token-graph-f-eval --token-graph-insight both \
    --entity-feval --entity-feval-version v2 \
    --ria-loop --ria-max-rounds 3 \
    --early-token-graph

# 5. Full 323q (3 domains, Pyserini)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval \
    --domains biology,economics,stackoverflow \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v26_full \
    --graph-top-k 50 --rerank-alpha 0.1 \
    --bm25-engine pyserini \
    --scoring-mode gedig_refine \
    --token-graph --token-graph-walk-score \
    --token-graph-f-eval --token-graph-insight both \
    --entity-feval --entity-feval-version v2 \
    --ria-loop --ria-max-rounds 3 \
    --early-token-graph
```

### CLI オプション一覧

| Option | Default | Description |
|--------|---------|-------------|
| **基本** | | |
| `--mode` | — | `bm25_only`, `graph_rerank`, `cot_retrieval` |
| `--scoring-mode` | `classic` | `classic`, `gedig`, `gedig_refine` ★ |
| `--bm25-engine` | `rank_bm25` | `rank_bm25` (Python), `pyserini` (Lucene) ★ |
| `--pyserini-k1` | 0.9 | Pyserini BM25 k1 パラメータ |
| `--pyserini-b` | 0.4 | Pyserini BM25 b パラメータ |
| `--graph-top-k` | 50 | グラフ構築に使う文書数 |
| `--rerank-alpha` | 0.1 | BM25 weight (低い = graph 重視) |
| `--limit` | — | クエリ数制限 (smoke test 用) |
| **geDIG Scoring (Spec H)** | | |
| `--gedig-lambda` | 1.0 | geDIG: GED vs IG balance |
| `--gedig-sp-beta` | 0.5 | geDIG: shortest-path weight |
| `--gedig-k-hop` | 2 | Local subgraph k-hop radius |
| `--gedig-mp-iterations` | 2 | Message passing iterations |
| `--gedig-mp-alpha` | 0.3 | Query influence weight |
| **RIA Iterative Expansion (Spec M)** | | |
| `--ria-loop` | false | RIA 反復クエリ拡張を有効化 |
| `--ria-max-rounds` | 3 | 最大 RIA ラウンド数 |
| `--ria-docs-per-round` | 50 | ラウンドあたりの新規取得文書数 |
| `--ria-feedback-top-k` | 5 | LLM フィードバック用 top-k 文書 |
| `--ria-beta0-target` | 1 | RIA 収束目標 β₀ |
| **Token Graph (Spec N/N.1)** | | |
| `--token-graph` | false | Token graph scoring 有効化 |
| `--token-graph-weight` | 0.15 | Graph scores とのブレンド比率 |
| `--token-graph-max-tokens` | 500 | spaCy パース対象のトークン上限 |
| `--token-graph-walk-score` | false | DG/AG 重み付き Walk Score |
| `--token-graph-dg-penalty` | 2.0 | DG エッジのコストペナルティ |
| `--token-graph-f-eval` | false | F-evaluation 有効化 |
| `--token-graph-insight` | `none` | Insight 注入モード: `none`, `graph_agg`, `path_bridge`, `both` |
| **Early Token Graph (Spec W)** | | |
| `--early-token-graph` | false | Phase 1.8 Early TG 有効化 (DG gap → CoT/RIA 指向) |
| `--early-tg-top-k` | 20 | Early TG に使う BM25 上位文書数 |
| **Entity F-eval (Spec O)** | | |
| `--entity-feval` | false | Entity F-eval scoring 有効化 |
| `--entity-feval-weight` | 0.1 | F-eval スコアのブレンド比率 |
| `--entity-feval-version` | `v1` | `v1` (percentile), `v2` (structural Δβ₁) |
| **Multi-CoT Ensemble (Spec P)** | | |
| `--n-cot-ensemble` | 1 | CoT 生成本数 (1=従来, 3=ensemble) |
| `--cot-cache-dir` | — | CoT キャッシュディレクトリ (再現性用) |
| `--cot-temperature` | 0.7 | Ensemble CoT 生成 temperature |

### BRIGHT 関連ファイル

| ファイル | 説明 |
|---------|------|
| **Pipeline Core** | |
| [src/bright_cot_pipeline.py](src/bright_cot_pipeline.py) | CoT × Graph re-ranking pipeline (main, ~2900行) |
| [src/bright_pipeline.py](src/bright_pipeline.py) | BM25 baseline + トークナイザ + グラフ rerank |
| [src/pyserini_bm25.py](src/pyserini_bm25.py) | Pyserini (Lucene) BM25 ラッパー (drop-in replacement) |
| [src/entity_graph.py](src/entity_graph.py) | Three-tier entity graph + TF-IDF features |
| [src/gedig_scoring.py](src/gedig_scoring.py) | MessagePassingNX + EdgeReevaluatorNX + GeDIGDocScorer |
| [src/token_graph.py](src/token_graph.py) | Per-document token graph + Walk Score + F-eval |
| **Support Modules** | |
| [src/answerer.py](src/answerer.py) | LLM API handler (OpenAI, python-dotenv) |
| [src/dense_retriever.py](src/dense_retriever.py) | E5-base-v2 + FAISS dense retrieval |
| [src/gedig_router.py](src/gedig_router.py) | geDIG routing (tier selection) |
| [src/episode_graph.py](src/episode_graph.py) | Episode-based graph construction |
| **Scripts** | |
| [scripts/run_bright.py](scripts/run_bright.py) | BRIGHT 実験ランナー (全モード対応) |
| [scripts/prepare_bright.py](scripts/prepare_bright.py) | BRIGHT データ準備 |
| [scripts/build_dense_index.py](scripts/build_dense_index.py) | Dense index 構築 |
| **Data** | |
| `data/bright/{domain}_docs.jsonl` | BRIGHT 文書コーパス |
| `data/bright/{domain}_docs_lucene_index/` | Pyserini Lucene インデックス |
| `data/bright/queries.jsonl` | BRIGHT クエリ + gold doc IDs |
| **Reports** | |
| [results/REPORT_SPEC_H_geDIG_scoring.md](results/REPORT_SPEC_H_geDIG_scoring.md) | Spec H: geDIG scoring |
| [results/REPORT_SPEC_I_dense_retrieval.md](results/REPORT_SPEC_I_dense_retrieval.md) | Spec I: Dense retrieval |
| [results/REPORT_SPEC_J_pointwise_reranking.md](results/REPORT_SPEC_J_pointwise_reranking.md) | Spec J: Pointwise reranking |
| [results/REPORT_SPEC_K_query_decomposition.md](results/REPORT_SPEC_K_query_decomposition.md) | Spec K: Query decomposition |
| [results/REPORT_SPEC_L_reasoning_reranking.md](results/REPORT_SPEC_L_reasoning_reranking.md) | Spec L: Reasoning reranking |
| [docs/spec_m_ria_report.md](docs/spec_m_ria_report.md) | Spec M: RIA |
| [docs/spec_n_token_graph_report.md](docs/spec_n_token_graph_report.md) | Spec N: Token Graph |

---

## v10: Entity-Graph Guided Paragraph Reordering (MuSiQue)

### Result

**統計的に有意な改善は確認できず。** ただし複数の再利用可能な知見を得た。

**500q フルラン (GPT-4o, MuSiQue distractor setting):**

| 条件 | EM | F1 | vs Baseline |
|------|----|----|-------------|
| Baseline A (全20パラ + CoT) | 47.4% | 0.614 | ref |
| v10d reorder_only | 48.6% | 0.620 | +1.2pt (有意でない) |

ホップ別では 4-hop に +5.3pt の傾向があるが N=76 で有意に達せず。

### 何をやったか

MuSiQue (20パラ/問題、2-4 hop) において、エンティティグラフから推論チェーンを
特定し、関連パラグラフを先頭に配置して LLM の注意力を暗黙的に誘導する試み。

**アプローチの進化 (v9 → v10d):**

| Version | アプローチ | 結果 (50q) |
|---------|-----------|-----------|
| v9 | タイトル相互参照 + ヒントテキスト | EM=60% (-2pt) |
| v10a | パラレベルグラフ + shortest path | EM=58-60% |
| v10b | 3シグナル混合エッジ | EM=56-60% |
| v10c | 文レベル Three-Tier エッジ分離 | EM=52-62% |
| v10d | v10c + チェーン拡張 | EM=68% (+6pt) → **500q で +1.2pt に縮小** |

### 確立された知見

| 知見 | 確実度 | 根拠 |
|------|--------|------|
| **Guided テキストは GPT-4o に害** | ★★★★★ | 全バージョン全条件で悪化 |
| **暗黙的誘導 > 明示的指示** | ★★★★★ | 一貫した結果 |
| **reorder_only は唯一 "損をしない" 介入** | ★★★★☆ | 500q で最悪 -0.4pt |
| **弱いモデルには reorder が逆効果** | ★★★★☆ | GPT-4o-mini で -12pt |
| 20パラ (~2,500 tokens) では "Lost in the Middle" が発生しない | ★★★★★ | Gold パラ位置とエラーに相関なし |
| エラーの 45% は推論の誤り (distractor entity 選択) | ★★★★★ | 500q エラー分析 |

### 主な教訓

1. **50q スモークテストは方向性のスクリーニングにしかならない** — 95% CI ≈ ±13pt
2. **GPT-4o の非決定性で同じ50問が ±8pt 揺れる** — N=50 での数値は無意味
3. **GPT-4o + 2,500 tokens はパラ位置に無関心** — コンテキストウィンドウの 2% では注意力の偏りが生じない
4. **Baseline のエラーは「情報が見えない」ではなく「推論を間違える」** — reorder で直る問題ではない

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/report_v10_entity_graph.md](docs/report_v10_entity_graph.md) | 完全な実験レポート (12セクション) |
| [docs/experiment_design_v10.md](docs/experiment_design_v10.md) | 実験設計書 |
| [src/entity_graph.py](src/entity_graph.py) | エンティティグラフ + チェーン抽出コア |
| [scripts/run_allcontext.py](scripts/run_allcontext.py) | 実験ランナー (baseline_a / v10_reorder_only / v10_pruned) |

### 次の打ち手 (未実施)

| # | 打ち手 | コスト | 期待値 |
|---|--------|-------|--------|
| 1 | フォーマット修正 (プロンプト改良) | ~$8 | +5~16pt (80問のフォーマット不一致を回収) |
| 2 | Oracle reorder テスト | ~$8 | reorder 仮説を正式に棄却/確認 |
| 3 | ~~ディストラクタ増量 (50-100パラ)~~ | — | **v11 で実施** |
| 4 | 推論改善にピボット | ~$8-20 | 質問分解、self-consistency 等 |

---

## v11: Pre-computed Topology Routing (MuSiQue)

### 仮説

コンテキストを 20 -> 50 パラに拡大して "Lost in the Middle" が発生する条件を作り、
事前構築グラフ + F 値ルーティングで性能劣化を回復できるか検証。

### アーキテクチャ

- **Offline**: 全パラから sentence-level 三層グラフを事前構築 (entity_graph.py 再利用)
- **Online**: クエリからサブグラフ抽出 -> F 値計算 -> System 1/System 2 ルーティング

```
Offline (問題ごとに 1 回):
  全 50 パラ -> sentence-level グラフ構築 -> beta_0, beta_1, centrality 事前計算

Online (クエリごと):
  質問 -> エンティティ抽出 -> グラフノードマッチ -> k-hop サブグラフ抽出
       -> F 値計算 -> System 1 (サブグラフのみ) / System 2 (全パラ、サブグラフ先頭)
```

### 実験条件

| ID | 条件 | パラ数 | 手法 |
|----|------|--------|------|
| B_20 | baseline_a | 20 | Plain CoT (既存結果) |
| B_50 | baseline_a | 50 | Plain CoT |
| V11_S | v11_subgraph | 50 | サブグラフのみ |
| V11_R | v11_routing | 50 | F 値ルーティング |

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/experiment_design_v11.md](docs/experiment_design_v11.md) | 実験設計書 |
| [src/corpus_graph.py](src/corpus_graph.py) | 事前グラフ + サブグラフ抽出 + F 値ルーティング |
| [scripts/build_scaled_data.py](scripts/build_scaled_data.py) | 50 パラデータ生成 |
| [scripts/run_allcontext.py](scripts/run_allcontext.py) | 実験ランナー (v11 モード追加) |

---

## v12: Open-World Topology-Guided Retrieval (FRAMES / BRIGHT)

### 仮説

geDIG の β₀-driven 反復検索で、Wikipedia (FRAMES) や大規模コーパス (BRIGHT) から
マルチホップ推論に必要な記事/文書を自動的に発見できるか検証。

v11 (コンテキストエンジニアリング) から真の RAG への移行:
- **β₀ > 1** → 情報ギャップを検出 → Wikipedia API で橋渡し記事を検索
- **F 値収束** → 検索の自然な停止条件
- **Component Gap Query (v8)** → ブリッジ検索クエリの自動生成

### アーキテクチャ

```
Question → Entity Extraction → Wikipedia Search (Initial)
                                       ↓
                              Entity Graph Construction
                                       ↓
                                   β₀ check
                                    ↓    ↓
                              β₀ = 1  β₀ > 1
                              (done)  (gap detected)
                                        ↓
                              Component Gap Query (LLM)
                                        ↓
                              Bridge Wikipedia Search
                                        ↓
                              Graph Reconstruction → β₀ check (repeat)
                                       ↓
                              Subgraph-first Context → LLM Answer
```

### FRAMES ベンチマーク

- 824 問: 2-11 Wikipedia 記事を要するマルチホップ推論
- 推論タイプ: Multiple constraints, Numerical, Temporal, Tabular

### BRIGHT ベンチマーク

- 1,384 クエリ: 推論集約型文書検索 (12 ドメイン, 1.33M 文書)
- BM25 = 14.5, SOTA = 63.4 nDCG@10

### ファイル

| ファイル | 説明 |
|---------|------|
| [docs/experiment_design_v12.md](docs/experiment_design_v12.md) | 実験設計書 |
| [src/wiki_retriever.py](src/wiki_retriever.py) | Wikipedia API 検索 + テキスト取得 |
| [src/open_world_pipeline.py](src/open_world_pipeline.py) | β₀-driven 反復検索パイプライン |
| [scripts/run_frames.py](scripts/run_frames.py) | FRAMES 実験ランナー |

---

## v2/v3: geDIG with Betti Numbers and Dual-Process Architecture (HotpotQA)

### Key Result

**Hybrid-E1 v3.1** — topology-guided System 1/System 2 switching with **model-dependent scaling**:

**500-question evaluation (primary reference):**

| Model | Hybrid-E1 EM | IRCoT EM | Δ | p-value | LLM Calls |
|:-----:|:---:|:---:|:-:|:------:|:---------:|
| **GPT-4o** | **51.2%** | 47.6% | **+3.6pt** | 0.086 | **2.2 vs 8** |
| GPT-4o-mini | 45.2% | **50.4%** | -5.2pt | **0.008** | 2.2 vs 8 |

Key findings:
- On GPT-4o, Hybrid-E1 **leads IRCoT at 3.6x fewer LLM calls**
- On GPT-4o-mini, IRCoT is significantly better — but Hybrid-E1 achieves 90% quality at 27% cost
- **Model scaling favors topology**: +6pt improvement (mini→4o) vs -3pt for IRCoT
- The gauge value F decides when to think fast (System 1) vs. slow (System 2) — **zero-cost routing**

> See [REPORT_v3_dual_process.md](REPORT_v3_dual_process.md) for the full experiment report.

---

## Overview

This experiment tests two hypotheses:

1. **(v2)** Adding **Betti number** (topological) terms to the geDIG gauge improves multi-hop QA performance
2. **(v3)** Using the gauge value as a **cognitive routing signal** (System 1/System 2 dual-process) improves quality-cost trade-off

### Extended Gauge Formula

```
F = ΔEPC_norm − λ·(ΔH_norm + γ₁·Δβ₁ − γ₀·Δβ₀)
```

- **ΔEPC** (metric): Cost of restructuring the knowledge graph
- **ΔH** (measure): Change in entropy / uncertainty
- **Δβ₁** (topology): Penalizes redundant cycle formation
- **Δβ₀** (topology): Rewards island merging (bridge question signal)

### v3 Dual-Process Architecture

```
Question → BM25 Retrieval → Build Knowledge Graph → Compute F (topology)
                                                         |
                                              ┌──────────┴──────────┐
                                         F < θ_dg              F >= θ_dg
                                        (confident)            (uncertain)
                                              |                      |
                                       ┌──────┴──────┐      ┌───────┴───────┐
                                       │  System 1    │      │   System 2     │
                                       │  Direct (1x) │      │   CoT (2-3x)  │
                                       └──────────────┘      └───────────────┘
```

---

## Full Results (GPT-4o-mini, 100 questions)

| Rank | Method | Category | EM | F1 | LLM Calls |
|:----:|--------|----------|:---:|:---:|:---------:|
| 1 | **Hybrid-E1 v3.1** | **geDIG+CoT** | **48.0%** | **0.622** | **~2.2** |
| 2 | IRCoT | Dynamic RAG | 46.0% | 0.637 | ~8 |
| 3 | GraphRAG | Static RAG | 43.0% | 0.589 | 1 |
| 4 | Hybrid-E1 v3.0 | geDIG+CoT | 40.0% | 0.600 | ~2.2 |
| 5 | geDIG-B | geDIG | 40.0% | 0.570 | 1 |
| 6 | Hybrid(B) | geDIG+CoT | 39.0% | 0.572 | ~1.5 |
| 7 | ReAct | Dynamic RAG | 39.0% | 0.536 | ~7 |
| 8 | E1-tuned | geDIG | 38.0% | 0.553 | 1 |
| 9 | geDIG-C | geDIG | 38.0% | 0.553 | 1 |
| 10 | geDIG-A | geDIG | 37.0% | 0.545 | 1 |
| 11 | geDIG-D | geDIG | 37.0% | 0.544 | 1 |
| 12 | BM25 | Baseline | 37.0% | 0.536 | 1 |
| 11 | ReAct | Dynamic RAG | 39.0% | 0.536 | ~7 |

---

## Experimental Conditions

### v2 Conditions (Betti number ablation)

| Condition | Config | structural_mode | gamma_0 | gamma_1 | Description |
|-----------|--------|-----------------|:-------:|:-------:|-------------|
| A | condition_a_sp.yaml | sp | 0 | 0 | v1 reproduction (no Betti) |
| B | condition_b_beta1.yaml | betti | 0 | 1.0 | beta_1 only (best v2) |
| C | condition_c_beta0.yaml | betti_full | 1.0 | 0 | beta_0 only |
| D | condition_d_betti_full.yaml | betti_full | 1.0 | 1.0 | Full Betti |

### v3 Conditions (Dual-process + tuning)

| Condition | Config | gamma_0 | gamma_1 | theta_dg | hybrid | Description |
|-----------|--------|:-------:|:-------:|:--------:|:------:|-------------|
| E1-tuned | condition_e1_tuned.yaml | 0.3 | 0.5 | -0.5 | no | Tuned params only |
| Hybrid(B) | condition_hybrid.yaml | 0 | 1.0 | 0.0 | yes | Untuned + CoT |
| **Hybrid-E1** | **condition_hybrid_e1.yaml** | **0.3** | **0.5** | **-0.5** | **yes** | **Tuned + CoT (best)** |

### Baselines

| Method | Implementation | Description |
|--------|---------------|-------------|
| BM25 | baselines/bm25_gpt.py | BM25 retrieval + GPT-4o-mini |
| GraphRAG | baselines/static_graphrag.py | Entity-overlap graph + centrality ranking |
| IRCoT | baselines/ircot.py | Interleaving Retrieval with CoT (Trivedi+ 2023) |
| ReAct | baselines/react_baseline.py | Reason + Act loop (Yao+ 2023) |

---

## Quick Start

```bash
# 1. Download data
PYTHONPATH=src .venv/bin/python3 experiments/hotpotqa_v2/scripts/download_data.py

# 2. Run smoke test (mock LLM, 10 examples)
LLM_PROVIDER=mock PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_experiment.py \
    --config experiments/hotpotqa_v2/configs/condition_hybrid_e1.yaml \
    --limit 10

# 3. Run full experiment (requires OPENAI_API_KEY)
set -a && source .env && set +a && \
PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_experiment.py \
    --config experiments/hotpotqa_v2/configs/condition_hybrid_e1.yaml \
    --data experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl \
    --output experiments/hotpotqa_v2/results/real_condition_hybrid_e1

# 4. Run baselines
PYTHONPATH=src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_baseline.py --baseline ircot \
    --data experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl \
    --output experiments/hotpotqa_v2/results/real_baseline_ircot

# 5. Compare conditions
python experiments/hotpotqa_v2/tools/compare_conditions.py \
    experiments/hotpotqa_v2/results/*/summary.json
```

## Tests

```bash
.venv/bin/python3 -m pytest experiments/hotpotqa_v2/test/ -v
```

## Directory Structure

```
hotpotqa_v2/
├── SPEC.md                        # Formal experiment specification
├── README.md                      # This file
├── DESIGN_v3_improvements.md      # v3 improvement design document
├── REPORT_v3_dual_process.md      # v3 experiment report
├── configs/                       # YAML configs for all conditions
├── src/                           # Core modules
│   ├── adapter.py                 # geDIG v2/v3 adapter (extended F + hybrid mode)
│   ├── answerer.py                # Shared LLM handler (mock/GPT-4o-mini)
│   ├── config.py                  # YAML config loader
│   ├── data_loader.py             # HotpotQA data loading
│   ├── evaluator.py               # EM/F1/SF-F1 metrics (type-stratified)
│   ├── graph_builder.py           # beta_0-sensitive knowledge graph construction
│   └── retriever.py               # BM25 retrieval module
├── baselines/                     # BM25, GraphRAG, IRCoT, ReAct baselines
│   ├── base.py                    # BaseRAG interface
│   ├── bm25_gpt.py                # BM25 + GPT baseline
│   ├── static_graphrag.py         # Static GraphRAG baseline
│   ├── ircot.py                   # IRCoT baseline (Trivedi+ 2023)
│   └── react_baseline.py          # ReAct baseline (Yao+ 2023)
├── scripts/                       # Experiment runners
│   ├── run_experiment.py          # geDIG condition runner
│   ├── run_baseline.py            # Baseline runner
│   ├── analyze_results.py         # Results analysis
│   └── tune_gamma.py              # Parameter tuning
├── tools/                         # Post-hoc analysis tools
├── test/                          # Unit tests (35 tests)
├── data/                          # Dataset files (.gitignored)
└── results/                       # Experiment outputs (.gitignored)
```

## Documents

| Document | Experiment | Description |
|----------|-----------|-------------|
| [docs/experiment_design_v11.md](docs/experiment_design_v11.md) | v11 (MuSiQue) | Pre-computed Topology Routing 実験設計書 |
| [docs/report_v10_entity_graph.md](docs/report_v10_entity_graph.md) | v10 (MuSiQue) | Entity-graph 実験レポート (仮説の生死判定、根本原因診断含む) |
| [docs/experiment_design_v10.md](docs/experiment_design_v10.md) | v10 (MuSiQue) | v10 実験設計書 |
| [SPEC.md](SPEC.md) | v2 (HotpotQA) | Formal experiment specification |
| [DESIGN_v3_improvements.md](DESIGN_v3_improvements.md) | v3 (HotpotQA) | v3 improvement design and System 1/2 architecture |
| [REPORT_v3_dual_process.md](REPORT_v3_dual_process.md) | v3 (HotpotQA) | Full v3 experiment report with 11-method comparison |
