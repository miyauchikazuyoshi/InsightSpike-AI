# BRIGHT Spec D: Unified Hybrid Pipeline 実験レポート

**日付**: 2026-03-12
**実験ID**: v12 BRIGHT Spec D
**担当**: InsightSpike-AI

---

## 1. エグゼクティブサマリー

BRIGHTベンチマーク（推論集約型文書検索）において、Dense retrieval (E5-base-v2 + FAISS)、
Tier D グラフエッジ（embedding類似度）、LLM rerank、および全クエリ強制 CoT を統合した
Unified Hybrid Pipeline を実装・評価した。

**結果**: nDCG@10 = 0.1564（BM25比 +130%）。しかし Spec B+C (Adaptive, 0.1596) を
わずかに下回り（-0.0032）、目標の 0.20 に未到達。

**主要発見**:
- Dense retrieval による新規 gold 文書発見は 17件/323q（0.05件/q）と低調
- CoT re-retrieval (272件) が引き続き gold 発見の主力
- 全クエリ強制 tier 3 が biology で大幅なレイテンシ増 (+63%) と性能低下を招来
- Tier D エッジ（平均76本/q）はグラフ密度を上げたが、スコアリング改善に直結せず
- **ボトルネックの本質**: ルーティング（AG/DG判定）の精度不足が最大の課題

---

## 2. 背景

### 2.1 Spec B+C の失敗分析

Spec B+C (Adaptive Re-retrieval) で nDCG=0 の 206 クエリを分析し、
3つの直交する失敗原因を特定:

| 原因 | 件数 | 割合 |
|------|------|------|
| C) 候補プール不足 (BM25限界) | 122 | 59.2% |
| B) Rerank失敗 (gold→top10外) | 45 | 21.8% |
| A) CoTスキップ損失 | 39 | 18.9% |

重複ゼロ → 全対策を同時投入する Unified パイプラインを設計。

### 2.2 アプローチ

3つの対策を1つのパイプラインに統合:

1. **Dense retrieval** (E5-base-v2): BM25の語彙ギャップを semantic search で補完
2. **Tier D グラフエッジ**: 文書間の embedding 類似度をグラフトポロジーに反映
3. **LLM rerank**: top-20 → top-10 の最終判定に LLM を使用
4. **CoT 強制実行**: 全クエリで CoT 生成（スキップ廃止）

### 2.3 設計思想: geDIG × Dense Embedding

従来の entity_graph の Tier 3 (TF-IDF cosine) は語彙一致ベース。
Dense embedding similarity を Tier D としてグラフに組み込むことで、
グラフのトポロジー自体を意味的に豊かにする geDIG の拡張。

```
グラフエッジ階層:
  Tier 1: Context (同段落隣接)          cost 0.05-0.10  語彙不要
  CoT:    CoT bridge                     cost 0.15       語彙一致
  Tier D: Dense embedding similarity     cost 0.15-0.45  意味的類似 ★新規
  Tier 2: Entity (共有エンティティ)      cost 0.20-0.50  語彙一致
  Tier 3: TF-IDF similarity              cost 0.50-0.80  語彙一致
```

---

## 3. 手法

### 3.1 パイプライン

```
[Unified Pipeline]
Phase 1a:  BM25(query) → top-100
Phase 1b:  Dense(query) → top-100 (new docs only)          ★C対策
Phase 1':  Merge → ~190 candidates (dedup)
Phase 1.5: routing_tier=3 固定 (全クエリ攻撃的)             ★A対策
Phase 2:   CoT生成 (全クエリ, gpt-4o-mini)
Phase 2.5a: BM25(CoT concepts) → 100 docs (aggressive)
Phase 2.5b: Dense(CoT text) → 50 docs                      ★C対策
Phase 2.5': Merge → ~340 candidates (dedup)
Phase 3:   Graph構築 (graph_top_k=50, Tier D付き)           ★B対策
Phase 4:   CoT ノード注入
Phase 5:   Graph scoring (PageRank)
Phase 6:   Combined ranking (α=0.1·BM25 + 0.9·Graph)
Phase 7:   LLM rerank top-20 → top-10 (gpt-4o-mini)        ★B対策
```

### 3.2 Dense Retrieval

- **モデル**: `intfloat/e5-base-v2` (384次元, L2正規化)
- **インデックス**: FAISS IndexFlatIP (内積 = cosine sim)
- **コーパス**: 214,660文書 (biology 57K + economics 50K + SO 107K)
- **オフライン構築**: CPU 3.6時間, 合計1.27GB
- **推論時**: Phase 1b で top-100 dense, Phase 2.5b で CoT テキストで top-50

### 3.3 Tier D グラフエッジ

graph_top_k=50 文書の pre-computed embedding 間の cosine similarity を計算。
threshold=0.85 以上のペアにエッジを追加。

- コスト: 0.15 (cos=1.0) → 0.45 (cos=0.85)
- 各パラグラフの first sentence (anchor node) 間に接続
- 50文書 → 最大1,225ペア → 平均76エッジ (threshold=0.85)

**デバッグ過程**: 初期閾値 0.5 では全1,225ペアがエッジ化（完全グラフ化）し、
PageRank が均一化。ペアワイズ cosine 分析で >0.5=100%, >0.85=12% と判明し、
閾値を 0.85 に引き上げ。

### 3.4 LLM Rerank

top-20 候補の先頭200文字をリスト化し、gpt-4o-mini にクエリ+CoT と共に提示。
最も関連性の高い順にドキュメント番号を返却させ、top-10 を再構成。

### 3.5 パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| initial_top_k | 100 | BM25 取得数 |
| dense_top_k | 100 | Dense 取得数 (Phase 1b) |
| dense_cot_top_k | 50 | Dense CoT 取得数 (Phase 2.5b) |
| cot_retrieval_top_k | 100 | BM25 CoT 取得数 (aggressive) |
| cot_retrieval_max_concepts | 20 | BM25 CoT クエリ概念数 |
| graph_top_k | 50 | グラフ構築文書数 |
| dense_sim_threshold | 0.85 | Tier D エッジ閾値 |
| rerank_alpha | 0.1 | BM25 vs Graph 重み |
| llm_rerank_top_k | 20 | LLM rerank 対象数 |
| model | gpt-4o-mini | CoT + LLM rerank |
| routing_tier | 3 (固定) | 全クエリ攻撃的 |

---

## 4. 結果

### 4.1 主要指標: 全条件比較 (323q, 3 domains)

| # | Condition | nDCG@10 | Recall@10 | MRR | vs BM25 |
|---|-----------|---------|-----------|-----|---------|
| 1 | BM25 | 0.0681 | 0.0831 | 0.0893 | — |
| 2 | Graph gk50 | 0.1078 | 0.1159 | 0.1510 | +58.3% |
| 3 | CoT+Graph | 0.1219 | 0.1180 | 0.1816 | +79.0% |
| 4 | CoT Re-retrieval (Spec A) | 0.1520 | 0.1573 | 0.2160 | +123.2% |
| 5 | Adaptive (Spec B+C) | 0.1596 | 0.1733 | 0.2257 | +134.4% |
| 6 | **Unified (Spec D)** | **0.1564** | **0.1728** | **0.2115** | **+129.7%** |

### 4.2 ドメイン別

| Domain | BM25 | Spec A | Spec B+C | **Spec D** | Δ vs B+C |
|--------|------|--------|----------|-----------|----------|
| biology (103q) | 0.0562 | 0.1885 | 0.2367 | **0.1989** | -16.0% |
| economics (103q) | 0.0591 | 0.1368 | 0.1350 | **0.1368** | +1.3% |
| stackoverflow (117q) | 0.0865 | 0.1334 | 0.1133 | **0.1362** | +20.2% |

**ドメイン別分析**:
- **Biology**: Spec B+C (0.2367) から大幅後退 (-16.0%)。
  Adaptive のルーティングが biology に効果的だった一方、Unified の強制 tier 3 が悪影響。
- **Economics**: ほぼ横ばい (+1.3%)。Dense retrieval の効果が限定的。
- **StackOverflow**: 最大の改善 (+20.2%)。Dense retrieval が SO の長い技術文書を捕捉。

### 4.3 nDCG>0 率

| Condition | nDCG=0 | nDCG>0 率 |
|-----------|--------|----------|
| BM25 | 78.9% | 21.1% |
| CoT+Graph | 73.1% | 26.9% |
| CoT Re-retrieval (Spec A) | 67.5% | 32.5% |
| Adaptive (Spec B+C) | 63.8% | 36.2% |
| **Unified (Spec D)** | **63.8%** | **36.2%** |

nDCG>0 率は Spec B+C と同等 (36.2%)。

### 4.4 Recall@10 / MRR 詳細

| Domain | Metric | Spec A | Spec B+C | **Spec D** |
|--------|--------|--------|----------|-----------|
| biology | Recall@10 | 0.1955 | 0.2386 | **0.2162** |
| biology | MRR | 0.2720 | 0.3529 | **0.2878** |
| economics | Recall@10 | 0.1282 | 0.1572 | **0.1256** |
| economics | MRR | 0.1781 | 0.1712 | **0.1858** |
| SO | Recall@10 | 0.1517 | 0.1300 | **0.1763** |
| SO | MRR | 0.1722 | 0.1617 | **0.1670** |

---

## 5. Dense Retrieval 診断

### 5.1 候補プール拡大

| 指標 | 値 |
|------|-----|
| BM25 top-100 | 100 docs/q |
| + Dense Phase 1b | +91.9 docs/q (平均) |
| + BM25 CoT (Phase 2.5a) | +100.0 docs/q |
| + Dense CoT (Phase 2.5b) | +50.0 docs/q |
| **マージ後総候補数** | **341.9 docs/q** |

候補プールは 100 → 342 に 3.4倍拡大。

### 5.2 Gold 文書発見 — チャネル別

| チャネル | 新規 Gold 総数 | 発見クエリ数 | 発見率 |
|---------|--------------|------------|--------|
| BM25 CoT re-retrieval | 272 | 129/323 | 39.9% |
| Dense Phase 1b + 2.5b | **17** | 13/323 | 4.0% |

CoT re-retrieval が Gold 発見の 94% を担当。
Dense retrieval の marginal gold 発見は極めて低い。

### 5.3 ドメイン別 Dense Gold 発見

| Domain | Dense 新規 Gold | Dense 発見クエリ |
|--------|---------------|----------------|
| biology | 4 | 3/103 |
| economics | 9 | 7/103 |
| stackoverflow | 4 | 3/117 |

### 5.4 なぜ Dense Retrieval の効果が限定的か

1. **語彙ギャップの性質**: BRIGHT の語彙ギャップは「専門用語」レベル。
   E5-base-v2 は汎用 embedding で、ドメイン特化の推論概念をカバーしきれない。
2. **CoT の代替性**: CoT が生成する推論概念は BM25 re-retrieval で十分に活用できており、
   Dense retrieval が追加的に発見できる Gold 文書は限定的。
3. **E5 の限界**: E5-base-v2 (384次元) は中規模モデル。
   より大きなモデル (e5-large, BGE-M3) でも改善幅は限定的と推測。

---

## 6. Tier D グラフエッジ診断

### 6.1 エッジ統計

| Domain | 平均 Tier D エッジ数 | 最小 | 最大 |
|--------|-------------------|------|------|
| biology | 101.0 | — | — |
| economics | 56.0 | — | — |
| stackoverflow | 72.5 | — | — |
| **全体** | **76.3** | — | — |

### 6.2 閾値選択の影響

ペアワイズ cosine similarity 分析 (graph_top_k=50 → 1,225ペア):

| 閾値 | エッジ率 | 平均エッジ数 | 効果 |
|------|---------|------------|------|
| 0.50 | 100% | 1,225 | 完全グラフ→PageRank均一化→nDCG=0 |
| 0.70 | ~100% | ~1,200 | ほぼ完全グラフ |
| 0.80 | ~47% | ~576 | 過密 |
| **0.85** | **~12%** | **~76** | **採用** |
| 0.90 | ~2% | ~25 | 疎すぎ |

**教訓**: BM25/Dense で取得した top-50 文書は互いに高い類似度を持つため（min cosine ~0.71）、
Dense embedding similarity による文書間エッジは「似た文書をさらに接続する」だけで、
グラフトポロジーに新しい情報をもたらしにくい。

---

## 7. LLM Rerank 診断

| 指標 | 値 |
|------|-----|
| 適用率 | 320/323 (99.1%) |
| 追加コスト | ~$0.08 (無視できる) |
| 追加レイテンシ | ~500ms/q |

LLM rerank は全クエリでほぼ動作したが、nDCG 改善への直接的寄与は
candidate pool の質に依存し、pool 自体が不十分な場合は効果が限定的。

---

## 8. コストとレイテンシ

| Condition | 平均レイテンシ | LLM コスト/323q | 備考 |
|-----------|-------------|----------------|----|
| BM25 | 4,635ms | $0 | — |
| CoT Re-retrieval (Spec A) | 12,069ms | ~$6 | — |
| Adaptive (Spec B+C) | 11,331ms | ~$4 | 43% CoTスキップ |
| **Unified (Spec D)** | **18,330ms** | **~$6.1** | Dense+LLM rerank |

### ドメイン別レイテンシ

| Domain | Spec A | Spec B+C | **Spec D** | Δ vs B+C |
|--------|--------|----------|-----------|----------|
| biology | 13,210ms | 9,220ms | **15,011ms** | +63% |
| economics | 11,680ms | 8,856ms | **13,294ms** | +50% |
| SO | 11,700ms | 15,917ms | **25,684ms** | +61% |

Unified は全ドメインで Adaptive より 50-63% 遅い。
主因: Dense retrieval のインデックス検索 + embedding ロード + LLM rerank。

---

## 9. 失敗分析

### 9.1 なぜ目標 (nDCG≥0.20) に未到達か

| 要因 | 影響 | 詳細 |
|------|------|------|
| Dense Gold 発見の低調さ | 大 | 17件/323q — CoT (272件) の 6% |
| Biology の後退 | 大 | 0.2367→0.1989 (-16%) |
| 強制 tier 3 の弊害 | 中 | Biology で不要な CoT が有害 |
| Tier D の限定的効果 | 中 | Top-k 文書間の高類似度 |

### 9.2 Biology 後退の原因

Adaptive (Spec B+C) では biology の 43/103 クエリが tier 1 or 2 にルーティングされ、
CoT なしの BM25+Graph で十分な性能を発揮していた。
Unified は全クエリ tier 3 を強制したため、本来 CoT が不要なクエリにも
CoT→Graph 処理が行われ、不正確な CoT がランキングを悪化させたケースがある。

→ **ルーティング精度が Adaptive の核心的強み** であることが判明。

### 9.3 StackOverflow の改善メカニズム

SO は Adaptive で 0.1133 → Unified で 0.1362 (+20.2%)。
Dense retrieval が SO の長い技術文書（平均 1,700 tokens）のセマンティック検索に貢献し、
BM25 の語彙一致限界を部分的に補完した。

### 9.4 対策→原因の成否

| Phase | 対策 | 対象原因 | 成否 |
|-------|------|---------|------|
| 1b | Dense(query) | C) プール不足 | △ — 候補拡大したが Gold 発見少 |
| 2 | CoT 強制実行 | A) CoTスキップ | ✗ — Biology で有害 |
| 2.5b | Dense(CoT text) | C) プール不足 | △ — CoT BM25 に対する追加効果小 |
| 3 | Tier D エッジ | B) Rerank失敗 | △ — グラフ密度上昇も改善限定的 |
| 7 | LLM rerank | B) Rerank失敗 | △ — pool 品質に依存 |

---

## 10. 教訓と次のステップ

### 10.1 本実験からの教訓

1. **ルーティング精度 > パイプライン複雑化**: 全クエリに最大パイプラインを適用するより、
   適切なルーティングで必要なクエリだけに CoT を実行する方が効果的。
2. **Dense retrieval ≠ silver bullet**: 汎用 embedding (E5-base-v2) による語彙ギャップ
   橋渡しの効果は限定的。BRIGHT の推論集約型ギャップには不十分。
3. **グラフエッジの質**: Top-k 文書間の embedding 類似度は高すぎて弁別力が低い。
   Dense エッジは文書「内」ではなく文書「間」の新しい接続に使うべき。
4. **コスト対効果**: Dense retrieval + LLM rerank でレイテンシ +62% だが、
   nDCG は -0.003 と Adaptive を下回った。

### 10.2 根本的課題: ルーティングの本質

現在の β₀ ベースルーティングは entity graph の connected components 数を閾値判定するだけで、
**クエリをグラフに注入していない**。これは geDIG の本来の設計とは異なる:

- **現行 (β₀ routing)**: Graph 構築 → β₀ 計算 → 閾値判定 → CoT on/off
- **本来 (geDIG routing)**: Graph 構築 → **クエリ注入** → Δβ₀, ΔH 測定 → geDIG 値計算 → 判定

geDIG 値はクエリがグラフにどう「統合」されるかの指標であり、
これをルーティングに使用することで、クエリ特性に応じた適応的パイプライン選択が可能になる。

### 10.3 次の仕様候補

| # | 仕様 | 概要 | 期待効果 |
|---|------|------|---------|
| **E** | **geDIG Routing** | クエリをグラフに注入し、geDIG値でAG/DG判定 | ルーティング精度向上 |
| F | Domain-specific Dense | ドメイン特化 embedding (PubMedBERT等) | Dense Gold 発見率向上 |
| G | Iterative CoT | CoT→Re-retrieve→CoT の反復 | 段階的概念発見 |

### 10.4 アーキテクチャの進化

```
v12 Phase 1: Graph Rerank         (BM25 → Graph)              → +58%
v12 Phase 2: CoT+Graph            (+ CoT injection)            → +79%
v12 Phase 3: CoT Re-retrieval     (+ vocabulary gap bridging)  → +123%
v12 Phase 4: Adaptive             (+ β₀ routing + aggressive)  → +134%
v12 Phase 5: Unified              (+ Dense + TierD + LLM RR)   → +130% ★後退
v12 Phase 6: ???                   (+ geDIG routing?)           → ???
```

---

## 11. BRIGHT リーダーボード比較

| 方法 | nDCG@10 | 備考 |
|------|---------|------|
| INF-X-Retriever (現 SOTA) | 63.4 | 専用モデル, 全12ドメイン |
| BM25 + GPT-4 + Llama rerank | 30.4 | 論文ベースライン |
| BM25 + GPT-4 reasoning | 27.0 | 論文ベースライン |
| BM25 (論文, 全12ドメイン) | 14.5 | 論文ベースライン |
| 我々 Adaptive (Spec B+C, 3ドメイン) | 16.0 | gpt-4o-mini |
| **我々 Unified (Spec D, 3ドメイン)** | **15.6** | gpt-4o-mini + E5 |
| 我々 CoT Re-retrieval (Spec A, 3ドメイン) | 15.2 | gpt-4o-mini |
| 我々 BM25 (3ドメイン) | 6.8 | 3ドメインのみ |

---

## 12. 結論

Unified Hybrid Pipeline (Spec D) は、Dense retrieval、Tier D グラフエッジ、
LLM rerank を統合した包括的なアプローチだが、目標の nDCG≥0.20 に到達せず、
Adaptive (Spec B+C) をわずかに下回った。

**最大の教訓**: パイプラインの複雑化（Dense + Tier D + LLM rerank）よりも、
**ルーティング精度の向上**が性能改善の最大レバレッジである。
現行の β₀ 閾値ルーティングは geDIG の本来の設計を活かしておらず、
クエリのグラフ注入によるトポロジー変化（geDIG値）に基づくルーティングが
次の重要な実験方向である。

Dense retrieval の限定的な効果は、汎用 embedding モデルの限界を示すとともに、
CoT が生成する推論概念が BM25 re-retrieval だけで十分に活用できることを再確認した。
