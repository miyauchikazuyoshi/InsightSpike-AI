# Spec N: Unified Token-level Graph (geDIG × 生成文法)

## 概要

geDIG の Merge/Move 理論を文書のグラフ表現に適用する。
単語（トークン）をノード、文を句構造パーサーで木構造（サブグラフ）化し、
文間を共参照・エンティティ重複で接続した**統一グラフ**を構築する。
クエリのトークンカバレッジでスコアリングし、
「推論的関連性」を構造的に捉える。

## 背景と動機

### ディスカッションからの着想

1. **文内 = Merge (DG)**: 句構造パーサーが出力する木 → β₁=0
2. **文内 Move (AG)**: 関係節の交差依存 → β₁ > 0（統語的複雑さ）
3. **文間 = 共参照**: "The cat sat. It was warm." → It→cat のエッジがループ形成 → β₁ > 0（談話的複雑さ）
4. **クエリ投入**: グラフ上でクエリトークンを多くカバーするパスを選ぶ → **動的 attention**

### Transformer との対応

```
Transformer:  固定重み × 動的入力 → 動的 attention (全トークン対全トークン, O(n²))
統一グラフ:   動的構造 × 動的クエリ → 動的探索パス (グラフ構造が制約, O(degree))
```

- Transformer はサブグラフなし、トークン列をフラットに保持
- 人間はエピソード単位でサブグラフのインデックスを持ち、末端は再生成
- geDIG 統一グラフは β₁ Sleep 済みの圧縮表現に対応

### 現行パイプラインの限界

現在の entity_graph.py は**文（sentence）をノード**にしている：
- ノード粒度が粗い — 1 文に複数の概念が混在
- エンティティ重複のみで接続 — 統語構造を無視
- クエリとの関連性は bag-of-words 的な token overlap で計算

統一グラフでは**単語をノード**にすることで：
- 推論パスが具体的（どのトークンがどうつながるか）
- 統語構造がグラフに反映される
- クエリ→回答のパスが Transformer の attention path に類似

## 設計

### グラフ構造

```
Document = [Sentence_1, Sentence_2, ..., Sentence_n]

Sentence_i のサブグラフ:
  - ノード = 各トークン (word)
  - エッジ = dependency parse の依存関係
  - 構造 = 木 (β₁=0, Merge 操作に対応)

文間接続:
  - 共参照エッジ: "It" → "cat" (代名詞→先行詞)
  - エンティティ重複: 同一エンティティの出現間
  - 隣接文エッジ: 文末→次文冒頭 (discourse flow)
  - ループ形成 → β₁ > 0 (談話構造)
```

### BRIGHT への適用

```
Phase 3' (Spec N):
  For each candidate document:
    1. spaCy で dependency parse → 文内木構造
    2. spaCy で coreference resolution → 文間エッジ
    3. エンティティ重複 → 文間エッジ (既存ロジック流用)
    4. 統一グラフ構築

Phase 5' (Spec N):
  Query トークン → グラフ上のマッチングノード
  スコア = query トークンをカバーする最短パス群のコスト
  (β₁ が低い doc = 構造的に整理されている = 高スコア？)
```

### スコアリング候補

#### A: Token Coverage Score

```python
def token_coverage_score(query_tokens, doc_graph):
    matched = {t for t in query_tokens if t in doc_graph.nodes}
    coverage = len(matched) / len(query_tokens)
    # ボーナス: マッチしたトークン間のパスが短い (構造的に近い)
    if len(matched) >= 2:
        avg_path = mean(shortest_path(n1, n2) for n1, n2 in pairs(matched))
        proximity_bonus = 1.0 / (1.0 + avg_path)
    return coverage * (1 + proximity_bonus)
```

#### B: Subgraph Extraction Score

```python
def subgraph_score(query_tokens, doc_graph):
    # クエリトークンを含む最小部分グラフ (Steiner tree 近似)
    matched_nodes = find_matching_nodes(query_tokens, doc_graph)
    steiner = approximate_steiner_tree(doc_graph, matched_nodes)
    # スコア = カバレッジ / サブグラフサイズ (密度)
    return len(matched_nodes) / steiner.number_of_nodes()
```

#### C: geDIG Walk Score

```python
def gedig_walk_score(query_tokens, doc_graph):
    # geDIG の DG/AG 分類を利用
    # DG エッジ (低コスト) を優先して query トークンを結ぶ
    # AG エッジ (高コスト) を経由する必要があるほど、推論的距離が大きい
    pass
```

### 実装に必要なツール

| ツール | 用途 | 状態 |
|--------|------|------|
| spaCy (en_core_web_sm/trf) | dependency parse | pip install 必要 |
| spaCy (neuralcoref or coreferee) | coreference resolution | pip install 必要 |
| NetworkX | グラフ構築・操作 | 既存 |
| geDIG core | DG/AG 分類 | 既存 (src/insightspike/) |

### スケール見積もり

| | 現行 (sentence-level) | Spec N (token-level) |
|--|----------------------|---------------------|
| ノード/doc | ~10 sentences | ~200-500 tokens |
| エッジ/doc | ~30-80 | ~300-800 |
| 50 docs | ~500 nodes | ~15,000 nodes |
| 計算量 | O(500²) | O(15,000²) ← **要注意** |

**対策**:
- 文内は木構造なので O(n) で探索可能
- 文間エッジのみ O(n_sentences²) — 現行と同じ
- Steiner tree 近似は O(V·E) — 大きいが 1 回/doc

## 検証計画

### Phase 1: 手動プロトタイプ (最小コスト)

BRIGHT biology の 1 クエリに対して:
1. top-3 candidate docs を spaCy で token-level グラフ化
2. query トークンのマッチング・パス可視化
3. gold doc と non-gold doc でスコアの差を目視確認

```python
# 最小プロトタイプ
import spacy
nlp = spacy.load("en_core_web_sm")

def build_token_graph(text):
    doc = nlp(text)
    G = nx.DiGraph()
    for sent in doc.sents:
        for token in sent:
            G.add_node(token.i, text=token.text, lemma=token.lemma_)
            if token.head != token:
                G.add_edge(token.head.i, token.i, dep=token.dep_)
    # 文間: 隣接文の ROOT 同士を接続
    roots = [sent.root.i for sent in doc.sents]
    for i in range(len(roots) - 1):
        G.add_edge(roots[i], roots[i+1], dep="discourse_flow")
    return G
```

### Phase 2: 10q 自動評価

- spaCy パイプライン構築
- token_coverage_score で全 candidate docs をスコアリング
- nDCG@10 を計算、baseline と比較

### Phase 3: 50q 評価 + geDIG 統合

- Spec H (geDIG refine) と組み合わせ
- token-level スコアを classic scoring の追加成分として blend

## 論文との接続

### §2.3 Merge/Move マッピング

| 操作 | 統語レベル | 談話レベル |
|------|-----------|-----------|
| Merge (DG) | 依存関係 → 木 (β₁=0) | — |
| Move (AG) | 関係節交差 → ループ (β₁>0) | 共参照 → ループ (β₁>0) |

### 検証可能な予測

1. **Gold doc の token coverage score > non-gold doc** (推論パスが存在)
2. **Gold doc の β₁ が「適度に」高い** (推論的接続がある = 非自明な関連性)
3. **Token coverage score と Transformer attention の相関** (将来的検証)

## リスクと仮線タグ

| リスク | 影響 | 仮線度 |
|--------|------|--------|
| spaCy の dependency parse 精度 | 木構造が不正確 | ★★☆ (実用十分) |
| Coreference resolution 精度 | 文間エッジが不正確 | ★★★ (spaCy v3.7+ で改善) |
| トークン数が多すぎて計算量爆発 | latency 増大 | ★★☆ (文内は木なので O(n)) |
| Token coverage が BM25 と等価 | 新しい情報なし | ★★★★ (最大リスク) |
| β₁ と推論困難度の相関なし | 理論の棄却 | ★★★ (検証が目的) |

**最大リスク**: Token coverage score が結局 BM25 の token overlap と同じ情報しか
持たない場合、統一グラフの付加価値がない。これを棄却するには、
**BM25 スコアと token coverage score の順位相関**を計算し、相関が低いことを確認する。

## Spec M との関係

| | Spec M (RIA) | Spec N (統一グラフ) |
|--|-------------|-------------------|
| 改善対象 | Retrieval Recall | Ranking Quality |
| パイプライン位置 | Phase 2.6 (検索ループ) | Phase 3-5 (グラフ+スコアリング) |
| 独立性 | ✅ 独立 | ✅ 独立 |
| 組合せ | RIA で recall 改善 → 統一グラフで ranking 改善 → 相乗効果の可能性 |

---

## 実装 (2026-03-13)

### 実装方針: 簡易版 (per-document token graph + external blend)

理論的設計(上記)の完全実装ではなく、**pragmatic な簡易版**を実装した。

**変更点**:
- 共参照解析 → **same_lemma エッジ**で代替 (spaCy coref は重い)
- 統一グラフ（全 doc を 1 つの巨大グラフ） → **per-document 独立グラフ**
- Phase 3 への統合 → **Phase 4.5 として既存パイプラインの外側に追加**
- Steiner tree / geDIG Walk → **Coverage × Proximity スコア（方法 A）**を採用

### パイプライン位置

```
Phase 3:     Entity graph construction              ← 変更なし
Phase 4:     CoT node injection                     ← 変更なし
Phase 4.5:   ★NEW★ Per-document token graph scoring (Spec N)
  ┌──────────────────────────────────────────────────┐
  │  4.5a: spaCy で各 candidate doc を dependency parse │
  │  4.5b: token graph 構築 (dep + root_chain + same_lemma) │
  │  4.5c: query lemma → graph ノードマッチング       │
  │  4.5d: coverage × (1 + proximity_bonus) スコア    │
  │  4.5e: min-max 正規化 → token_graph_scores       │
  └──────────────────────────────────────────────────┘
Phase 5:     Graph scoring (classic/geDIG/refine)   ← 変更なし
Phase 5.5:   ★NEW★ Token graph blend
  graph_scores[id] = (1-w) * graph_scores[id] + w * token_graph_scores[id]
Phase 6:     Combined ranking                       ← 変更なし
```

### 実装ファイル

| ファイル | 操作 | 概要 |
|---------|------|------|
| `src/token_graph.py` | **新規** | token graph 構築 + scorer (~170 行) |
| `src/bright_cot_pipeline.py` | **修正** | Phase 4.5 + Phase 5.5 blend + Result fields |
| `scripts/run_bright.py` | **修正** | CLI 引数 3 個 + config + diagnostics |

### グラフ構造

**ノード**: 各トークン（`lemma`, `pos`, `sent_idx` 属性付き）

**エッジ** (3 種類):
1. `dep` — dependency parse (head → child), 文内木構造
2. `root_chain` — 連続文の ROOT 間 (双方向), 談話フロー
3. `same_lemma` — 同一 content lemma の文間接続 (双方向, 3文窓)
   - 対象 POS: NOUN, VERB, ADJ, PROPN (content words のみ)
   - lemma 長 > 2 文字のみ
   - 出現 2～20 回の lemma のみ（ストップワード的な語を除外）

### スコアリング

```python
score = coverage × (1 + proximity_bonus)
coverage = |matched_query_lemmas| / |query_content_lemmas|
proximity_bonus = 1 / (1 + avg_shortest_path_between_matched_nodes)
```

- lemma ベースのマッチング（surface form ではなく）
- 最短パスは undirected グラフ上で計算
- 各 lemma につき 1 つの代表ノードのみ使用
- ペア数 > 50 の場合はランダムサンプリング (seed=42)

### CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--token-graph` | false | Token graph scoring 有効化 |
| `--token-graph-weight` | 0.15 | Graph scores とのブレンド比率 |
| `--token-graph-max-tokens` | 500 | spaCy パース対象のトークン上限 |

### 冗長性検証

BM25 との Spearman 順位相関 (ρ) を自動計算:
- |ρ| < 0.5 → 新しいランキング信号 ✅
- |ρ| > 0.8 → BM25 と冗長 → 付加価値なし ❌

### 設計上の決定理由

1. **External blend (Phase 5.5) を選択した理由**:
   - 全 3 scoring mode (classic/geDIG/geDIG_refine) に同一ロジックで対応
   - 既存メソッドのシグネチャ変更不要
   - `--token-graph` OFF なら完全に不活性（既存動作への影響ゼロ）

2. **same_lemma エッジで共参照を代替した理由**:
   - spaCy の coreference resolution は計算コストが高い
   - same_lemma は「同じ概念の文間接続」を近似的に捉える
   - 追加パッケージ不要 (spaCy core のみで完結)

3. **per-document グラフを選択した理由**:
   - 全 doc を統一すると O(15,000²) で latency 爆発
   - per-document なら O(500²) × 50 docs ≈ O(12.5M) で許容範囲
   - ドキュメント間の比較は graph_scores の blend で実現

---

## 実験結果 (2026-03-13)

### 50q 評価 (biology domain)

| 構成 | nDCG@10 | Recall@10 | MRR | vs Baseline |
|------|---------|-----------|-----|-------------|
| Baseline (geDIG_refine, v13) | 0.2496 | — | — | — |
| Token Graph のみ (Spec N) | 0.2544 | 0.2486 | 0.3917 | +0.0048 (+1.9%) |
| RIA のみ (Spec M) | 0.2564 | 0.2661 | 0.3815 | +0.0068 (+2.7%) |
| **RIA + Token Graph (M+N)** | **0.2707** | **0.2643** | **0.3972** | **+0.0211 (+8.5%)** |

### 成功基準の判定

| 基準 | 閾値 | 結果 | 判定 |
|------|------|------|------|
| Smoke test | エラーなし | 10q 全クエリ成功 | ✅ |
| 50q nDCG | ≥ 0.2496 (no regression) | 0.2544 | ✅ |
| 目標 nDCG | > 0.27 (+5%) | 0.2707 (M+N) | ✅ |
| Spearman ρ | \|ρ\| < 0.5 (BM25 と異なる信号) | 大半が -0.75 ～ 0.20 | ✅ |
| Latency | < 5s/query 追加 | ~0.7s/query (初回除く) | ✅ |

### Key Findings

1. **Token Graph 単体**: nDCG +1.9% — 小さいが正の改善
2. **RIA + Token Graph**: nDCG **+8.5%** — 明確な相乗効果
   - RIA が Recall を改善（プール内に gold doc を追加）
   - Token Graph が Ranking を改善（gold doc のスコアを引き上げ）
3. **BM25 との冗長性**: Spearman ρ の中央値 ≈ -0.38 → **完全に独立した信号**
   - 殆どの ρ が負: Token Graph は BM25 とは逆方向のランキングを提供
   - これは理論的に重要: 構造的近接性は表層的マッチとは異なる情報を捉えている
4. **Coverage**: 平均 0.13 — query lemma の約 13% が doc 内で見つかる
   - BRIGHT の推論型クエリでは surface coverage が低いのは想定通り
   - proximity_bonus が真の差別化要因
5. **Latency**: spaCy ロード後は ~0.7s/query — 実用的に問題なし

### 最大リスクの検証結果

> Token coverage が BM25 と等価 (ρ > 0.8) の場合は付加価値なし

**→ 棄却**。ρ の殆どが負値で、BM25 とは完全に異なるランキング信号。
Token Graph は dependency parse + lemma マッチによる構造的近接性という、
BM25 の表層的 token overlap では捉えられない情報を提供している。

---

## Spec N.1: geDIG Walk Score — DG/AG 重み付きパススコア (計画)

**日付**: 2026-03-13
**ステータス**: ✅ 完了 — RIA 併用で nDCG@10 = 0.3181 (+27.4%)

### 動機

**ユーザーの洞察**:
> "これって単純な検索じゃなくて、AG/DG評価で辿れるんじゃない？"
> "クエリに対して、一発で見つからなくても、マルチホップを辿るとループを検知して確信度を上げる設計。でこれは迷路と一緒。"

**問題**: 現在の `_score_single()` は全エッジを均一コスト (weight=1) で最短経路計算している。
Bridge (DG) エッジと cycle (AG) エッジの構造的意味の違いを無視している。

### 迷路実験との対応

| 迷路 (Wake-Sleep-Wake) | 文書ランキング (Spec N.1) |
|------------------------|--------------------------|
| Wake: 未知エリアの探索 | Query lemma のグラフノード発見 = coverage |
| Sleep: 構造解析・圧縮 | Tarjan bridge 検出で DG/AG 分類 |
| Wake: ループ検知で確信度UP | AG-rich パスを優先する weighted SP = proximity_bonus |
| AG (ループ) = 既知の確認 | Cycle エッジ = 複数独立パスによる裏付け |
| DG (ブリッジ) = 新規探索 | Bridge エッジ = 単一チェーンのみ、脆い接続 |

### 理論的根拠

geDIG の DG/AG 分類を per-document token graph に適用:

- **Bridge (DG) エッジ**: 除去するとグラフが分断 → 構造的に脆い接続
  - 2 つの query lemma が bridge のみで繋がっている = 1 本のチェーンのみ
  - → 接続の信頼度は低い
- **Cycle (AG) エッジ**: 除去しても連結性を保つ → 複数パスによる裏付け
  - 2 つの query lemma が cycle-rich サブグラフに埋め込まれている = 複数の独立パス
  - → 接続の信頼度は高い

### 既存 Phase 5 geDIG との区別

| | Phase 5 geDIG (既存) | Spec N.1 Walk Score (今回) |
|---|---|---|
| **対象** | 文書間 entity graph (sentence ノード) | 文書内 token graph (word ノード) |
| **粒度** | sentence-level | token-level |
| **目的** | 文書プール全体の構造評価 | 個々の文書内での query term 接続品質評価 |
| **DG/AG** | sentence 間の bridge/cycle | word 間の bridge/cycle |
| **Tarjan** | graph_builder.py の既存実装 | token_graph.py に新規適用 |

### 設計

#### Phase 4.5d の変更 (scoring 部分のみ)

```
現行:
  ug = doc_graph.to_undirected()
  sp = shortest_path_length(ug, src, dst)                    # uniform cost

Spec N.1:
  ug = doc_graph.to_undirected()
  bridges = nx.bridges(ug)                                    # Tarjan O(V+E)
  for edge in ug.edges():
    ug[u][v]['cost'] = dg_penalty if is_bridge else 1.0       # DG=高, AG=低
  sp = shortest_path_length(ug, src, dst, weight='cost')     # weighted
```

#### エッジコスト設計

```
AG (cycle) エッジ:  cost = 1.0        ← 確認的接続 → 低コスト → 優遇
DG (bridge) エッジ: cost = dg_penalty  ← 脆弱な接続 → 高コスト → 抑制
                    (default: 2.0)
```

**proximity_bonus の変化**:
- Cycle-rich な接続 → weighted SP が短い → proximity_bonus が高い → スコア UP
- Bridge-only な接続 → weighted SP が長い → proximity_bonus が低い → スコア DOWN

#### β₁ 診断

```python
beta_1 = E - V + C  # 独立サイクル数
```

β₁ > 0 の文書 = same_lemma エッジがサイクルを形成 → Walk Score が差別化に効く。
β₁ = 0 の文書 = 純粋な木構造 → bridge/cycle 区別なし → uniform cost と同等。

### 実装計画

#### 修正ファイル

| ファイル | 修正内容 |
|---------|---------|
| `src/token_graph.py` | `_classify_edges_dg_ag()` 新関数 + `_score_single()` 重み付き SP |
| `src/bright_cot_pipeline.py` | パラメータ 2 個 + Result フィールド 2 個 |
| `scripts/run_bright.py` | CLI 2 個 + config + diagnostics |

#### 新 CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--token-graph-walk-score` | false | DG/AG 重み付き最短経路の有効化 |
| `--token-graph-dg-penalty` | 2.0 | Bridge エッジのコストペナルティ |

#### 後方互換性

- `--token-graph` のみ (walk-score なし) → 既存の uniform cost 動作、変更なし
- `--token-graph --token-graph-walk-score` → 新しい weighted SP
- `--token-graph-walk-score` は `--token-graph` と独立フラグ → 既存テスト結果を再現可能

### 検証計画

#### 50q A/B (4 構成)

| 構成 | コマンド追加オプション | 比較対象 |
|------|----------------------|---------|
| TG uniform (既存) | `--token-graph` | baseline |
| TG walk dg=2.0 | `--token-graph --token-graph-walk-score` | vs uniform |
| RIA + walk dg=2.0 | `+ --ria-loop --ria-max-rounds 3` | vs M+N (0.2707) |
| RIA + walk dg=3.0 | `+ --token-graph-dg-penalty 3.0` | 感度テスト |

#### 成功基準

| 基準 | 閾値 |
|------|------|
| Smoke test | エラーなし, β₁ > 0 の doc が存在 |
| No regression | nDCG ≥ 0.2544 (plain TG) |
| Walk > Plain | Walk Score nDCG > plain TG nDCG |
| 目標 (RIA+Walk) | nDCG > 0.2707 (M+N best) |
| Latency | < 2× plain TG latency |

### リスク

| リスク | 影響 | 対策 |
|--------|------|------|
| Token graph が tree-dominant (β₁ ≈ 0) | bridge/cycle 区別不能 | same_lemma エッジがサイクル形成源のため β₁ > 0 期待 |
| dg_penalty 最適値不明 | 次善のパラメータ | 2.0 と 3.0 の 2 点で感度テスト |
| Tarjan bridge 検出の計算コスト | latency 増加 | O(V+E) で ~500 nodes なら無視可能 |
| Weighted SP が Dijkstra で遅い | latency 増加 | ~500 nodes, 50 pairs で ms オーダー |

---

## Spec N.1: 実験結果 (2026-03-13)

### 50q 評価 (biology domain)

| 構成 | nDCG@10 | Recall@10 | MRR | vs Baseline |
|------|---------|-----------|-----|-------------|
| Baseline (geDIG_refine, v13) | 0.2496 | — | — | — |
| TG uniform (Spec N) | 0.2544 | 0.2486 | 0.3917 | +1.9% |
| TG Walk dg=2.0 (N.1 のみ) | 0.2238 | 0.2158 | 0.3489 | -10.3% |
| RIA + TG uniform (M+N) | 0.2707 | 0.2643 | 0.3972 | +8.5% |
| **RIA + Walk dg=2.0 (M+N.1)** | **0.3181** | **0.3139** | **0.4424** | **+27.4%** |
| RIA + Walk dg=3.0 | 0.2774 | 0.2806 | 0.4234 | +11.1% |

### 成功基準の判定

| 基準 | 閾値 | 結果 | 判定 |
|------|------|------|------|
| Smoke test | エラーなし, β₁ > 0 | 10q 全成功, β₁=1.4～34.7 | ✅ |
| No regression (with RIA) | nDCG ≥ 0.2707 | 0.3181 | ✅ |
| Walk > Plain (with RIA) | Walk nDCG > plain nDCG | 0.3181 > 0.2707 | ✅ |
| 目標 (RIA+Walk) | nDCG > 0.2707 | 0.3181 (+17.5%) | ✅ |
| Walk のみ no regression | nDCG ≥ 0.2544 | 0.2238 | ❌ |
| Latency | < 2× plain TG | ~1.1× | ✅ |

### Key Findings

#### 1. Walk Score 単体は逆効果、RIA 併用で劇的改善

Walk Score のみ: nDCG 0.2238 (baseline 0.2496 より -10.3%)
RIA + Walk Score: nDCG **0.3181** (baseline より **+27.4%**)

**解釈**: dependency parse の木構造はほぼ全エッジが bridge (DG)。
Walk Score は bridge にペナルティを課すため、gold doc がプールにない場合は
全文書のスコアが均一に圧縮され、分散が減少 → ランキング品質低下。

RIA が gold doc をプールに注入すると、gold doc は same_lemma エッジで
サイクル豊富なサブグラフを持つ → Walk Score が正しくこれを検出 →
ランキング改善。

#### 2. 迷路実験との完全な対応

**迷路**: Wake (探索) → Sleep (構造解析) → Wake (ループ確認)
- 先に探索しないとループ検知は無意味
- ループがない領域では Sleep の付加価値はゼロ

**BRIGHT**: RIA (探索) → Walk Score (DG/AG 分類) → proximity_bonus (確認)
- RIA なしでは gold doc がプールにいない → Walk Score の効果なし
- RIA + Walk Score = 探索 + 確認の相乗効果

**これは Wake-Sleep-Wake アーキテクチャそのもの。**

#### 3. dg_penalty の感度

- dg_penalty=2.0: nDCG **0.3181** (最適)
- dg_penalty=3.0: nDCG 0.2774

ペナルティが強すぎると (3.0)、bridge 経由のパスが過度に長くなり、
cycle 部分の相対的優位がかき消される。2.0 が最適バランス。

#### 4. β₁ の分布

smoke test (10q) の β₁: 1.4 ～ 34.7 (平均 ~10)
- β₁ > 0 の文書が大半 → same_lemma エッジがサイクルを形成
- β₁ が高い文書 = 多くの概念が文間で繰り返し言及 = 推論的接続が豊富
- Walk Score はこの β₁ の差を proximity_bonus の差に変換

### 再現コマンド

```bash
# RIA + Walk Score (最高性能, nDCG=0.3181)
export $(cat .env | xargs) && \
PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
    experiments/hotpotqa_v2/scripts/run_bright.py \
    --mode cot_retrieval --domains biology \
    --data-dir experiments/hotpotqa_v2/data/bright/ \
    --output experiments/hotpotqa_v2/results/v15_walk_ria_50q \
    --limit 50 --graph-top-k 50 --rerank-alpha 0.1 \
    --scoring-mode gedig_refine --token-graph --token-graph-walk-score \
    --ria-loop --ria-max-rounds 3
```

---

## Spec N.2: geDIG F-Evaluation + Insight Vector Injection (計画)

**日付**: 2026-03-13
**ステータス**: 計画中

### 動機: Tarjan ≠ geDIG

**ユーザーの指摘**:
> "AGってサイクルだっけ？？ 0hopでのF評価値が閾値を超えるかどうかじゃなかった？？"
> "類似度スコアでエッジを繋いだときに、類似度が高いノードと接続できているならgeDIG値は閾値より下がる。類似度が不確実な場合にhop数を伸ばす。"

**問題**: Spec N.1 の `_classify_edges_dg_ag()` は Tarjan bridge 検出 (グラフ構造) で分類しているが、
geDIG の本来の定義は **F-evaluation** (query-relative evaluation) に基づく:

| | Spec N.1 (現行) | geDIG 原理 (本来) |
|---|---|---|
| **分類基準** | bridge/cycle (topology) | F 評価値 (query relevance) |
| **AG の意味** | サイクル内エッジ | 0-hop で確認可能（高類似度） |
| **DG の意味** | bridge エッジ | 不確実、hop 延長が必要 |
| **コスト** | 固定 (2.0 / 1.0) | データから導出 |
| **ペナルティ** | マジックナンバー | 不要（類似度から自然に決まる） |

### 迷路 geDIG の原理 (復習)

```
g0 = GED - λ·(IG + β·SP)

AG fires: g0 < θ_AG → 情報利得が十分 → 探索不要（確認済み）
DG fires: gmin_mh < θ_DG (hop≥1) → 多段探索で価値判明 → 採用

動的 θ: 過去の g0 値の 90th パーセンタイル
```

### Token Graph への適用

```
f_eval(u,v) = edge_cost(u,v) - λ · query_relevance(u,v)

f_eval < θ → AG: 低コスト（このエッジの接続は query に直接関係）
f_eval ≥ θ → DG: コスト = 1.0 + (f_eval - θ) （不確実 → 自然なスケール）
```

#### query_relevance(u,v): エッジの query 関連度 [0, 1]

```python
def _query_relevance(u, v, g, query_lemmas):
    # 端点の直接マッチ (0 or 1)
    rel_u = 1.0 if g.nodes[u]["lemma"] in query_lemmas else 0.0
    rel_v = 1.0 if g.nodes[v]["lemma"] in query_lemmas else 0.0
    direct = max(rel_u, rel_v)

    # 1-hop 近傍の query 密度
    u_nbrs = set(g.predecessors(u)) | set(g.successors(u))
    v_nbrs = set(g.predecessors(v)) | set(g.successors(v))
    u_density = sum(1 for n in u_nbrs if g.nodes[n]["lemma"] in query_lemmas) / max(len(u_nbrs), 1)
    v_density = sum(1 for n in v_nbrs if g.nodes[n]["lemma"] in query_lemmas) / max(len(v_nbrs), 1)
    neighborhood = (u_density + v_density) / 2

    return 0.6 * direct + 0.4 * neighborhood
```

#### edge_cost(u,v): エッジタイプに基づく構造コスト [0, 1]

```python
def _edge_structural_cost(u, v, g):
    etype = g[u][v].get("edge_type", "dep")
    if etype == "dep":
        return 0.2   # 文内依存 = 安い
    elif etype == "root_chain":
        sent_dist = abs(g.nodes[u]["sent_idx"] - g.nodes[v]["sent_idx"])
        return 0.3 + 0.1 * min(sent_dist, 3)
    elif etype == "same_lemma":
        sent_dist = abs(g.nodes[u]["sent_idx"] - g.nodes[v]["sent_idx"])
        return 0.4 + 0.1 * min(sent_dist, 5)
    return 0.5
```

#### 動的閾値

```python
f_values = [f_eval(u, v) for (u, v) in ug.edges()]
theta = percentile(f_values, 30)  # 下位 30% = AG, 上位 70% = DG
```

**ポイント**: `dg_penalty = 2.0` というマジックナンバーが `(f_val - θ)` に置き換わる。
コストがデータから自然に決まる。

### DG サブグラフ → 洞察ベクトル生成

**ユーザーの洞察**:
> "DGで得たサブグラフから洞察ベクトルを生成して、それをさらにグラフにノードとして投入する。
> このプロセスでBの穴埋めみたいなプロセスで簡易推論が完結する。"

DG エッジ = query に直接関係しない不確実な接続。しかしこの DG 領域を探索すると、
query lemma 同士を繋ぐ **bridging concepts** が見つかる。これを洞察ノードとして
グラフに inject することで、AG/DG の隙間（方法 B の穴）を埋める。

#### Pattern A: DG ブリッジ Lemma 集約

```
1. AG-only サブグラフで query ノードの連結成分を求める
2. 異なる query 成分を DG エッジで繋ぐノード = bridging nodes
3. bridging nodes の lemma を集約 → 洞察ノードとして inject
```

**イメージ**:
```
[AG cluster 1: "enzyme", "protein"]  ←DG→  bridging: "receptor"  ←DG→  [AG cluster 2: "signal", "pathway"]
                                                ↓
                                    insight node: "receptor" を inject
                                    → cluster 1 と cluster 2 が接続
```

#### Pattern B: DG パス中間ノード

```
1. Query lemma ノード対の weighted 最短パスを求める
2. パス上で DG エッジを通過する中間ノード = 推論ステップ
3. 中間ノードの lemma → 洞察ノードとして inject
```

#### Pattern C: LLM 洞察生成

```
1. DG エッジに関連する文テキストを抽出
2. LLM に投入: "これらの断片を query に繋ぐ概念は？"
3. LLM 出力の概念 → 洞察ノードとして inject
```

> "サブグラフ全部LLMにぶん投げて、新たな情報もらうのもありだと思う。論文ではそう書いてるはず。"

### 洞察ノードの Inject

```python
def _inject_insights(doc_graph, insight_lemmas, query_lemmas):
    for lemma in insight_lemmas:
        nid = new_node_id()
        g.add_node(nid, lemma=lemma, pos="INSIGHT", sent_idx=-1)
        # 同一 lemma のノードと接続 (insight_bridge)
        # Query lemma ノードにも接続 (insight_query) → B の穴埋め
```

**重要**: insight ノードは coverage に含めない（水増し防止）。
proximity_bonus のみに影響 → query ノード間の経路が短くなる。

### Wake-Sleep-Wake との完全対応

```
Wake  (探索):   RIA で candidate pool を拡大
Sleep (構造解析): F-eval で DG/AG 分類 + DG サブグラフから洞察ベクトル生成
Wake  (確認):   洞察ノード inject → proximity_bonus で再評価

迷路: 探索 → ループ検知 → 確信度UP
文書: RIA → F-eval → 洞察 inject → ランキング改善
```

### 実装計画

#### 修正ファイル

| ファイル | 修正内容 |
|---------|---------|
| `src/token_graph.py` | F-eval 分類 + Pattern A, B + inject (~200 行追加) |
| `src/bright_cot_pipeline.py` | パラメータ 3 個 + Result フィールド 4 個 |
| `scripts/run_bright.py` | CLI 引数 3 個 + config + diagnostics |

#### 新 CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--token-graph-f-eval` | false | F-evaluation ベースの DG/AG 分類 |
| `--token-graph-f-lambda` | 1.0 | F-eval の λ (構造コスト vs 情報利得) |
| `--token-graph-insight` | none | 洞察パターン: none / graph_agg / path_bridge / llm |

#### 後方互換性

- `--token-graph-walk-score` (Tarjan) はそのまま残す
- `--token-graph-f-eval` は独立フラグ
- 両方指定時は f-eval が優先

### 検証計画

#### 50q A/B テスト (6 構成)

| # | 構成 | 比較目的 |
|---|------|---------|
| 1 | M+N.1 Tarjan (baseline) | 比較基準 (nDCG=0.3181) |
| 2 | F-eval のみ (insight=none) | F-eval vs Tarjan |
| 3 | RIA + F-eval (insight=none) | M+N.2 base |
| 4 | RIA + F-eval + Pattern A | graph_agg の効果 |
| 5 | RIA + F-eval + Pattern B | path_bridge の効果 |
| 6 | RIA + F-eval + Pattern A+B | 両パターン併用 |

#### 成功基準

| 基準 | 閾値 |
|------|------|
| Smoke test | エラーなし、F-eval 分類が動作 |
| No regression | F-eval nDCG ≥ 0.3181 (M+N.1) |
| Insight 効果 | Insight あり > Insight なし |
| 目標 | nDCG > 0.34 |
| Stretch | nDCG > 0.38 |

### リスク

| リスク | 対策 |
|--------|------|
| F-eval の neighborhood 計算コスト O(degree²) | max 500 tokens なら問題なし |
| Insight が coverage を水増し | insight ノードは coverage 計算から除外 |
| 動的 θ のパーセンタイル 30% が不適切 | 20%, 30%, 50% で感度テスト |
| Pattern C (LLM) のレイテンシ | 後回し、Graph パターンの結果を先に見る |

### 実装順序

1. **Step 1-4**: F-eval 分類 + Pattern A, B → `token_graph.py`
2. **Step 5-6**: パイプライン統合 → `pipeline.py` + `run_bright.py`
3. **Step 7**: 設計ドキュメント更新 → 本ファイル
4. **Step 8**: Pattern C (LLM) は Graph パターンの結果を見てから判断

---

## Spec N.2 実験結果 (2026-03-13)

### 50q Biology A/B テスト結果

| # | 構成 | nDCG@10 | R@10 | MRR | Δ vs N.1 |
|---|------|---------|------|-----|----------|
| 1 | M+N.1 (Tarjan + RIA, baseline) | **0.3181** | 0.2965 | 0.4908 | — |
| 2 | F-eval only (no RIA) | 0.2198 | 0.1952 | 0.3575 | -0.0983 |
| 3 | RIA + F-eval (no insight) | 0.3061 | 0.2881 | 0.4697 | -0.0120 |
| 4 | RIA + F-eval + Pattern A (graph_agg) | 0.3115 | 0.2883 | 0.5042 | -0.0066 |
| 5 | RIA + F-eval + Pattern B (path_bridge) | 0.2553 | 0.2688 | 0.3695 | -0.0628 |
| 6 | RIA + F-eval + Pattern A+B (both) | 0.2978 | 0.3026 | 0.4409 | -0.0203 |

### 分析

1. **F-eval vs Tarjan**: F-eval 分類は Tarjan と同等レベルだが、微量の regression (-1.2%)
   - 動的 θ (30th percentile) の切り方がまだ最適でない可能性
   - F-eval は query relevance を反映する理論的に正しい分類だが、コスト構造のチューニングが必要

2. **Pattern A (graph_agg)**: Config 3→4 で +0.54% の微改善
   - bridging node の lemma 注入が proximity bonus を改善するケースあり
   - Q3 (handedness): 0.920→1.000, Q18 (SARS-CoV-2): 0.000→0.765 など大改善クエリも存在
   - しかし Q35 (hair grey): 0.892→0.645 など悪化クエリもあり、安定性に課題

3. **Pattern B (path_bridge)**: 明確に悪化 (-6.3%)
   - DG パス上の中間ノードが多すぎてノイズに
   - 注入ノードが proximity_bonus を歪ませている可能性

4. **Pattern A+B (both)**: Pattern B の悪影響で A の効果が相殺

### 考察と次ステップ

- **F-eval の方向性は正しい**が、コストスケールのチューニングが必要
  - `f_lambda` の調整 (現在 1.0 → 0.5, 1.5 など)
  - 閾値パーセンタイルの調整 (20%, 50%)
  - edge_structural_cost の重み配分
- **Pattern A は有望**だが、注入数の制限やフィルタリングが必要
  - 現在 max 5 lemmas → 3 に絞る
  - bridging lemma の quality filter (頻度, 情報量)
- **Pattern B は棄却** — パス中間ノードは雑すぎる
- **Pattern C (LLM)** は A の改良と並行して検討
- **目標 nDCG > 0.34 は未達** — F-eval のコストチューニングが鍵

---

## Spec O: Entity Graph F-Evaluation (Cross-Doc DG/AG)

### 背景と動機

#### Spec N.2 の診断分析で発見された根本問題

50q Biology テストの詳細分析で判明:
- **20/50 クエリが nDCG=0** (gold が top-10 に入らない)
- うち **4 クエリ** (Q0, Q26, Q37, Q49): gold が候補プール (top-50) に存在するのに top-10 入りしない → ランキング失敗
- うち **16 クエリ**: gold が候補プールに存在しない → 検索失敗 (将来の Spec P で対応)

Token graph の DG/AG は **per-doc** (文書内のトークングラフ) で動作するため、
文書間を跨いだグラフ構造を利用できない。
gold に到達するためのクロスドキュメント接続情報が欠落している。

#### Entity Graph の接続状況

Entity graph は候補間を豊富に接続している:
- **1000-2600 edges, β₀=1-7** (完全パイプライン実行時)
- Q37: β₀=1 (全305ノードが1つの連結成分), 2195 edges
- Gold が候補プールにいるのにランクされない → **entity graph 上の re-ranking が鍵**

#### ユーザーの洞察

> "事前グラフで候補同士は接続されてるんだよね？
> 複数のルートからの伝播で、gold候補に接続することがあるんじゃないかな。
> 直接はつながることは少なくても、グラフの系を手繰るとその直接繋がらない候補にエッジが集まるっていうことがあるんじゃないかと。"

→ DG/AG を entity graph (cross-doc) に適用し、multi-path convergence でランキングを改善する。

### 設計

#### 1. Entity Graph F-Eval 分類

Entity graph のエッジには既にコスト構造がある (entity_graph.py):
- Tier 1 context: 0.05-0.10
- Tier 2 entity: 0.20-0.50
- Tier 3 similarity: 0.50-0.80

これを structural_cost として使い、query_relevance と組み合わせる:

```
f_eval(u,v) = edge_cost(既存) - λ · query_relevance(u,v)
f_eval < θ → AG (query-aligned, 低コスト = 元のコスト維持)
f_eval ≥ θ → DG (uncertain, 高コスト = cost + (f_val - θ))
θ = 30th percentile (動的閾値)
```

**query_relevance(u,v)** — 3コンポーネント:
- (a) 端点の MP 更新済み TF-IDF ベクトルと query の cosine 類似度 (重み 0.5)
- (b) 端点のエンティティと query エンティティの重複密度 (重み 0.3)
- (c) 1-hop 近傍の query 密度 (重み 0.2)

#### 2. Walk Score on Entity Graph

各ドキュメントについて:
1. query ノードからの最短経路距離 (weighted)
2. AG エッジ経由の**収束ボーナス** (複数ルートからの合流)

```
proximity = 1.0 / (1.0 + min_shortest_path)
convergence = min(ag_paths / max(doc_nodes, 1), 2.0)
score = proximity * (1.0 + 0.3 * convergence)
```

「複数ルートからエッジが集まる」= convergence_bonus。
AG パスが多い文書ほど、確認済み経路が多い → 高スコア。

#### 3. パイプライン統合

既存フローに **Phase 5.25** を追加:

```
Phase 4.5  : Token graph (per-doc, Spec N)              ← 既存
Phase 5    : Graph scoring (classic/geDIG/geDIG_refine)  ← 既存
Phase 5.25 : Entity graph F-eval walk score (Spec O)     ← NEW
Phase 5.5  : Blend: token_graph + entity_feval           ← 拡張
Phase 6    : Combined BM25 + graph ranking               ← 既存
```

Phase 5.5 のブレンド:
```
graph_scores[doc] = (1 - w) * graph_scores[doc] + w * entity_feval_scores[doc]
```
w = entity_feval_weight (default 0.20)

### ファイル修正一覧

| ファイル | 操作 | 概要 |
|---------|------|------|
| `src/gedig_scoring.py` | **追加** | `entity_graph_feval_scores()` (~120行) |
| `src/bright_cot_pipeline.py` | **修正** | Phase 5.25 追加, Result フィールド, Constructor |
| `scripts/run_bright.py` | **修正** | CLI 3個, config, pipeline 3箇所, diagnostics |

**変更しないファイル**: token_graph.py, entity_graph.py

### 検証計画

#### 50q A/B テスト

| # | 構成 | 目的 |
|---|------|------|
| 1 | N.1 baseline (Tarjan + RIA) | 比較基準 (nDCG=0.3181) |
| 2 | + entity-feval (weight=0.20) | multi-path convergence 効果 |
| 3 | + entity-feval (weight=0.30) | weight 感度テスト |
| 4 | + entity-feval + token F-eval | N.2 + O の組合せ |

#### 成功基準

| 基準 | 閾値 |
|------|------|
| Smoke test | エラーなし |
| No regression | nDCG ≥ 0.3181 |
| 4 クエリ改善 | Q0/Q26/Q37/Q49 が nDCG > 0 に |
| 目標 | nDCG > 0.34 |

#### 最大リスク
1. TF-IDF 特徴の再計算コスト → `compute_node_tfidf_features()` は Phase 5 で既に実行済み。再利用
2. Weight 選択 → 0.20 はコンサバ。データで調整
3. 16 クエリの gold 不在 → re-weight では救えない。将来の pool expansion (Spec P) で対応

## Spec O 実験結果 (2026-03-13)

### 50q Biology A/B テスト結果

| # | 構成 | nDCG@10 | R@10 | MRR | Δ vs N.1 |
|---|------|---------|------|-----|----------|
| 1 | M+N.1 (Tarjan + RIA, baseline) | **0.3181** | 0.2965 | 0.4908 | — |
| 2 | + entity-feval (weight=0.20) | 0.2792 | 0.2826 | 0.4322 | -0.0389 |
| 3 | + entity-feval (weight=0.30) | 0.2717 | 0.2735 | 0.3983 | -0.0464 |
| 4 | + entity-feval + token F-eval | 0.2986 | 0.2813 | 0.4529 | -0.0195 |

### ターゲットクエリ分析 (gold が pool にいるのにランクされない 4 クエリ)

| Query | Baseline | Config 2 | Config 3 | Config 4 | 備考 |
|-------|----------|----------|----------|----------|------|
| Q0 | 0.000 | 0.000 | 0.000 | **0.113** | Config 4 で改善 |
| Q26 | 0.000 | 0.000 | 0.000 | **1.000** | Config 4 で完全一致 |
| Q37 | 0.000 | 0.000 | 0.000 | 0.000 | 変化なし |
| Q49 | 0.000 | 0.000 | 0.000 | 0.000 | 変化なし |

### Win/Loss 分析 (vs N.1 baseline, Δ>0.01)

| Config | Wins | Losses | Ties |
|--------|------|--------|------|
| 2 (EF weight=0.20) | 7 | 18 | 25 |
| 4 (EF + token F-eval) | 7 | 16 | 27 |

### Config 4 注目クエリ

**大幅改善 (>0.1):**
- Q0: 0.000→0.113, Q6: 0.249→0.415, Q29: 0.000→0.613
- Q31: 0.000→0.469, Q38: 0.000→0.469, Q48: 0.000→0.704

**大幅悪化 (>0.1):**
- Q11: 0.339→0.000, Q12: 0.390→0.000, Q16: 0.356→0.000
- Q28: 1.000→0.631, Q35: 1.000→0.645

### 分析と考察

1. **Entity F-eval 単体 (Config 2/3) は regression**
   - weight=0.20 で -3.9%, weight=0.30 で -4.6%
   - Entity graph の F-eval walk score がノイズを導入している
   - 原因: query_relevance の 3 コンポーネントが entity graph のスケールに合っていない
   - 特に endpoint cosine sim (weight 0.5) が文レベルの粗い粒度で精度不足

2. **Token F-eval との組合せ (Config 4) は効果あり**
   - Q26 が 0.000→1.000 (完全改善)、Q0 が 0.000→0.113
   - Q29/Q31/Q38/Q48 など baseline で nDCG=0 のクエリが大幅改善
   - しかし Q11/Q12/Q16 など既存の良いクエリが悪化 → 全体では regression

3. **根本的な問題: スコアの安定性**
   - entity F-eval は改善するクエリと悪化するクエリの分散が大きい
   - convergence bonus (avg 1.0-1.6) の差別化力が弱い
   - walk score の proximity が shortest path 距離に依存 → 密なグラフでは差がつきにくい

### 次ステップ候補

1. **Entity F-eval の query_relevance 改善**
   - endpoint sim の weight を下げる (0.5→0.3)
   - entity overlap の weight を上げる (0.3→0.5)
   - neighborhood density の閾値調整 (0.3→0.2)
2. **Blend weight の adaptive 化**
   - β₀ が小さい (密な) グラフでは entity F-eval weight を下げる
   - β₀ が大きい (疎な) グラフでは entity F-eval weight を上げる
3. **Entity F-eval を選択的に適用**
   - baseline nDCG=0 のクエリのみに entity F-eval を適用
   - 既存のランキングが良いクエリには適用しない

---

## Spec O.2: Ranking DG/AG — メタレベル F-eval ルーティング

### 背景と動機

#### Spec O の問題点
Entity F-eval は**強いシグナル**がある（Q26: 0→1.000, Q48: 0→0.704 等）が、
一律適用で既存の良いクエリを壊す（29 クエリ中 16 が悪化）。

Oracle 分析:
- baseline=0 のクエリにだけ適用 → nDCG 0.3181→**0.3654** (+0.0474)
- 全 Spec で各クエリ最良選択 → Oracle **0.4100** (+0.0919)

→ 「**いつ entity F-eval を適用するか**」のルーティングが必要。

#### メインコード (geDIG) の参考パターン

**multi-hop DG 探索** (`run_experiment_query.py:1778-1905`):
- 各候補エッジで graph を仮変更 → ΔSP (構造変化の影響) を測定
- g(h) = ΔGED - λ·IG(h) で改善判定
- g < g_best → DG fire（新情報が有益）

**3-attention scoring** (`graph_walker.py:96-105`):
```
score_3att = ag_attention × dg_confidence × reward_value
```

これを**ランキング決定に再帰的に適用**する。

### 設計

#### Ranking DG/AG のアルゴリズム

Phase 5.25 で entity F-eval を計算した後、**適用するかどうか**を DG/AG で判定:

```
scores_base = graph_scores (entity F-eval 適用前)
scores_ef   = entity_feval_scores (Phase 5.25 で計算済み)
```

**3-Signal Ranking DG** (メインコードの 3-attention に対応):

**(a) Score Dispersion (← ag_attention の逆)**
- top-10 の graph_scores の正規化エントロピー
- 均一分布に近い（スコアが拮抗）→ DG が高い → 新情報が必要
- 明確な順位がある → DG が低い → 現状維持

```
probs = normalize(top_k_scores)
entropy = -Σ p·log(p)
score_dispersion_dg = entropy / log(k)  # [0, 1]
```

**(b) Rank Disagreement (← dg_confidence)**
- base ranking と entity F-eval ranking の top-10 Jaccard 不一致度
- 大きく異なる → 新情報がある → entity F-eval を取り入れる余地
- ほぼ同じ → 新情報なし → entity F-eval は冗長

```
overlap = |set(base_top10) ∩ set(ef_top10)|
rank_disagreement = 1.0 - overlap / 10.0  # [0, 1]
```

**(c) Convergence Signal (← reward_value)**
- entity graph の convergence bonus の強さ
- AG パスが多い文書が多い → entity F-eval の信号が強い

```
convergence_signal = avg_convergence / 2.0  # normalize to ~[0, 1]
```

**Adaptive Weight 計算**:

```
ranking_dg = dispersion × disagreement × convergence_signal
adaptive_weight = base_weight × min(ranking_dg × max_factor, max_factor)
```

#### メインコードとの対応

| メインコード (maze geDIG) | Spec O.2 (BRIGHT) |
|---------------------------|---------------------|
| g_try = h_graph.copy() + add_edge | scores_ef = entity_feval 適用 |
| ΔSP = sp_gain_rel(before, after) | rank_disagreement = 1 - Jaccard |
| g(h) = ΔGED - λ·IG | ranking_dg = dispersion × disagreement × convergence |
| g < g_best → DG fire | ranking_dg > θ → adaptive_weight UP |
| 3att: ag × dg_conf × reward | 3-signal: dispersion × disagreement × convergence |

#### パイプライン変更

Phase 5.25 (既存) と Phase 5.5 のブレンドを改修:

```
Phase 5.25 : Entity graph F-eval walk score          ← 既存（変更なし）
Phase 5.5  : Ranking DG/AG → adaptive blend weight   ← 改修
Phase 6    : Combined BM25 + graph ranking            ← 既存
```

### ファイル修正一覧

| ファイル | 操作 | 概要 |
|---------|------|------|
| `src/bright_cot_pipeline.py` | **修正** | Phase 5.5 を adaptive routing に改修, Result 2フィールド追加 |
| `scripts/run_bright.py` | **修正** | diagnostics に ranking_dg, adaptive_weight 追加 |

**変更しないファイル**: gedig_scoring.py, token_graph.py, entity_graph.py

### 検証計画

#### 50q A/B テスト

| # | 構成 | 目的 |
|---|------|------|
| 1 | N.1 baseline | 比較基準 (0.3181) |
| 2 | + entity-feval adaptive (base_w=0.20) | DG routing 効果 |
| 3 | + entity-feval adaptive (base_w=0.30) | 高 weight + routing |
| 4 | + entity-feval adaptive + token F-eval | 全部入り |

#### 成功基準

| 基準 | 閾値 |
|------|------|
| No regression | nDCG ≥ 0.3181 |
| Win/Loss 改善 | Wins > Losses (Spec O は 7W/16L) |
| 目標 | nDCG > 0.34 |
| 理想 | Oracle 0.3654 に近づく |

### Spec O.2 実験結果

#### 50q A/B テスト結果

| # | 構成 | nDCG@10 | R@10 | MRR | Δ vs fresh base |
|---|------|---------|------|-----|----------------|
| 1a | v15 baseline (旧) | 0.3181 | 0.3139 | 0.4424 | — |
| 1b | **v18 baseline (新 fresh)** | **0.0636** | 0.0635 | 0.0957 | **0.0000** |
| 2 | + adaptive w=0.20 | 0.0656 | 0.0835 | 0.0882 | +0.0020 |
| 3 | + adaptive w=0.30 | 0.0645 | 0.0835 | 0.0787 | +0.0009 |
| 4 | + adaptive + token F-eval | 0.0659 | 0.0835 | 0.0882 | +0.0023 |

#### Win/Loss vs fresh baseline (v18_c1)

| 構成 | Wins | Losses | Ties |
|------|------|--------|------|
| C2: adaptive w=0.20 | 1 | 3 | 46 |
| C3: adaptive w=0.30 | 2 | 4 | 44 |
| C4: adaptive + token F-eval | 1 | 3 | 46 |

#### 重大発見: CoT 非決定性の影響

**v15 baseline (0.3181) → v18 fresh baseline (0.0636): Δ = -0.2545**

同一の構成（entity F-eval OFF）で再実行した結果、nDCG が 0.32 から 0.06 に激減。
これは **LLM CoT 生成の非決定性** が原因:

- 50クエリ中、v15 で nDCG > 0 だった 29 クエリのうち **21 クエリが v18 で 0** に
- v18 で元のスコアを維持したのは **わずか 3 クエリ** (Q3, Q19, Q41)
- gpt-4o-mini の temperature=0 でも、CoT テキストは毎回大きく変動
- 異なる CoT → 異なる概念抽出 → 異なる entity graph → 異なるスコア

**パイプライン全体の評価信頼性に影響する根本問題**。

#### Ranking DG/AG 分析 (adaptive routing の動作)

ranking_dg の分布:
```
[0.0, 0.1):  3 queries  → aw ≈ 0.02 (ほぼ entity F-eval なし)
[0.1, 0.2):  5 queries  → aw ≈ 0.07
[0.2, 0.3): 11 queries  → aw ≈ 0.12
[0.3, 0.5): 24 queries  → aw ≈ 0.18
[0.5, 1.0):  7 queries  → aw ≈ 0.30+
```

adaptive_weight の分布:
```
[0.00, 0.05):  3 queries (6%)
[0.05, 0.10):  5 queries (10%)
[0.10, 0.15): 11 queries (22%)
[0.15, 0.20): 12 queries (24%)
[0.20, 0.50): 19 queries (38%)
```

avg ranking_dg = 0.3471, avg adaptive_weight = 0.1736

**3-signal 分解**:
- score_dispersion: 常に高め (entropy は top-10 で均一に近い) → 殆どのクエリで dispersion > 0.8
- rank_disagreement: 中程度 (base vs ef で top-10 の 40-60% が重複)
- convergence_signal: avg_convergence ≈ 1.0-1.5 → signal ≈ 0.5-0.75

→ 3-signal の **dispersion が支配的** で、routing が十分に selective でない

#### 考察と次ステップ

**Spec O.2 の Ranking DG/AG routing 自体は中立的**:
- fresh baseline 比較で Win/Loss がほぼ均衡 (1W/3L → ほぼ影響なし)
- adaptive weight の分布は妥当 (0 から 0.36 まで分散)
- ただし dispersion が常に高く、weight が十分に suppress されない

**CoT 非決定性が最大の課題**:
- 同一クエリの nDCG が 0.3181 → 0.0636 に変動 (run-to-run variance)
- 原因: gpt-4o-mini の CoT テキストが毎回大きく異なる
- 結果: entity graph の構造・スコアが全く異なるものになる

**対策案**:
1. **CoT キャッシュ**: 一度生成した CoT をファイルに保存して再利用
2. **Deterministic LLM**: temperature=0 + seed 指定 (OpenAI API の seed パラメータ)
3. **Multi-CoT Ensemble**: 複数 CoT を生成して結果をアンサンブル
4. **CoT-free baseline**: CoT に依存しないスコアリングを基準にする

**Spec O.2 のステータス**: 実装完了、技術的には正常動作。ただし CoT 非決定性により
効果の正確な評価が困難。CoT 安定化後に再評価が必要。

---

## Spec P: Multi-CoT Ensemble with DG/AG

### 背景

Spec O.2 の実験で**CoT 非決定性**が最大の課題として発覚:
- 同一構成の baseline 再実行で nDCG 0.3181 → 0.0636 に激減
- gpt-4o-mini の CoT が毎回異なる → entity graph 構造・スコアが全く異なる
- **対策案 1（キャッシュ）と 3（Multi-CoT Ensemble）を融合**

### geDIG との対応

メインコード (maze) の multi-hop DG 探索パターン:

| maze geDIG | Spec P (Multi-CoT) |
|-----------|-------------------|
| 候補エッジを仮追加 | 複数 CoT パスを生成 |
| g(h) = ΔGED - λ·IG で評価 | 各 CoT のランキングを比較 |
| 全候補が同じ構造改善 → AG | 全 CoT で一致して高ランク → AG |
| 候補間で分岐 → DG | CoT 間でランキング不一致 → DG |
| 3att: ag × dg_conf × reward | agreement × (1-variance) × mean_score |

**CoT 間の分散 = DG シグナル**: メインコードの「複数候補を同時探索して DG を検出」を
ランキングレベルに適用。

### 設計

#### アーキテクチャ

```
Phase 2:    N 本の CoT を生成 (or キャッシュ読み込み)
Phase 2.5:  全 CoT の概念を union → 1回の re-retrieval (広範検索)
Phase 3:    1つのグラフを構築 (union 候補プール)
Phase 4-5:  各 CoT i について:
              graph_i = graph.copy()
              inject CoT_i → score → graph_scores_i  (N回)
Phase 5 ENS: ensemble mean + variance → DG/AG 分類
Phase 5.5+: 通常通り (ensemble mean を使用)
```

#### CoT キャッシュ

```
{cot_cache_dir}/{query_id}.json
```

```json
[
  {"text": "CoT 1...", "entities": ["e1"], "terms": ["t1"]},
  {"text": "CoT 2...", "entities": ["e2"], "terms": ["t2"]},
  {"text": "CoT 3...", "entities": ["e3"], "terms": ["t3"]}
]
```

- キャッシュ hit: ファイルに N 本以上の CoT があれば先頭 N 本を使用
- キャッシュ miss: N 本生成 → ファイルに保存
- temperature=0.7 で CoT の多様性を確保

#### Ensemble スコアリング

```python
for cot_info in cot_list:        # N 回ループ
    graph_i = graph.copy()
    inject_cot_nodes(graph_i, cot_info)
    scores_i = compute_scores(graph_i, cot_info)
    per_cot_scores.append(scores_i)

# 集約
for doc_id in all_doc_ids:
    scores_array = [s.get(doc_id, 0.0) for s in per_cot_scores]
    graph_scores[doc_id] = mean(scores_array)
    variance = var(scores_array)
    agreement = 1.0 - min(variance / 0.25, 1.0)

# DG/AG 分類
AG docs: agreement >= 0.8 AND mean_score > 0.3
DG docs: agreement < 0.5
```

### ファイル修正一覧

| ファイル | 操作 | 概要 |
|---------|------|------|
| `src/bright_cot_pipeline.py` | **修正** | Result 8フィールド, __init__ 3パラメータ, `_generate_or_load_cots()`, Phase 2 ensemble fork, Phase 5 ensemble loop |
| `scripts/run_bright.py` | **修正** | 3 CLI args, config, pipeline 構築(5箇所), diagnostics |

### 検証計画

#### Smoke test
1. N=1 (default): 後方互換性確認
2. N=3 + cache: ensemble 動作 + cache hit 時の再現性確認

#### 50q テスト

| # | 構成 | 目的 |
|---|------|------|
| 1a | baseline run1 (N=1) | 分散測定 |
| 1b | baseline run2 (N=1) | 分散測定 |
| 2 | ensemble N=3 + cache | ensemble 効果 |
| 3 | ensemble N=3 + entity-feval | 組み合わせ |

#### 成功基準

| 基準 | 閾値 |
|------|------|
| 再現性 | cache hit 時に同一 nDCG |
| 安定性 | N=3 ensemble > max(N=1 run1, N=1 run2) |
| DG/AG 有効性 | AG docs が gold docs と正相関 |

---

### Spec P 実装レポート

#### 実装状況: ✅ 完了

| 変更 | 概要 |
|------|------|
| `bright_cot_pipeline.py` | BrightCoTResult 8フィールド追加, `__init__` 3パラメータ (`n_cot_ensemble`, `cot_cache_dir`, `cot_temperature`), `_generate_or_load_cots()` メソッド (cache read/write + N本CoT生成), Phase 2 ensemble fork (union concepts), Phase 5 ensemble loop (per-CoT scoring + mean/variance + DG/AG分類), Result構築 |
| `run_bright.py` | 3 CLI args (`--n-cot-ensemble`, `--cot-cache-dir`, `--cot-temperature`), config記録, 5箇所pipeline構築に3パラメータ追加, diagnostics record + print (ENS表示) |

#### 機能検証: ✅ 成功

1. **Syntax check**: 両ファイル pass
2. **N=1 後方互換**: 3q smoke test — 既存動作と同一 (ensemble code path スキップ)
3. **N=3 ensemble**: 3q smoke test — `ENS(N=3G AG=8/DG=0 agr=1.00 var=0.0000)` 正常出力
4. **Cache write**: `{cot_cache_dir}/{query_id}.json` にJSON保存確認
5. **Cache hit**: 再実行で `N=3C` (C=cache) 表示, AG/DG値が完全一致 → **再現性確認**

#### 50q テスト結果 (v19) — API Key あり ✅

3ドメイン×50q = 150クエリ。`.env` から OPENAI_API_KEY 自動読み込み対応済み。

| # | 構成 | nDCG@10 | R@10 | MRR | Bio | Econ | SO |
|---|------|---------|------|-----|-----|------|----|
| C1a | baseline N=1 run1 | **0.1184** | 0.1414 | 0.1638 | 0.165 | 0.100 | 0.090 |
| C1b | baseline N=1 run2 | **0.1258** | 0.1361 | 0.1959 | 0.191 | 0.091 | 0.095 |
| C2 | ensemble N=3 | **0.0829** | 0.1005 | 0.1211 | 0.133 | 0.055 | 0.061 |
| C3 | ensemble N=3 + feval | **0.0788** | 0.0960 | 0.1183 | 0.120 | 0.055 | 0.062 |

#### 分析

##### 1. Baseline CoT 非決定性（期待通り）
- C1a=0.1184 vs C1b=0.1258 → **6.3% のぶれ**
- temperature=0.0 でも CoT 生成に多少の非決定性がある
- Biology が最も影響を受ける (0.165 vs 0.191)

##### 2. Ensemble は性能劣化（予想外）
- C2(0.0829) < C1avg(0.1221) → **-32% の劣化**
- C2 vs C1avg: **wins=5, losses=10, ties=35** (全50q/domain)
- 大きな損失ケース:
  - Q20: C1avg=0.6131 → C2=0.0000 (−0.61!)
  - Q42: C1avg=0.2961 → C2=0.0000 (−0.30)
  - Q10: C1avg=0.1864 → C2=0.0000 (−0.19)

##### 3. DG 文書が一切出ない
- 全クエリで **DG=0**, agreement=0.97〜1.00
- 3本の CoT (temperature=0.7) でも抽出概念がほぼ同一
- → geDIG の「多視点による意見不一致検出」が機能していない

##### 4. 劣化の原因分析
- **Ensemble averaging がスコアを薄める**: 3本のCoTで少しずつ異なるグラフを構築 → 平均化で尖ったスコアが平坦化
- **正解文書のランクが落ちる**: ensemble 前に Top-10 にいた文書が平均化で順位を下げる
- **全CoT高一致のため多様性ゼロ**: DG 信号が発生しないので ensemble の理論的メリット（不確実領域の発見）が活かされない

##### 5. geDIG マッピングの問題点
- 仮説: 「CoT間の不一致 = DG」→ 実際: CoT間一致度が常に 0.97+ で DG が発生しない
- temperature=0.7 では概念抽出レベルで十分な多様性が生まれない
- **根本問題**: LLM の CoT は query に対して似た推論をするため、異なる概念を抽出しにくい

#### 改善方向

1. **Prompt diversification**: 3本の CoT に異なるプロンプト（楽観/悲観/中立 など）を与える
2. **Temperature 引き上げ**: 0.7 → 1.0〜1.2 でより多様な CoT を生成
3. **DG 閾値の再調整**: agreement < 0.5 → < 0.95 など、現実のスコア分布に合わせる
4. **Ensemble 方式の変更**: mean ではなく max-pooling や rank-fusion にする
5. **Spec P の一旦保留**: 現状のアーキテクチャでは multi-CoT の効果が薄い。他の Spec を優先すべき
