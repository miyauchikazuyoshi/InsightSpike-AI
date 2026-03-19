# Phase 1 = 特殊 geDIG：一般化へのロードマップ

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。


**Version**: 0.2 (Draft)
**Date**: 2026-01-30
**Author**: Kazuyoshi Miyauchi
**Status**: Research Memo

> **v0.2 更新**: Phase 5（仮説ノード創発）を追加。理解 vs 閃き（エッジ操作 vs ノード創発）の区別を明記。RAG での閃きモードを追加。

---

## 1. 核心の認識

**Phase 1 は「特殊 geDIG」である。一般 geDIG ではない。**

```
Phase 1 = 「迷路特化 geDIG」

固定されているもの:
- 状態表現: (x, y, dx, dy, wall, visits, ...) ← 迷路専用
- similarity の重み: 手動ヒューリスティック
- 正例/負例の定義: ゴール到達 / revisit ← タスク依存

学習しているもの:
- edge_weight だけ
```

**「F の片側（IG）を、特定ドメインで、固定表現の上で学習」**している状態。

---

## 2. 一般化へのハードル

| 段階 | 何を自律化するか | ハードル | 状態 |
|------|-----------------|---------|------|
| **Phase 1** | edge_weight（評価） | 低 | ← 今ここ |
| **Phase 2** | similarity の重み（検索幾何） | 中 | 設計済 |
| **Phase 3** | 状態表現そのもの（特徴選択） | 高 | 構想中 |
| **Phase 4** | 正例/負例の定義（タスク依存性の脱却） | 最難関 | 未着手 |
| **Phase 5** | 仮説ノードの創発（閃きモード） | 最難関 | 構想中 |

### Phase 4 が「真の一般 geDIG」

正例/負例を「タスクごとに手で定義する」から「geDIG 自身が発見する」へ。

これができると、新しいドメインに対して：
1. 最初は「何が正例/負例か分からない」状態で探索
2. AG/DG の発火パターンから「これは良い/悪い」を自己発見
3. 発見した正例/負例で edge_weight と similarity を学習

### Phase 5 が「閃く geDIG」— 理解から創造へ

Phase 1-4 は全て「既存ノード間のエッジ操作」。これは「理解」のプロセス。

**閃き（Eureka）は、新しい仮説ノードの創発**。

参照: [`docs/research/insight_vs_understanding.md`](insight_vs_understanding.md)

---

## 2.1 理解 vs 閃き：エッジ操作 vs ノード創発

**Phase 1-4 と Phase 5 の本質的な違い**

```
Phase 1-4: 理解（Understanding）
  - 操作対象: エッジ（既存ノード間の接続）
  - プロセス: 分析的、段階的、予測可能
  - 結果: 知識の整理・最適化

Phase 5: 閃き（Insight/Eureka）
  - 操作対象: ノード（新しい仮説概念の創発）
  - プロセス: 創造的、非線形、予測困難
  - 結果: 新しい関連性の発見
```

### 理解（Understanding）— 現在の geDIG

```python
# 既存ノード間のエッジを操作
edge_weight[(A, B)] += α  # 正例
edge_weight[(A, C)] -= β  # 負例

# 結果: A-B の接続が強化、A-C が抑制
# 例: 「ML は AI のサブセット」という関係の発見
```

### 閃き（Insight）— 仮説ノード創発

```python
# グラフの空白地帯でメッセージパッシング
gap = find_knowledge_gap(A, B)  # A と B は遠い

# 橋渡し概念（仮説ノード）を生成
hypothesis_node = generate_bridge_hypothesis(
    source=A,
    target=B,
    neighbors=find_gap_neighbors(gap)
)

# 結果: 新しいノード H が A-H-B を繋ぐ
# 例: 「量子ビット」と「脳」を「回路」で結ぶ発見
```

### geDIG 式との対応

```
理解: F = ΔEPC - λ·ΔIG
  - ΔEPC: エッジの追加/削除コスト
  - ΔIG: 既存ノード間の予測改善

閃き: F' = ΔEPC' - λ·ΔIG'
  - ΔEPC': ノード追加 + エッジ追加のコスト（高い）
  - ΔIG': 新しい経路による予測改善（大きい可能性）

閃きが起きる条件: F' < 0 かつ ΔIG' >> ΔIG
  → 「コストは高いが、それを上回る情報利得」
```

### AG/DG の拡張

```
現在（理解）:
  AG: 「このエッジは曖昧か？」 → 探索を開く
  DG: 「このエッジは確定か？」 → コミット

拡張（閃き）:
  AG': 「この空白地帯に何かありそうか？」 → 仮説生成を開始
  DG': 「この仮説ノードは有効か？」 → ノードをコミット
```

### 実装の段階

1. **Phase 1-4（理解）を完成させる** — エッジ操作の自律化
2. **空白地帯検出を追加** — 「繋がっていないが繋がりそう」を検知
3. **メッセージパッシング実装** — 空白地帯での仮説収束
4. **仮説ノード評価** — 新規性 × 有効性 で採否判定
5. **理解と閃きの統合** — 状況に応じたモード選択

---

## 3. ドメイン別の「特殊 geDIG」定義

### 3.1 迷路（Maze）— Phase 1 実装済

| 要素 | 定義 |
|------|------|
| **状態** | `(x, y)` + 局所観測 |
| **行動** | 4方向移動 |
| **正例** | ゴール到達、新規セル到達 |
| **負例** | revisit、blocked、dead-end |
| **similarity** | 位置・方向ベクトルの重み付き距離 |
| **edge_weight** | 遷移 `(s, a, s')` への正例/負例スコア蓄積 |

### 3.2 ARC — Phase 1' 設計中

| 要素 | 定義 |
|------|------|
| **状態** | 入力グリッド + 適用済み変換列 |
| **行動** | DSL プリミティブの選択・適用 |
| **正例** | train 例での一致、簡潔な記述 |
| **負例** | near-miss（一部 train で破綻）、overfit（冗長） |
| **similarity** | タスク/変換の埋め込み距離 |
| **edge_weight** | 変換 `(state, op, result)` への成功/失敗スコア |

### 3.3 RAG（Retrieval-Augmented Generation）— 新規

| 要素 | 定義 |
|------|------|
| **状態** | クエリ + 検索結果セット + 生成コンテキスト |
| **行動** | どのドキュメントを選ぶか、どう統合するか |
| **正例** | 回答品質向上に寄与した検索結果 |
| **負例** | ハルシネーション誘発、冗長、無関係 |
| **similarity** | クエリ-ドキュメント埋め込み距離 |
| **edge_weight** | `(query_type, doc, outcome)` への評価スコア |

---

## 4. RAG への geDIG 適用：詳細設計

### 4.1 問題設定

従来の RAG:
```
query → retrieve(top-k by similarity) → generate(context) → answer
```

問題点:
- similarity が高くても「良い検索結果」とは限らない
- 過去の成功/失敗が活かされない
- ハルシネーション誘発ドキュメントが繰り返し選ばれる

### 4.2 geDIG-RAG の設計

```
query → retrieve(top-k by similarity × confidence) → generate → answer
                              ↑
                        edge_weight から計算
```

#### 状態の定義

```python
state = {
    "query": query_embedding,
    "query_type": classify(query),  # factual / reasoning / creative / ...
    "retrieved_docs": [doc1, doc2, ...],
    "context_so_far": current_context,
}
```

#### 行動の定義

```python
action = {
    "select_doc": doc_id,           # どのドキュメントを選ぶか
    "integration": "append" | "replace" | "summarize",  # どう統合するか
}
```

#### 正例/負例の定義

**正例（edge_weight を上げる）**:
- `answer_quality_improved`: 回答の正確性/有用性が向上
- `no_hallucination`: 生成された回答が検索結果と整合
- `user_accepted`: ユーザーが回答を採用（明示的フィードバック）
- `citation_used`: 生成時に実際に引用された

**負例（edge_weight を下げる）**:
- `hallucination_induced`: 検索結果にない情報を生成（検出可能）
- `redundant`: 既にコンテキストにある情報の重複
- `irrelevant`: 回答に寄与しなかった（attention weight で推定）
- `user_rejected`: ユーザーが回答を拒否

#### edge_weight の更新

```python
# エッジ: (query_type, doc_id) → weight
w = defaultdict(float)

# 正例
if answer_quality_improved:
    for doc in used_docs:
        w[(query_type, doc.id)] += α

# 負例
if hallucination_detected:
    for doc in retrieved_but_misused:
        w[(query_type, doc.id)] -= β_hallucination

if doc_was_redundant:
    w[(query_type, doc.id)] -= β_redundant
```

#### 検索時の統合

```python
def retrieve_with_gedig(query, top_k=10):
    query_type = classify(query)
    query_emb = embed(query)

    candidates = []
    for doc in corpus:
        sim = similarity(query_emb, doc.embedding)
        edge_w = w.get((query_type, doc.id), 0.0)
        confidence = sigmoid(edge_w)

        # 乗算型統合
        score = sim * confidence
        candidates.append((doc, score))

    return sorted(candidates, key=lambda x: -x[1])[:top_k]
```

### 4.3 Sleep フェーズ（RAG）

**Wake（オンライン）**:
- クエリを受けて検索・生成
- 正例/負例をログに記録

**Sleep（オフライン）**:
- ログから edge_weight を更新
- ハルシネーション誘発ドキュメントのパターンを学習
- query_type ごとの「良いドキュメント」プロファイルを構築

### 4.4 geDIG 指標との対応

```
F = ΔEPC - λ·ΔIG

ΔEPC（構造コスト）:
- 検索結果の数（多いほどコスト高）
- ドキュメントの長さ（長いほどコスト高）
- コンテキストウィンドウの消費

ΔIG（情報利得）:
- 回答品質の改善量
- ハルシネーション減少量
- ユーザー満足度

F < 0 のとき「良い検索」:
- 少ない検索結果で、大きな回答品質向上
```

### 4.5 AG/DG の対応

**AG（Attention Gate）**: 「もっと検索が必要か？」
- `g0 > θ_AG`: 現在のコンテキストでは曖昧/不十分 → 追加検索
- トリガー: 生成モデルの不確実性が高い、回答が短すぎる

**DG（Decision Gate）**: 「この検索結果を採用するか？」
- `min(g0, g_min) ≤ θ_DG`: この検索結果は信頼できる → 採用
- トリガー: edge_weight が高い、similarity が高い、冗長でない

### 4.6 RAG での閃きモード（Phase 5）

RAG における「閃き」は、**クエリとドキュメントを繋ぐ新しい概念の創発**。

#### 理解モード（通常の RAG）

```
Query: "量子コンピュータの応用は？"
  ↓ 検索
Docs: [量子暗号, 量子シミュレーション, 量子機械学習]
  ↓ 生成
Answer: 「量子コンピュータは暗号、シミュレーション、機械学習に応用されます」

→ 既存ドキュメントの整理・要約
```

#### 閃きモード（仮説生成 RAG）

```
Query: "量子コンピュータと脳の共通点は？"
  ↓ 検索
Docs: [量子ビット, 神経ネットワーク, 情報処理]  ← 直接答えがない
  ↓ 空白地帯検出
Gap: 「量子」と「脳」は遠いが、「情報処理」で繋がりそう
  ↓ 仮説生成
Hypothesis: 「重ね合わせ原理が共通？」
  ↓ 検証
Answer: 「両者は重ね合わせ原理で動作する可能性—量子状態と神経振動が
         情報処理パラダイムを共有しているかもしれない」

→ 新しい視点・仮説の提示
```

#### 閃きの評価基準（RAG）

```python
def is_eureka_rag(state):
    return (
        state.query_doc_gap > θ_gap and      # クエリとドキュメントが遠い
        state.hypothesis_generated and        # 仮説が生成された
        state.hypothesis_grounded and         # 仮説がドキュメントに根拠を持つ
        state.novelty > θ_novelty             # 既存回答と異なる
    )
```

#### 正例/負例（閃きモード）

**正例**:
- 仮説が後続の検索で支持された
- ユーザーが「新しい視点だ」と評価
- 仮説から派生した追加質問が生まれた

**負例**:
- 仮説がドキュメントと矛盾
- ハルシネーション判定された
- ユーザーが「的外れ」と評価

---

## 5. 一般化の戦略：特殊から始めて抽象を抽出

```
迷路（Phase 1）     ← 最小検証
    ↓
    動作原理の確認
    ↓
ARC（Phase 1'）     ← ドメイン移植
    ↓
    共通パターンの抽出
    ↓
RAG（Phase 1''）    ← 実用ドメイン
    ↓
    さらに共通パターンを抽出
    ↓
一般 geDIG（Phase 2-4）
    - 状態表現の自律化
    - similarity の自律化
    - 正例/負例定義の自律化
```

**各ドメインの「特殊 geDIG」を実装する過程で、共通の抽象が見えてくる**。

それを Phase 2-4 で形式化する。

---

## 6. 現在地と次のステップ

### 現在地

- [x] 迷路 Phase 1: edge_weight の基本設計
- [x] 設計ドキュメント: similarity × confidence 統合
- [ ] 迷路 Phase 1: 実装・検証
- [ ] ARC Phase 1': 設計
- [ ] RAG Phase 1'': 設計

### 次のステップ

1. **迷路で edge_weight 実装を完了**
   - 乗算型統合 `score = similarity * sigmoid(w)`
   - Q-learning との比較実験

2. **RAG への写像を具体化**
   - 既存の RAG 実験（`experiments/rag_reranking/`）との接続
   - edge_weight ログの設計

3. **共通抽象の抽出**
   - 迷路・ARC・RAG で共通する部分を特定
   - Phase 2 への入力として整理

---

## References

- `docs/design/episode_memory_autodesign.md` — Phase 0-3 の詳細設計
- `docs/design/graph_pattern_sleep_semantic_space.md` — similarity × confidence 統合
- `docs/research/insight_vs_understanding.md` — 理解 vs 閃きの詳細設計
- `experiments/rag_reranking/` — RAG 実験（既存）
- `docs/research/gedig_origin_story.md` — geDIG の起源
