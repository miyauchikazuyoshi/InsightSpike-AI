# Maze Sleep Phase 2: Index + Graph による記憶と推論

**Version**: 0.1 (Draft)
**Date**: 2026-01-31
**Status**: Design Phase

---

## 概要

Phase 1 の迷路実験（`maze-query-hub-prototype`）から得られた知見を基に、
**インデックス + グラフ構造** を核とした新しい Sleep 実装を設計する。

### 動機

Phase 1 では生データによるベクトル距離計算で類似度を算出していた。
Sleep 相を本格的に実装するにあたり、以下の課題が明らかになった：

1. **「同じ」と「似てる」の区別が曖昧** - 再認（recognition）と想起（recall）の分離が不十分
2. **ベクトル形式への依存** - 8次元ベクトルの設計がドメイン固有のヒューリスティック
3. **DGエッジの活用が不明確** - グラフ構造をどう検索・推論に使うか

### 核心的アイデア

```
インデックス: 「これは何か？過去に見たか？」（識別）
グラフ構造:   「これに関連するものは何か？」（関連）

ベクトルは手段。本質はインデックス + グラフ。
```

---

## 主要コンセプト

### 1. 再認 (Recognition) vs 想起 (Recall)

```
再認: 「これは前に来た場所だ」→ 過去の経験をそのまま適用
想起: 「これに似た場所があった」→ 過去を参考に推論
```

- 再認 = インデックスでの exact match（EPC ≈ 0）
- 想起 = 類似検索（EPC > 0）

### 2. 予測符号化 (Predictive Coding)

```
予測誤差 ≈ 0  → 「既知」→ 情報を取り入れない → 既存記憶を適用
予測誤差 > 0  → 「新規」→ 情報を取り入れる → 学習・記憶
```

AG/DG との対応：
- AG: 予測誤差が大きい → 注意を向ける → 探索範囲拡大
- DG: 記憶すべきと判断 → グラフに統合 → エッジ確定

### 3. 特徴 vs ラベル

エピソードベクトルの再解釈：

```
構造特徴（類似度計算用）:
  - 位置、方向、状態属性

ラベル（評価用）:
  - outcome: 良い/悪い（+1, 0, -1）
  - purpose: 目的との関係
```

統合: `score = similarity(構造特徴) × confidence(ラベル)`

### 4. F値による類似度の調整

```
raw_similarity = ベクトル距離（生データ）
F = ΔEPC − λ·ΔIG

adjusted_score = similarity × sigmoid(-F)
```

F が低い（良い）ほどスコアが高くなる。

---

## 参照

- `docs/research/gpt_bert_gedig_perspective.md` - GPT/BERT的タスク構造との対応
- `docs/design/graph_pattern_sleep_semantic_space.md` - 検索と評価の分離
- `experiments/maze-query-hub-prototype/` - Phase 1 実装
