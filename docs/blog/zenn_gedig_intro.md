---
title: "RAGの「いつ更新するか」問題を解く ― geDIG入門"
emoji: "🧠"
type: "tech"
topics: ["rag", "llm", "knowledgegraph", "ai", "machinelearning"]
published: false
---

# RAGの「いつ更新するか」問題を解く

RAG（Retrieval-Augmented Generation）は「**何を**取得するか」の最適化で大きく進歩しました。BM25、DPR、Contriever、ColBERT... 検索精度は年々向上しています。

でも、こんな経験はありませんか？

```
ユーザー: 「東京の人口は？」

RAG: 「関連文書を5件取得しました！」
  → 文書1: 東京の人口は1400万人（2023年）  ← 正解
  → 文書2: 東京の面積は2194km²           ← ノイズ
  → 文書3: 東京は日本の首都              ← ノイズ
  → 文書4: 東京都の人口推移              ← 冗長
  → 文書5: 東京オリンピック2020          ← ノイズ

→ 結果: 正解は含まれているが、ノイズも多い
→ LLMのコンテキストが汚染される
→ 回答の質が低下
```

問題は「**いつ**新しい情報を受け入れるか」の判断基準がないことです。

## geDIGとは

geDIG（Graph Edit Distance + Information Gain）は、この「いつ」を判断するための**統一ゲージ**です。

核心となる式：

```
F = 構造コスト − λ × 情報利得
```

- **F が小さい** → 良い更新（低コスト、高利得）
- **F が大きい** → 悪い更新（高コスト、低利得）

## 直感的な理解

人間が新しい情報を聞いたとき、無意識にやっていることを考えてみましょう。

**例1**: 「地球は平らだ」と言われたら
- 情報利得: 小（地平線の説明にはなる...かも）
- 構造コスト: **巨大**（既存知識と矛盾）
- 判断: **却下** 🙅

**例2**: 「東京の人口は1400万人」と言われたら
- 情報利得: 大（質問に直接回答）
- 構造コスト: 小（既存知識と整合）
- 判断: **受け入れ** 🙆

geDIGはこの判断を計算可能にしたものです。

## 2段階のゲート

geDIGには2つのゲートがあります。

### AG（Ambiguity Gate）—「もっと探すべき？」

ローカルな構造が曖昧なときに発火します。

```python
if g0 > theta_AG:
    # 曖昧だ、もっと検索しよう
    expand_search()
```

### DG（Decision Gate）—「これは良い接続？」

マルチホップ評価で「本当にショートカットになるか」を確認します。

```python
if g_min < theta_DG:
    # 確信あり、この情報を受け入れよう
    commit_to_graph()
```

## 実際の効果

### HotPotQA（マルチホップQA）での結果

| 手法 | EM | F1 | 遅延(ms) |
|------|-----|-----|---------|
| BM25 | 36.6% | 52.3% | 820 |
| **geDIG** | **37.5%** | **53.8%** | 873 |
| 差分 | +2.4% | +2.9% | +6.5% |

7,405件のフルデータセットで、geDIGはBM25を上回りました。遅延の増加はわずか53ms。

### 迷路探索での結果

「知識グラフの更新」を迷路に置き換えた概念実証：

| 手法 | 成功率 | ステップ数 | 地図の圧縮率 |
|------|--------|-----------|------------|
| ランダム | 45% | 210 | 0% |
| 貪欲法 | 92% | 85 | 0% |
| **geDIG** | **98%** | **69** | **95%** |

geDIGはゴールを効率的に見つけるだけでなく、**不要な情報を捨てて最小限の地図を構築**します。

## 使ってみる

### インストール

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
pip install -e .
```

### 最小サンプル

```python
from insightspike import create_agent

# エージェント作成（mock LLMで動作確認）
agent = create_agent(provider="mock")

# 質問を処理
result = agent.process_question("geDIGとは何ですか？")
print(result.response)
```

### geDIGの判断を直接見る

```python
from insightspike.algorithms.gedig_core import GedigCore

# geDIGコア初期化
core = GedigCore(
    lambda_weight=1.0,  # 構造と情報のバランス
    gamma=1.0,          # 最短路の重み
)

# 評価
result = core.calculate(
    graph_before=current_graph,
    graph_after=graph_with_new_node,
    linkset_info=linkset,
)

print(f"F値: {result.f_score}")
print(f"構造コスト: {result.epc_norm}")
print(f"情報利得: {result.ig_norm}")

if result.f_score < threshold:
    print("→ この更新を受け入れます")
else:
    print("→ この更新は却下します")
```

## なぜこれが重要か

geDIGは単なるアルゴリズムではなく、**設計原理**です。

> 構造と情報のバランスで、いつ変化するかを決める

この原理は：
- 🧠 脳が何を記憶するかの判断
- 🌱 細胞がいつ分裂するかの判断
- 🏢 組織がいつ変革するかの判断

にも共通するかもしれません。

もしこの原理が根本的なものなら、私たちは「自分で学ぶべきタイミングを知っているAI」を設計できるかもしれません。

## デモを試す

**ブラウザで今すぐ試せます！**

👉 **[geDIG Demo on Hugging Face](https://huggingface.co/spaces/miyaukaz/gedig-demo)**

- サンプル質問でgeDIGの判断を体験
- BM25との比較を可視化
- F値、AG/DGの状態をリアルタイム表示

## もっと知る

- **[GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI)** - コードとドキュメント
- **[5分でわかるgeDIG](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes_ja.md)** - 概要
- **[論文 (arXiv)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)** - 詳細な理論
- **[インタラクティブPlayground](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/examples/playground.py)** - ローカルで触る

## まとめ

| 問題 | geDIGの解決策 |
|------|--------------|
| いつ検索すべき？ | AG（曖昧性ゲート）が判断 |
| いつ受け入れるべき？ | DG（決定ゲート）が判断 |
| ノイズをどう排除？ | F値が高い情報を却下 |
| 判断基準は？ | 構造コスト − λ × 情報利得 |

RAGの「**何を**取得するか」から「**いつ**更新するか」へ。

geDIGはその答えの一つを提供します。

---

**フィードバック・質問はGitHub Issuesまたは[@kazuyoshim5436](https://twitter.com/kazuyoshim5436)まで！**
