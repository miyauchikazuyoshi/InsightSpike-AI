# InsightSpike-AI — geDIG: AIに「閃き」を与える

[![CI (Lite)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-lite.yml)
[![CI (Unit)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci-unit.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.pdf)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-green)](https://miyauchikazuyoshi.github.io/InsightSpike-AI)

---

## なぜアインシュタインは、アマチュアの独立研究者として、相対論を発見できたのか？

1905年、特許庁の審査官だったアインシュタインは、物理学のアカデミアの外部にいながら、電磁気学と古典力学という矛盾する2つの理論を統一した。

**彼は何をしたのか？**

私たちの仮説：彼は異なる理論の**構造的パターン**を認識し、それらを**同型にする最小の変換**（ローレンツ変換）を発見した。これが「閃き」の実体である。

```
電磁気学の構造 ─┬─→ 矛盾 ←─┬─ 古典力学の構造
               │           │
               └──→ T* ←──┘
                    │
                    ▼
            T* = ローレンツ変換
            （2つの構造を同型にする最小編集）
```

**geDIG**は、この「閃き」を数式化した計算論的モデルである。

---

## 核心の方程式

```
T* = argmin_T GED(T(G₁), G₂)
```

> 2つの知識構造を同型にする最小変換 T* を発見せよ。
> その変換こそが「閃き」である。

### 人間の認知に関する仮説

人間の脳は、**構造的類似性**と**確率的関連性**を等価に扱っている：

```
「AとBは構造が似ている」
    ↓ 脳内で変換
「AとBは関係がある確率が高い」
```

この変換を定量化したのが geDIG の統一ゲージである：

```
F = ΔEPC_norm − λ · ΔIG
```

| 項 | 意味 | 認知的対応 |
|----|------|-----------|
| ΔEPC_norm | 構造変化のコスト | 「どれだけ考え方を変えるか」 |
| ΔIG | 情報利得 | 「どれだけ理解が深まるか」 |
| F | 統一ゲージ | 「閃きの価値」（小さいほど良い） |

---

## デモ：AIが「閃く」瞬間

### 新規アナロジーの自動発見

16の異なるドメイン（物理学、生物学、心理学、芸術...）の知識グラフから、アルゴリズムが自動発見した意外なアナロジー：

| 発見 | 構造的意味 |
|------|-----------|
| **革命 ≈ 感情** | 社会革命は「集団的感情反応」として理解できる |
| **コンパイラ ≈ 岩石サイクル** | コード変換と地質変成は「段階的変形プロセス」 |
| **免疫系 ≈ 情報拡散** | ウイルス感染とバイラルマーケティングは同じ構造 |
| **遺伝子発現 ≈ 学習** | 分子レベルと認知レベルで「学習」の構造が同じ |

```bash
# 自分で試す
poetry run python experiments/isomorphism_discovery/novel_analogy_discovery.py
```

**これらは誰も教えていない。アルゴリズムが構造から発見した。**

---

## 3つのレベルの「閃き」

```
Level 3: 同型発見 ─────────────────────────────
         T* = argmin_T GED(T(G₁), G₂)
         「矛盾を解消する変換の発見」
         例: アインシュタインの相対論
                    │
Level 2: アナロジー検出 ────────────────────────
         SS(G₁, G₂) > θ
         「異なるドメイン間の構造的対応」
         例: ボーアの原子モデル（太陽系≈原子）
                    │
Level 1: パターンマッチ ────────────────────────
         sim(a, b) = cos(φ(a), φ(b))
         「要素間の類似性」
         例: 通常のRAG検索
```

geDIGは3つのレベルすべてをカバーする統一フレームワークである。

---

## 検証結果

### 科学史の再現

| 発見 | 年 | 構造類似度 | geDIGによる検出 |
|------|-----|-----------|----------------|
| ボーアの原子モデル | 1913 | 0.995 | ✓ |
| ケクレのベンゼン環 | 1865 | 0.967 | ✓ |
| ダーウィンの自然選択 | 1859 | 0.985 | ✓ |

### Cross-Domain QA

| 条件 | F1スコア |
|------|---------|
| 構造類似度なし | 0.062 |
| 構造類似度あり | **0.660** |
| **改善幅** | **+60%** |

### スケーラビリティ

| ノード数 | 処理時間 |
|---------|---------|
| 100 | 32ms |
| 500 | 1.6s |
| 1000 | 5.5s |

---

## 分子設計AIとの接続

創薬AIは分子グラフの編集距離で「同じ薬効を持つ異なる分子」を発見する（Scaffold Hopping）。

geDIGは同じ数学で「同じ説明力を持つ異なる理論」を発見する。

```
分子設計AI                    geDIG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
分子グラフ        ←→        知識グラフ
分子編集距離      ←→        GED
同じ薬効の異分子  ←→        同じ説明力の異理論
Scaffold Hopping ←→        Theory Unification
```

**創薬で新薬を発見するアルゴリズムが、知識で新理論を発見できる。**

---

## クイックスタート

```bash
# セットアップ
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
poetry install

# 新規アナロジー発見
poetry run python experiments/isomorphism_discovery/novel_analogy_discovery.py

# 同型発見の基本テスト
poetry run python -c "
from src.insightspike.algorithms.isomorphism_discovery import discover_insight, create_test_graphs

solar, atom, _ = create_test_graphs()
transform = discover_insight(solar, atom)
print(transform)
print(transform.to_insight_description())
"
```

---

## アプリケーション

### 現在動作するもの

- **RAG最適化**: 「いつ検索するか」の自律的判断（HotpotQAで EM +3.5pt）
- **迷路探索**: 部分観測環境での効率的探索（15×15で成功率98%）
- **アナロジー検出**: 異分野間の構造的類似性発見（F1 +60%）
- **同型発見**: 知識構造間の最小変換発見（1000ノード対応）

### 将来のビジョン

- **科学発見支援**: 研究者の論文から意外な分野間接続を発見
- **教育**: 「〇〇は△△に似ている」の自動生成
- **創作支援**: 物語構造の異分野転用
- **創薬との統合**: 分子設計AIとの技術共有

---

## 技術詳細

### 統一ゲージ

```
F = ΔEPC_norm − λ · ΔIG

ΔEPC_norm: 正規化編集経路コスト（構造変化量）
ΔIG = ΔH_norm + γ · ΔSP_rel（情報利得）
  - ΔH_norm: エントロピー変化
  - ΔSP_rel: 最短路改善
```

### 二段ゲート（AG/DG）

```
AG (Attention Gate): 0-hopでの曖昧さ検知 → 探索トリガー
DG (Decision Gate): Multi-hopでの安定性確認 → 決定トリガー
```

### 理論的背景

- **自由エネルギー原理 (FEP)**: 脳は「驚き」を最小化する
- **最小記述長 (MDL)**: 最良の仮説は最も圧縮できるもの
- **グラフ編集距離 (GED)**: 構造変換の最小コスト

geDIGはこれらを操作的に橋渡しする。

---

## ドキュメント

- [5分でわかるgeDIG](docs/concepts/gedig_in_5_minutes.md) / [日本語](docs/concepts/gedig_in_5_minutes_ja.md)
- [直感的ガイド](docs/concepts/intuition.md) / [日本語](docs/concepts/intuition_ja.md)
- [論文 (v6)](docs/paper/geDIG_onegauge_improved_v6.pdf)
- [同型発見の設計書](docs/design/level3_isomorphism_discovery.md)
- [実験結果](docs/experiments/structural_similarity_results.md)

---

## 論文

### 主論文
- [geDIG: 単一ゲージによる動的知識グラフ制御](docs/paper/geDIG_onegauge_improved_v6.pdf)

### 投稿予定
- JSAI 2026: HotpotQAベンチマーク + 閃きの計算的モデル
- 独立論文: "Graph Edit Distance as a Computational Model of Scientific Insight"

---

## コラボレーション募集

「閃きの計算論的モデル」という野心的な目標に向けて、以下の専門家を募集：

| 役割 | 貢献 |
|------|------|
| **認知科学者** | 人間の閃きとgeDIGの対応検証 |
| **分子設計AI研究者** | Scaffold Hoppingとの技術統合 |
| **理論物理学者** | FEP-MDLブリッジの数学的厳密化 |
| **MLエンジニア** | 大規模知識グラフでの検証 |

**連絡先**: miyauchikazuyoshi@gmail.com / X: @kazuyoshim5436

---

## ライセンス

Apache-2.0

## 特許出願

- JP 2025-082988, 2025-082989（出願中）

---

> 「創薬AIが分子の同型を探すように、geDIGは理論の同型を探す。
>  その編集操作こそが閃きの実体である。」
