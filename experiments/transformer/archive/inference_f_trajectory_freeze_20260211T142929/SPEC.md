# 推論時F軌跡実験 仕様書

**作成日**: 2026-02-03
**ステータス**: 📝 仕様検討中（EPC, H, SPの定義について議論中）

---

## 1. 概要

### 1.1 目的

Transformerの推論時（1回のforward pass）において、入力ベクトルが層を通過するごとに**F値が単調減少する**かを検証する。

### 1.2 核心的仮説

> **「各層は、入力の無秩序状態から出力の秩序状態へ、Fを下げながら遷移している」**

```
Layer 0 (入力埋め込み):  F_0 = 高い（無構造・曖昧）
Layer 1:                 F_1 < F_0
Layer 2:                 F_2 < F_1
  ...
Layer L (出力):          F_L = 最小（構造解決済み）
```

### 1.3 理論的意義

geDIGは**2つの並行する過程**でFの減少を捉える：

| 過程 | 観測対象 | 何が変化するか | 検証方法 |
|------|----------|---------------|----------|
| **学習** | モデル（重み） | Attention構造 | Pythiaチェックポイント |
| **推論** | ベクトル（表現） | Hidden state構造 | 任意モデルのforward pass |

```
学習過程:  step 0 → step N   モデルのF↓（重みが最適化）
推論過程:  Layer 0 → Layer L  ベクトルのF↓（表現が構造化）
```

**両者は対をなす**：
- 学習 = 「F↓を実現する変換」を獲得する過程
- 推論 = 獲得した変換で「F↓」を実行する過程

どちらも同じgeDIG原理に従う。

---

## 2. 手法

### 2.1 AttentionベースとHidden Stateベースの関係

geDIGは2つの視点から構造を捉えられる：

| 視点 | 観測対象 | 適用場面 |
|------|----------|----------|
| **Attention** | Attention行列 | 学習過程（モデル内部の変化） |
| **Hidden State** | 表現ベクトルの類似度 | 推論過程（ベクトルの変化） |

```
Attentionベース:
  構造 = Attention行列（明示的なグラフ）
  用途 = 学習中のモデル構造変化を追跡

Hidden Stateベース:
  構造 = トークン間類似度（暗黙のグラフ）
  用途 = 推論中のベクトル変化を追跡
```

**両者は補完関係**にあり、同じF原理の異なる表現。

### 2.2 Hidden Stateからの構造抽出

各層の hidden state `h ∈ R^{S×D}` （S=系列長, D=次元）から、トークン間の「暗黙のグラフ」を構築する。

```python
# トークン間類似度行列（暗黙のグラフの隣接行列）
sim_matrix = cosine_similarity(h, h)  # shape: (S, S)
```

この類似度行列が「構造」を表す：
- 高い類似度 = 強いエッジ（関連するトークン）
- 低い類似度 = 弱いエッジ（無関係なトークン）

### 2.3 geDIG成分の定義（Hidden Stateベース）

#### EPC（構造変化コスト）

```python
def compute_epc(h_before, h_after):
    """
    層間のhidden state変化量
    = 表現空間での「移動距離」
    """
    sim_before = cosine_similarity_matrix(h_before)
    sim_after = cosine_similarity_matrix(h_after)

    # 類似度構造の変化量
    epc = torch.abs(sim_after - sim_before).mean()
    return epc
```

**解釈**: EPCが大きい = 構造を大きく変えた = コストが高い

#### H（エントロピー）

```python
def compute_entropy(h):
    """
    類似度分布のエントロピー
    高H = 全トークンが均等に類似（無構造）
    低H = 特定トークンに類似が集中（構造化）
    """
    sim = cosine_similarity_matrix(h)

    # 各行を確率分布として正規化
    sim_norm = softmax(sim, dim=-1)

    # シャノンエントロピー
    entropy = -(sim_norm * log(sim_norm + eps)).sum(dim=-1).mean()
    return entropy
```

#### SP（ショートカット純度）

```python
def compute_sp(h, anchor_idx=0, k_ratio=0.2):
    """
    アンカートークンへの類似度集中度

    anchor_idx:
      - BERT: 0 (CLS)
      - GPT: -1 (最終トークン) または質問の最後
    """
    sim = cosine_similarity_matrix(h)

    # アンカー列の類似度
    to_anchor = sim[:, anchor_idx]

    # 上位k%の占有率
    k = max(1, int(len(to_anchor) * k_ratio))
    top_k = torch.topk(to_anchor, k).values.sum()
    total = to_anchor.sum()

    sp = top_k / (total + eps)
    return sp
```

#### F値（統合指標）

```python
def compute_F(h_before, h_after, anchor_idx=0, lambda_=1.0, gamma=0.5, entropy_sign=-1):
    """
    層間のF値を計算

    entropy_sign:
      -1: 集中化が利得（推論時）
      +1: 拡散が利得（探索時）
    """
    epc = compute_epc(h_before, h_after)

    h_before_entropy = compute_entropy(h_before)
    h_after_entropy = compute_entropy(h_after)
    delta_h = h_after_entropy - h_before_entropy

    sp_before = compute_sp(h_before, anchor_idx)
    sp_after = compute_sp(h_after, anchor_idx)
    delta_sp = sp_after - sp_before

    F = epc - lambda_ * (entropy_sign * delta_h + gamma * delta_sp)
    return F, epc, delta_h, delta_sp
```

---

## 3. 実験設計

### 3.1 対象モデル

| モデル | タイプ | アンカー | 優先度 |
|--------|--------|----------|--------|
| bert-base-uncased | Encoder | CLS (idx=0) | 高 |
| gpt2 | Decoder | 最終トークン | 高 |
| distilbert-base-uncased | Encoder | CLS (idx=0) | 中 |
| pythia-70m | Decoder | 最終トークン | 中 |

### 3.2 テストデータ

```python
test_sentences = [
    # 短文
    "Hello world.",
    "The cat sat on the mat.",

    # 中文
    "Machine learning is transforming how we interact with technology.",

    # 長文
    "Natural language processing enables computers to understand, interpret, and generate human language in ways that are both meaningful and useful.",

    # 質問文（生成タスク想定）
    "What is the capital of France?",
    "How does photosynthesis work?",
]
```

### 3.3 測定項目

各サンプル・各モデルで以下を記録：

```python
{
    "sample": "Hello world.",
    "model": "bert-base-uncased",
    "num_layers": 12,
    "trajectory": {
        "F": [F_0, F_1, F_2, ..., F_L],
        "EPC": [EPC_1, EPC_2, ..., EPC_L],
        "H": [H_0, H_1, ..., H_L],
        "SP": [SP_0, SP_1, ..., SP_L],
    },
    "is_monotonic_F": true/false,
    "total_F_decrease": F_0 - F_L,
}
```

### 3.4 評価指標

1. **単調減少率**: F軌跡が単調減少しているサンプルの割合
2. **総減少量**: F_0 - F_L の平均
3. **最大減少層**: 最もΔFが大きい層の特定
4. **相関**: 入力長、タスク、モデルとの関係

---

## 4. 期待される結果

### 4.1 仮説が正しい場合

```
Layer:    0    1    2    3    4    5    ...   L
F値:     高   ↓    ↓    ↓    ↓    ↓    ...   低
```

- F軌跡が単調減少（または概ね減少）
- 浅層で大きな減少、深層で収束
- モデル・タスク問わず同じ傾向

### 4.2 仮説が間違っている場合

- F軌跡が非単調（増減を繰り返す）
- モデルやタスクで傾向がバラバラ
- 特定の層でFが急増する

### 4.3 部分的に正しい場合

- 特定の層範囲でのみF↓
- 浅層でF↑、深層でF↓（または逆）
- モデルタイプ（Encoder/Decoder）で異なる

---

## 5. 実装計画

### 5.1 ファイル構成

```
inference_f_trajectory/
├── SPEC.md                    # 本仕様書
├── gedig_hidden.py            # Hidden stateベースのgeDIG実装
├── measure_trajectory.py      # F軌跡測定スクリプト
├── analyze_results.py         # 結果分析・可視化
└── results/
    ├── trajectory_bert.json
    ├── trajectory_gpt2.json
    ├── trajectory_plot.png
    └── summary.json
```

### 5.2 実装順序

1. **gedig_hidden.py**: Hidden stateベースのEPC, H, SP, F計算
2. **measure_trajectory.py**: 複数モデル・サンプルでF軌跡を測定
3. **analyze_results.py**: 単調性検証、可視化、統計

### 5.3 依存関係

```
torch
transformers
numpy
matplotlib
```

---

## 6. 迷路実験との対応

| 迷路 | Transformer推論 |
|------|-----------------|
| 開始位置（空グラフ） | 入力埋め込み（Layer 0） |
| step 1: エッジ追加 | Layer 1: 構造形成開始 |
| step N: ゴール到達 | Layer L: 出力（構造解決） |
| F_start = 高い | F_0 = 高い |
| F_goal = 最小 | F_L = 最小 |
| 各stepでF↓ | 各LayerでF↓ |

**同じ原理が両ドメインを支配している**ことの検証。

---

## 7. 成功基準

### 7.1 最小成功

- [ ] 1つのモデルで、過半数のサンプルでF軌跡が概ね減少

### 7.2 中程度の成功

- [ ] 複数モデル（BERT, GPT2）で同様の傾向
- [ ] 層別の特徴（浅層で急減少など）が明確

### 7.3 完全成功

- [ ] モデル・タスク問わずF↓が普遍的
- [ ] 迷路実験と定量的に比較可能
- [ ] 論文のメイン結果として使用可能

---

## 8. リスクと対策

| リスク | 対策 |
|--------|------|
| F軌跡が非単調 | 層範囲を限定、成分別に分析 |
| SPの定義がCausal LMで不適切 | 最終トークン、またはSP無しで検証 |
| 計算コスト | 小さいモデル（distilbert, gpt2-small）から開始 |
| hidden stateの類似度が情報不足 | attention併用、または別の類似度指標 |

---

## 9. 次のステップ

1. **gedig_hidden.py** を実装
2. **BERT** で初期検証（5サンプル程度）
3. 結果を見て仕様を調整
4. 本格実験（複数モデル・多サンプル）

---

*この実験が成功すれば、geDIGは「学習原理」から「推論原理」へと適用範囲が拡大し、Transformer（およびベクトル変換一般）の統一理論となる可能性がある。*
