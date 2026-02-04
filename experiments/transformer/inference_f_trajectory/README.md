# 推論時F軌跡実験

**ステータス**: 📝 仕様検討中

## 概要

Transformerの推論時（1回のforward pass）において、**層を通過するごとにF値が単調減少するか**を検証する実験。

## 核心的仮説

```
Layer 0 (入力埋め込み):  F_0 = 高い（無構造・曖昧）
Layer 1:                 F_1 < F_0
  ...
Layer L (出力):          F_L = 最小（構造解決済み）
```

## 学習過程との関係

| 過程 | 観測対象 | 何が変化するか | 検証方法 |
|------|----------|---------------|----------|
| **学習** | モデル（重み） | Attention構造 | Pythiaチェックポイント |
| **推論** | ベクトル（表現） | Hidden state構造 | 任意モデルのforward pass |

両者は**対をなす**：
- 学習 = 「F↓を実現する変換」を獲得する過程
- 推論 = 獲得した変換で「F↓」を実行する過程

## 検討中の論点

### Hidden StateベースのgeDIG定義

```python
# 類似度行列（暗黙のグラフ）
sim = cosine_similarity(h, h)

# vs Attention（QK^T）を使うべきか？
```

### 位置符号の活用

```python
# 位置距離を重み付けに使用
# 「遠いのに類似度が高い」= ショートカット
shortcut_score = (sim * position_distance).mean()
```

### Q, K, V分離の意味

```
QK^T = h @ W_Q @ W_K.T @ h.T = h @ W_QK @ h.T

cosine(h, h) = h @ h.T  ← W_QK = I の特殊ケース
```

## 関連実験

- **学習過程の検証**: [../pythia_checkpoints/](../pythia_checkpoints/) （実験完了）

## ファイル構成

```
inference_f_trajectory/
├── README.md    # 本ファイル
├── SPEC.md      # 詳細仕様書
└── (実装予定)
```

詳細仕様は [SPEC.md](./SPEC.md) を参照。

---

*Last updated: 2026-02-03*
