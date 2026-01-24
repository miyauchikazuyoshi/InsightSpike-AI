# Flash-geDIG Specification
**"Structure as a First-Class Citizen in Deep Learning"**

## 1. コンセプト (Philosophy)
*   **Zero-Copy**: CPU/NetworkXへのデータ転送を一切行わず、全ての計算をGPU上のTensor演算で完結させる。
*   **Differentiable**: 全工程が微分可能（Differentiable）であり、学習時の損失関数（Loss）として直接利用可能。
*   **Plug-and-Play**: 既存のTransformerモデル（HuggingFace等）に、わずか数行で組み込める手軽さ。

## 2. API Design

### 2.1. Functional API (推奨)
最も手軽に使うためのステートレスな関数群。

```python
import torch
import insightspike.gedig.functional as F_gedig

# attention: (Batch, Heads, Seq, Seq)
f_score, metrics = F_gedig.compute_f_score(
    attention_matrix, 
    mask=None,
    lambda_param=1.0,  # コスト/価値のバランス
    gamma=0.5          # 通信効率(SP)の重み
)

print(f"Structure Score: {f_score.mean().item()}")
# metrics['epc'] -> 配線コスト
# metrics['entropy'] -> エントロピー（迷い）
# metrics['sp'] -> 通信効率
```

### 2.2. Modular API (nn.Module)
学習ループやモデルの一部として組み込むためのクラス。

```python
from insightspike.gedig import FlashGeDIGLoss

# 前処理でインスタンス化（パラメータ設定済み）
gedig_loss = FlashGeDIGLoss(alpha=0.01)

# Training Loop
outputs = model(inputs)
loss = outputs.loss + gedig_loss(outputs.attentions)
loss.backward()  # 構造化への圧力をかけて逆伝播
```

## 3. Core Algorithms (High-Speed Approximations)

GPU高速化のために、グラフ理論指標を「微分可能な行列演算」に近似翻訳します。

| 指標 | 厳密な定義 (NetworkX) | **Flash-geDIG (Tensor Approx)** | 計算オーダー |
| :--- | :--- | :--- | :--- |
| **Edges** | 閾値カットと本数カウント | **Soft Thresholding** (Sigmoid関数による連続的な重み和) | $O(N^2)$ |
| **Entropy** | 確率分布のシャノンエントロピー | **Normalized Entropy** (Tensor演算) | $O(N^2)$ |
| **SP (Efficiency)** | 全点間最短パス長の平均 (BFS/Dijkstra) | **Matrix Powers** (隣接行列の累乗 $A^k$ による到達性近似) | $O(k \cdot N^3)$ |

*   **SP近似の補足**: $N$が大きい場合、行列累乗は重いため、デフォルトでは $k=4$ 程度の「局所的な効率性（Local Efficiency）」で代用し、超高速化を図ります。

## 4. Use Cases

1.  **診断 (Diagnostics)**: 学習済みモデルの「思考の深さ」を層ごとに可視化。
2.  **正則化 (Regularization)**: 学習時にF値を最大化し、汎化性能と解釈性を向上。
3.  **剪定 (Pruning)**: F値が低い（構造化されていない）Headを自動的に削除し、モデルを軽量化。
4.  **RAG Reranking**: 検索結果の中で「最も構造的に整合する（F値が高い）」情報を採用。

## 5. Next Steps
1.  `src/insightspike/gedig/` パッケージを作成。
2.  `functional.py` と `module.py` を実装。
3.  `train_f_regularized.py` からロジックを移植・純化。
4.  ユニットテストで勾配（Gradient）が流れることを確認。
