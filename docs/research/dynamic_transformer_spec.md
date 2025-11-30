# Dynamic Transformer Architecture via geDIG

## 1. 概要 (Overview)

本ドキュメントは、geDIG (Generalized Differential Information Gain) のグラフ力学を、Transformer の内部動作（Attention/FFN）と数学的に対応付け、**「推論しながら自らの構造を最適化する次世代 Transformer (Dynamic Transformer)」** の設計仕様を定義するものです。

## 2. コア・アイソモーフィズム (Core Isomorphism)

既存の Transformer と geDIG 知識グラフは、以下の対応関係（同型性）を持ちます。

| Transformer Component | geDIG Graph Component | 役割 | 性質 |
| :--- | :--- | :--- | :--- |
| **Token** | **Node** | 情報の単位 | 状態ベクトルを持つ |
| **Attention Head** | **Edge** | 情報の伝播経路 | $W_Q, W_K, W_V$ を持つ |
| **Attention Score** | **Edge Weight ($w_{ij}$)** | 接続の強さ | 動的に変動する |
| **FFN (Feed-Forward)** | **Consolidated Edge** | 長期記憶・直感 | Phase 2 で固定化される |

### 2.1 エッジの拡張 (QKV-Augmented Edges)

従来、グラフのエッジは単なるスカラー重み $w_{ij}$ でしたが、これを拡張し、Transformer の Attention Head と等価な機能を持たせます。

各エッジ $e_{ij}$ は以下の属性を持ちます：
*   **$W_Q^{(e)}, W_K^{(e)}, W_V^{(e)}$**: クエリ・キー・バリュー射影行列（またはその低ランク近似/キャッシュ）。
*   **$h_{ij}$**: Attention Score (活性度)。
    $$ h_{ij} = \text{softmax} \left( \frac{(x_i W_Q^{(e)}) (x_j W_K^{(e)})^T}{\sqrt{d_k}} \right) $$

これにより、**「グラフ = スパースな接続を持つ Transformer」** と見なすことができます。

## 3. 動的サイクル (The Dynamic Cycle)

静的な重みを持つ従来の Transformer と異なり、Dynamic Transformer は **Phase 1 (覚醒)** と **Phase 2 (睡眠)** のサイクルを通じて、自律的に構造を変化させます。

### 3.1 Phase 1: Awake (Dynamic Routing / Exploration)
*   **動作:** 推論（Inference）および探索。
*   **Attention:**
    *   既存のエッジ（Strong Attention）だけでなく、**AG (Attention Gate)** を開いて未知のノードへの接続（Weak Attention）を試行する。
    *   これは「全結合 Attention」の一部を動的に計算することに相当する。
*   **学習:**
    *   DG (Decision Gate) が発火（洞察を獲得）した場合、そのパスを **「短期記憶 (Short-term Edge)」** として一時保存する。

### 3.2 Phase 2: Sleep (Consolidation / Structural Plasticity)
*   **動作:** オフライン最適化。
*   **FFN化 (Consolidation):**
    *   Phase 1 で頻繁に活性化したエッジ（パス）を、**FFN の重みとして焼き付ける**。
    *   数理的には、複数の Attention Head の合成関数を、単一の線形変換（または MLP）で近似・蒸留することに相当する。
    *   $$ \text{FFN}(x) \approx \sum_{(i,j) \in \text{Path}} \text{Attn}(x_i, x_j) $$
*   **剪定 (Pruning):**
    *   貢献度の低いエッジ（Attention Head）を削除し、計算リソース（スパース性）を回復させる。

## 4. 期待される効果 (Impact)

1.  **推論コストの劇的な削減:**
    *   「熟考（多ホップのAttention探索）」が「直感（1ホップのFFN）」に変換されるため、学習が進むほど推論が高速化する。
2.  **無限のコンテキスト長:**
    *   必要な情報はグラフ構造（長期記憶）として外部化・内部化されるため、コンテキストウィンドウの制限を受けない。
3.  **解釈可能性 (Explainability):**
    *   なぜその答えが出たのか、グラフ上のパス（思考の軌跡）として追跡可能になる。

## 5. 実装ロードマップ (Draft)

1.  **Step 1:** `EdgeAttributes` に QKV ベクトルを追加する。
2.  **Step 2:** `SignalPropagator` を Attention 機構に置き換える（単純な減衰ではなく、内積による活性伝播）。
3.  **Step 3:** Phase 2 (Sleep) での「エッジ → FFN」蒸留ロジックを実装する。
