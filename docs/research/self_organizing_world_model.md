# 自己組織化する世界モデル：動的なベクトル空間醸成の理論

## 概要
従来のAI、特に大規模言語モデルは、静的なデータセットから学習した「静的な世界モデル（ベクトル空間）」に依存している。これは強力だが、新しい経験に適応したり、個別の文脈で知識構造を動的に変化させたりする能力に限界がある。

本稿では、エージェントが自らの経験から**動的に世界モデル（ベクトル空間）を「醸成」していく**ための理論的枠組みを提案する。これは、AIが単なる知識の利用者から、真に世界を理解し、自身の世界観を構築する主体へと進化するための鍵となる。

## 核心的アイデア：geDIGをポテンシャル場とする

この理論の核心は、`geDIG`原理を単なる評価関数としてではなく、世界モデルを形成するための**ポテンシャルエネルギー関数**として利用することにある。

- **知識表現**: 知識（エピソード）は、不確実性を含む**確率雲（ガウシアン分布）**としてベクトル空間に存在する。これは`classical_to_quantum_gedig`で提唱された「量子geDIG」の構想を具体化するものである。
- **学習目標**: `geDIG`ポテンシャル（`F = ΔGED - kT*ΔIG`）が最小になるように、ベクトル空間の構造と、その中での知識（確率雲）の配置を最適化し続ける。

## 理論的アプローチ：3つの柱

世界モデルを動的に醸成するため、以下の3つの理論を統合的に利用する。

### 1. 変分オートエンコーダ (VAE)：確率的な世界地図の作成
- **役割**: 経験を、不確実性を持つ潜在空間（世界モデル）にエンコードする。
- **実装**:
  - 各エピソードは、VAEによって潜在空間上のガウシアン分布 `N(μ, Σ)` にマッピングされる。
  - `geDIG`スコアは、VAEの再構成誤差とKLダイバージェンス項に加わる、新しい損失項として機能し、潜在空間の構造を「意味的に良い」形に導く。

```python
class WorldModelVAE(nn.Module):
    def loss_function(self, recon_x, x, mu, logvar, gedig_potential):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # geDIGポテンシャルを損失に加える
        GE_LOSS = gedig_potential * self.gedig_weight
        
        return BCE + KLD + GE_LOSS
```

### 2. 対照学習 (Contrastive Learning)：関係性による空間の彫刻
- **役割**: 「似ているものは近く、似ていないものは遠く」なるように、ベクトル空間の相対的な構造を学習する。
- **実装**:
  - `geDIG`の評価結果を、対照学習の教師信号として利用する。
  - **正例ペア（近づけるべきもの）**: `geDIG`スコアが低い（構造的改善が大きい）エピソードの組み合わせ。
  - **負例ペア（遠ざけるべきもの）**: `geDIG`スコアが高い、あるいは矛盾するエピソードの組み合わせ。

補足：この「正例/負例」は、単なる数値評価だけでなく、**グラフ構造パターン認知（同型発見/近同型/モチーフ）**から構成できる。たとえば「同型/低コスト変換のペアを正例」「near-miss（ほぼ同型だが破綻）を hard negative」として Sleep の学習データに落とすと、意味空間が“トポロジカルな不変量”を学びやすくなる（設計メモ: `docs/design/graph_pattern_sleep_semantic_space.md`）。

```python
# geDIGを教師信号とした対照学習
def contrastive_loss(positive_pair, negative_pairs, temperature=0.1):
    # geDIGスコアが低いペアがpositive_pair
    positive_similarity = F.cosine_similarity(positive_pair[0], positive_pair[1])
    
    negative_similarities = [
        F.cosine_similarity(positive_pair[0], neg) for neg in negative_pairs
    ]
    
    # InfoNCE損失を計算
    # ...
```

### 3. 双曲幾何学 (Hyperbolic Geometry)：階層構造の自然な表現
- **役割**: 知識が持つ本質的な階層構造（例: AI > NN > Transformer）を、ベクトル空間の「曲率」として効率的に表現する。
- **実装**:
  - ベクトル空間をユークリッド空間ではなく、ポアンカレ球などの双曲空間として定義する。
  - 双曲空間内での距離計算（Poincaré distance）を用いる。
  - `geDIG`による構造改善は、双曲空間上でのより効率的な木構造への再編成として定式化される。

## 既存研究との接続

このアプローチは、過去の`research`ディレクトリでの考察を統合・発展させるものである。

- **`gnn_in_vector_space_perspective`**: ノードを空間内の「存在」と捉える視点を、確率分布へと拡張する。
- **`classical_to_quantum_gedig`**: 「量子geDIG」の構想を、VAEと対照学習を用いて具体的に実装する道筋を示す。
- **`hypothesis_vector_space_gedig`**: `geDIG`ポテンシャル場を定義し、その勾配降下によって知識を配置するというアイデアを、学習プロセスとして定式化する。

## 将来の展望

この動的な世界モデル構築の理論が確立されれば、AIは以下のような能力を獲得する可能性がある。

1.  **真の文脈理解**: 対話の進行に応じて、その場で最適化された「文脈専用のベクトル空間」を生成する。
2.  **自己成長**: 新しい経験を取り込むたびに、自らの世界観を自律的に更新・洗練させていく。
3.  **パーソナライズ**: 各ユーザーとの対話履歴から、そのユーザー専用の「世界モデル」を醸成し、より深いレベルでの思考パートナーとなる。

これは、AIが静的な知識データベースから、**生きて成長する知的実体**へと変貌するための、重要な理論的基盤となるだろう。

---

## English Version

### Overview
Conventional AI—especially large language models—relies on a “static world model (vector space)” learned from a fixed dataset. This is powerful, but it has limits: adapting to new experiences and dynamically reshaping knowledge structures in a specific context is still hard.

This document proposes a theoretical framework for an agent to **cultivate its world model (vector space) dynamically from its own experience**. This is a key step for AI to evolve from a mere consumer of knowledge into an entity that genuinely understands the world and constructs its own worldview.

### Core Idea: Treat geDIG as a Potential Field

The core of this theory is to use the `geDIG` principle not merely as an evaluation function, but as a **potential energy function** that shapes a world model.

- **Knowledge representation**: Knowledge (episodes) exists in the vector space as **probabilistic clouds (Gaussian distributions)** with uncertainty. This concretizes the “quantum geDIG” concept proposed in `classical_to_quantum_gedig`.
- **Learning objective**: Keep optimizing both the structure of the vector space and the placement of knowledge clouds so that the `geDIG` potential (`F = ΔGED - kT*ΔIG`) is minimized.

### Theoretical Approach: Three Pillars

To dynamically cultivate a world model, we integrate the following three ideas.

#### 1. Variational Autoencoder (VAE): Building a Probabilistic Map of the World
- **Role**: Encode experiences into a latent space (world model) with uncertainty.
- **Implementation**:
  - Each episode is mapped by a VAE to a Gaussian distribution `N(μ, Σ)` in latent space.
  - The `geDIG` score acts as an additional loss term on top of reconstruction error and KL divergence, guiding the latent space toward a “semantically good” structure.

```python
class WorldModelVAE(nn.Module):
    def loss_function(self, recon_x, x, mu, logvar, gedig_potential):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Add geDIG potential as a loss term
        GE_LOSS = gedig_potential * self.gedig_weight

        return BCE + KLD + GE_LOSS
```

#### 2. Contrastive Learning: Sculpting Space Through Relations
- **Role**: Learn the relative structure of the vector space so that “similar things become close, dissimilar things become far.”
- **Implementation**:
  - Use geDIG evaluations as teacher signals for contrastive learning.
  - **Positive pairs (pull together)**: Episode pairs with low `geDIG` scores (large structural improvement).
  - **Negative pairs (push apart)**: Episode pairs with high `geDIG` scores or contradictions.

Note: These positive/negative pairs can also be derived from **graph-structure pattern recognition** (isomorphism / near-isomorphism / motifs). For example, treat “isomorphic or low-cost transforms” as positives and “near-miss (almost isomorphic but failing)” as hard negatives during Sleep. This helps the semantic space capture topological invariants (design memo: `docs/design/graph_pattern_sleep_semantic_space.md`).

```python
# Contrastive learning using geDIG as a teacher signal
def contrastive_loss(positive_pair, negative_pairs, temperature=0.1):
    # positive_pair is a pair with a low geDIG score
    positive_similarity = F.cosine_similarity(positive_pair[0], positive_pair[1])

    negative_similarities = [
        F.cosine_similarity(positive_pair[0], neg) for neg in negative_pairs
    ]

    # Compute InfoNCE loss
    # ...
```

#### 3. Hyperbolic Geometry: Natural Representation of Hierarchies
- **Role**: Represent the inherent hierarchical structure of knowledge (e.g., AI > NN > Transformer) efficiently as curvature in the space.
- **Implementation**:
  - Define the embedding space not as Euclidean, but as a hyperbolic space such as the Poincaré ball.
  - Use hyperbolic distance (Poincaré distance) for measuring relationships.
  - Structural improvements driven by `geDIG` can be formalized as reorganizing knowledge into a more efficient tree-like structure in hyperbolic space.

### Connection to Prior Notes

This approach integrates and extends past discussions in the `research` directory.

- **`gnn_in_vector_space_perspective`**: Extends the view of nodes as “entities” in space into a probabilistic distribution view.
- **`classical_to_quantum_gedig`**: Provides a path to concretize “quantum geDIG” via VAE and contrastive learning.
- **`hypothesis_vector_space_gedig`**: Formalizes the idea of defining a `geDIG` potential field and placing knowledge by gradient descent as a learning process.

### Future Outlook

If this theory of dynamic world-model construction is established, AI may acquire capabilities like:

1. **True contextual understanding**: Generate an on-the-fly, context-specific vector space optimized for the current dialogue.
2. **Self-growth**: Autonomously update and refine its worldview whenever new experiences are incorporated.
3. **Personalization**: Cultivate a user-specific world model from interaction history, becoming a deeper thinking partner.

This could serve as an important theoretical foundation for AI to transform from a static knowledge database into a **living, growing intellectual entity**.
