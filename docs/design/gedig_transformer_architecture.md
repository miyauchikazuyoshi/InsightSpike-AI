# geDIG-Transformer Architecture Design

**Version**: 0.1 (Draft)
**Date**: 2026-01-26
**Author**: Kazuyoshi Miyauchi
**Status**: Proposal

---

## Executive Summary

本設計書は、geDIG原理（F = ΔEPC - λΔIG）に基づくTransformerアーキテクチャを定義する。従来のTransformerが「全てを計算する」のに対し、geDIG-Transformerは「何を計算するか」「いつ計算を止めるか」を動的に決定する。

**Key Innovations**:
1. **AG Gate**: 入力の「驚き度」に基づくスパース化
2. **DG Gate**: F値に基づくEarly Exit / Update Rejection
3. **F-Loss**: 構造最適化を学習目標に組み込み
4. **Wake-Sleep Dynamics**: 推論中の動的構造更新

---

## 1. Motivation

### 1.1 現行Transformerの限界

| 問題 | 詳細 | 計算コスト |
|------|------|-----------|
| 全結合Attention | 全トークンペアを計算 | O(n²) |
| 固定深度 | 簡単な入力も全層通過 | O(L) layers always |
| 静的構造 | 推論中に構造不変 | No adaptation |
| 学習/推論分離 | 推論中に学習不可 | Separate phases |

### 1.2 geDIG原理による解決

```
F = ΔEPC - λΔIG

ΔEPC: 構造コストの変化（エッジ数、計算量）
ΔIG:  情報利得の変化（エントロピー減少、予測改善）
```

**原理**: F最小化 = 最小コストで最大の情報利得を得る構造を選択

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      geDIG-Transformer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: x ∈ ℝ^{n × d}                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Embedding Layer                       │   │
│  │            + Positional Encoding                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ╔═════════════════════════════════════════════════════════╗   │
│  ║              geDIG Block (×L layers)                    ║   │
│  ╠═════════════════════════════════════════════════════════╣   │
│  ║                                                         ║   │
│  ║  Input: h^{l-1}                                         ║   │
│  ║       │                                                 ║   │
│  ║       ▼                                                 ║   │
│  ║  ┌───────────────┐                                      ║   │
│  ║  │   AG Gate     │ → Compute surprise S(h)              ║   │
│  ║  │   (Sparse)    │ → Select active tokens: A ⊂ [1..n]   ║   │
│  ║  └───────┬───────┘                                      ║   │
│  ║          │                                              ║   │
│  ║          ▼                                              ║   │
│  ║  ┌───────────────┐                                      ║   │
│  ║  │   Sparse      │ → Attention only on A × A            ║   │
│  ║  │   Attention   │ → O(|A|²) instead of O(n²)           ║   │
│  ║  └───────┬───────┘                                      ║   │
│  ║          │                                              ║   │
│  ║          ▼                                              ║   │
│  ║  ┌───────────────┐                                      ║   │
│  ║  │   DG Gate     │ → Compute ΔF                         ║   │
│  ║  │   (Decision)  │ → Accept/Reject/Exit                 ║   │
│  ║  └───────┬───────┘                                      ║   │
│  ║          │                                              ║   │
│  ║          ├─── Exit? ──→ [Output Layer] ──→ Output       ║   │
│  ║          │                                              ║   │
│  ║          ▼                                              ║   │
│  ║  ┌───────────────┐                                      ║   │
│  ║  │     FFN       │ → Consolidated pathways              ║   │
│  ║  │  (Intuition)  │ → Fixed transformations              ║   │
│  ║  └───────┬───────┘                                      ║   │
│  ║          │                                              ║   │
│  ║          ▼                                              ║   │
│  ║       h^{l}                                             ║   │
│  ║                                                         ║   │
│  ╚═════════════════════════════════════════════════════════╝   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Output Layer                          │   │
│  │              (Task-specific head)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│    Output                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Attention Field Construction (Core Theory)

### 3.1 Attention as Graph Adjacency Matrix

**基本的対応**:

```
標準Transformer           geDIG Graph
─────────────────────────────────────────
Token i                   Node i
Attention(i,j)            Edge weight w_ij
Attention Matrix A        Adjacency Matrix W
Multi-Head                Multiple edge types
```

**数学的定義**:

標準Attention:
```
A_ij = softmax_j(q_i · k_j / √d_k)

where:
  q_i = W_Q · h_i    (query)
  k_j = W_K · h_j    (key)
  v_j = W_V · h_j    (value)
```

geDIG的解釈:
```
A_ij = P(edge i→j exists | current state)
     = P(token j is relevant to token i)
```

**Attention Matrix = 確率的隣接行列**

---

### 3.2 Edge Processing Cost (EPC) in Attention

**定義**: Attention行列の維持コスト

```
EPC(A) = Σ_ij c(A_ij)

where c(A_ij) = cost function for edge (i,j)
```

**具体的なコスト関数の選択肢**:

| コスト関数 | 式 | 意味 |
|-----------|---|------|
| **L0 (Count)** | `Σ_ij 𝟙[A_ij > ε]` | 非ゼロエッジ数 |
| **L1 (Sparse)** | `Σ_ij |A_ij|` | エッジ重みの総和 |
| **Entropy** | `-Σ_ij A_ij log A_ij` | Attention の集中度 |
| **Compute** | `|Active_i| × |Active_j|` | 実計算量 |

**推奨**: L1 + Compute の組み合わせ

```python
def compute_epc(attention_matrix, active_mask):
    """
    EPC = sparsity_cost + compute_cost
    """
    # L1 sparsity
    sparsity_cost = attention_matrix.abs().sum()

    # Compute cost (number of active pairs)
    n_active = active_mask.sum()
    compute_cost = n_active ** 2

    return alpha * sparsity_cost + beta * compute_cost
```

---

### 3.3 Information Gain (IG) in Attention

**定義**: Attention適用による情報利得

```
IG(A) = H(h_before) - H(h_after | A)
      = Entropy減少 + 予測精度向上
```

**具体的な計算方法**:

#### 3.3.1 Entropy-based IG

```python
def compute_entropy(h):
    """
    Hidden stateのエントロピー推定

    高次元での直接計算は困難なため、プロキシを使用:
    - 分散ベース: 高分散 = 高エントロピー
    - Rank-based: 表現の有効次元数
    """
    # Option 1: Variance-based proxy
    variance = h.var(dim=-1)  # [batch, seq_len]
    entropy = torch.log(variance + 1e-8)

    # Option 2: Effective rank (より正確だが高コスト)
    # U, S, V = torch.svd(h)
    # p = S / S.sum()
    # entropy = -(p * torch.log(p + 1e-8)).sum()

    return entropy.mean()

def compute_ig(h_before, h_after):
    """Information Gain from attention"""
    H_before = compute_entropy(h_before)
    H_after = compute_entropy(h_after)
    return H_before - H_after  # Positive = entropy decreased = good
```

#### 3.3.2 Prediction-based IG

```python
def compute_ig_predictive(h_before, h_after, predictor):
    """
    予測精度の改善としてのIG

    「良いattention」= 次の状態をより正確に予測できる
    """
    # Predict next state from before
    pred_before = predictor(h_before)

    # Prediction error
    error_before = F.mse_loss(pred_before, h_after.detach())

    # IG = error reduction
    # (実際には次層の入力との比較が必要)
    return -error_before  # Lower error = higher IG
```

---

### 3.4 F-Minimizing Attention Field Construction

**目標**: F = EPC - λIG を最小化する Attention Matrix を構築

#### 3.4.1 標準Attentionとの関係

標準Transformer:
```
A = softmax(QK^T / √d)   # F を考慮しない
```

geDIG-Transformer:
```
A = argmin_A [EPC(A) - λIG(A)]   # F最小化
  subject to: A_ij ≥ 0, Σ_j A_ij = 1
```

**問題**: argmin は微分不可能

**解決**: F を正則化項として追加

```
A = softmax(QK^T / √d - γ · F_penalty)
```

#### 3.4.2 F-Aware Attention Score

```python
class FMinimizingAttention(nn.Module):
    def __init__(self, d_model, n_heads, lambda_ig=1.0, gamma=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.lambda_ig = lambda_ig
        self.gamma = gamma

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # F-penalty predictor
        self.f_penalty = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def compute_edge_f_penalty(self, h_i, h_j):
        """
        各エッジ(i,j)のF penaltyを予測

        高いpenalty = このエッジは構造的に無駄
        """
        # Concatenate source and target
        edge_repr = torch.cat([h_i, h_j], dim=-1)
        penalty = self.f_penalty(edge_repr)
        return penalty.squeeze(-1)

    def forward(self, h, attention_mask=None):
        batch, seq_len, d_model = h.shape

        # Standard Q, K, V
        Q = self.W_q(h)  # [batch, seq, d_model]
        K = self.W_k(h)
        V = self.W_v(h)

        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch, seq, seq]

        # F-penalty for each potential edge
        # Efficient: use outer product approximation
        h_expanded_i = h.unsqueeze(2).expand(-1, -1, seq_len, -1)
        h_expanded_j = h.unsqueeze(1).expand(-1, seq_len, -1, -1)
        edge_features = torch.cat([h_expanded_i, h_expanded_j], dim=-1)
        f_penalties = self.f_penalty(edge_features).squeeze(-1)
        # [batch, seq, seq]

        # F-aware scores: penalize high-F edges
        f_aware_scores = scores - self.gamma * f_penalties

        # Apply mask if provided
        if attention_mask is not None:
            f_aware_scores = f_aware_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )

        # Softmax
        attention = F.softmax(f_aware_scores, dim=-1)

        # Apply attention
        output = torch.matmul(attention, V)
        output = self.W_o(output)

        return output, attention, f_penalties
```

---

### 3.5 AG Gate: When to Compute Attention

**問題**: 全トークンペアのF-penaltyを計算 → O(n²) で意味がない

**解決**: AG Gateで「計算すべきトークン」を事前選択

#### 3.5.1 AG Gate の役割

```
Input tokens: [t_1, t_2, ..., t_n]
                    ↓
              AG Gate (O(n))
                    ↓
Active tokens: [t_3, t_7, t_12, ...]  (k << n)
                    ↓
         Attention only on Active (O(k²))
```

#### 3.5.2 Surprise Score の計算

**「驚き」の定義**: 予測からの逸脱

```python
class AGGate(nn.Module):
    def __init__(self, d_model, context_size=5):
        super().__init__()
        self.context_size = context_size

        # Local context predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model * context_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def compute_local_context(self, h):
        """
        各トークンの局所コンテキストを計算
        """
        batch, seq_len, d_model = h.shape

        # Pad for context window
        pad_size = self.context_size // 2
        h_padded = F.pad(h, (0, 0, pad_size, pad_size), mode='constant', value=0)

        # Gather context for each position
        contexts = []
        for i in range(seq_len):
            ctx = h_padded[:, i:i+self.context_size, :]
            ctx = ctx.reshape(batch, -1)
            contexts.append(ctx)

        return torch.stack(contexts, dim=1)  # [batch, seq_len, d_model * context_size]

    def forward(self, h):
        """
        Surprise score = ||h_actual - h_predicted||²

        高い驚き = 局所コンテキストから予測できない = 処理すべき
        """
        batch, seq_len, d_model = h.shape

        # Compute local context
        contexts = self.compute_local_context(h)  # [batch, seq, d*ctx]

        # Predict each token from its context
        h_predicted = self.predictor(contexts)  # [batch, seq, d]

        # Surprise = prediction error
        surprise = ((h - h_predicted) ** 2).mean(dim=-1)  # [batch, seq]

        # Normalize
        surprise = (surprise - surprise.mean()) / (surprise.std() + 1e-8)

        # Active mask
        active_mask = surprise > self.threshold

        return active_mask, surprise
```

#### 3.5.3 Surprise と Information Gain の関係

**理論的接続**:

```
高い Surprise ≈ 高い Potential IG

理由:
- Surprise = 局所コンテキストから予測できない
- = 新しい情報を持っている可能性が高い
- = Attentionで処理する価値がある
- = IG が高くなる可能性
```

**ただし**: Surprise が高くても IG が低いケースもある（ノイズ）

→ DG Gate で事後的にフィルタリング

---

### 3.6 Attention Field の段階的構築

#### 3.6.1 Phase 1: Candidate Generation (AG)

```
Step 1: 全トークンの Surprise を計算 (O(n))
Step 2: Active Set A = {i : Surprise(i) > θ} を決定
Step 3: 候補エッジ集合 E_candidate = A × A
```

#### 3.6.2 Phase 2: Edge Scoring

```
Step 4: 候補エッジのみ Attention Score を計算 (O(|A|²))
Step 5: 各エッジの F-penalty を計算
Step 6: F-aware Score = Standard Score - γ × F-penalty
```

#### 3.6.3 Phase 3: Edge Selection (DG)

```
Step 7: Softmax で確率化
Step 8: 低スコアエッジを刈り込み (optional top-k)
Step 9: 最終 Attention Matrix を確定
```

#### 3.6.4 Full Pipeline

```python
class GeDIGAttentionField(nn.Module):
    def __init__(self, d_model, n_heads, lambda_ig=1.0):
        super().__init__()
        self.ag_gate = AGGate(d_model)
        self.attention = FMinimizingAttention(d_model, n_heads, lambda_ig)
        self.dg_gate = DGGate(d_model, lambda_ig)

    def forward(self, h):
        """
        geDIG Attention Field Construction Pipeline
        """
        batch, seq_len, d_model = h.shape

        # === Phase 1: AG Gate ===
        active_mask, surprise = self.ag_gate(h)
        # active_mask: [batch, seq_len]

        # === Phase 2: Sparse Attention with F-penalty ===
        # Create attention mask from active tokens
        attn_mask = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        # attn_mask: [batch, seq, seq] - True where both i and j are active

        # Compute attention only where needed
        h_attended, attention_weights, f_penalties = self.attention(
            h, attention_mask=attn_mask.float()
        )

        # === Phase 3: DG Gate ===
        h_out, decision, delta_F = self.dg_gate(h, h_attended, active_mask)

        return h_out, {
            'active_mask': active_mask,
            'surprise': surprise,
            'attention_weights': attention_weights,
            'f_penalties': f_penalties,
            'decision': decision,
            'delta_F': delta_F
        }
```

---

### 3.7 Attention Field の視覚化と解釈

#### 3.7.1 構造の可視化

```python
def visualize_attention_field(metrics, tokens):
    """
    geDIG Attention Field の可視化
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (a) Surprise scores
    ax = axes[0, 0]
    ax.bar(range(len(tokens)), metrics['surprise'][0].cpu())
    ax.axhline(y=0, color='r', linestyle='--', label='threshold')
    ax.set_title('Surprise Score (AG Gate)')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)

    # (b) Active mask
    ax = axes[0, 1]
    active = metrics['active_mask'][0].cpu().float()
    ax.imshow(active.unsqueeze(0), aspect='auto', cmap='Greens')
    ax.set_title('Active Tokens')

    # (c) Attention weights
    ax = axes[1, 0]
    attn = metrics['attention_weights'][0].cpu()
    im = ax.imshow(attn, cmap='Blues')
    ax.set_title('Attention Weights')
    plt.colorbar(im, ax=ax)

    # (d) F-penalties
    ax = axes[1, 1]
    f_pen = metrics['f_penalties'][0].cpu()
    im = ax.imshow(f_pen, cmap='Reds')
    ax.set_title('F-Penalties (Red = High Cost Edge)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
```

#### 3.7.2 解釈

| 可視化 | 意味 | 良い状態 |
|--------|------|---------|
| Surprise | 各トークンの「新規性」 | 重要トークンで高い |
| Active Mask | 処理対象 | 内容語がActive、機能語がInactive |
| Attention | 情報フロー | 意味的に関連するペアで高い |
| F-Penalty | エッジコスト | 無関係ペアで高い |

---

### 3.8 理論的保証

#### 3.8.1 F最小化の収束

**命題**: 適切な学習率で、geDIG Attention FieldはF最小点に収束する

**証明スケッチ**:
1. F = EPC - λIG は下に有界（EPCは非負、IGは有界）
2. 各ステップでΔF < 0 となるエッジのみ採用
3. → F は単調減少
4. → 有界単調列は収束

**注意**: 局所最適に陥る可能性あり → Temperature schedulingで対応

#### 3.8.2 計算量

| 操作 | 標準Transformer | geDIG-Transformer |
|------|-----------------|-------------------|
| Surprise計算 | - | O(n) |
| Attention Score | O(n²) | O(k²), k = |A| |
| F-penalty | - | O(k²) |
| 総計算量 | O(n²) | O(n + k²) |

k = αn (α < 1) のとき:
```
O(n + α²n²) ≈ O(α²n²) << O(n²)  when α << 1
```

---

### 3.9 Multi-Head Attention as Multiple Edge Types

#### 3.9.1 Head = Edge Type の対応

```
標準MHA                      geDIG Graph
────────────────────────────────────────────────
Head 1: syntactic attention   Edge type 1: 構文関係
Head 2: semantic attention    Edge type 2: 意味関係
Head 3: positional attention  Edge type 3: 位置関係
...                           ...
```

**グラフ理論的解釈**: Multi-relational Graph（複数のエッジタイプを持つグラフ）

#### 3.9.2 Head-wise AG/DG Gating

```python
class MultiHeadGeDIGAttention(nn.Module):
    def __init__(self, d_model, n_heads, lambda_ig=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Per-head AG gates (各ヘッドが異なるSurprise基準を持つ)
        self.ag_gates = nn.ModuleList([
            AGGate(self.d_k) for _ in range(n_heads)
        ])

        # Per-head F-penalty predictors
        self.f_penalties = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_k * 2, self.d_k),
                nn.GELU(),
                nn.Linear(self.d_k, 1)
            ) for _ in range(n_heads)
        ])

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, h):
        batch, seq_len, d_model = h.shape

        # Project to Q, K, V
        Q = self.W_q(h).view(batch, seq_len, self.n_heads, self.d_k)
        K = self.W_k(h).view(batch, seq_len, self.n_heads, self.d_k)
        V = self.W_v(h).view(batch, seq_len, self.n_heads, self.d_k)

        # Transpose for head-first processing
        Q = Q.transpose(1, 2)  # [batch, heads, seq, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        all_head_outputs = []
        all_head_metrics = []

        for head_idx in range(self.n_heads):
            q = Q[:, head_idx]  # [batch, seq, d_k]
            k = K[:, head_idx]
            v = V[:, head_idx]

            # Head-specific AG Gate
            active_mask, surprise = self.ag_gates[head_idx](q)

            # Standard attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Head-specific F-penalty
            # (simplified: use q,k instead of full h for efficiency)
            q_exp = q.unsqueeze(2).expand(-1, -1, seq_len, -1)
            k_exp = k.unsqueeze(1).expand(-1, seq_len, -1, -1)
            edge_feat = torch.cat([q_exp, k_exp], dim=-1)
            f_pen = self.f_penalties[head_idx](edge_feat).squeeze(-1)

            # F-aware scores
            f_aware_scores = scores - 0.1 * f_pen

            # Apply active mask
            attn_mask = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
            f_aware_scores = f_aware_scores.masked_fill(~attn_mask, float('-inf'))

            # Softmax and apply
            attn_weights = F.softmax(f_aware_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            head_output = torch.matmul(attn_weights, v)
            all_head_outputs.append(head_output)

            all_head_metrics.append({
                'active_mask': active_mask,
                'surprise': surprise,
                'attention': attn_weights,
                'f_penalty': f_pen
            })

        # Concatenate heads
        output = torch.cat(all_head_outputs, dim=-1)  # [batch, seq, d_model]
        output = self.W_o(output)

        return output, all_head_metrics
```

#### 3.9.3 Head Specialization の期待

学習が進むと、各ヘッドが異なる役割に特化することが期待される：

| Head | Surprise基準 | 捕捉する関係 |
|------|-------------|-------------|
| Head 1 | 局所コンテキスト逸脱 | 構文的依存関係 |
| Head 2 | 意味ベクトル距離 | 意味的類似性 |
| Head 3 | 位置パターン逸脱 | 長距離依存 |
| Head 4 | 稀少パターン検出 | 例外・特殊ケース |

**検証方法**: 学習後に各ヘッドのAttentionパターンを分析

---

### 3.10 Dynamic Edge Creation/Deletion

#### 3.10.1 推論中のグラフ動的変化

geDIG-Transformerでは、推論中にグラフ構造（Attention Pattern）が動的に変化：

```
Layer 1: 初期グラフ（AG発火トークン間の仮接続）
    ↓ DG判定
Layer 2: 有効エッジのみ残存 + 新規AG発火
    ↓ DG判定
Layer 3: さらに精緻化
    ...
    ↓
Final: 最適化されたAttention構造
```

#### 3.10.2 Wake-Sleep Dynamics（推論中学習）

**Phase 1 (Wake / Forward Pass)**:
```python
# 通常の推論
with torch.no_grad():
    output, metrics = model(input_ids)

# メトリクスを記録
wake_memory.append({
    'input': input_ids,
    'attention_patterns': metrics['attention_weights'],
    'f_values': metrics['delta_F']
})
```

**Phase 2 (Sleep / Consolidation)** - Optional:
```python
# 頻繁に使われるパターンをFFNに固定化
if len(wake_memory) > consolidation_threshold:
    frequent_patterns = analyze_patterns(wake_memory)

    # FFN重みを更新して「直感」として焼き付け
    model.consolidate(frequent_patterns)

    wake_memory.clear()
```

#### 3.10.3 Consolidation: Attention → FFN

**概念**: 頻繁に使われるAttentionパターンを、FFNの重みとして「固定化」

```python
class ConsolidatingGeDIGBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadGeDIGAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.pattern_memory = []

    def consolidate(self, patterns, learning_rate=0.01):
        """
        頻繁なAttentionパターンをFFNに蒸留

        理論: Attention(x) ≈ FFN(x) for frequent patterns
        """
        for pattern in patterns:
            # Pattern: (input_repr, attention_output)
            x, y = pattern['input'], pattern['output']

            # FFNをこのパターンにフィット
            ffn_out = self.ffn(x)
            loss = F.mse_loss(ffn_out, y)

            # 軽微な重み更新
            loss.backward()
            with torch.no_grad():
                for param in self.ffn.parameters():
                    param -= learning_rate * param.grad
                    param.grad.zero_()

    def forward(self, h, use_intuition=True):
        # Try FFN first (intuition / System 1)
        if use_intuition:
            ffn_out = self.ffn(h)
            confidence = self.estimate_confidence(h, ffn_out)

            if confidence > self.confidence_threshold:
                # FFN output is reliable → skip attention (fast path)
                return ffn_out, {'used_intuition': True}

        # Fall back to full attention (deliberation / System 2)
        attn_out, metrics = self.attention(h)
        return attn_out, {'used_intuition': False, **metrics}
```

**System 1 / System 2 の実現**:
- **System 1 (直感)**: FFN経由の高速パス（固定化済みパターン）
- **System 2 (熟考)**: Attention経由の遅いパス（新規パターン）

---

### 3.11 Attention Field と F の関係まとめ

```
┌─────────────────────────────────────────────────────────────┐
│                    F = EPC - λIG                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EPC (Edge Processing Cost)                                 │
│  ├─ |Active Tokens|²  ... 計算するペア数                    │
│  ├─ Σ|A_ij|          ... Attention重みの総和                │
│  └─ Head数 × 上記    ... Multi-headのコスト                 │
│                                                             │
│  IG (Information Gain)                                      │
│  ├─ H(before) - H(after)  ... エントロピー減少              │
│  ├─ Prediction improvement ... 予測精度向上                 │
│  └─ Task performance      ... タスク性能への寄与            │
│                                                             │
│  最適Attention Field:                                        │
│  ├─ 最小限のActiveトークン (低EPC)                          │
│  ├─ 意味的に重要なエッジのみ (低EPC, 高IG)                  │
│  ├─ Head間で役割分担 (効率的なIG獲得)                       │
│  └─ 頻出パターンはFFN化 (EPC→0, IG維持)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.12 Similarity Measure Selection for Entropy Calculation

#### 3.12.1 問題設定

geDIGのエントロピー計算は「類似度」に基づく：
```
weights = similarities between nodes/edges
probabilities = normalize(weights)
H = -Σ p log p
```

**問いかけ**: どの類似度尺度を採用すべきか？

#### 3.12.2 コサイン類似度の限界

**標準的な選択**: コサイン類似度
```python
sim(a, b) = (a · b) / (||a|| × ||b||)
```

**前提条件**:
- ベクトル空間が「完成」している
- 方向（角度）が意味的関係を表現している
- 正規化が適切に行われている

**問題**: 以下の状況ではコサイン類似度は機能しない可能性がある：

| 状況 | 問題 |
|------|------|
| 学習初期 | 空間がランダムに近く、方向に意味がない |
| 動的更新中 | 空間が常に変化し、方向の意味が不安定 |
| 未学習領域 | 訓練データ外の入力では方向が信頼できない |
| スパース活性化 | 多くの次元がゼロで、角度計算が不安定 |

**仮説**: 未完成なベクトル空間では、「方向」より「距離」や「大きさ」の方が信頼できる情報である。

#### 3.12.3 代替案の検討

**Option A: L2距離ベース**
```python
sim(a, b) = 1 / (1 + ||a - b||)
```
- ✅ 空間内の「位置」を直接使用
- ✅ 正規化不要
- ⚠️ スケール依存（次元数に影響される）

**Option B: ガウシアンカーネル (RBF)**
```python
sim(a, b) = exp(-||a - b||² / (2σ²))
```
- ✅ 距離ベースで未完成空間に強い
- ✅ σで局所性を制御可能
- ✅ 確率的解釈が可能
- ✅ 微分可能で学習に適する
- ✅ カーネル法との理論的接続

**Option C: 内積（非正規化）**
```python
sim(a, b) = a · b
```
- ✅ 方向と大きさの両方を考慮
- ⚠️ スケールが発散しやすい
- ⚠️ 負の値を取りうる

**Option D: ノルム比**
```python
sim(a, b) = min(||a||, ||b||) / max(||a||, ||b||)
```
- ✅ 「活性度」の類似性
- ❌ 方向情報を完全に無視

**Option E: L2距離 + ノルム比 の複合**
```python
sim(a, b) = α × (1/(1+dist)) + (1-α) × norm_ratio
```
- ✅ 位置と活性度の両方
- ⚠️ ハイパーパラメータ(α)が増える

#### 3.12.4 決定: ガウシアンカーネル（推奨）

**Primary Choice: Gaussian Kernel (RBF)**

```python
def gedig_similarity(a: Tensor, b: Tensor, sigma: float = None) -> Tensor:
    """
    geDIG標準の類似度関数

    Args:
        a: [batch, n, d] or [n, d]
        b: [batch, m, d] or [m, d]
        sigma: bandwidth parameter (adaptive if None)

    Returns:
        similarity: [batch, n, m] or [n, m]
    """
    # Ensure 3D
    if a.dim() == 2:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Pairwise squared L2 distance
    # ||a - b||² = ||a||² + ||b||² - 2(a·b)
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)  # [batch, n, 1]
    b_sq = (b ** 2).sum(dim=-1, keepdim=True)  # [batch, m, 1]
    ab = torch.matmul(a, b.transpose(-2, -1))  # [batch, n, m]

    dist_sq = a_sq + b_sq.transpose(-2, -1) - 2 * ab  # [batch, n, m]
    dist_sq = torch.clamp(dist_sq, min=0.0)  # Numerical stability

    # Adaptive sigma: median heuristic
    if sigma is None:
        sigma = torch.sqrt(dist_sq.median() + 1e-8)

    # Gaussian kernel
    similarity = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))

    if squeeze_output:
        similarity = similarity.squeeze(0)

    return similarity
```

**選定理由**:

1. **未完成空間への堅牢性**: 距離ベースなので、方向が意味を持たない段階でも機能
2. **適応的スケーリング**: median heuristicでσを自動調整
3. **確率的解釈**: カーネル密度推定との接続
4. **理論的基盤**: 再生核ヒルベルト空間(RKHS)理論
5. **実績**: SVMやGaussian Processで広く使用され、堅牢性が実証済み

#### 3.12.5 Sentence-BERTとの関連

**Sentence-BERT (SBERT) の学習過程**:

```
Input: (sentence_A, sentence_B, label)
       label ∈ {similar, dissimilar}

Objective:
  similar pairs → close in embedding space
  dissimilar pairs → far in embedding space

Loss: Contrastive Loss or Triplet Loss
  L = Σ max(0, ||f(A) - f(P)||² - ||f(A) - f(N)||² + margin)
```

**SBERTとgeDIG-Transformerの類似点**:

| SBERT | geDIG-Transformer |
|-------|-------------------|
| 類似文を近くに配置 | 関連トークンを近くに配置 |
| Contrastive Learning | F最小化 |
| 埋め込み空間の構築 | Attention構造の構築 |
| L2距離で評価 | ガウシアンカーネルで評価 |

**SBERT学習の知見の活用**:

1. **Warmup期間**: 最初は類似度学習を緩やかに（σを大きく）
2. **Hard Negative Mining**: F削減に寄与しないエッジを重点的に学習
3. **In-batch Negatives**: バッチ内の他サンプルを負例として活用
4. **温度パラメータ**: σを学習可能にして適応

```python
class AdaptiveGaussianSimilarity(nn.Module):
    """
    SBERTの知見を取り入れた適応的ガウシアン類似度
    """
    def __init__(self, d_model, init_sigma=1.0, learnable=True):
        super().__init__()
        self.d_model = d_model

        if learnable:
            # Learnable log-sigma (for numerical stability)
            self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))
        else:
            self.register_buffer('log_sigma', torch.tensor(math.log(init_sigma)))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def forward(self, a, b, temperature=1.0):
        """
        Args:
            a, b: input tensors
            temperature: scaling factor (higher = softer similarities)
        """
        dist_sq = self._pairwise_dist_sq(a, b)

        # Effective sigma with temperature
        effective_sigma = self.sigma * temperature

        similarity = torch.exp(-dist_sq / (2 * effective_sigma ** 2 + 1e-8))

        return similarity

    def _pairwise_dist_sq(self, a, b):
        # Efficient pairwise squared distance
        a_sq = (a ** 2).sum(dim=-1, keepdim=True)
        b_sq = (b ** 2).sum(dim=-1, keepdim=True)
        ab = torch.matmul(a, b.transpose(-2, -1))
        return torch.clamp(a_sq + b_sq.transpose(-2, -1) - 2 * ab, min=0.0)
```

#### 3.12.6 学習スケジュールとの連携

**SBERTスタイルのカリキュラム**:

| Phase | σ (bandwidth) | 効果 |
|-------|---------------|------|
| 初期 (Epoch 1-5) | 大きい (σ=10) | 粗い構造を学習 |
| 中期 (Epoch 6-15) | 中程度 (σ=1) | 細かい構造を学習 |
| 後期 (Epoch 16+) | 適応的 | データに応じて自動調整 |

```python
def get_sigma_schedule(epoch, max_epochs=30):
    """
    Curriculum-based sigma scheduling
    """
    if epoch < 5:
        # Warmup: large sigma (soft similarities)
        return 10.0
    elif epoch < 15:
        # Refinement: medium sigma
        progress = (epoch - 5) / 10
        return 10.0 * (1 - progress) + 1.0 * progress
    else:
        # Fine-tuning: adaptive (return None for median heuristic)
        return None
```

#### 3.12.7 実験計画: 類似度尺度の比較

将来的に以下の比較実験を実施予定：

| Experiment | Similarity | Task | Metric |
|------------|-----------|------|--------|
| Exp-A | Cosine | GLUE | Accuracy |
| Exp-B | Gaussian (fixed σ) | GLUE | Accuracy |
| Exp-C | Gaussian (adaptive σ) | GLUE | Accuracy |
| Exp-D | Gaussian (learnable σ) | GLUE | Accuracy |
| Exp-E | L2 + Norm Ratio | GLUE | Accuracy |

**Ablation questions**:
1. 未完成空間（学習初期）での各尺度の安定性
2. σの設定が性能に与える影響
3. Learnable σの収束挙動

---

## 4. Component Specifications

### 3.1 AG Gate (Ambiguity Gate)

**目的**: 処理すべきトークンを選択（注意の配分）

#### 3.1.1 数学的定義

```
S(h_i) = ||h_i - μ(h)||² / σ²(h)    # Surprise score

A = {i : S(h_i) > θ_AG}              # Active set

θ_AG = adaptive threshold (learnable or percentile-based)
```

#### 3.1.2 実装

```python
class AGGate(nn.Module):
    def __init__(self, d_model, method='learned'):
        super().__init__()
        self.method = method

        if method == 'learned':
            # Learned surprise predictor
            self.predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            )
            self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, h, return_scores=False):
        """
        Args:
            h: [batch, seq_len, d_model]
        Returns:
            active_mask: [batch, seq_len] boolean
            surprise_scores: [batch, seq_len] (optional)
        """
        if self.method == 'learned':
            scores = self.predictor(h).squeeze(-1)  # [batch, seq_len]
            scores = torch.sigmoid(scores)
        elif self.method == 'statistical':
            mean = h.mean(dim=1, keepdim=True)
            std = h.std(dim=1, keepdim=True) + 1e-6
            scores = ((h - mean) ** 2).mean(dim=-1) / (std.mean(dim=-1) ** 2)

        # Adaptive threshold (top-k alternative)
        if self.training:
            # Soft mask during training (Gumbel-softmax style)
            active_mask = torch.sigmoid((scores - self.threshold) * 10)
        else:
            # Hard mask during inference
            active_mask = scores > self.threshold

        if return_scores:
            return active_mask, scores
        return active_mask
```

#### 3.1.3 設計選択

| オプション | Pros | Cons |
|-----------|------|------|
| **Learned threshold** | タスク適応的 | 学習が不安定になりうる |
| **Percentile (top-k%)** | 安定 | 固定スパース率 |
| **Statistical (z-score)** | 解釈可能 | タスク非依存 |

**推奨**: 初期実験は Percentile (top-50%)、安定後 Learned に移行

---

### 3.2 Sparse Attention

**目的**: Activeトークン間のみAttention計算

#### 3.2.1 数学的定義

```
Q_A = W_Q · h_A,  K_A = W_K · h_A,  V_A = W_V · h_A

Attn_sparse = softmax(Q_A · K_A^T / √d_k) · V_A

# Activeでないトークンは前層の値を保持
h'_i = h^{l-1}_i  if i ∉ A
h'_i = Attn_sparse_i  if i ∈ A
```

#### 3.2.2 実装

```python
class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, active_mask):
        """
        Args:
            h: [batch, seq_len, d_model]
            active_mask: [batch, seq_len] boolean or soft weights
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = h.shape

        # Compute Q, K, V for all (needed for cross-attention to inactive)
        Q = self.W_q(h).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(h).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(h).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Create sparse attention mask
        # Active tokens attend to all, but only active tokens are updated
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        attn_output = self.W_o(attn_output)

        # Apply active mask: only update active tokens
        if active_mask.dtype == torch.bool:
            output = h.clone()
            output[active_mask] = attn_output[active_mask]
        else:
            # Soft mask: interpolate
            active_mask = active_mask.unsqueeze(-1)
            output = active_mask * attn_output + (1 - active_mask) * h

        return output
```

#### 3.2.3 計算量削減

| Activeトークン率 | 計算量 | 削減率 |
|-----------------|--------|--------|
| 100% (従来) | O(n²) | 0% |
| 50% | O(0.25n²) | 75% |
| 25% | O(0.0625n²) | 94% |
| 10% | O(0.01n²) | 99% |

---

### 3.3 DG Gate (Decision Gate)

**目的**: 構造更新の承認/棄却、Early Exit判定

#### 3.3.1 数学的定義

```
F(h) = EPC(h) - λ · IG(h)

EPC(h) = ||W||_1 + α · ActiveRatio    # 構造コスト
IG(h)  = H(h^{l-1}) - H(h^l)          # 情報利得（エントロピー減少）

ΔF = F(h^l) - F(h^{l-1})

Decision:
  - ΔF < θ_exit  → Early Exit (十分収束)
  - ΔF >= 0      → Reject update (改善なし)
  - otherwise    → Accept update (改善あり)
```

#### 3.3.2 実装

```python
class DGGate(nn.Module):
    def __init__(self, d_model, lambda_ig=1.0):
        super().__init__()
        self.lambda_ig = lambda_ig

        # Learnable components
        self.epc_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        self.exit_threshold = nn.Parameter(torch.tensor(-0.1))
        self.reject_threshold = nn.Parameter(torch.tensor(0.0))

    def compute_entropy(self, h):
        """Estimate entropy of hidden states"""
        # Use variance as proxy for entropy
        return h.var(dim=-1).mean(dim=-1)  # [batch]

    def compute_epc(self, h, active_ratio):
        """Compute Edge Processing Cost"""
        base_epc = self.epc_estimator(h.mean(dim=1)).squeeze(-1)  # [batch]
        return base_epc + active_ratio

    def compute_F(self, h, active_ratio):
        """Compute Free Energy"""
        epc = self.compute_epc(h, active_ratio)
        entropy = self.compute_entropy(h)
        return epc - self.lambda_ig * entropy

    def forward(self, h_before, h_after, active_ratio):
        """
        Args:
            h_before: [batch, seq_len, d_model] - before attention
            h_after: [batch, seq_len, d_model] - after attention
            active_ratio: float - proportion of active tokens
        Returns:
            output: [batch, seq_len, d_model]
            decision: 'exit', 'reject', or 'accept'
            delta_F: [batch] - change in free energy
        """
        F_before = self.compute_F(h_before, active_ratio)
        F_after = self.compute_F(h_after, active_ratio)
        delta_F = F_after - F_before

        # Batch-level decision (simplification; could be per-sample)
        mean_delta_F = delta_F.mean()

        if not self.training:
            if mean_delta_F < self.exit_threshold:
                return h_after, 'exit', delta_F
            elif mean_delta_F >= self.reject_threshold:
                return h_before, 'reject', delta_F
            else:
                return h_after, 'accept', delta_F
        else:
            # During training, always accept but return delta_F for loss
            return h_after, 'accept', delta_F
```

---

### 3.4 geDIG Block (統合)

```python
class GeDIGBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, lambda_ig=1.0):
        super().__init__()

        # Gates
        self.ag_gate = AGGate(d_model, method='learned')
        self.dg_gate = DGGate(d_model, lambda_ig=lambda_ig)

        # Core components
        self.attention = SparseAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h, return_metrics=False):
        """
        Args:
            h: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            should_exit: bool
            metrics: dict (optional)
        """
        metrics = {}

        # 1. AG Gate: Select active tokens
        active_mask, surprise_scores = self.ag_gate(h, return_scores=True)
        active_ratio = active_mask.float().mean().item()
        metrics['active_ratio'] = active_ratio
        metrics['surprise_scores'] = surprise_scores

        # 2. Sparse Attention
        h_normed = self.norm1(h)
        h_attn = self.attention(h_normed, active_mask)
        h_attn = h + h_attn  # Residual

        # 3. DG Gate: Decide accept/reject/exit
        h_out, decision, delta_F = self.dg_gate(h, h_attn, active_ratio)
        metrics['decision'] = decision
        metrics['delta_F'] = delta_F

        if decision == 'exit':
            if return_metrics:
                return h_out, True, metrics
            return h_out, True

        # 4. FFN (only if not exiting)
        h_normed = self.norm2(h_out)
        h_ffn = self.ffn(h_normed)
        h_out = h_out + h_ffn  # Residual

        if return_metrics:
            return h_out, False, metrics
        return h_out, False
```

---

### 3.5 Full Model

```python
class GeDIGTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        lambda_ig=1.0
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            GeDIGBlock(d_model, n_heads, d_ff, dropout, lambda_ig)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, return_metrics=False):
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
            metrics: dict (optional)
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.embedding(input_ids) + self.pos_encoding(positions)

        # Process through blocks
        all_metrics = []
        actual_layers = 0

        for i, block in enumerate(self.blocks):
            h, should_exit, metrics = block(h, return_metrics=True)
            actual_layers += 1
            metrics['layer'] = i
            all_metrics.append(metrics)

            if should_exit and not self.training:
                break

        # Output
        h = self.final_norm(h)
        logits = self.output_head(h)

        if return_metrics:
            return logits, {
                'actual_layers': actual_layers,
                'total_layers': len(self.blocks),
                'layer_metrics': all_metrics
            }
        return logits
```

---

## 4. Training Procedure

### 4.1 Loss Function

```python
def gedig_loss(logits, targets, metrics, alpha=0.1, beta=0.01):
    """
    Combined loss: Task loss + F-regularization + Sparsity bonus
    """
    # 1. Task loss (standard cross-entropy)
    task_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100
    )

    # 2. F-regularization: encourage F decrease
    delta_F_total = sum(
        m['delta_F'].mean() for m in metrics['layer_metrics']
    )
    f_loss = F.relu(delta_F_total)  # Penalize F increase

    # 3. Sparsity bonus: reward using fewer active tokens
    avg_active_ratio = sum(
        m['active_ratio'] for m in metrics['layer_metrics']
    ) / len(metrics['layer_metrics'])
    sparsity_loss = avg_active_ratio  # Lower is better

    # 4. Early exit bonus: reward using fewer layers
    layer_ratio = metrics['actual_layers'] / metrics['total_layers']
    exit_loss = layer_ratio  # Lower is better

    total_loss = task_loss + alpha * f_loss + beta * (sparsity_loss + exit_loss)

    return total_loss, {
        'task_loss': task_loss.item(),
        'f_loss': f_loss.item(),
        'sparsity_loss': sparsity_loss,
        'exit_loss': exit_loss,
        'avg_active_ratio': avg_active_ratio,
        'actual_layers': metrics['actual_layers']
    }
```

### 4.2 Training Schedule

| Phase | Epochs | Focus | α (F-reg) | β (sparsity) |
|-------|--------|-------|-----------|--------------|
| 1. Warmup | 1-5 | Task learning | 0.0 | 0.0 |
| 2. F-introduction | 6-15 | Introduce F-loss | 0.01→0.1 | 0.0 |
| 3. Sparsity | 16-30 | Add sparsity | 0.1 | 0.01→0.1 |
| 4. Fine-tune | 31+ | Balance all | 0.1 | 0.1 |

### 4.3 Curriculum

```python
class GeDIGTrainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def get_loss_weights(self):
        """Curriculum-based loss weights"""
        if self.epoch < 5:
            return {'alpha': 0.0, 'beta': 0.0}
        elif self.epoch < 15:
            progress = (self.epoch - 5) / 10
            return {'alpha': 0.1 * progress, 'beta': 0.0}
        elif self.epoch < 30:
            progress = (self.epoch - 15) / 15
            return {'alpha': 0.1, 'beta': 0.1 * progress}
        else:
            return {'alpha': 0.1, 'beta': 0.1}
```

---

## 5. Evaluation Metrics

### 5.1 Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Task Accuracy | 標準タスク精度 | ≥ Baseline |
| Perplexity | 言語モデル品質 | ≤ Baseline |

### 5.2 Efficiency Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Active Ratio | 平均Activeトークン率 | < 50% |
| Layer Utilization | 平均使用レイヤー数 | < 75% |
| FLOPs | 推論計算量 | < 50% of Baseline |
| Latency | 推論時間 | < 70% of Baseline |

### 5.3 Interpretability Metrics

| Metric | Description |
|--------|-------------|
| Surprise Correlation | AG score と 人間の「重要」判断の相関 |
| Exit Accuracy | Early Exitした入力の精度 |
| F-Trajectory | 層ごとのF値推移 |

---

## 6. Implementation Roadmap

### Phase 1: Minimal PoC (2 weeks)

**Goal**: AG Gate のみで効果検証

| Task | Days | Deliverable |
|------|------|-------------|
| AG Gate実装 | 2 | `ag_gate.py` |
| 既存Transformerに統合 | 2 | `gedig_transformer_v0.py` |
| GLUE (SST-2) で評価 | 3 | Accuracy vs Active Ratio |
| 分析・レポート | 2 | `phase1_report.md` |

**Success Criteria**:
- Active Ratio < 70% で Accuracy 95% 維持

### Phase 2: DG Gate + Early Exit (2 weeks)

**Goal**: F値ベースのEarly Exit

| Task | Days | Deliverable |
|------|------|-------------|
| F計算の実装 | 2 | `f_calculator.py` |
| DG Gate実装 | 3 | `dg_gate.py` |
| Early Exit統合 | 2 | `gedig_transformer_v1.py` |
| 評価・チューニング | 3 | Layer utilization vs Accuracy |

**Success Criteria**:
- 平均Layer使用 < 80% で Accuracy 維持

### Phase 3: Full Training (3 weeks)

**Goal**: F-Loss統合、End-to-end学習

| Task | Days | Deliverable |
|------|------|-------------|
| F-Loss実装 | 2 | `gedig_loss.py` |
| Curriculum実装 | 2 | `trainer.py` |
| GLUE Full評価 | 5 | 全8タスク |
| WikiText Perplexity | 3 | Language modeling |
| 分析・論文準備 | 5 | `gedig_transformer_paper.md` |

**Success Criteria**:
- GLUE平均 ≥ BERT-base
- FLOPs < 50% of BERT-base

### Phase 4: Scaling & Applications (Ongoing)

- Larger models (BERT-large scale)
- Pre-training from scratch
- Domain adaptation
- Multi-task learning

---

## 7. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AG Gateが収束しない | Medium | High | Percentile fallback |
| Early Exitで精度低下 | Medium | High | Conservative thresholds |
| 学習不安定 | High | Medium | Curriculum, gradient clipping |
| GPU効率悪化 | Medium | Medium | 実装最適化、カスタムカーネル |
| Baseline未達 | Low | High | Phase 1で早期検証 |

---

## 8. Expected Results

### 8.1 Optimistic Scenario

| Metric | Baseline (BERT-base) | geDIG-Transformer |
|--------|---------------------|-------------------|
| GLUE Average | 79.6 | **80.5** (+0.9) |
| FLOPs | 100% | **35%** |
| Latency | 100% | **45%** |
| Active Ratio | 100% | **40%** |
| Avg Layers | 12 | **7.5** |

### 8.2 Conservative Scenario

| Metric | Baseline (BERT-base) | geDIG-Transformer |
|--------|---------------------|-------------------|
| GLUE Average | 79.6 | 78.5 (-1.1) |
| FLOPs | 100% | **60%** |
| Latency | 100% | **65%** |
| Active Ratio | 100% | **55%** |
| Avg Layers | 12 | **9** |

---

## 9. Theoretical Significance

### 9.1 geDIG原理の実証

本実装が成功すれば、以下が実証される：

1. **構造=確率の等価性**: AG/DG gatingがTransformerで機能
2. **F最小化の普遍性**: 言語タスクでもF最小化が有効
3. **スパース計算の原理的根拠**: 「なぜスパースが良いか」の説明

### 9.2 次のステップへの橋渡し

- **マルチモーダル**: 同じAG/DG原理で視覚・音声を統合
- **継続学習**: 推論中のDG gateでオンライン学習
- **脳との対応**: AG/DGと神経活動の相関分析

---

## Appendix A: Hyperparameters

```yaml
# Model
d_model: 512
n_heads: 8
n_layers: 12
d_ff: 2048
max_seq_len: 512
dropout: 0.1

# geDIG specific
lambda_ig: 1.0
ag_threshold_init: 0.5
ag_method: 'learned'  # or 'percentile'
dg_exit_threshold: -0.1
dg_reject_threshold: 0.0

# Training
batch_size: 32
learning_rate: 2e-5
warmup_steps: 10000
max_epochs: 30
alpha_final: 0.1  # F-loss weight
beta_final: 0.1   # Sparsity weight
```

---

## Appendix B: Related Work

| Work | Relation to geDIG-Transformer |
|------|------------------------------|
| **Sparse Transformers** (Child et al., 2019) | 固定スパースパターン。geDIGは動的 |
| **Early Exit** (Schwartz et al., 2020) | Confidence-based。geDIGはF-based |
| **Adaptive Attention** (Sukhbaatar et al., 2019) | 学習されたスパン。geDIGは入力依存 |
| **Universal Transformers** (Dehghani et al., 2019) | 適応的深度。geDIGはF-based exit |
| **Switch Transformer** (Fedus et al., 2021) | MoE routing。geDIGはAG/DG gating |

---

## Appendix C: File Structure

```
insightspike-ai/
├── src/
│   └── gedig_transformer/
│       ├── __init__.py
│       ├── gates/
│       │   ├── ag_gate.py
│       │   └── dg_gate.py
│       ├── attention/
│       │   └── sparse_attention.py
│       ├── blocks/
│       │   └── gedig_block.py
│       ├── model.py
│       ├── loss.py
│       └── trainer.py
├── experiments/
│   └── gedig_transformer/
│       ├── train_sst2.py
│       ├── train_glue.py
│       ├── eval_efficiency.py
│       └── configs/
│           ├── base.yaml
│           └── large.yaml
└── docs/
    └── design/
        └── gedig_transformer_architecture.md  # This file
```

---

*End of Design Document*
