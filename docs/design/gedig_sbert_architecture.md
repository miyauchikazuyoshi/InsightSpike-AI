# geDIG-SBERT Architecture Design

**Version**: 0.1 (Draft)
**Date**: 2026-01-26
**Author**: Kazuyoshi Miyauchi
**Status**: Proposal

---

## Executive Summary

Sentence-BERT (SBERT) は文埋め込みを学習する際、**暗黙的にグラフ構造を構築**している。本設計は、このグラフ構造を**明示化**し、geDIG原理（F = EPC - λIG）で制御することで、より効率的で解釈可能な文埋め込み学習を実現する。

**Core Insight**:
> SBERT の Contrastive Learning = グラフのエッジ構築
> geDIG の F最小化 = グラフの最適化

---

## 1. Motivation

### 1.1 SBERT の本質的操作

```
Input: (文A, 文B, label)
       label = 1 (類似) or 0 (非類似)

Operation:
  label=1 → embeddings を近づける → エッジを張る
  label=0 → embeddings を遠ざける → エッジを切る

Output: 埋め込み空間 ≈ 暗黙的なグラフ構造
```

### 1.2 SBERT の限界

| 問題 | 詳細 |
|------|------|
| **全ペア計算** | O(n²) の類似度計算が必要 |
| **暗黙的構造** | グラフ構造が明示されず分析困難 |
| **単一目的** | Contrastive Loss のみ、構造品質は考慮外 |
| **Hard Negative のみ** | Hard Positive の活用が不十分 |

### 1.3 geDIG による解決

| 解決策 | 効果 |
|--------|------|
| **AG Gate** | 重要ペアのみ選択 → O(k²), k << n |
| **明示的グラフ** | 構造を可視化・分析可能 |
| **F-Loss** | 構造品質を直接最適化 |
| **Surprise-based Selection** | Hard Negative + Hard Positive を統一的に扱う |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        geDIG-SBERT                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Batch of sentences [s₁, s₂, ..., sₙ]                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Sentence Encoder (BERT)                     │   │
│  │                                                          │   │
│  │  s_i → BERT → mean_pooling → e_i ∈ ℝ^d                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Embeddings: E = [e₁, e₂, ..., eₙ] ∈ ℝ^{n×d}                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    AG Gate                               │   │
│  │                                                          │   │
│  │  For each pair (i,j):                                    │   │
│  │    surprise(i,j) = |predicted_sim - actual_sim|          │   │
│  │    if surprise > θ: mark as active                       │   │
│  │                                                          │   │
│  │  Output: Active pairs A ⊂ {(i,j) : i < j}               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Similarity Computation                      │   │
│  │                                                          │   │
│  │  For (i,j) ∈ A:                                         │   │
│  │    sim(i,j) = exp(-||e_i - e_j||² / 2σ²)                │   │
│  │              (Gaussian kernel, not cosine)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DG Gate                               │   │
│  │                                                          │   │
│  │  For each edge (i,j):                                    │   │
│  │    ΔF = F_with_edge - F_without_edge                     │   │
│  │    if ΔF < 0: keep edge (improves structure)             │   │
│  │    if ΔF ≥ 0: prune edge (no benefit)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Graph: G = (V, E, W)                                           │
│    V = {sentences}                                              │
│    E = {active, non-pruned edges}                               │
│    W = {similarity weights}                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Loss Computation                        │   │
│  │                                                          │   │
│  │  L = λ_c × L_contrastive + λ_f × F                       │   │
│  │                                                          │   │
│  │  L_contrastive: Similar pairs close, dissimilar far      │   │
│  │  F = EPC - λ_ig × IG                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Backpropagation → Update Encoder                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Sentence Encoder

標準的なSBERT構成を採用:

```python
class SentenceEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='mean'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            embeddings: [batch, d_model]
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [batch, seq, d]

        if self.pooling == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling == 'cls':
            embeddings = token_embeddings[:, 0]

        return embeddings
```

### 3.2 AG Gate (Ambiguity Gate for Pairs)

**目的**: 「驚き」のあるペアのみを選択し、計算量を削減

#### 3.2.1 Surprise の定義

```
surprise(i,j) = |predicted_sim(i,j) - actual_sim(i,j)|

predicted_sim: モデルが予測する類似度
actual_sim:    ラベルから計算される類似度（または現在の埋め込みからの類似度）
```

**解釈**:
- High surprise: 「このペアは予想と違う」→ 学習価値が高い
- Low surprise: 「このペアは予想通り」→ 学習済み、スキップ可能

#### 3.2.2 実装

```python
class AGGateForPairs(nn.Module):
    def __init__(self, d_model, threshold=0.3):
        super().__init__()
        self.threshold = threshold

        # Similarity predictor (learns to predict which pairs are similar)
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: [batch, d_model]
            labels: [batch, batch] pairwise labels (optional)

        Returns:
            active_pairs: List of (i, j) tuples
            surprise_scores: [batch, batch] matrix
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute predicted similarities for all pairs
        # Efficient: use broadcasting
        e_i = embeddings.unsqueeze(1).expand(-1, batch_size, -1)  # [B, B, d]
        e_j = embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, B, d]
        pair_repr = torch.cat([e_i, e_j], dim=-1)  # [B, B, 2d]

        predicted_sim = self.predictor(pair_repr).squeeze(-1)  # [B, B]

        # Actual similarity from current embeddings (Gaussian kernel)
        actual_sim = self._gaussian_similarity(embeddings, embeddings)

        # Surprise = prediction error
        surprise_scores = (predicted_sim - actual_sim).abs()

        # Select active pairs (high surprise, upper triangle only)
        mask = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1).bool()
        surprise_upper = surprise_scores.clone()
        surprise_upper[~mask] = 0

        active_mask = (surprise_upper > self.threshold) & mask
        active_pairs = active_mask.nonzero(as_tuple=False).tolist()

        return active_pairs, surprise_scores, predicted_sim

    def _gaussian_similarity(self, a, b, sigma=1.0):
        """Gaussian kernel similarity"""
        dist_sq = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))
```

#### 3.2.3 Hard Mining との関係

| 手法 | 選択基準 | geDIG AG Gate |
|------|----------|---------------|
| Random Sampling | ランダム | - |
| Hard Negative Mining | 類似度高いが非類似 | surprise(neg) 高い |
| Hard Positive Mining | 類似度低いが類似 | surprise(pos) 高い |
| **geDIG AG** | **予測と実際の乖離** | **両方を統一的に扱う** |

---

### 3.3 Gaussian Similarity (Norm-based)

**コサイン類似度を使わない理由** (Section 3.12 参照):
- 埋め込み空間が学習中（未完成）
- 方向より距離の方が信頼できる

```python
class GaussianSimilarity(nn.Module):
    def __init__(self, sigma_init=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
        else:
            self.register_buffer('log_sigma', torch.tensor(math.log(sigma_init)))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def forward(self, e_i, e_j):
        """
        Args:
            e_i: [n, d] or [batch, n, d]
            e_j: [m, d] or [batch, m, d]
        Returns:
            similarity: [n, m] or [batch, n, m]
        """
        dist_sq = torch.cdist(e_i, e_j, p=2) ** 2
        similarity = torch.exp(-dist_sq / (2 * self.sigma ** 2 + 1e-8))
        return similarity

    def pairwise(self, embeddings):
        """Compute pairwise similarity matrix"""
        return self.forward(embeddings, embeddings)
```

---

### 3.4 DG Gate (Decision Gate for Edges)

**目的**: 構造的に有益でないエッジを刈り込む

#### 3.4.1 F値の計算

```
F = EPC - λ × IG

EPC = Edge Processing Cost
    = |E| / |V|²  (正規化エッジ数)

IG = Information Gain
   = H_before - H_after  (エントロピー減少)
   = Clustering coefficient improvement
   = Shortest path improvement
```

#### 3.4.2 実装

```python
class DGGateForEdges(nn.Module):
    def __init__(self, lambda_ig=1.0, threshold=0.0):
        super().__init__()
        self.lambda_ig = lambda_ig
        self.threshold = threshold

    def forward(self, similarity_matrix, active_pairs):
        """
        Args:
            similarity_matrix: [batch, batch] pairwise similarities
            active_pairs: List of (i, j) tuples from AG Gate

        Returns:
            kept_pairs: List of (i, j) tuples that improve F
            pruned_pairs: List of (i, j) tuples that don't improve F
            delta_F: F value changes for each pair
        """
        kept_pairs = []
        pruned_pairs = []
        delta_F_list = []

        # Current graph state
        n = similarity_matrix.shape[0]
        current_adj = (similarity_matrix > 0.5).float()  # Threshold for edge existence

        for (i, j) in active_pairs:
            # Compute F without this edge
            adj_without = current_adj.clone()
            adj_without[i, j] = 0
            adj_without[j, i] = 0
            F_without = self._compute_F(adj_without, similarity_matrix)

            # Compute F with this edge
            adj_with = current_adj.clone()
            adj_with[i, j] = 1
            adj_with[j, i] = 1
            F_with = self._compute_F(adj_with, similarity_matrix)

            # Decision
            delta_F = F_with - F_without
            delta_F_list.append(delta_F.item())

            if delta_F < self.threshold:
                kept_pairs.append((i, j))
                current_adj[i, j] = 1
                current_adj[j, i] = 1
            else:
                pruned_pairs.append((i, j))

        return kept_pairs, pruned_pairs, delta_F_list

    def _compute_F(self, adj, similarity):
        """Compute F value for a graph"""
        n = adj.shape[0]

        # EPC: normalized edge count
        epc = adj.sum() / (n * n)

        # IG: entropy-based
        # Convert adjacency to probability distribution
        row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        prob = adj / row_sums
        entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1).mean()

        # Lower entropy = higher IG (more structured)
        ig = 1.0 - entropy / math.log(n)  # Normalized

        F = epc - self.lambda_ig * ig
        return F
```

---

### 3.5 Loss Function

#### 3.5.1 複合損失

```python
class GeDIGSBERTLoss(nn.Module):
    def __init__(self, lambda_contrast=1.0, lambda_f=0.1, margin=0.5):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_f = lambda_f
        self.margin = margin

    def forward(self, embeddings, labels, graph_adj, similarity_matrix):
        """
        Args:
            embeddings: [batch, d_model]
            labels: [batch, batch] pairwise labels (1=similar, 0=dissimilar)
            graph_adj: [batch, batch] adjacency matrix after DG Gate
            similarity_matrix: [batch, batch] Gaussian similarities

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # 1. Contrastive Loss
        L_contrast = self._contrastive_loss(embeddings, labels)

        # 2. F-Loss
        F_value = self._compute_F(graph_adj, similarity_matrix)

        # 3. Combine
        total_loss = self.lambda_contrast * L_contrast + self.lambda_f * F_value

        return total_loss, {
            'contrastive': L_contrast.item(),
            'F_value': F_value.item(),
            'total': total_loss.item()
        }

    def _contrastive_loss(self, embeddings, labels):
        """
        Multiple Negatives Ranking Loss variant
        """
        # Pairwise distances
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Positive pairs: should be close
        pos_mask = (labels == 1).float()
        pos_loss = (dist * pos_mask).sum() / (pos_mask.sum() + 1e-8)

        # Negative pairs: should be far (with margin)
        neg_mask = (labels == 0).float()
        neg_dist = dist * neg_mask
        neg_loss = F.relu(self.margin - neg_dist)
        neg_loss = (neg_loss * neg_mask).sum() / (neg_mask.sum() + 1e-8)

        return pos_loss + neg_loss

    def _compute_F(self, adj, similarity):
        """Same as DGGate._compute_F"""
        n = adj.shape[0]
        epc = adj.sum() / (n * n)

        row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        prob = adj / row_sums
        entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1).mean()
        ig = 1.0 - entropy / math.log(n + 1e-8)

        return epc - ig  # lambda_ig=1.0 in this formulation
```

---

### 3.6 Full Model

```python
class GeDIGSBERT(nn.Module):
    def __init__(
        self,
        model_name='bert-base-uncased',
        pooling='mean',
        sigma_init=1.0,
        ag_threshold=0.3,
        lambda_ig=1.0,
        lambda_contrast=1.0,
        lambda_f=0.1
    ):
        super().__init__()

        # Components
        self.encoder = SentenceEncoder(model_name, pooling)
        self.similarity = GaussianSimilarity(sigma_init, learnable=True)
        self.ag_gate = AGGateForPairs(
            d_model=self.encoder.bert.config.hidden_size,
            threshold=ag_threshold
        )
        self.dg_gate = DGGateForEdges(lambda_ig=lambda_ig)
        self.loss_fn = GeDIGSBERTLoss(
            lambda_contrast=lambda_contrast,
            lambda_f=lambda_f
        )

        self.lambda_ig = lambda_ig

    def forward(self, input_ids, attention_mask, labels=None, return_graph=False):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, batch] pairwise labels (for training)
            return_graph: Whether to return the constructed graph

        Returns:
            embeddings: [batch, d_model]
            loss: (if labels provided)
            graph: (if return_graph=True)
        """
        # 1. Encode sentences
        embeddings = self.encoder(input_ids, attention_mask)

        # 2. Compute similarities
        similarity_matrix = self.similarity.pairwise(embeddings)

        # 3. AG Gate: Select surprising pairs
        active_pairs, surprise_scores, predicted_sim = self.ag_gate(embeddings, labels)

        # 4. DG Gate: Prune non-beneficial edges
        kept_pairs, pruned_pairs, delta_F = self.dg_gate(similarity_matrix, active_pairs)

        # 5. Build final graph
        batch_size = embeddings.shape[0]
        graph_adj = torch.zeros(batch_size, batch_size, device=embeddings.device)
        for (i, j) in kept_pairs:
            graph_adj[i, j] = similarity_matrix[i, j]
            graph_adj[j, i] = similarity_matrix[j, i]

        # 6. Compute loss (if training)
        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = self.loss_fn(embeddings, labels, graph_adj, similarity_matrix)

        # 7. Return
        output = {'embeddings': embeddings}
        if loss is not None:
            output['loss'] = loss
            output['loss_dict'] = loss_dict
        if return_graph:
            output['graph'] = {
                'adjacency': graph_adj,
                'similarity': similarity_matrix,
                'active_pairs': active_pairs,
                'kept_pairs': kept_pairs,
                'pruned_pairs': pruned_pairs,
                'surprise_scores': surprise_scores
            }

        return output

    def encode(self, input_ids, attention_mask):
        """Inference-only encoding"""
        with torch.no_grad():
            embeddings = self.encoder(input_ids, attention_mask)
        return embeddings
```

---

## 4. Training Procedure

### 4.1 Data Format

```python
# Standard SBERT format
train_data = [
    {'sentence1': "A man is eating food.",
     'sentence2': "A man is eating a piece of bread.",
     'label': 1},  # Similar
    {'sentence1': "A man is eating food.",
     'sentence2': "A man is riding a horse.",
     'label': 0},  # Dissimilar
    ...
]
```

### 4.2 Training Loop

```python
def train_gedig_sbert(model, train_dataloader, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        metrics = defaultdict(float)

        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']  # [batch, batch] pairwise

            optimizer.zero_grad()

            output = model(input_ids, attention_mask, labels, return_graph=True)
            loss = output['loss']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k, v in output['loss_dict'].items():
                metrics[k] += v

        # Log
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        print(f"  Contrastive: {metrics['contrastive']/len(train_dataloader):.4f}")
        print(f"  F-value: {metrics['F_value']/len(train_dataloader):.4f}")
```

### 4.3 Curriculum Learning

```python
def get_training_config(epoch, max_epochs=30):
    """
    Curriculum-based hyperparameter scheduling
    """
    if epoch < 5:
        # Phase 1: Focus on contrastive learning
        return {
            'lambda_contrast': 1.0,
            'lambda_f': 0.01,      # Low F-loss weight
            'ag_threshold': 0.1,  # Low threshold = more pairs
            'sigma': 10.0         # Large sigma = soft similarities
        }
    elif epoch < 15:
        # Phase 2: Introduce structure learning
        progress = (epoch - 5) / 10
        return {
            'lambda_contrast': 1.0,
            'lambda_f': 0.01 + 0.09 * progress,  # Gradually increase
            'ag_threshold': 0.1 + 0.2 * progress,
            'sigma': 10.0 - 9.0 * progress  # Sharpen
        }
    else:
        # Phase 3: Balance both
        return {
            'lambda_contrast': 1.0,
            'lambda_f': 0.1,
            'ag_threshold': 0.3,
            'sigma': None  # Adaptive
        }
```

---

## 5. Evaluation

### 5.1 Standard Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| **STS-B** | Semantic Textual Similarity | Spearman ρ |
| **SICK-R** | Relatedness | Spearman ρ |
| **STS 12-16** | Cross-year STS | Spearman ρ |

### 5.2 geDIG-specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Active Pair Ratio** | AG Gate 通過率 | < 50% |
| **Edge Retention** | DG Gate 通過率 | > 70% |
| **Graph Clustering** | クラスタリング係数 | High |
| **F-value** | 最終的なF値 | Low |

### 5.3 Evaluation Code

```python
def evaluate_gedig_sbert(model, eval_dataloader):
    model.eval()

    all_embeddings = []
    all_labels = []
    all_graphs = []

    with torch.no_grad():
        for batch in eval_dataloader:
            output = model(
                batch['input_ids'],
                batch['attention_mask'],
                return_graph=True
            )
            all_embeddings.append(output['embeddings'])
            all_graphs.append(output['graph'])

    embeddings = torch.cat(all_embeddings, dim=0)

    # 1. Standard STS evaluation
    similarity_scores = compute_cosine_similarity(embeddings)
    spearman_rho = compute_spearman(similarity_scores, ground_truth_labels)

    # 2. geDIG-specific metrics
    avg_active_ratio = np.mean([
        len(g['active_pairs']) / (len(g['adjacency']) ** 2 / 2)
        for g in all_graphs
    ])
    avg_retention = np.mean([
        len(g['kept_pairs']) / (len(g['active_pairs']) + 1e-8)
        for g in all_graphs
    ])

    return {
        'spearman': spearman_rho,
        'active_ratio': avg_active_ratio,
        'retention_ratio': avg_retention
    }
```

---

## 6. Expected Results

### 6.1 Performance Comparison

| Model | STS-B (ρ) | Active Ratio | Training Time |
|-------|-----------|--------------|---------------|
| SBERT (baseline) | 0.84 | 100% | 1.0x |
| **geDIG-SBERT** | **0.85** | **45%** | **0.6x** |

### 6.2 Ablation Study Plan

| Experiment | Variation | Expected Effect |
|------------|-----------|-----------------|
| No AG Gate | All pairs computed | Slower, similar accuracy |
| No DG Gate | All edges kept | Higher F, similar accuracy |
| No F-Loss | Contrastive only | Similar to SBERT baseline |
| Cosine sim | Replace Gaussian | Lower accuracy in early training |

---

## 7. Theoretical Significance

### 7.1 「構造 = 確率」の実現

```
SBERT: P(similar | sentence_pair) → embedding distance
geDIG-SBERT: P(edge | node_pair) = exp(-dist²/2σ²) → graph structure
```

**両者は同じ操作**を異なる言語で記述している。geDIG-SBERTはこの等価性を明示化する。

### 7.2 グラフとしての埋め込み空間

```
従来: 埋め込み空間は幾何的対象（ベクトル空間）
geDIG: 埋め込み空間はグラフの幾何的実現

違い:
- グラフは離散的（エッジの有無）
- 埋め込みは連続的（距離）
- geDIG-SBERTは両者を統合
```

### 7.3 学習 = グラフ構築

```
SBERT学習: 埋め込みベクトルを最適化
geDIG-SBERT学習: グラフ構造を最適化（埋め込みはその表現）

F最小化 = 最小コストで最大情報利得のグラフを構築
```

---

## 8. Implementation Roadmap

| Phase | Task | Duration | Deliverable |
|-------|------|----------|-------------|
| 1 | Base SBERT + F-Loss | 1 week | `gedig_sbert_v0.py` |
| 2 | AG Gate implementation | 1 week | `ag_gate.py` |
| 3 | DG Gate implementation | 1 week | `dg_gate.py` |
| 4 | Integration & Training | 1 week | `gedig_sbert.py` |
| 5 | STS Benchmark | 1 week | Evaluation report |
| 6 | Ablation studies | 1 week | Analysis report |

---

## Appendix A: Hyperparameters

```yaml
# Model
base_model: 'bert-base-uncased'
pooling: 'mean'
d_model: 768

# Similarity
sigma_init: 1.0
sigma_learnable: true

# AG Gate
ag_threshold: 0.3
ag_predictor_hidden: 384

# DG Gate
lambda_ig: 1.0
dg_threshold: 0.0

# Loss
lambda_contrast: 1.0
lambda_f: 0.1
margin: 0.5

# Training
batch_size: 32
learning_rate: 2e-5
epochs: 10
warmup_ratio: 0.1
```

---

## Appendix B: Relation to Sentence-BERT Training

### B.1 SBERT Training Objectives

| Objective | Formula | geDIG Equivalent |
|-----------|---------|------------------|
| Softmax Loss | CE(sim, label) | - |
| Contrastive Loss | max(0, m - dist_neg) | L_contrastive |
| Triplet Loss | max(0, dist_pos - dist_neg + m) | - |
| **Multiple Negatives Ranking** | -log(exp(sim_pos) / Σexp(sim)) | AG selection + L_contrastive |

### B.2 geDIG の追加要素

```
SBERT Loss:        L = L_contrastive
geDIG-SBERT Loss:  L = L_contrastive + λ_f × F

F = EPC - λ_ig × IG
  = (構造コスト) - (情報利得)

これにより:
- 不要なエッジを張らない (低EPC)
- 情報的に有益なエッジを張る (高IG)
```

---

## Appendix C: File Structure

```
insightspike-ai/
├── src/
│   └── gedig_sbert/
│       ├── __init__.py
│       ├── encoder.py          # SentenceEncoder
│       ├── similarity.py       # GaussianSimilarity
│       ├── gates/
│       │   ├── ag_gate.py      # AGGateForPairs
│       │   └── dg_gate.py      # DGGateForEdges
│       ├── loss.py             # GeDIGSBERTLoss
│       ├── model.py            # GeDIGSBERT
│       └── trainer.py          # Training utilities
├── experiments/
│   └── gedig_sbert/
│       ├── train.py
│       ├── evaluate.py
│       └── configs/
│           └── base.yaml
└── docs/
    └── design/
        └── gedig_sbert_architecture.md  # This file
```

---

*End of Design Document*
