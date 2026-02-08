"""
Hidden State-based geDIG Implementation

Computes geDIG components (EPC, H, SP, F) from Transformer hidden states
instead of attention matrices. This is model-agnostic and can be applied
to any model that outputs hidden states per layer.

Reference: SPEC.md Section 2.3
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GedigResult:
    """Result of geDIG computation between two layers."""
    f_value: float          # F = EPC - λ(s·ΔH + γΔSP)
    epc: float              # Edit Path Cost (structure change)
    delta_h: float          # Entropy change
    delta_sp: float         # Shortcut Purity change
    h_before: float         # Entropy of h_before
    h_after: float          # Entropy of h_after
    sp_before: float        # SP of h_before
    sp_after: float         # SP of h_after


def cosine_similarity_matrix(h: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for hidden states.

    Args:
        h: Hidden states of shape (seq_len, hidden_dim)

    Returns:
        Similarity matrix of shape (seq_len, seq_len)
    """
    # Normalize along hidden dimension
    h_norm = F.normalize(h, p=2, dim=-1)
    # Compute cosine similarity
    sim = torch.mm(h_norm, h_norm.t())
    return sim


def compute_epc(h_before: torch.Tensor, h_after: torch.Tensor, method: str = "vector") -> float:
    """
    Compute Edit Path Cost: how much did the vectors move.

    Args:
        h_before: Hidden states before (seq_len, hidden_dim)
        h_after: Hidden states after (seq_len, hidden_dim)
        method:
            "vector" - L2 distance of vectors (案1: 機能分離)
            "similarity" - change in similarity matrix (旧実装)

    Returns:
        EPC value
    """
    if method == "vector":
        # 案1: ベクトルの移動距離
        # 正規化してからL2距離を測る（スケール不変）
        h_before_norm = F.normalize(h_before, p=2, dim=-1)
        h_after_norm = F.normalize(h_after, p=2, dim=-1)

        # 各トークンの移動距離の平均
        distances = torch.norm(h_after_norm - h_before_norm, p=2, dim=-1)
        epc = distances.mean()
        return epc.item()

    elif method == "similarity":
        # 旧実装: 類似度行列の変化
        sim_before = cosine_similarity_matrix(h_before)
        sim_after = cosine_similarity_matrix(h_after)
        epc = torch.abs(sim_after - sim_before).mean()
        return epc.item()

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_entropy(h: torch.Tensor, temperature: float = 1.0, method: str = "hidden") -> float:
    """
    Compute entropy.

    Args:
        h: Hidden states (seq_len, hidden_dim)
        temperature: Softmax temperature for normalization
        method:
            "similarity" - v1: entropy of similarity distribution
            "hidden" - v2: entropy of hidden state values (確率エントロピー)

    Returns:
        Mean entropy across all tokens
    """
    eps = 1e-10

    if method == "similarity":
        # v1: 類似度分布のエントロピー
        sim = cosine_similarity_matrix(h)
        sim_norm = F.softmax(sim / temperature, dim=-1)
        entropy_per_token = -(sim_norm * torch.log(sim_norm + eps)).sum(dim=-1)
        return entropy_per_token.mean().item()

    elif method == "hidden":
        # v2: hidden state自体のエントロピー（確率エントロピー）
        # 各トークンのhidden stateを確率分布と見なす
        h_abs = torch.abs(h)
        h_prob = h_abs / (h_abs.sum(dim=-1, keepdim=True) + eps)

        # シャノンエントロピー（次元数で正規化）
        entropy_per_token = -(h_prob * torch.log(h_prob + eps)).sum(dim=-1)
        max_entropy = np.log(h.shape[-1])  # 最大エントロピー（均等分布）
        normalized_entropy = entropy_per_token / max_entropy

        return normalized_entropy.mean().item()

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_gini(values: torch.Tensor) -> float:
    """
    Compute Gini coefficient of a distribution.
    Returns value in [0, 1]: 0 = perfect equality, 1 = perfect inequality
    """
    values = values.flatten()
    # Shift to non-negative
    values = values - values.min()
    values = torch.sort(values)[0]
    n = len(values)

    if n == 0 or values.sum() == 0:
        return 0.0

    # Standard Gini formula
    index = torch.arange(1, n + 1, dtype=torch.float32, device=values.device)
    gini = (2 * torch.sum(index * values) - (n + 1) * torch.sum(values)) / (n * torch.sum(values) + 1e-10)
    return max(0.0, min(1.0, gini.item()))  # Clamp to [0, 1]


def compute_positional_correlation(sim: torch.Tensor) -> float:
    """
    Compute correlation between similarity and positional distance.

    High negative correlation = nearby tokens are similar (structured)
    No correlation = similarity independent of position (liquid)

    Returns: correlation coefficient (typically negative for structured text)
    """
    seq_len = sim.shape[0]

    # Position distance matrix |i - j|
    pos = torch.arange(seq_len, dtype=torch.float32, device=sim.device)
    dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))

    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(seq_len, dtype=torch.bool, device=sim.device)
    sim_flat = sim[mask]
    dist_flat = dist[mask]

    # Pearson correlation
    sim_mean = sim_flat.mean()
    dist_mean = dist_flat.mean()

    sim_centered = sim_flat - sim_mean
    dist_centered = dist_flat - dist_mean

    cov = (sim_centered * dist_centered).mean()
    std_sim = sim_centered.std()
    std_dist = dist_centered.std()

    if std_sim < 1e-10 or std_dist < 1e-10:
        return 0.0

    corr = cov / (std_sim * std_dist)
    return corr.item()


def compute_sp(
    h: torch.Tensor,
    anchor_idx: int = 0,
    k_ratio: float = 0.2,
    method: str = "positional"
) -> float:
    """
    Compute Shortcut Purity / Structural Strength.

    Args:
        h: Hidden states (seq_len, hidden_dim)
        anchor_idx: Index of anchor token (for "anchor" method)
        k_ratio: Top-k ratio for concentration measurement
        method:
            "anchor" - v1: concentration toward anchor token
            "gini" - v2: Gini coefficient of similarity
            "positional" - v3: correlation with positional distance (構造強度)

    Returns:
        SP value
    """
    sim = cosine_similarity_matrix(h)
    seq_len = sim.shape[0]

    if method == "anchor":
        # v1: アンカーへの集中度
        if anchor_idx < 0:
            anchor_idx = seq_len + anchor_idx
        to_anchor = sim[:, anchor_idx]
        k = max(1, int(seq_len * k_ratio))
        top_k_values = torch.topk(to_anchor, k).values
        eps = 1e-10
        sp = top_k_values.sum() / (to_anchor.sum() + eps)
        return sp.item()

    elif method == "gini":
        # v2: 類似度分布のGini係数（構造強度）
        mask = ~torch.eye(seq_len, dtype=torch.bool, device=sim.device)
        off_diag = sim[mask]
        gini = compute_gini(off_diag)
        return gini

    elif method == "positional":
        # v3: 類似度と位置距離の相関（構造強度）
        # 負の相関 = 近いトークンが類似 = 構造化
        # 相関なし = 位置と無関係 = 液相
        corr = compute_positional_correlation(sim)

        # SP = -corr として、高SP = 構造化（近いほど類似）
        # corrは通常負なので、-corrは正になる
        sp = -corr
        return sp

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_gedig(
    h_before: torch.Tensor,
    h_after: torch.Tensor,
    anchor_idx: int = 0,
    lambda_: float = 1.0,
    gamma: float = 0.5,
    entropy_sign: int = -1,
    k_ratio: float = 0.2,
    temperature: float = 1.0,
    epc_method: str = "vector",
    h_method: str = "hidden",
    sp_method: str = "gini"
) -> GedigResult:
    """
    Compute full geDIG metrics between two layers.

    F = EPC - λ(s·ΔH + γΔSP)

    Args:
        h_before: Hidden states of previous layer (seq_len, hidden_dim)
        h_after: Hidden states of current layer (seq_len, hidden_dim)
        anchor_idx: Anchor token index (0 for BERT, -1 for GPT)
        lambda_: Information gain weight
        gamma: SP weight relative to entropy
        entropy_sign: -1 for inference (concentration = gain), +1 for exploration
        k_ratio: Top-k ratio for SP computation
        temperature: Softmax temperature for entropy
        epc_method: "vector" or "similarity"
        h_method: "hidden" (v2: 確率エントロピー) or "similarity" (v1)
        sp_method: "gini" (v2: 構造強度) or "anchor" (v1)

    Returns:
        GedigResult with all components
    """
    # Compute components
    epc = compute_epc(h_before, h_after, method=epc_method)

    h_entropy_before = compute_entropy(h_before, temperature, method=h_method)
    h_entropy_after = compute_entropy(h_after, temperature, method=h_method)
    delta_h = h_entropy_after - h_entropy_before

    sp_before = compute_sp(h_before, anchor_idx, k_ratio, method=sp_method)
    sp_after = compute_sp(h_after, anchor_idx, k_ratio, method=sp_method)
    delta_sp = sp_after - sp_before

    # Compute F
    # F = EPC - λ(s·ΔH + γΔSP)
    # entropy_sign = -1: entropy decrease is good (inference)
    # entropy_sign = +1: entropy increase is good (exploration)
    information_gain = entropy_sign * delta_h + gamma * delta_sp
    f_value = epc - lambda_ * information_gain

    return GedigResult(
        f_value=f_value,
        epc=epc,
        delta_h=delta_h,
        delta_sp=delta_sp,
        h_before=h_entropy_before,
        h_after=h_entropy_after,
        sp_before=sp_before,
        sp_after=sp_after
    )


def compute_trajectory(
    hidden_states: list[torch.Tensor],
    anchor_idx: int = 0,
    lambda_: float = 1.0,
    gamma: float = 0.5,
    entropy_sign: int = -1,
    k_ratio: float = 0.2,
    temperature: float = 1.0,
    epc_method: str = "vector",
    h_method: str = "hidden",
    sp_method: str = "gini"
) -> list[GedigResult]:
    """
    Compute geDIG trajectory across all layers.

    Args:
        hidden_states: List of hidden states per layer [(seq_len, hidden_dim), ...]
        epc_method: "vector" or "similarity"
        h_method: "hidden" (v2) or "similarity" (v1)
        sp_method: "gini" (v2) or "anchor" (v1)
        Other args: Same as compute_gedig

    Returns:
        List of GedigResult for each layer transition (len = num_layers - 1)
    """
    results = []

    for i in range(len(hidden_states) - 1):
        h_before = hidden_states[i]
        h_after = hidden_states[i + 1]

        result = compute_gedig(
            h_before, h_after,
            anchor_idx=anchor_idx,
            lambda_=lambda_,
            gamma=gamma,
            entropy_sign=entropy_sign,
            k_ratio=k_ratio,
            temperature=temperature,
            epc_method=epc_method,
            h_method=h_method,
            sp_method=sp_method
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Quick test with random data
    print("Testing gedig_hidden.py...")

    seq_len = 10
    hidden_dim = 64
    num_layers = 6

    # Simulate hidden states that become more structured over layers
    hidden_states = []
    for layer in range(num_layers):
        # Add some structure as layers progress
        h = torch.randn(seq_len, hidden_dim)
        # Make first token increasingly dominant
        h[0] = h[0] * (1 + layer * 0.5)
        hidden_states.append(h)

    # Compute trajectory
    trajectory = compute_trajectory(hidden_states, anchor_idx=0)

    print(f"\nTrajectory across {num_layers} layers:")
    print("-" * 60)
    for i, result in enumerate(trajectory):
        print(f"Layer {i} → {i+1}:")
        print(f"  F={result.f_value:.4f}, EPC={result.epc:.4f}, "
              f"ΔH={result.delta_h:.4f}, ΔSP={result.delta_sp:.4f}")

    print("\nDone!")
