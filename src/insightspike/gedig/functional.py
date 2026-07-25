
"""
Flash-geDIG Functional API
==========================

Core tensor-native implementations of geDIG metrics.
All functions are stateless, differentiable, and GPU-ready.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import math

if TYPE_CHECKING:
    from gedig.adapters.transformer import TransformerFEvalResult
else:
    # Keep importing the adapter lazy while allowing typing.get_type_hints()
    # to resolve the public function at runtime.
    TransformerFEvalResult = Any


def compute_structural_profile(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    lambda_param: float = 1.0,
    gamma: float = 0.5,
    temperature: float = 0.1,
    percentile: float = 0.9,
    max_path_length: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the legacy single-state Flash structural profile.

    This is an absolute diagnostic over one attention state. It is not the
    canonical geDIG before/after delta, so its components intentionally do not
    use ``delta_*`` names and it has no universal optimization direction.

    Args:
        attention: Attention weights (Batch, Heads, Seq, Seq)
        attention_mask: Mask for valid tokens (Batch, Seq). 1 for valid, 0 for padding.
        lambda_param: Weight balancing Cost (EPC) vs Value (Entropy + SP).
        gamma: Weight for Structure Potential (SP) relative to Entropy.
        temperature: Temperature for soft thresholding (sigmoid).
        percentile: Percentile for dynamic thresholding of edges.
        max_path_length: Maximum path length for SP approximation (Matrix Power).

    Returns:
        profile_values: Tensor of profile values (Batch, Heads)
        metrics: Absolute ``epc``, ``h``, ``sp``, and ``clustering`` tensors.
    """
    # 1. Preprocess: Apply mask if provided
    if attention_mask is not None:
        # Expand mask to (Batch, 1, Seq, Seq) for broadcasting over heads
        mask_2d = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
        mask_2d = mask_2d.float()
        attention = attention * mask_2d

    # 2. Compute Components
    epc = _compute_soft_density(attention, temperature, percentile)
    entropy = _compute_entropy(attention, attention_mask)
    structure_potential = _compute_soft_path_efficiency(
        attention,
        temperature,
        percentile,
        max_path_length,
    )
    clustering = _compute_soft_clustering(
        attention,
        temperature,
        percentile,
    )

    # 3. Compute F
    # ... (omitted comments)
    f_values = epc - lambda_param * (
        entropy + gamma * structure_potential
    )

    return f_values, {
        "epc": epc,
        "h": entropy,
        "sp": structure_potential,
        "clustering": clustering,
    }


def compute_f_score(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    lambda_param: float = 1.0,
    gamma: float = 0.5,
    temperature: float = 0.1,
    percentile: float = 0.9,
    max_path_length: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compatibility wrapper for the historical single-state Flash API.

    The returned numbers and legacy ``delta_*`` metric keys are preserved.
    New diagnostic code should call :func:`compute_structural_profile`; code
    that needs canonical geDIG semantics should call
    :func:`compute_delta_f_score`.
    """

    profile, metrics = compute_structural_profile(
        attention,
        attention_mask=attention_mask,
        lambda_param=lambda_param,
        gamma=gamma,
        temperature=temperature,
        percentile=percentile,
        max_path_length=max_path_length,
    )
    return profile, {
        "delta_epc": metrics["epc"],
        "delta_h": metrics["h"],
        "delta_sp": metrics["sp"],
        "delta_clustering": metrics["clustering"],
    }


def compute_delta_f_score(
    before_attention: torch.Tensor,
    after_attention: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    lambda_param: float = 1.0,
    gamma: float = 0.5,
    temperature: float = 10.0,
    percentile: float = 0.9,
    use_betti: bool = False,
) -> "TransformerFEvalResult":
    """Compute canonical before/after geDIG F via ``TransformerFEval``.

    Lower F is better under the repository-wide judgment convention. The
    returned adapter result exposes ``F`` with shape ``(batch, heads)``,
    scalar ``F_mean``, and the component means without repackaging them.
    """

    from gedig.adapters.transformer import TransformerFEval

    return TransformerFEval(
        lambda_param=lambda_param,
        gamma=gamma,
        percentile=percentile,
        temperature=temperature,
        use_betti=use_betti,
    ).compute(before_attention, after_attention, mask)

# Helpers

def _clamp_percentile(percentile: float) -> float:
    p = float(percentile)
    if p <= 0.0:
        return 1e-6
    if p >= 1.0:
        return 1.0 - 1e-6
    return p


def _compute_soft_threshold(attn_flat: torch.Tensor, percentile: float) -> torch.Tensor:
    """Compute a differentiable threshold for soft adjacency."""
    p = _clamp_percentile(percentile)
    if hasattr(torch, "quantile"):
        threshold = torch.quantile(attn_flat, p, dim=-1, keepdim=True)
    else:
        n = attn_flat.shape[-1]
        k = max(1, min(int(round(p * n)), n - 1))
        threshold = torch.kthvalue(attn_flat, k, dim=-1).values.unsqueeze(-1)
    return threshold.unsqueeze(-1)

# ... (existing functions)

def _compute_soft_clustering(
    attention: torch.Tensor,
    temperature: float,
    percentile: float
) -> torch.Tensor:
    """
    Approximates Clustering Coefficient using triangle counts (Trace(A^3)).
    """
    batch_size, num_heads, seq_len, _ = attention.shape
    
    # Soft Adjacency
    attn_flat = attention.view(batch_size, num_heads, -1)
    threshold = _compute_soft_threshold(attn_flat, percentile)
    adj = torch.sigmoid((attention - threshold) / temperature)
    
    # Remove self-loops for clustering (we care about triangles between DIFFERENT nodes)
    eye = torch.eye(seq_len, device=attention.device).unsqueeze(0).unsqueeze(0)
    adj = adj * (1.0 - eye)
    
    # Compute A^3
    adj_2 = torch.matmul(adj, adj)
    adj_3 = torch.matmul(adj_2, adj)
    
    # Trace(A^3) = Sum of diagonal elements
    # shape: (Batch, Heads, Seq, Seq) -> diagonal -> (Batch, Heads, Seq) -> sum -> (Batch, Heads)
    trace_A3 = torch.diagonal(adj_3, dim1=-2, dim2=-1).sum(dim=-1)
    
    # Normalize by Max Possible Triangles? 
    # Or just use raw intensity as a "Semantic Density" signal?
    # Max triangles in complete graph: N(N-1)(N-2)/6 ~ N^3
    # Let's normalize by N^3 roughly to keep it in [0, 1] range.
    max_triangles = seq_len * seq_len * seq_len
    clustering = trace_A3 / (max_triangles + 1e-10)
    
    return clustering


def _compute_soft_density(
    attention: torch.Tensor, 
    temperature: float, 
    percentile: float
) -> torch.Tensor:
    """Compute soft edge density (EPC)."""
    batch_size, num_heads, seq_len, _ = attention.shape
    
    # Dynamic thresholding per head
    attn_flat = attention.view(batch_size, num_heads, -1)
    threshold = _compute_soft_threshold(attn_flat, percentile)

    # Soft threshold (Sigmoid)
    edge_probs = torch.sigmoid((attention - threshold) / temperature)

    # Density = Sum(probs) / Max_Edges
    max_edges = seq_len * seq_len
    density = edge_probs.sum(dim=(-2, -1)) / max_edges

    return density


def _compute_entropy(
    attention: torch.Tensor, 
    attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Compute normalized Shannon entropy."""
    batch_size, num_heads, seq_len, _ = attention.shape
    
    # 1. Normalize attention to be a valid probability distribution
    # (Attention is usually Softmaxed, but masking might zero out rows)
    attn_flat = attention.view(batch_size, num_heads, -1)
    attn_sum = attn_flat.sum(dim=-1, keepdim=True) + 1e-10
    attn_norm = attn_flat / attn_sum

    # 2. Compute Entropy: -Sum(p * log(p))
    # Add epsilon to mask zero values (log(0) -> -inf)
    log_attn = torch.log(attn_norm + 1e-10)
    entropy = -(attn_norm * log_attn).sum(dim=-1)

    # 3. Normalize by Max Entropy (log N)
    if attention_mask is not None:
        # Effective sequence length per sample
        valid_count = attention_mask.sum(dim=-1).float() # (Batch,)
        max_entropy = torch.log(valid_count * valid_count + 1e-10) # Max possible edges?
        # Note: Entropy is usually per ROW (source token).
        # Standard attention entropy is average of row entropies.
        # But this code treats the whole matrix as one distribution?
        # Let's check `train_f_regularized.py`.
        # It normalizes `attn_flat = attention.view(...)` so it sums to 1 over the whole matrix.
        # Yes, it treats the whole matrix as a distribution.
        # So Max Entropy is log(N*N).
        max_entropy = max_entropy.unsqueeze(1) # Broadcast to heads
    else:
        max_entropy = math.log(seq_len * seq_len)

    entropy_norm = entropy / (max_entropy + 1e-10)
    return entropy_norm


def _compute_soft_path_efficiency(
    attention: torch.Tensor,
    temperature: float,
    percentile: float,
    max_path_length: int
) -> torch.Tensor:
    """
    Approximates Structure Potential (SP) using matrix powers.
    SP ~ Global Efficiency = 1 / L_avg
    """
    batch_size, num_heads, seq_len, _ = attention.shape

    # 1. Soft Adjacency
    attn_flat = attention.view(batch_size, num_heads, -1)
    threshold = _compute_soft_threshold(attn_flat, percentile)
    adj = torch.sigmoid((attention - threshold) / temperature)

    # Add self-loops (identity) to ensure reachability doesn't decay purely by loss
    # and to represent "staying" at a node.
    eye = torch.eye(seq_len, device=attention.device).unsqueeze(0).unsqueeze(0)
    adj = adj + eye

    # 2. Matrix Powers for Reachability
    # Efficiency is proportional to "how easily can we reach nodes?"
    # We sum contributions from paths of length 1, 2, ..., k
    # Weight decays with path length (1/k).
    
    aggregated_efficiency = torch.zeros(batch_size, num_heads, device=attention.device)
    adj_power = adj.clone()

    for path_len in range(1, max_path_length + 1):
        if path_len > 1:
            adj_power = torch.matmul(adj_power, adj)
            # Clamp to prevent explosion in soft-logic, though sigmoid output is < 1.
            # Matmul of <1 adds up. 
            # We treat values > 0.5 as "connected".
            # For soft logic, we can just normalize or clip.
            # Keep values in a stable range without hard clipping.
            adj_power = adj_power / (1.0 + adj_power)

        # "Reachability" = Fraction of pairs connected with intensity > 0.5
        # This mirrors Global Efficiency's inverse path length.
        # (If connected at step 1, dist=1, contrib=1/1)
        # (If connected at step 2, dist=2, contrib=1/2) 
        # But we need to be careful not to double count direct connections as step 2.
        # This approximation assumes contributions accumulate. 
        # For a "Flash" metric, simple monotonic correlation with efficiency is enough.
        
        weight = 1.0 / path_len
        reachability = torch.sigmoid((adj_power - 0.5) / temperature).mean(dim=(-2, -1))
        aggregated_efficiency += weight * reachability

    return aggregated_efficiency / max_path_length
