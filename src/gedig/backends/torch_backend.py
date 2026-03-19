"""PyTorch backend for F-eval components (differentiable).

Used by transformer experiments. Operates on attention tensors (B, H, S, S).

Extracts from:
  - thermodynamic_gedig.py: DifferentiableGeDIG._compute_*
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for torch_backend. pip install torch")


# ─── Graph Snapshot ──────────────────────────────────────────────

class TorchGraphSnapshot:
    """Attention tensor wrapped as a GraphSnapshot.

    Wraps (B, H, S, S) attention with soft thresholding.

    Parameters
    ----------
    attention : torch.Tensor
        Shape (B, H, S, S).
    mask : torch.Tensor or None
        Shape (B, S). 1 for valid, 0 for padding.
    percentile : float
        Fraction of edges kept after thresholding.
    temperature : float
        Sigmoid temperature for soft thresholding.
    """

    def __init__(
        self,
        attention: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        percentile: float = 0.9,
        temperature: float = 10.0,
    ):
        _require_torch()
        self.attention = attention
        self.mask = mask
        self.percentile = percentile
        self.temperature = temperature

        # Compute soft edges
        self.soft_edges, self.valid_counts = self._compute_soft_edges()

    def _compute_soft_edges(self):
        B, H, S, _ = self.attention.shape

        if self.mask is not None:
            mask_2d = self.mask.unsqueeze(1).unsqueeze(2) * self.mask.unsqueeze(1).unsqueeze(3)
            attn = self.attention * mask_2d
            valid_counts = self.mask.sum(dim=1, keepdim=True).float()
        else:
            attn = self.attention
            valid_counts = torch.full((B, 1), S, device=self.attention.device, dtype=torch.float)

        # Percentile threshold
        attn_flat = attn.view(B, H, -1)
        k = max(1, int((1 - self.percentile) * S * S))
        threshold = torch.kthvalue(attn_flat, k, dim=-1).values
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)

        soft_edges = torch.sigmoid(self.temperature * (attn - threshold))
        return soft_edges, valid_counts

    def node_count(self) -> int:
        return self.attention.shape[-1]

    def edge_count(self) -> int:
        return int(self.soft_edges.sum().item())

    def edge_set(self) -> set:
        return set()  # Continuous graph — no discrete edge set


# ─── EPC (Soft Edge Density Change) ─────────────────────────────

class TorchEPC:
    """Soft edge density change via sigmoid thresholding.

    ΔEPC = |soft_edges_after - soft_edges_before| / max_edges
    """

    def compute(
        self,
        before: TorchGraphSnapshot,
        after: TorchGraphSnapshot,
    ) -> "torch.Tensor":
        max_edges = after.valid_counts.squeeze(-1) ** 2  # (B,)
        edge_diff = torch.abs(after.soft_edges - before.soft_edges)
        delta_epc = edge_diff.sum(dim=(-2, -1)) / (max_edges.unsqueeze(1) + 1e-9)
        return delta_epc  # (B, H)


# ─── Entropy ─────────────────────────────────────────────────────

class TorchEntropy:
    """Shannon entropy change of attention distributions."""

    def compute(
        self,
        before: TorchGraphSnapshot,
        after: TorchGraphSnapshot,
    ) -> "torch.Tensor":
        h_before = self._entropy(before.attention)
        h_after = self._entropy(after.attention)
        return h_after - h_before  # (B, H)

    def _entropy(self, attention: "torch.Tensor") -> "torch.Tensor":
        attn_norm = attention / (attention.sum(dim=-1, keepdim=True) + 1e-9)
        entropy = -(attn_norm * torch.log(attn_norm + 1e-9)).sum(dim=-1)
        return entropy.mean(dim=-1)  # (B, H)


# ─── Structure Potential: SP (Path Efficiency) ───────────────────

class TorchSP:
    """Path efficiency via matrix power approximation.

    efficiency = E + E²/2 + E³/3
    """

    def compute(
        self,
        before: TorchGraphSnapshot,
        after: TorchGraphSnapshot,
    ) -> "torch.Tensor":
        max_edges = after.valid_counts.squeeze(-1) ** 2
        sp_before = self._path_efficiency(before.soft_edges, max_edges)
        sp_after = self._path_efficiency(after.soft_edges, max_edges)
        return sp_after - sp_before  # (B, H)

    def _path_efficiency(
        self,
        soft_edges: "torch.Tensor",
        max_edges: "torch.Tensor",
    ) -> "torch.Tensor":
        reach_2 = torch.matmul(soft_edges, soft_edges)
        reach_3 = torch.matmul(reach_2, soft_edges)
        efficiency = soft_edges + reach_2 / 2 + reach_3 / 3
        sp = efficiency.sum(dim=(-2, -1)) / (max_edges.unsqueeze(1) + 1e-9)
        return sp / 3


# ─── Structure Potential: Betti-1 ────────────────────────────────

class TorchBetti:
    """Differentiable β₁ via Laplacian eigenvalue counting.

    β₁ = E - V + C where C is approximated by counting
    near-zero eigenvalues of the graph Laplacian.
    """

    def __init__(self, eps: float = 0.1, t_soft: float = 20.0):
        self.eps = eps
        self.t_soft = t_soft

    def compute(
        self,
        before: TorchGraphSnapshot,
        after: TorchGraphSnapshot,
    ) -> "torch.Tensor":
        b1_before = self._betti_1(before.soft_edges, before.valid_counts)
        b1_after = self._betti_1(after.soft_edges, after.valid_counts)
        return b1_after - b1_before  # (B, H)

    def _betti_1(
        self,
        soft_edges: "torch.Tensor",
        valid_counts: "torch.Tensor",
    ) -> "torch.Tensor":
        B, H, S, _ = soft_edges.shape

        # Symmetrize
        A = (soft_edges + soft_edges.transpose(-2, -1)) / 2

        # Edge count
        E = A.sum(dim=(-2, -1)) / 2  # (B, H)

        # Vertex count
        V = valid_counts.squeeze(-1)  # (B,)

        # Laplacian eigenvalues for connected components
        degree = A.sum(dim=-1)
        L = torch.diag_embed(degree) - A
        eigenvalues = torch.linalg.eigvalsh(L)  # (B, H, S)

        # Soft count of near-zero eigenvalues
        C_soft = torch.sigmoid(self.t_soft * (self.eps - eigenvalues)).sum(dim=-1)

        # β₁ = E - V + C, normalized
        beta_1 = E - V.unsqueeze(1) + C_soft
        max_edges = V ** 2
        return beta_1 / (max_edges.unsqueeze(1) + 1e-9)
