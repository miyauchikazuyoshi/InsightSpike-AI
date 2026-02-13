"""Metrics for fixed-model geDIG inference experiments.

Definitions implemented in this module:
  - H(l): vocab entropy from hidden state projected by unembedding matrix.
  - EPC(l): normalized Frobenius change of pairwise distance matrices.
  - SP(l): Spearman correlation of predicted depth vectors across layers.
  - B1(l): First Betti number from layer-wise distance graph.
  - F(l): delta_EPC(l) - lambda * (delta_H(l) + gamma * delta_structural(l)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

EPS = 1e-12
SP_MODES = {"spearman", "betti1", "both"}


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks with stable tie handling."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        average_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """Compute Spearman correlation without scipy dependency."""
    if x.numel() != y.numel() or x.numel() < 2:
        return None

    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = y.detach().cpu().numpy().astype(np.float64, copy=False)

    rx = _rankdata(x_np)
    ry = _rankdata(y_np)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom <= EPS:
        return None
    return float(np.dot(rx, ry) / denom)


def load_projection_matrix(
    path: Optional[str],
    hidden_dim: int,
    default_proj_dim: Optional[int],
) -> torch.Tensor:
    """Load projection matrix B from .npy or build identity projection."""
    if path:
        matrix = np.load(Path(path))
        if matrix.ndim != 2:
            raise ValueError(f"projection matrix must be rank-2: {path}")
        if matrix.shape[1] != hidden_dim:
            raise ValueError(
                f"projection dim mismatch for {path}: "
                f"expected second dim {hidden_dim}, got {matrix.shape[1]}"
            )
        return torch.tensor(matrix, dtype=torch.float32)

    proj_dim = hidden_dim
    if default_proj_dim is not None:
        if default_proj_dim <= 0:
            raise ValueError("default_proj_dim must be > 0")
        proj_dim = min(hidden_dim, int(default_proj_dim))

    matrix = torch.zeros((proj_dim, hidden_dim), dtype=torch.float32)
    matrix[:, :proj_dim] = torch.eye(proj_dim, dtype=torch.float32)
    return matrix


def compute_vocab_entropy(
    hidden: torch.Tensor,
    unembed_weight: torch.Tensor,
    temperature: float = 1.0,
    chunk_tokens: int = 8,
    token_indices: Optional[Sequence[int]] = None,
) -> float:
    """Compute Shannon entropy from vocab distribution (logit-lens style)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")

    hidden = hidden.to(dtype=torch.float32)
    unembed_weight = unembed_weight.to(dtype=torch.float32)
    if token_indices is not None:
        picked = [int(i) for i in token_indices if 0 <= int(i) < hidden.shape[0]]
        if picked:
            hidden = hidden[picked]

    entropies: List[torch.Tensor] = []
    for start in range(0, hidden.shape[0], chunk_tokens):
        chunk = hidden[start : start + chunk_tokens]
        logits = (chunk @ unembed_weight.t()) / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropies.append(entropy)

    return float(torch.cat(entropies, dim=0).mean().item())


def pairwise_distance_matrix(projected_hidden: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance matrix."""
    return torch.cdist(projected_hidden, projected_hidden, p=2)


def _fro_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(value, ord="fro")


def _normalize_distance_matrix(dist_mat: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize pairwise distance matrix scale for cross-layer comparability."""
    if mode == "none":
        return dist_mat
    if mode != "median":
        raise ValueError(f"unsupported distance_norm mode: {mode}")

    n = int(dist_mat.shape[0])
    if n < 2:
        return dist_mat
    tri_u = torch.triu_indices(n, n, offset=1)
    vals = dist_mat[tri_u[0], tri_u[1]]
    if vals.numel() == 0:
        return dist_mat
    denom = torch.median(vals)
    return dist_mat / (denom + EPS)


def _count_components(adj: np.ndarray) -> int:
    """Count connected components for an undirected adjacency matrix."""
    n = int(adj.shape[0])
    if n == 0:
        return 0

    visited = np.zeros((n,), dtype=bool)
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adj[node])
            for nb in neighbors:
                if visited[nb]:
                    continue
                visited[nb] = True
                stack.append(int(nb))

    return components


def _betti_1_from_distance_matrix(
    dist_mat: torch.Tensor,
    threshold: Optional[float] = None,
    k_neighbors: int = 5,
) -> float:
    """Compute first Betti number from a pairwise distance matrix.

    If `k_neighbors > 0`, uses a symmetric k-NN graph.
    Otherwise uses threshold graph (`distance < threshold`), where threshold
    defaults to the median of upper-triangle distances.
    """
    n = int(dist_mat.shape[0])
    if n < 2:
        return 0.0

    dist = dist_mat.detach().cpu().numpy().astype(np.float64, copy=False)
    adj = np.zeros((n, n), dtype=bool)

    if k_neighbors > 0:
        k_eff = min(int(k_neighbors), n - 1)
        for i in range(n):
            row = dist[i]
            order = np.argsort(row, kind="mergesort")
            picked = 0
            for j in order:
                j = int(j)
                if j == i:
                    continue
                adj[i, j] = True
                picked += 1
                if picked >= k_eff:
                    break
        # Undirected graph
        adj = np.logical_or(adj, adj.T)
    else:
        if threshold is None:
            tri_u = np.triu_indices(n, k=1)
            vals = dist[tri_u]
            if vals.size == 0:
                return 0.0
            threshold = float(np.median(vals))
        adj = dist < float(threshold)
        np.fill_diagonal(adj, False)
        adj = np.logical_or(adj, adj.T)

    edges = int(np.count_nonzero(np.triu(adj, k=1)))
    components = _count_components(adj)
    beta1 = edges - n + components
    return float(max(beta1, 0))


@dataclass
class LayerCurves:
    H: List[Optional[float]]
    EPC: List[Optional[float]]
    SP: List[Optional[float]]
    B1: List[Optional[float]]
    delta_H: List[Optional[float]]
    delta_EPC: List[Optional[float]]
    delta_SP: List[Optional[float]]
    delta_B1: List[Optional[float]]

    def as_dict(self) -> Dict[str, List[Optional[float]]]:
        return {
            "H": self.H,
            "EPC": self.EPC,
            "SP": self.SP,
            "B1": self.B1,
            "delta_H": self.delta_H,
            "delta_EPC": self.delta_EPC,
            "delta_SP": self.delta_SP,
            "delta_B1": self.delta_B1,
        }


def compute_layer_curves(
    hidden_states: Sequence[torch.Tensor],
    unembed_weight: torch.Tensor,
    b_dist: torch.Tensor,
    b_depth: torch.Tensor,
    temperature: float = 1.0,
    vocab_chunk_tokens: int = 8,
    sp_mode: str = "both",
    betti_k_neighbors: int = 5,
    betti_threshold: Optional[float] = None,
    entropy_token_indices: Optional[Sequence[int]] = None,
    distance_norm: str = "none",
) -> LayerCurves:
    """Compute H/EPC and structural metrics (SP/B1) layer by layer."""
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include at least embedding + 1 layer")
    if sp_mode not in SP_MODES:
        raise ValueError(f"sp_mode must be one of {sorted(SP_MODES)}, got: {sp_mode}")

    num_layers = len(hidden_states)
    h_values: List[Optional[float]] = [None] * num_layers
    epc_values: List[Optional[float]] = [None] * num_layers
    sp_values: List[Optional[float]] = [None] * num_layers
    b1_values: List[Optional[float]] = [None] * num_layers
    delta_h: List[Optional[float]] = [None] * num_layers
    delta_epc: List[Optional[float]] = [None] * num_layers
    delta_sp: List[Optional[float]] = [None] * num_layers
    delta_b1: List[Optional[float]] = [None] * num_layers

    dist_mats: List[torch.Tensor] = []
    depth_vectors: List[torch.Tensor] = []
    use_sp = sp_mode in {"spearman", "both"}
    use_betti = sp_mode in {"betti1", "both"}

    for idx, layer_hidden in enumerate(hidden_states):
        layer_hidden = layer_hidden.to(dtype=torch.float32)
        h_values[idx] = compute_vocab_entropy(
            hidden=layer_hidden,
            unembed_weight=unembed_weight,
            temperature=temperature,
            chunk_tokens=vocab_chunk_tokens,
            token_indices=entropy_token_indices,
        )

        z_dist = layer_hidden @ b_dist.t()
        dm = pairwise_distance_matrix(z_dist)
        dm = _normalize_distance_matrix(dm, mode=distance_norm)
        dist_mats.append(dm)
        if use_betti:
            b1_values[idx] = _betti_1_from_distance_matrix(
                dm,
                threshold=betti_threshold,
                k_neighbors=betti_k_neighbors,
            )

        if use_sp:
            z_depth = layer_hidden @ b_depth.t()
            depth = torch.sum(z_depth * z_depth, dim=-1)
            depth_vectors.append(depth)

        if idx >= 1:
            prev = dist_mats[idx - 1]
            curr = dist_mats[idx]
            epc = _fro_norm(curr - prev) / (_fro_norm(prev) + EPS)
            epc_values[idx] = float(epc.item())

            if use_sp:
                sp_values[idx] = spearman_corr(depth_vectors[idx], depth_vectors[idx - 1])

            if use_betti and b1_values[idx - 1] is not None and b1_values[idx] is not None:
                delta_b1[idx] = float(b1_values[idx] - b1_values[idx - 1])

            if h_values[idx - 1] is not None and h_values[idx] is not None:
                delta_h[idx] = float(h_values[idx] - h_values[idx - 1])

        if idx >= 2:
            if epc_values[idx] is not None and epc_values[idx - 1] is not None:
                delta_epc[idx] = float(epc_values[idx] - epc_values[idx - 1])
            if use_sp and sp_values[idx] is not None and sp_values[idx - 1] is not None:
                delta_sp[idx] = float(sp_values[idx] - sp_values[idx - 1])

    return LayerCurves(
        H=h_values,
        EPC=epc_values,
        SP=sp_values,
        B1=b1_values,
        delta_H=delta_h,
        delta_EPC=delta_epc,
        delta_SP=delta_sp,
        delta_B1=delta_b1,
    )


def compute_f_curve(
    delta_epc: Sequence[Optional[float]],
    delta_h: Sequence[Optional[float]],
    delta_sp: Sequence[Optional[float]],
    lambda_param: float,
    gamma: float,
) -> List[Optional[float]]:
    """Compute F(l) from delta components.

    Note: `delta_sp` is the structural term and may be either `delta_SP` or
    `delta_B1`, depending on experiment configuration.
    """
    if not (len(delta_epc) == len(delta_h) == len(delta_sp)):
        raise ValueError("all delta curves must have same length")

    f_values: List[Optional[float]] = [None] * len(delta_epc)
    for idx in range(len(delta_epc)):
        e = delta_epc[idx]
        h = delta_h[idx]
        s = delta_sp[idx]
        if e is None or h is None or s is None:
            continue
        f_values[idx] = float(e - lambda_param * (h + gamma * s))
    return f_values


def linear_fit_r2(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Fit y = ax + b on finite points and return slope/intercept/R^2."""
    points = [(idx, val) for idx, val in enumerate(values) if val is not None and np.isfinite(val)]
    if len(points) < 2:
        return {"slope": None, "intercept": None, "r2": None, "n_points": len(points)}

    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot <= EPS else 1.0 - ss_res / ss_tot

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_points": len(points),
    }


def mean_optional_curves(curves: Sequence[Sequence[Optional[float]]]) -> List[Optional[float]]:
    """Mean across curves while ignoring None/NaN values."""
    if not curves:
        return []

    length = max(len(curve) for curve in curves)
    out: List[Optional[float]] = [None] * length

    for idx in range(length):
        values = []
        for curve in curves:
            if idx >= len(curve):
                continue
            value = curve[idx]
            if value is None or not np.isfinite(value):
                continue
            values.append(float(value))
        if values:
            out[idx] = float(np.mean(values))
    return out


def monotonic_nonincreasing(values: Sequence[Optional[float]]) -> Optional[bool]:
    """Check monotonic non-increasing behavior on finite points."""
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    return all(finite[idx + 1] <= finite[idx] for idx in range(len(finite) - 1))


def grid_search_f(
    delta_epc: Sequence[Optional[float]],
    delta_h: Sequence[Optional[float]],
    delta_sp: Sequence[Optional[float]],
    lambda_values: Iterable[float],
    gamma_values: Iterable[float],
) -> Optional[Dict[str, object]]:
    """Grid-search lambda/gamma by maximizing R^2 of F-layer line fit."""
    best: Optional[Dict[str, object]] = None
    for lambda_value in lambda_values:
        for gamma_value in gamma_values:
            f_curve = compute_f_curve(
                delta_epc=delta_epc,
                delta_h=delta_h,
                delta_sp=delta_sp,
                lambda_param=lambda_value,
                gamma=gamma_value,
            )
            fit = linear_fit_r2(f_curve)
            r2 = fit["r2"]
            slope = fit["slope"]
            if r2 is None or slope is None:
                continue

            candidate = {
                "lambda": float(lambda_value),
                "gamma": float(gamma_value),
                "fit": fit,
                "f_curve": f_curve,
            }
            if best is None:
                best = candidate
                continue

            best_r2 = float(best["fit"]["r2"])  # type: ignore[index]
            best_slope = float(best["fit"]["slope"])  # type: ignore[index]
            if (r2 > best_r2 + 1e-12) or (
                abs(r2 - best_r2) <= 1e-12 and slope < best_slope
            ):
                best = candidate

    return best
