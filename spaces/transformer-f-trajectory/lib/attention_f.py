"""Compute attention-based F per layer × head — the canonical formula used
in the JSAI 2026 paper (Section 3) and in experiments/transformer/results/phase1.

Uses the official `AttentionGeDIGCalculator` from
`src/insightspike/algorithms/gedig/attention.py` so this demo reproduces
the exact same numbers as the paper.

Formula (per head, per layer):
    F = ΔEPC − λ·γ·ΔSP − λ·ΔH

    ΔEPC = |E| / L²              (edge density of thresholded attention)
    ΔSP  = path efficiency        (1 / mean shortest-path on weakly conn comp)
    ΔH   = normalised entropy     (H(attn) / log L²)

Defaults match the paper: λ=1.0, γ=0.5, top-10% percentile threshold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Use the canonical calculator from the main codebase. We import the
# `attention` module directly to avoid the package `__init__` pulling in
# heavyweight optional dependencies like `torch_geometric`.
_ATTN_MOD = (
    Path(__file__).resolve().parents[3]
    / "src" / "insightspike" / "algorithms" / "gedig" / "attention.py"
)
import importlib.util as _ilu

_MOD_NAME = "_canonical_attention_gedig"
_spec = _ilu.spec_from_file_location(_MOD_NAME, _ATTN_MOD)
_mod = _ilu.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _mod  # required for dataclass introspection
_spec.loader.exec_module(_mod)
AttentionGeDIGCalculator = _mod.AttentionGeDIGCalculator
AttentionGeDIGConfig = _mod.AttentionGeDIGConfig


@dataclass
class AttentionTrajectory:
    """Per-layer aggregated F values from a single forward pass.

    Mirrors the schema of experiments/transformer/results/phase1/score_full.json
    aggregated to per-layer (mean over heads).
    """

    text: str
    model_name: str
    num_layers: int
    num_heads: int
    num_tokens: int

    # Real attention
    f_per_layer: List[float]              # length = num_layers, mean over heads
    epc_per_layer: List[float]
    h_per_layer: List[float]
    sp_per_layer: List[float]

    # Per (layer × head) F values, for heatmap
    f_layer_head: List[List[float]]       # shape (num_layers, num_heads)

    # Random baseline (mean over a few random matrices, per layer)
    f_random_per_layer: List[float]

    # Aggregate
    f_mean_real: float
    f_mean_random: float
    delta_f: float                        # f_mean_real − f_mean_random
    win_rate: float                       # fraction of (layer×head) where Real > Random

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_tokens": self.num_tokens,
            "f_per_layer": self.f_per_layer,
            "epc_per_layer": self.epc_per_layer,
            "h_per_layer": self.h_per_layer,
            "sp_per_layer": self.sp_per_layer,
            "f_layer_head": self.f_layer_head,
            "f_random_per_layer": self.f_random_per_layer,
            "f_mean_real": self.f_mean_real,
            "f_mean_random": self.f_mean_random,
            "delta_f": self.delta_f,
            "win_rate": self.win_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttentionTrajectory":
        return cls(**d)


def _attentions(model, tokenizer, text: str, device: str):
    """Run forward pass and return (attention tensors, valid_mask).

    attentions: tuple of tensors, one per layer, shape (batch=1, heads, L, L)
    valid_mask: bool array shape (L,) marking non-padding tokens
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # tuple of (1, H, L, L)
    valid = inputs["attention_mask"].squeeze(0).bool().cpu().numpy()
    return attentions, valid


def compute(
    model,
    tokenizer,
    text: str,
    *,
    model_name: str = "bert-base-uncased",
    lambda_: float = 0.5,
    gamma: float = 0.5,
    percentile: float = 0.9,
    device: str = "cpu",
    n_random: int = 3,
    rng_seed: int = 42,
) -> AttentionTrajectory:
    """Compute attention-based F-trajectory for one sentence.

    Returns per-layer mean F (over heads), plus a random-attention baseline.
    """
    cfg = AttentionGeDIGConfig(
        lambda_param=lambda_,
        gamma=gamma,
        threshold=0.0,
        use_percentile=True,
        percentile=percentile,
        undirected_sp=True,
    )
    calc = AttentionGeDIGCalculator(cfg)

    attentions, valid = _attentions(model, tokenizer, text, device)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    L = valid.shape[0]
    num_valid_tokens = int(valid.sum())

    f_layer_head: List[List[float]] = []
    epc_per_layer: List[float] = []
    h_per_layer: List[float] = []
    sp_per_layer: List[float] = []
    f_real_per_layer: List[float] = []

    rng = np.random.default_rng(rng_seed)
    f_rand_per_layer: List[float] = []
    real_wins = 0
    total_compared = 0

    for layer_idx, attn_layer in enumerate(attentions):
        head_arr = attn_layer.squeeze(0).cpu().numpy()  # (H, L, L)
        f_heads: List[float] = []
        epc_heads: List[float] = []
        h_heads: List[float] = []
        sp_heads: List[float] = []

        for h_idx in range(num_heads):
            attn = head_arr[h_idx]  # (L, L)
            result = calc.compute(attn, valid)
            f_heads.append(float(result.F))
            epc_heads.append(float(result.delta_epc))
            h_heads.append(float(result.delta_h))
            sp_heads.append(float(result.delta_sp))

        # Per-layer random baseline (mean over n_random shuffles + heads)
        rand_f_samples: List[float] = []
        for _ in range(n_random):
            for h_idx in range(num_heads):
                # Random attention: row-stochastic random matrix
                raw = rng.uniform(0.0, 1.0, size=(L, L))
                raw = raw / raw.sum(axis=1, keepdims=True).clip(min=1e-12)
                rand_result = calc.compute(raw, valid)
                rand_f_samples.append(float(rand_result.F))

        f_layer_head.append(f_heads)
        epc_per_layer.append(float(np.mean(epc_heads)))
        h_per_layer.append(float(np.mean(h_heads)))
        sp_per_layer.append(float(np.mean(sp_heads)))
        f_real_per_layer.append(float(np.mean(f_heads)))
        f_rand_per_layer.append(float(np.mean(rand_f_samples)))

        # win counts per head pair
        for h_idx, f_real in enumerate(f_heads):
            # Compare each real head against the random baseline mean for this layer
            if f_real > f_rand_per_layer[-1]:
                real_wins += 1
            total_compared += 1

    f_mean_real = float(np.mean(f_real_per_layer))
    f_mean_random = float(np.mean(f_rand_per_layer))
    return AttentionTrajectory(
        text=text,
        model_name=model_name,
        num_layers=num_layers,
        num_heads=num_heads,
        num_tokens=num_valid_tokens,
        f_per_layer=f_real_per_layer,
        epc_per_layer=epc_per_layer,
        h_per_layer=h_per_layer,
        sp_per_layer=sp_per_layer,
        f_layer_head=f_layer_head,
        f_random_per_layer=f_rand_per_layer,
        f_mean_real=f_mean_real,
        f_mean_random=f_mean_random,
        delta_f=f_mean_real - f_mean_random,
        win_rate=real_wins / max(total_compared, 1),
    )


def load_model(model_name: str = "bert-base-uncased", device: str = "cpu"):
    """Load HuggingFace model + tokenizer for attention-based analysis."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    model.to(device)
    return model, tokenizer
