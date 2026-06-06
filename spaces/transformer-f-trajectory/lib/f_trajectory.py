"""Compute F-trajectory across Transformer layers.

This is a thin wrapper around the research code at:
    experiments/transformer/inference_f_trajectory/gedig_hidden.py

Used by app.py to compute F per layer for an input sentence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Import the research module (we keep the demo as a thin layer over it
# so improvements to the research code flow through automatically).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESEARCH_DIR = _PROJECT_ROOT / "experiments" / "transformer" / "inference_f_trajectory"
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

from gedig_hidden import compute_trajectory  # noqa: E402


@dataclass
class FTrajectory:
    """F-trajectory across all layers for one input sentence."""

    text: str
    model_name: str
    num_layers: int
    num_tokens: int

    # Per-layer arrays of length num_layers
    f_per_layer: List[float]
    cumulative_f: List[float]
    epc_per_layer: List[float]
    delta_h_per_layer: List[float]
    delta_sp_per_layer: List[float]

    # Aggregate metrics
    total_f: float
    mean_f: float
    monotonic: bool

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_tokens": self.num_tokens,
            "f_per_layer": self.f_per_layer,
            "cumulative_f": self.cumulative_f,
            "epc_per_layer": self.epc_per_layer,
            "delta_h_per_layer": self.delta_h_per_layer,
            "delta_sp_per_layer": self.delta_sp_per_layer,
            "total_f": self.total_f,
            "mean_f": self.mean_f,
            "monotonic": self.monotonic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FTrajectory":
        return cls(**d)


def _hidden_states(model, tokenizer, text: str, device: str) -> List[torch.Tensor]:
    """Run forward pass and return per-layer hidden states."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Each element: (batch, seq_len, hidden_dim). Drop the batch axis.
    return [hs.squeeze(0).detach().cpu() for hs in outputs.hidden_states]


def compute(
    model,
    tokenizer,
    text: str,
    *,
    model_name: str = "bert-base-uncased",
    anchor_idx: int = 0,
    lambda_: float = 1.0,
    gamma: float = 0.5,
    epc_method: str = "vector",
    device: str = "cpu",
) -> FTrajectory:
    """Compute F-trajectory for one input sentence.

    Args:
        model, tokenizer: HuggingFace model/tokenizer (loaded with output_hidden_states=True able)
        text: Input sentence.
        model_name: For display in result.
        anchor_idx: Token index used as graph anchor (0 = [CLS] for encoder, -1 = last for decoder).
        lambda_, gamma: F formula coefficients.
        epc_method: "vector" (L2 of token vector motion) or "similarity".
        device: "cpu" / "mps" / "cuda".

    Returns:
        FTrajectory.
    """
    hidden_states = _hidden_states(model, tokenizer, text, device)
    num_layers = len(hidden_states) - 1  # transitions between layers
    num_tokens = hidden_states[0].shape[0]

    results = compute_trajectory(
        hidden_states,
        anchor_idx=anchor_idx,
        lambda_=lambda_,
        gamma=gamma,
        epc_method=epc_method,
    )

    f_per_layer = [r.f_value for r in results]
    epc_per_layer = [r.epc for r in results]
    delta_h_per_layer = [r.delta_h for r in results]
    delta_sp_per_layer = [r.delta_sp for r in results]

    cumulative_f: List[float] = []
    running = 0.0
    for v in f_per_layer:
        running += v
        cumulative_f.append(running)

    total_f = cumulative_f[-1] if cumulative_f else 0.0
    mean_f = float(np.mean(f_per_layer)) if f_per_layer else 0.0

    monotonic = all(b >= a for a, b in zip(cumulative_f[:-1], cumulative_f[1:]))

    return FTrajectory(
        text=text,
        model_name=model_name,
        num_layers=num_layers,
        num_tokens=num_tokens,
        f_per_layer=f_per_layer,
        cumulative_f=cumulative_f,
        epc_per_layer=epc_per_layer,
        delta_h_per_layer=delta_h_per_layer,
        delta_sp_per_layer=delta_sp_per_layer,
        total_f=total_f,
        mean_f=mean_f,
        monotonic=monotonic,
    )


def load_model(model_name: str = "bert-base-uncased", device: str = "cpu"):
    """Load HuggingFace model + tokenizer ready for F-trajectory computation."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer
