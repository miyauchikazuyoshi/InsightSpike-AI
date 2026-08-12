"""Transformer adapter: maps attention tensors to unified geDIG F-eval.

Wraps the existing DifferentiableGeDIG pattern using the unified core:
  attention tensor (B, H, S, S) → TorchGraphSnapshot → FEval

Usage:
    from gedig.adapters.transformer import TransformerFEval

    f_eval = TransformerFEval(lambda_param=1.0, gamma=0.5, use_betti=False)
    result = f_eval.compute(before_attention, after_attention, mask=mask)
    # result.F_mean  → scalar F value
    # result.delta_epc, result.delta_h, result.delta_sp  → components

This adapter produces numerically equivalent results to the original
DifferentiableGeDIG.forward() in thermodynamic_gedig.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gedig.backends.torch_backend import (
    TorchGraphSnapshot,
    TorchEPC,
    TorchEntropy,
    TorchSP,
    TorchBetti,
)
from gedig.core.f_eval import FEval


@dataclass
class TransformerFEvalResult:
    """F-eval result for transformer attention.

    Extends FEvalResult with tensor-shaped outputs (B, H) and
    backward-compatible fields matching DifferentiableGeDIG.forward().
    """
    F: Any = None              # (B, H) tensor
    F_mean: Any = None         # scalar tensor
    delta_epc: Any = None      # scalar tensor
    delta_h: Any = None        # scalar tensor
    delta_sp: Any = None       # scalar tensor (always computed for comparison)
    delta_b1: Any = None       # scalar tensor (0 if use_betti=False)
    use_betti: bool = False
    f_formula: str = ""


class TransformerFEval:
    """Unified F-eval for transformer attention graphs.

    Drop-in replacement for DifferentiableGeDIG.forward() using
    the unified core backends. This is a before/after delta evaluator;
    lower F is better under the canonical judgment convention.

    Parameters
    ----------
    lambda_param : float
        Weight for (H + γ·B) term.
    gamma : float
        Weight for B relative to H.
    percentile : float
        Fraction of edges kept after soft thresholding.
    temperature : float
        Sigmoid temperature for soft thresholding.
    use_betti : bool
        If True, use β₁ instead of SP for the third component.
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        percentile: float = 0.9,
        temperature: float = 10.0,
        use_betti: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TransformerFEval")

        self.lambda_param = lambda_param
        self.gamma = gamma
        self.percentile = percentile
        self.temperature = temperature
        self.use_betti = use_betti

        # Backend components
        self.epc = TorchEPC()
        self.entropy = TorchEntropy()
        self.sp = TorchSP()
        self.betti = TorchBetti()

        # The F formula itself lives in the core FEval — this adapter only
        # chooses which structure-potential backend (SP or β₁) is injected.
        self._f_eval = FEval(
            epc=self.epc,
            entropy=self.entropy,
            sp=self.betti if use_betti else self.sp,
            lambda_param=lambda_param,
            gamma=gamma,
        )

    def compute(
        self,
        before_attention: "torch.Tensor",
        after_attention: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> TransformerFEvalResult:
        """Compute F between two attention states.

        Parameters
        ----------
        before_attention : torch.Tensor
            Reference attention (B, H, S, S).
        after_attention : torch.Tensor
            Current attention (B, H, S, S).
        mask : torch.Tensor or None
            Valid token mask (B, S).

        Returns
        -------
        TransformerFEvalResult with F (B, H) and component means.
        """
        # Create snapshots
        before = TorchGraphSnapshot(
            before_attention, mask,
            percentile=self.percentile,
            temperature=self.temperature,
        )
        after = TorchGraphSnapshot(
            after_attention, mask,
            percentile=self.percentile,
            temperature=self.temperature,
        )

        # Delegate F = ΔEPC - λ(ΔH + γΔB) to the unified core; components
        # come back per-(B, H) because the injected backends are torch.
        core = self._f_eval.compute(before, after)
        F = core.f_value
        delta_epc = core.delta_epc  # (B, H)
        delta_h = core.delta_h      # (B, H)

        if self.use_betti:
            delta_b1 = core.delta_b                   # (B, H)
            delta_sp = self.sp.compute(before, after)  # reported for comparison
        else:
            delta_sp = core.delta_b                   # (B, H)
            delta_b1 = torch.zeros_like(delta_sp)

        return TransformerFEvalResult(
            F=F,
            F_mean=F.mean(),
            delta_epc=delta_epc.mean(),
            delta_h=delta_h.mean(),
            delta_sp=delta_sp.mean(),
            delta_b1=delta_b1.mean(),
            use_betti=self.use_betti,
            f_formula="ΔEPC - λ(ΔH + γΔB)" if self.use_betti else "ΔEPC - λ(ΔH + γΔSP)",
        )
