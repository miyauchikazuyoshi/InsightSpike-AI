"""Unified F-eval: F = ΔEPC - λ(ΔH + γΔB).

Composes three component Protocols (EPC, Entropy, StructurePotential)
into a single F score. Backend implementations provide the components.

Usage:
    from gedig.core import FEval
    from gedig.backends.networkx_backend import NxEPC, NxEntropy, NxSP

    f_eval = FEval(
        epc=NxEPC(),
        entropy=NxEntropy(),
        sp=NxSP(),
        lambda_param=1.0,
        gamma=0.5,
    )
    result = f_eval.compute(before_snapshot, after_snapshot)
    print(result.f_value)  # F score
"""

from __future__ import annotations

from .protocols import (
    EPCComputer,
    EntropyComputer,
    FEvalResult,
    GraphSnapshot,
    StructurePotentialComputer,
)


class FEval:
    """Unified geDIG F-eval computation.

    F = ΔEPC - λ(ΔH + γΔB)

    Three components are injected via Protocol:
      - epc: structural change cost (GED / edge density)
      - entropy: information ordering change
      - sp: structure potential (path efficiency or β₁)

    Parameters
    ----------
    epc : EPCComputer
        Computes ΔEPC between before/after graph states.
    entropy : EntropyComputer
        Computes ΔH between before/after graph states.
    sp : StructurePotentialComputer
        Computes ΔSP or ΔB between before/after graph states.
    lambda_param : float
        Weight for the (H + γ·B) term. Default 1.0.
    gamma : float
        Weight for B relative to H within the second term. Default 0.5.
    """

    def __init__(
        self,
        epc: EPCComputer,
        entropy: EntropyComputer,
        sp: StructurePotentialComputer,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
    ):
        self.epc = epc
        self.entropy = entropy
        self.sp = sp
        self.lambda_param = lambda_param
        self.gamma = gamma

    def compute(
        self,
        before: GraphSnapshot,
        after: GraphSnapshot,
    ) -> FEvalResult:
        """Compute F between two graph states.

        Returns FEvalResult with f_value and individual components.
        """
        d_epc = self.epc.compute(before, after)
        d_h = self.entropy.compute(before, after)
        d_b = self.sp.compute(before, after)

        f = d_epc - self.lambda_param * (d_h + self.gamma * d_b)

        return FEvalResult(
            f_value=f,
            delta_epc=d_epc,
            delta_h=d_h,
            delta_b=d_b,
            metadata={
                "lambda": self.lambda_param,
                "gamma": self.gamma,
            },
        )

    def __repr__(self) -> str:
        return (
            f"FEval(lambda={self.lambda_param}, gamma={self.gamma}, "
            f"epc={type(self.epc).__name__}, "
            f"entropy={type(self.entropy).__name__}, "
            f"sp={type(self.sp).__name__})"
        )
