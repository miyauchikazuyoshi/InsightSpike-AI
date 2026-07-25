
"""
Flash-geDIG Module API
======================

PyTorch Modules for integrating geDIG into models and training loops.
"""

from typing import Dict, Literal, Optional, Tuple, List
import torch
import torch.nn as nn
from . import functional as F_gedig

class FlashGeDIGLoss(nn.Module):
    """
    Differentiable single-state Flash-profile loss module.
    
    The objective direction is experiment-specific; canonical before/after
    delta F is exposed by ``compute_delta_f_score`` instead.
    
    Example:
        criterion = FlashGeDIGLoss(
            lambda_param=1.0,
            objective="maximize",  # historical experiment behavior
        )
        loss = task_loss + criterion(model_outputs.attentions)
    """
    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        temperature: float = 0.1,
        percentile: float = 0.9,
        max_path_length: int = 4,
        *,
        alpha: float = 1.0,
        objective: Literal["minimize", "maximize"] = "maximize",
    ):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if objective not in {"minimize", "maximize"}:
            raise ValueError(
                "objective must be either 'minimize' or 'maximize'"
            )
        self.alpha = alpha
        self.objective = objective
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.temperature = temperature
        self.percentile = percentile
        self.max_path_length = max_path_length

    def forward(
        self, 
        attentions: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the mean structural-profile score across layers and heads.
        
        Args:
            attentions: Tuple of attention tensors (Batch, Heads, Seq, Seq) from HuggingFace model.
            attention_mask: Padding mask (Batch, Seq)
            
        Returns:
            loss: Signed, alpha-scaled structural-profile objective. The
                  default ``objective="maximize"`` preserves the historical
                  ``-profile`` behavior. Canonical delta-F optimization uses
                  the separate before/after API and a minimize objective.
        """
        f_scores = []
        for ptr, layer_attn in enumerate(attentions):
            # layer_attn: (Batch, Heads, Seq, Seq)
            f_val, _ = F_gedig.compute_structural_profile(
                layer_attn,
                attention_mask,
                lambda_param=self.lambda_param,
                gamma=self.gamma,
                temperature=self.temperature,
                percentile=self.percentile,
                max_path_length=self.max_path_length
            )
            f_scores.append(f_val.mean())
            
        if not f_scores:
            return torch.tensor(0.0, device=attentions[0].device if attentions else 'cpu')
            
        f_mean = torch.stack(f_scores).mean()
        
        direction = 1.0 if self.objective == "minimize" else -1.0
        return self.alpha * direction * f_mean


class GeDIGObserver(nn.Module):
    """
    Passive observer module that computes metrics without affecting gradients (by default).
    Useful for diagnostics/logging.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @torch.no_grad()
    def measure(
        self, 
        attentions: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, float]]:
        """
        Returns a list of dicts (one per layer) containing metrics.
        """
        results = []
        for layer_idx, layer_attn in enumerate(attentions):
            f_val, metrics = F_gedig.compute_structural_profile(
                layer_attn, 
                attention_mask,
                **self.config
            )
            
            layer_res = {
                "layer": layer_idx,
                "f_mean": f_val.mean().item(),
                "epc": metrics["epc"].mean().item(),
                "entropy": metrics["h"].mean().item(),
                "sp": metrics["sp"].mean().item(),
            }
            results.append(layer_res)
        return results
