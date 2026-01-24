
"""
Flash-geDIG Module API
======================

PyTorch Modules for integrating geDIG into models and training loops.
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from . import functional as F_gedig

class FlashGeDIGLoss(nn.Module):
    """
    Differentiable geDIG Loss module.
    
    Add this value to your task loss to encourage structural inference.
    
    Example:
        criterion = FlashGeDIGLoss(lambda_param=1.0)
        loss = task_loss + criterion(model_outputs.attentions)
    """
    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        temperature: float = 0.1,
        percentile: float = 0.9,
        max_path_length: int = 4
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.temperature = temperature
        self.percentile = percentile
        self.max_path_length = max_path_length

    def forward(
        self, 
        attentions: Tuple[torch.Tensor], 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean F-score across all layers and heads.
        
        Args:
            attentions: Tuple of attention tensors (Batch, Heads, Seq, Seq) from HuggingFace model.
            attention_mask: Padding mask (Batch, Seq)
            
        Returns:
            loss: Scalar tensor representing the structural loss (Negative F to be minimized? 
                  Note: F is Structural Fitness. High F is good. 
                  So we return -F (or -alpha*F) to be minimized?
                  The user usually does `loss + alpha * gedig_loss`. 
                  If this module returns "F", user maximizes it.
                  Let's return NEGATIVE F_mean so minimizing this module MAXIMIZES Structure.
        """
        f_scores = []
        for ptr, layer_attn in enumerate(attentions):
            # layer_attn: (Batch, Heads, Seq, Seq)
            f_val, _ = F_gedig.compute_f_score(
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
        
        # We want to MAXIMIZE F (Structural Fitness).
        # Optimization minimizes Loss.
        # So return -F.
        return -f_mean


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
        attentions: Tuple[torch.Tensor], 
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, float]]:
        """
        Returns a list of dicts (one per layer) containing metrics.
        """
        results = []
        for layer_idx, layer_attn in enumerate(attentions):
            f_val, metrics = F_gedig.compute_f_score(
                layer_attn, 
                attention_mask,
                **self.config
            )
            
            layer_res = {
                "layer": layer_idx,
                "f_mean": f_val.mean().item(),
                "epc": metrics['delta_epc'].mean().item(),
                "entropy": metrics['delta_h'].mean().item(),
                "sp": metrics['delta_sp'].mean().item()
            }
            results.append(layer_res)
        return results
