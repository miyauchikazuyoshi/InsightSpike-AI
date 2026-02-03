#!/usr/bin/env python3
"""
F-Regularized Training Experiment

Tests the causal hypothesis: Does minimizing geDIG F during training improve performance?

Experiment design:
- Baseline: standard CrossEntropy fine-tuning
- Treatment: L_total = L_CE + alpha * F_mean
- Compare across multiple alpha values: [0, 0.001, 0.01, 0.1, 1.0]

Key insight: If F-regularization improves performance, this demonstrates
that geDIG F is not just correlated with good attention, but causally
contributes to it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# -----------------------------------------------------------------------------
# Differentiable geDIG Calculator
# -----------------------------------------------------------------------------

@dataclass
class DifferentiableGeDIG:
    """
    Computes geDIG F in a differentiable manner for backpropagation.

    F = delta_epc - lambda * (delta_h + gamma * delta_sp)

    Key adaptations for differentiability:
    - Edge density: soft thresholding with sigmoid
    - Entropy: directly from attention weights
    - Path efficiency: approximated via soft adjacency powers
    """
    lambda_param: float = 1.0
    gamma: float = 0.5
    temperature: float = 0.1  # for soft thresholding
    percentile: float = 0.9
    max_path_length: int = 4  # for SP approximation

    def compute_F(
        self,
        attention: torch.Tensor,  # (batch, heads, seq, seq)
        attention_mask: Optional[torch.Tensor] = None,  # (batch, seq)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute differentiable geDIG F.

        Returns dict with F, delta_epc, delta_h, delta_sp (all differentiable).
        """
        batch_size, num_heads, seq_len, _ = attention.shape

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to (batch, 1, seq, seq)
            mask_2d = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
            mask_2d = mask_2d.float()
            attention = attention * mask_2d

        # Compute per-head metrics
        delta_epc = self._compute_soft_density(attention)  # (batch, heads)
        delta_h = self._compute_entropy(attention, attention_mask)  # (batch, heads)
        delta_sp = self._compute_soft_path_efficiency(attention, attention_mask)  # (batch, heads)

        # Combine: F = delta_epc - lambda * (delta_h + gamma * delta_sp)
        F_values = delta_epc - self.lambda_param * (delta_h + self.gamma * delta_sp)

        return {
            "F": F_values,  # (batch, heads)
            "F_mean": F_values.mean(),  # scalar
            "delta_epc": delta_epc,
            "delta_h": delta_h,
            "delta_sp": delta_sp,
        }

    def _compute_soft_density(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Soft edge density using percentile-based thresholding.

        Instead of hard threshold, use sigmoid around the percentile value.
        """
        batch_size, num_heads, seq_len, _ = attention.shape

        # Compute threshold per head (percentile of attention values)
        attn_flat = attention.view(batch_size, num_heads, -1)  # (B, H, S*S)
        k = int(self.percentile * seq_len * seq_len)
        threshold = torch.kthvalue(attn_flat, k, dim=-1).values  # (B, H)
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # Soft thresholding with sigmoid
        edge_probs = torch.sigmoid((attention - threshold) / self.temperature)

        # Density = sum of edge probabilities / max edges
        max_edges = seq_len * seq_len
        density = edge_probs.sum(dim=(-2, -1)) / max_edges  # (B, H)

        return density

    def _compute_entropy(
        self,
        attention: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Normalized entropy of attention distribution.

        H_norm = -sum(p * log(p)) / log(N)
        """
        batch_size, num_heads, seq_len, _ = attention.shape

        # Flatten attention per head
        attn_flat = attention.view(batch_size, num_heads, -1)  # (B, H, S*S)

        # Normalize to ensure sum = 1 per head
        attn_norm = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute entropy
        log_attn = torch.log(attn_norm + 1e-10)
        entropy = -(attn_norm * log_attn).sum(dim=-1)  # (B, H)

        # Normalize by max entropy
        if attention_mask is not None:
            # Count valid positions per batch
            valid_count = attention_mask.sum(dim=-1).float()  # (B,)
            max_entropy = torch.log(valid_count * valid_count + 1e-10).unsqueeze(1)  # (B, 1)
        else:
            max_entropy = math.log(seq_len * seq_len)

        entropy_norm = entropy / (max_entropy + 1e-10)

        return entropy_norm

    def _compute_soft_path_efficiency(
        self,
        attention: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate path efficiency using soft adjacency matrix powers.

        Key insight: (A^k)_ij > 0 means there's a path of length k from i to j.
        We use soft thresholding to create adjacency, then compute reachability.

        Path efficiency = 1 / avg_shortest_path ≈ sum of inverse path lengths
        """
        batch_size, num_heads, seq_len, _ = attention.shape

        # Create soft adjacency matrix
        attn_flat = attention.view(batch_size, num_heads, -1)
        k = int(self.percentile * seq_len * seq_len)
        threshold = torch.kthvalue(attn_flat, k, dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        adj = torch.sigmoid((attention - threshold) / self.temperature)

        # Add self-loops for numerical stability
        eye = torch.eye(seq_len, device=attention.device).unsqueeze(0).unsqueeze(0)
        adj = adj + eye

        # Compute path contributions for each length
        # For path length k: contribution = (1/k) * (A^k has path)
        path_efficiency = torch.zeros(batch_size, num_heads, device=attention.device)

        adj_power = adj.clone()
        for path_len in range(1, self.max_path_length + 1):
            if path_len > 1:
                adj_power = torch.matmul(adj_power, adj)
                adj_power = torch.clamp(adj_power, 0, 1)  # Normalize

            # Contribution from paths of this length
            # Higher weight for shorter paths
            weight = 1.0 / path_len
            reachability = (adj_power > 0.5).float().mean(dim=(-2, -1))
            path_efficiency = path_efficiency + weight * reachability

        # Normalize
        path_efficiency = path_efficiency / self.max_path_length

        return path_efficiency


# -----------------------------------------------------------------------------
# F-Regularized Model Wrapper
# -----------------------------------------------------------------------------

class FRegularizedModel(nn.Module):
    """
    Wrapper that adds geDIG F regularization to the loss.
    """

    def __init__(
        self,
        base_model: nn.Module,
        alpha: float = 0.1,
        gedig_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.gedig = DifferentiableGeDIG(**(gedig_config or {}))

        # Storage for logging
        self._last_gedig_metrics: Optional[Dict[str, float]] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Get base model outputs with attentions
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            **kwargs,
        )

        if labels is not None and self.alpha > 0:
            # Compute F regularization from all attention layers
            all_attentions = outputs.attentions  # tuple of (B, H, S, S)

            f_values = []
            for layer_attn in all_attentions:
                gedig_out = self.gedig.compute_F(layer_attn, attention_mask)
                f_values.append(gedig_out["F_mean"])

            f_mean = torch.stack(f_values).mean()

            # Modified loss: L_total = L_CE + alpha * F_mean
            # Note: We want to MINIMIZE F, so we add it to the loss
            total_loss = outputs.loss + self.alpha * f_mean

            # Store metrics for logging
            self._last_gedig_metrics = {
                "f_mean": f_mean.item(),
                "ce_loss": outputs.loss.item(),
                "total_loss": total_loss.item(),
            }

            # Return modified outputs WITHOUT attentions to avoid size mismatch in eval
            return SequenceClassifierOutput(
                loss=total_loss,
                logits=outputs.logits,
                hidden_states=None,
                attentions=None,
            )

        # For eval without labels, also don't return attentions
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=None,
            attentions=None,
        )


# -----------------------------------------------------------------------------
# Custom Trainer with F Logging
# -----------------------------------------------------------------------------

class FRegularizedTrainer(Trainer):
    """Trainer that logs geDIG metrics."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # Log geDIG metrics if available
        if hasattr(model, "_last_gedig_metrics") and model._last_gedig_metrics:
            self.log(model._last_gedig_metrics)

        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------

def run_experiment(
    alpha: float,
    model_name: str = "distilbert-base-uncased",
    train_samples: int = 1000,
    eval_samples: int = 500,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a single F-regularization experiment.
    """
    set_seed(seed)

    if output_dir is None:
        output_dir = Path(f"results/transformer_gedig/f_reg/alpha_{alpha}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running experiment: alpha={alpha}")
    print(f"{'='*60}\n")

    # Load data
    ds_train = load_dataset("glue", "sst2", split=f"train[:{train_samples}]")
    ds_eval = load_dataset("glue", "sst2", split=f"validation[:{eval_samples}]")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    train_ds = ds_train.map(tokenize_fn, batched=True)
    eval_ds = ds_eval.map(tokenize_fn, batched=True)

    # Remove unused columns
    cols_to_remove = [c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "label")]
    train_ds = train_ds.remove_columns(cols_to_remove).with_format("torch")
    eval_ds = eval_ds.remove_columns(cols_to_remove).with_format("torch")

    # Load model
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if alpha > 0:
        model = FRegularizedModel(base_model, alpha=alpha)
    else:
        model = base_model

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        report_to=[],
        load_best_model_at_end=False,
        seed=seed,
    )

    # Metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    # Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = FRegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    train_result = trainer.train()

    # Final evaluation
    eval_result = trainer.evaluate()

    # Compute final geDIG metrics on eval set
    final_f_metrics = compute_final_gedig_metrics(model, eval_ds, tokenizer, data_collator)

    # Summary
    result = {
        "alpha": alpha,
        "model_name": model_name,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "epochs": epochs,
        "seed": seed,
        "final_accuracy": eval_result.get("eval_accuracy"),
        "final_loss": eval_result.get("eval_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "final_gedig": final_f_metrics,
    }

    # Save
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nResult: accuracy={result['final_accuracy']:.4f}, F_mean={final_f_metrics.get('f_mean', 'N/A')}")

    return result


def compute_final_gedig_metrics(
    model: nn.Module,
    eval_dataset,
    tokenizer,
    data_collator,
) -> Dict[str, float]:
    """Compute geDIG metrics on the full eval set."""
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()

    dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator)
    gedig = DifferentiableGeDIG()

    all_f_values = []
    all_epc = []
    all_h = []
    all_sp = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Handle wrapped model
            if hasattr(model, "base_model"):
                outputs = model.base_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_attentions=True,
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_attentions=True,
                )

            for layer_attn in outputs.attentions:
                metrics = gedig.compute_F(layer_attn, batch.get("attention_mask"))
                all_f_values.append(metrics["F"].mean().item())
                all_epc.append(metrics["delta_epc"].mean().item())
                all_h.append(metrics["delta_h"].mean().item())
                all_sp.append(metrics["delta_sp"].mean().item())

    return {
        "f_mean": np.mean(all_f_values),
        "f_std": np.std(all_f_values),
        "delta_epc_mean": np.mean(all_epc),
        "delta_h_mean": np.mean(all_h),
        "delta_sp_mean": np.mean(all_sp),
    }


def run_alpha_sweep(
    alphas: List[float] = [0.0, 0.001, 0.01, 0.1, 1.0],
    seeds: List[int] = [42, 123, 456],
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run experiments across multiple alpha values and seeds.
    """
    results = []

    for alpha in alphas:
        for seed in seeds:
            result = run_experiment(alpha=alpha, seed=seed, **kwargs)
            results.append(result)

    return results


def analyze_results(results: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    """
    Analyze results and create summary statistics.
    """
    import pandas as pd

    df = pd.DataFrame(results)

    # Group by alpha
    summary = df.groupby("alpha").agg({
        "final_accuracy": ["mean", "std"],
        "final_loss": ["mean", "std"],
    }).reset_index()

    summary.columns = ["alpha", "acc_mean", "acc_std", "loss_mean", "loss_std"]

    # Best alpha
    best_idx = summary["acc_mean"].idxmax()
    best_alpha = summary.loc[best_idx, "alpha"]
    best_acc = summary.loc[best_idx, "acc_mean"]

    # Baseline (alpha=0) comparison
    baseline = summary[summary["alpha"] == 0.0]
    if not baseline.empty:
        baseline_acc = baseline["acc_mean"].values[0]
        improvement = best_acc - baseline_acc
    else:
        baseline_acc = None
        improvement = None

    analysis = {
        "summary": summary.to_dict(),
        "best_alpha": best_alpha,
        "best_accuracy": best_acc,
        "baseline_accuracy": baseline_acc,
        "improvement": improvement,
        "conclusion": (
            f"Best alpha={best_alpha} achieves {best_acc:.4f} accuracy. "
            f"{'Improvement over baseline: ' + f'{improvement:.4f}' if improvement else 'No baseline for comparison.'}"
        ),
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, default=str))

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\n{summary.to_string()}")
    print(f"\n{analysis['conclusion']}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="F-Regularization Experiment")
    parser.add_argument("--alpha", type=float, default=None, help="Single alpha value to test")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run full alpha sweep")
    parser.add_argument("--alphas", type=str, default="0,0.001,0.01,0.1,1.0", help="Comma-separated alphas for sweep")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/transformer_gedig/f_reg"))
    args = parser.parse_args()

    if args.alpha_sweep or args.alpha is None:
        alphas = [float(x) for x in args.alphas.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]

        results = run_alpha_sweep(
            alphas=alphas,
            seeds=seeds,
            model_name=args.model,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        # Save all results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        all_results_path = args.output_dir / "all_results.json"
        all_results_path.write_text(json.dumps(results, indent=2))

        # Analyze
        analyze_results(results, args.output_dir / "analysis.json")
    else:
        run_experiment(
            alpha=args.alpha,
            model_name=args.model,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir / f"alpha_{args.alpha}",
        )


if __name__ == "__main__":
    main()
