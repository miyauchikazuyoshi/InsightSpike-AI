
"""
Neuro-Pruning Tool
==================

Prunes Transformer heads based on their Structural Fitness (geDIG F-score).
Removes "neural clutter" (low F-score heads) to compress the model.

Usage:
    python experiments/neuro_pruning/prune_by_structure.py --model bert-base-uncased --amount 0.2
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    DataCollatorWithPadding,
)
from datasets import load_dataset

# Add src to path to import insightspike
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from insightspike.gedig import compute_f_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_heads(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20
) -> np.ndarray:
    """
    Compute average F-score for each head in the model.
    Returns: Global score matrix (num_layers, num_heads)
    """
    logger.info("Diagnosing model structure (Calculating F-scores)...")
    model.eval()
    
    # Storage for cumulative scores
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    total_f_scores = torch.zeros(num_layers, num_heads, device=device)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask").to(device) if "attention_mask" in batch else None

            # Forward pass to get attentions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            
            # shape of attentions: tuple of (batch, num_heads, seq, seq)
            attentions = outputs.attentions 

            for layer_idx, layer_attn in enumerate(attentions):
                # Compute F-score efficiently on GPU
                f_val, _ = compute_f_score(
                    layer_attn, 
                    attention_mask=attention_mask,
                    temperature=0.1,  # Sharp decision boundary for diagnosis
                    percentile=0.9
                )
                # f_val shape: (batch, num_heads)
                
                # Sum over batch
                total_f_scores[layer_idx] += f_val.sum(dim=0)
            
            count += input_ids.size(0)
            
    # Average
    avg_f_scores = total_f_scores / count
    return avg_f_scores.cpu().numpy()


def _tokenize_dataset(raw_dataset, tokenizer, with_labels: bool):
    def tokenize_function(examples):
        if "sentence" in examples:
            return tokenizer(examples["sentence"], truncation=True, max_length=128)
        if "sentence1" in examples and "sentence2" in examples:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
        if "text" in examples:
            return tokenizer(examples["text"], truncation=True, max_length=128)
        raise ValueError("Unsupported dataset schema for tokenization")

    tokenized = raw_dataset.map(tokenize_function, batched=True)
    keep = {"input_ids", "attention_mask"}
    if "token_type_ids" in tokenized.column_names:
        keep.add("token_type_ids")
    if with_labels:
        if "label" not in tokenized.column_names:
            raise ValueError("Dataset must include 'label' for evaluation")
        tokenized = tokenized.rename_column("label", "labels")
        keep.add("labels")
    tokenized = tokenized.remove_columns([col for col in tokenized.column_names if col not in keep])
    tokenized.set_format("torch")
    return tokenized


def _load_dataset_split(
    dataset: str,
    subset: Optional[str],
    split: str,
    samples: int,
    seed: Optional[int],
):
    ds = load_dataset(dataset, subset, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if samples > 0:
        limit = min(samples, len(ds))
        ds = ds.select(range(limit))
    return ds


def evaluate_classifier(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += outputs.loss.item() * labels.size(0)

    if total == 0:
        return {"accuracy": 0.0, "avg_loss": 0.0, "total": 0}

    return {
        "accuracy": correct / total,
        "avg_loss": loss_sum / total,
        "total": total,
    }


def select_heads_to_prune(
    f_scores: np.ndarray, 
    amount: float
) -> Dict[int, List[int]]:
    """
    Identify bottom `amount` (percentage) of heads based on F-score.
    Returns: Dict {layer_idx: [head_idx_to_prune, ...]}
    """
    num_layers, num_heads = f_scores.shape
    total_heads = num_layers * num_heads
    num_to_prune = int(total_heads * amount)
    
    logger.info(f"Selecting {num_to_prune} heads to prune (out of {total_heads})...")
    
    # Flatten and sort
    flat_indices = np.argsort(f_scores.flatten())
    prune_indices = flat_indices[:num_to_prune] # Lowest scores
    
    heads_to_prune = {}
    for idx in prune_indices:
        layer = int(idx // num_heads) # Cast to int for JSON serialization
        head = int(idx % num_heads)
        if layer not in heads_to_prune:
            heads_to_prune[layer] = []
        heads_to_prune[layer].append(head)
        
    return heads_to_prune


def main():
    parser = argparse.ArgumentParser(description="Neuro-Pruning by Structural Fitness")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--amount", type=float, default=0.1, help="Pruning ratio (0.0 - 1.0)")
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset name")
    parser.add_argument("--subset", type=str, default="sst2", help="Dataset subset")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed for diagnostic/eval sampling")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--diagnostic_split", type=str, default="train", help="Split for head diagnosis")
    parser.add_argument("--diagnostic_samples", type=int, default=100, help="Max diagnostic samples (0 = full split)")
    parser.add_argument("--max_batches", type=int, default=20, help="Max batches for head diagnosis")
    parser.add_argument("--eval", action="store_true", help="Evaluate classifier accuracy before/after pruning")
    parser.add_argument("--eval_split", type=str, default="validation", help="Dataset split for evaluation")
    parser.add_argument("--eval_samples", type=int, default=200, help="Max eval samples (0 = full split)")
    parser.add_argument("--eval_seed", type=int, default=None, help="Shuffle seed for eval sampling")
    parser.add_argument("--skip_save", action="store_true", help="Skip saving model weights")
    parser.add_argument("--output_dir", type=Path, default=Path("pruned_model"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 1. Load Model & Data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.eval:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
    else:
        model = AutoModel.from_pretrained(args.model)
    model.to(device)
    
    # Load calibration data (small subset)
    raw_dataset = _load_dataset_split(
        args.dataset,
        args.subset,
        args.diagnostic_split,
        args.diagnostic_samples,
        args.seed,
    )
    tokenized_datasets = _tokenize_dataset(raw_dataset, tokenizer, with_labels=False)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_datasets, batch_size=args.batch_size, collate_fn=collator)

    # 2. Diagnose (Calculate F-scores)
    original_params = model.num_parameters()
    logger.info(f"Original Parameters: {original_params}")
    
    f_scores = diagnose_heads(model, dataloader, device, max_batches=args.max_batches)
    
    # Log scores (optional: visualize later)
    # print("F-Scores matrix:\n", f_scores)

    # 3. Select Heads
    heads_to_prune = select_heads_to_prune(f_scores, args.amount)
    logger.info(f"Heads to prune: {heads_to_prune}")

    pre_eval = None
    post_eval = None

    eval_seed = None
    if args.eval:
        eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
        eval_dataset = _load_dataset_split(
            args.dataset,
            args.subset,
            args.eval_split,
            args.eval_samples,
            eval_seed,
        )
        eval_tokenized = _tokenize_dataset(eval_dataset, tokenizer, with_labels=True)
        eval_loader = DataLoader(eval_tokenized, batch_size=args.batch_size, collate_fn=collator)
        pre_eval = evaluate_classifier(model, eval_loader, device)

    # 4. Prune
    model.prune_heads(heads_to_prune)
    
    # 5. Save & Verify
    pruned_params = model.num_parameters()
    reduction = (original_params - pruned_params) / original_params * 100
    
    logger.info(f"Pruned Parameters: {pruned_params}")
    logger.info(f"Reduction: {reduction:.2f}%")

    if args.eval:
        post_eval = evaluate_classifier(model, eval_loader, device)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_save:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # Save meta-info
    prune_counts = {int(layer): len(heads) for layer, heads in heads_to_prune.items()}
    with open(args.output_dir / "pruning_info.json", "w") as f:
        json.dump(
            {
                "original_model": args.model,
                "pruning_amount": args.amount,
                "pruned_heads": heads_to_prune,
                "pruned_heads_per_layer": prune_counts,
                "f_scores": f_scores.tolist(),
                "param_reduction_percent": reduction,
                "seed": args.seed,
                "diagnostic_split": args.diagnostic_split,
                "diagnostic_samples": args.diagnostic_samples,
                "max_batches": args.max_batches,
                "eval_split": args.eval_split if args.eval else None,
                "eval_samples": args.eval_samples if args.eval else None,
                "eval_seed": eval_seed if args.eval else None,
                "pre_eval": pre_eval,
                "post_eval": post_eval,
            },
            f,
            indent=2,
        )

    report_path = args.output_dir / "pruning_report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Neuro-Pruning Report\n\n")
        fh.write(f"- model: {args.model}\n")
        fh.write(f"- pruning_ratio: {args.amount}\n")
        fh.write(f"- param_reduction_percent: {reduction:.2f}\n")
        fh.write(f"- pruned_heads_per_layer: {prune_counts}\n")
        if pre_eval and post_eval:
            fh.write(f"- pre_eval_accuracy: {pre_eval['accuracy']:.4f}\n")
            fh.write(f"- post_eval_accuracy: {post_eval['accuracy']:.4f}\n")
            fh.write(f"- pre_eval_loss: {pre_eval['avg_loss']:.4f}\n")
            fh.write(f"- post_eval_loss: {post_eval['avg_loss']:.4f}\n")
        
    if args.skip_save:
        logger.info("Skipped saving model weights.")
    else:
        logger.info(f"Pruned model saved to {args.output_dir}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
