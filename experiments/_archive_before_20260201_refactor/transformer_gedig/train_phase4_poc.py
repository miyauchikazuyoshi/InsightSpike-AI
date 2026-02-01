#!/usr/bin/env python3
"""
Phase4 PoC: small fine-tuning run on SST-2 to produce checkpoints for geDIG tracking.
- Trains DistilBERT on a tiny SST-2 subset (200 train / 200 eval).
- Saves checkpoints every N steps.
- Writes a summary JSON with per-checkpoint eval accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


def load_sst2_splits(train_count: int = 200, eval_count: int = 200) -> Dict[str, List]:
    ds_train = load_dataset("glue", "sst2", split=f"train[:{train_count}]")
    ds_eval = load_dataset("glue", "sst2", split=f"validation[:{eval_count}]")
    return {"train": ds_train, "eval": ds_eval}


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)


def compute_accuracy(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


def main() -> None:
    set_seed(42)
    model_name = "distilbert-base-uncased"
    out_dir = Path("results/transformer_gedig/checkpoints/distilbert_sst2_poc")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_sst2_splits()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = {
        "train": raw["train"].map(lambda x: tokenize_function(tokenizer, x), batched=True),
        "eval": raw["eval"].map(lambda x: tokenize_function(tokenizer, x), batched=True),
    }
    tokenized = {
        k: v.remove_columns([c for c in v.column_names if c not in ("input_ids", "attention_mask", "label")]).with_format("torch")
        for k, v in tokenized.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=10,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        report_to=[],
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # collect checkpoint accuracies
    checkpoints = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    ckpt_metrics: Dict[str, float] = {}
    for ckpt in checkpoints:
        model_ck = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        trainer_ck = Trainer(
            model=model_ck,
            args=args,
            eval_dataset=tokenized["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_accuracy,
        )
        m = trainer_ck.evaluate()
        ckpt_metrics[ckpt.name] = m.get("eval_accuracy")

    summary = {
        "final_eval_accuracy": eval_metrics.get("eval_accuracy"),
        "checkpoints": ckpt_metrics,
        "train_samples": len(tokenized["train"]),
        "eval_samples": len(tokenized["eval"]),
    }
    Path(out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] saved checkpoints under {out_dir}, summary: {summary}")


if __name__ == "__main__":
    main()
