#!/usr/bin/env python3
"""
Tiny grokking-style experiment on modular addition to capture a sharp generalization jump.

Generates (a, b) pairs with label (a+b mod p). Trains a 2-layer tiny Transformer.
Saves checkpoints every save_steps and logs train/val accuracy over steps.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModAddDataset(Dataset):
    def __init__(self, p: int, size: int, seed: int = 42):
        super().__init__()
        rng = random.Random(seed)
        self.p = p
        self.data: List[Tuple[int, int, int]] = []
        for _ in range(size):
            a = rng.randrange(p)
            b = rng.randrange(p)
            self.data.append((a, b, (a + b) % p))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a, b, y = self.data[idx]
        return {
            "input_ids": torch.tensor([a, b], dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
        }


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq=2]
        x = self.embed(input_ids)  # [b, seq, d]
        x = x.transpose(0, 1)  # [seq, b, d]
        x = self.enc(x)  # [seq, b, d]
        x = x.mean(dim=0)  # [b, d]
        logits = self.cls(x)  # [b, vocab]
        return logits


@dataclass
class Config:
    p: int = 113
    train_size: int = 512
    val_size: int = 2000
    batch_size: int = 128
    steps: int = 15000
    lr: float = 1e-3
    weight_decay: float = 0.1
    save_steps: int = 500
    seed: int = 42
    out_dir: Path = Path("results/transformer_gedig/grokking_mod_add")


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device))
            preds = logits.argmax(dim=-1)
            labels = batch["label"].to(device)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_ds = ModAddDataset(cfg.p, cfg.train_size, seed=cfg.seed)
    val_ds = ModAddDataset(cfg.p, cfg.val_size, seed=cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TinyTransformer(cfg.p, d_model=64, n_heads=2, n_layers=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log: List[Dict] = []
    global_step = 0
    for epoch in range(math.ceil(cfg.steps * cfg.batch_size / len(train_ds))):
        for batch in train_loader:
            global_step += 1
            model.train()
            logits = model(batch["input_ids"].to(DEVICE))
            loss = F.cross_entropy(logits, batch["label"].to(DEVICE))
            loss.backward()
            opt.step()
            opt.zero_grad()

            if global_step % cfg.save_steps == 0 or global_step == 1:
                val_acc = accuracy(model, val_loader, DEVICE)
                train_acc = accuracy(model, train_loader, DEVICE)
                entry = {"step": global_step, "train_acc": train_acc, "val_acc": val_acc, "loss": float(loss.item())}
                log.append(entry)
                ckpt_path = cfg.out_dir / f"checkpoint-{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[step {global_step}] loss={loss.item():.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
            if global_step >= cfg.steps:
                break
        if global_step >= cfg.steps:
            break

    cfg_dict = cfg.__dict__.copy()
    cfg_dict["out_dir"] = str(cfg_dict["out_dir"])
    summary = {"config": cfg_dict, "log": log}
    (cfg.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {cfg.out_dir/'run_summary.json'} with {len(log)} eval points")


if __name__ == "__main__":
    main()
