#!/usr/bin/env python3
"""
geDIG v2 検証実験

実験仮説:
1. Transformer学習は geDIG F を改善（低下）させる方向に進む
2. F改善方向への介入は精度向上に寄与する
3. SP（ショートカット純度）が学習で向上する

実験設計:
- Phase 1: 微視的観察（学習ステップごとのF変化）
- Phase 2: 学習時介入（Baseline vs F↓ vs F↑）
- Phase 3: アブレーション（entropy_sign=1 vs -1）
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F_torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# geDIG v2
from gedig_v2 import GeDIGv2, GeDIGv2Result


# =============================================================================
# Experiment 1: 微視的観察
# =============================================================================

@dataclass
class MicroscopicResult:
    """微視的観察の結果"""
    total_steps: int = 0
    f_history: List[float] = field(default_factory=list)
    sp_history: List[float] = field(default_factory=list)
    h_history: List[float] = field(default_factory=list)
    epc_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)

    # 統計
    f_improved_steps: int = 0  # F が下がったステップ数
    sp_improved_steps: int = 0  # SP が上がったステップ数

    def compute_stats(self) -> Dict[str, float]:
        if len(self.f_history) < 2:
            return {}

        f_deltas = np.diff(self.f_history)
        sp_deltas = np.diff(self.sp_history)

        return {
            "total_steps": self.total_steps,
            "f_improvement_rate": (f_deltas < 0).mean(),
            "sp_improvement_rate": (sp_deltas > 0).mean(),
            "f_initial": self.f_history[0],
            "f_final": self.f_history[-1],
            "f_total_change": self.f_history[-1] - self.f_history[0],
            "sp_initial": self.sp_history[0],
            "sp_final": self.sp_history[-1],
            "sp_total_change": self.sp_history[-1] - self.sp_history[0],
            "mean_f": np.mean(self.f_history),
            "mean_sp": np.mean(self.sp_history),
        }


def run_microscopic_observation(
    model_name: str = "distilbert-base-uncased",
    num_samples: int = 500,
    num_steps: int = 100,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    entropy_sign: int = 1,  # geDIG基本原則
    output_dir: Optional[Path] = None,
) -> MicroscopicResult:
    """
    微視的観察: 学習の各ステップでgeDIG成分を追跡

    仮説: 学習が進むにつれて
    - F が下がる（良い変化）
    - SP が上がる（ショートカット形成）
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"entropy_sign: {entropy_sign} ({'延伸利得' if entropy_sign == 1 else '集中利得'})")

    # モデル
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, attn_implementation="eager"
    ).to(device)

    # データ
    dataset = load_dataset("glue", "sst2", split=f"train[:{num_samples}]")

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["sentence", "idx"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # geDIG v2
    gedig = GeDIGv2(
        lambda_param=1.0,
        gamma=0.5,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=[0],  # CLS
        entropy_sign=entropy_sign,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    result = MicroscopicResult()

    model.train()
    step = 0
    prev_attn = None  # 前ステップのattentionを保持

    print(f"\nRunning microscopic observation ({num_steps} steps)...")
    print("Reference: Previous step attention (step-to-step change)")

    for batch in tqdm(dataloader, total=min(num_steps, len(dataloader))):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        mask = batch.get("attention_mask")

        # Forward
        outputs = model(**batch, output_attentions=True)
        loss = outputs.loss

        # 現在のattention（最終層）
        current_attn = outputs.attentions[-1].detach()
        B, H, S, _ = current_attn.shape

        # 参照状態: 前ステップのattention（なければ一様分布）
        if prev_attn is None or prev_attn.shape != current_attn.shape:
            ref_attn = gedig.compute_reference_attention(B, H, S, mask, device)
        else:
            ref_attn = prev_attn

        # geDIG計算（Before → After の変化量）
        gedig_result = gedig(ref_attn, current_attn, mask)

        # 次ステップ用に保存
        prev_attn = current_attn.clone()

        # 記録
        result.f_history.append(gedig_result.F_mean)
        result.sp_history.append(gedig_result.sp_after)
        result.h_history.append(gedig_result.h_after)
        result.epc_history.append(gedig_result.delta_epc)
        result.loss_history.append(loss.item())

        # 学習ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    result.total_steps = step

    # 統計
    stats = result.compute_stats()

    print(f"\n{'='*60}")
    print("MICROSCOPIC OBSERVATION RESULTS (geDIG v2)")
    print(f"{'='*60}")
    print(f"Total steps: {stats.get('total_steps', 0)}")
    print(f"F improvement rate: {stats.get('f_improvement_rate', 0)*100:.1f}%")
    print(f"SP improvement rate: {stats.get('sp_improvement_rate', 0)*100:.1f}%")
    print(f"F: {stats.get('f_initial', 0):.4f} → {stats.get('f_final', 0):.4f} (Δ={stats.get('f_total_change', 0):.4f})")
    print(f"SP: {stats.get('sp_initial', 0):.4f} → {stats.get('sp_final', 0):.4f} (Δ={stats.get('sp_total_change', 0):.4f})")

    # 保存
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_data = {
            "config": {
                "model_name": model_name,
                "num_samples": num_samples,
                "num_steps": num_steps,
                "entropy_sign": entropy_sign,
            },
            "stats": stats,
            "f_history": result.f_history,
            "sp_history": result.sp_history,
            "h_history": result.h_history,
            "loss_history": result.loss_history,
        }

        filename = f"microscopic_entropy_sign_{entropy_sign}.json"
        (output_dir / filename).write_text(json.dumps(save_data, indent=2))
        print(f"Saved to {output_dir / filename}")

    return result


# =============================================================================
# Experiment 2: 学習時介入
# =============================================================================

def run_training_intervention(
    model_name: str = "distilbert-base-uncased",
    num_train_samples: int = 2000,
    num_eval_samples: int = 500,
    num_epochs: int = 3,
    alpha: float = 0.1,  # F正則化の強さ
    entropy_sign: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    学習時介入実験

    3つのモデルを比較:
    - Baseline: 通常学習（CE損失のみ）
    - Positive: F↓方向への誘導（良い変化を促進）
    - Negative: F↑方向への誘導（悪い変化を促進）

    仮説: Positive > Baseline > Negative の精度順
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"alpha: {alpha}, entropy_sign: {entropy_sign}")

    # データ準備
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    train_dataset = load_dataset("glue", "sst2", split=f"train[:{num_train_samples}]")
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.remove_columns(["sentence", "idx"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")

    eval_dataset = load_dataset("glue", "sst2", split=f"validation[:{num_eval_samples}]")
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.remove_columns(["sentence", "idx"])
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def train_model(mode: str) -> Dict[str, Any]:
        """モードに応じてモデルを学習"""
        print(f"\n{'='*40}")
        print(f"Training: {mode.upper()}")
        print(f"{'='*40}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, attn_implementation="eager"
        ).to(device)

        gedig = GeDIGv2(
            entropy_sign=entropy_sign,
            anchor_indices=[0],
        ).to(device)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collator)
        eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=collator)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        train_losses = []
        eval_accs = []
        f_values = []
        sp_values = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_losses = []
            epoch_f = []
            epoch_sp = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                mask = batch.get("attention_mask")

                outputs = model(**batch, output_attentions=True)
                ce_loss = outputs.loss

                current_attn = outputs.attentions[-1]
                B, H, S, _ = current_attn.shape
                ref_attn = gedig.compute_reference_attention(B, H, S, mask, device)

                gedig_result = gedig(ref_attn, current_attn, mask)

                # モードに応じた損失
                if mode == "baseline":
                    total_loss = ce_loss
                elif mode == "positive":
                    # F↓を促進 = Fを損失に加える
                    total_loss = ce_loss + alpha * gedig_result.F
                else:  # negative
                    # F↑を促進 = -Fを損失に加える
                    total_loss = ce_loss - alpha * gedig_result.F

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(ce_loss.item())
                epoch_f.append(gedig_result.F_mean)
                epoch_sp.append(gedig_result.sp_after)

            train_losses.extend(epoch_losses)
            f_values.extend(epoch_f)
            sp_values.extend(epoch_sp)

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += len(batch["labels"])

            accuracy = correct / total
            eval_accs.append(accuracy)

            print(f"  Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, "
                  f"acc={accuracy*100:.1f}%, F={np.mean(epoch_f):.4f}, SP={np.mean(epoch_sp):.4f}")

        return {
            "final_accuracy": eval_accs[-1],
            "eval_accuracy_history": eval_accs,
            "mean_f": np.mean(f_values[-len(epoch_f):]),  # 最終エポック
            "mean_sp": np.mean(sp_values[-len(epoch_sp):]),
        }

    # 3モード実行
    results = {}
    for mode in ["baseline", "positive", "negative"]:
        results[mode] = train_model(mode)

    # 結果表示
    print(f"\n{'='*60}")
    print("TRAINING INTERVENTION RESULTS (geDIG v2)")
    print(f"{'='*60}")

    baseline_acc = results["baseline"]["final_accuracy"]
    positive_acc = results["positive"]["final_accuracy"]
    negative_acc = results["negative"]["final_accuracy"]

    print(f"\nFinal Accuracy:")
    print(f"  Baseline:  {baseline_acc*100:.1f}%")
    print(f"  Positive:  {positive_acc*100:.1f}% (F↓ direction)")
    print(f"  Negative:  {negative_acc*100:.1f}% (F↑ direction)")

    print(f"\nDifferences:")
    print(f"  Positive - Baseline: {(positive_acc - baseline_acc)*100:+.1f}%")
    print(f"  Negative - Baseline: {(negative_acc - baseline_acc)*100:+.1f}%")
    print(f"  Positive - Negative: {(positive_acc - negative_acc)*100:+.1f}%")

    # 判定
    if positive_acc > negative_acc + 0.01:
        conclusion = "POSITIVE > NEGATIVE: geDIG F is a valid training signal"
    elif negative_acc > positive_acc + 0.01:
        conclusion = "NEGATIVE > POSITIVE: geDIG F direction may be inverted"
    else:
        conclusion = "INCONCLUSIVE: No significant difference"

    print(f"\nConclusion: {conclusion}")

    results["conclusion"] = conclusion
    results["config"] = {
        "alpha": alpha,
        "entropy_sign": entropy_sign,
        "num_train_samples": num_train_samples,
        "num_epochs": num_epochs,
    }

    # 保存
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"intervention_alpha_{alpha}_entropy_{entropy_sign}.json"
        (output_dir / filename).write_text(json.dumps(results, indent=2))
        print(f"Saved to {output_dir / filename}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="geDIG v2 Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["microscopic", "intervention", "ablation", "all"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--entropy-sign", type=int, default=1,
                        help="1=延伸利得(default), -1=集中利得(ablation)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/transformer/results/gedig_v2"))
    args = parser.parse_args()

    if args.experiment in ["microscopic", "all"]:
        print("\n" + "="*60)
        print("EXPERIMENT 1: Microscopic Observation")
        print("="*60)
        run_microscopic_observation(
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            entropy_sign=args.entropy_sign,
            output_dir=args.output_dir,
        )

    if args.experiment in ["intervention", "all"]:
        print("\n" + "="*60)
        print("EXPERIMENT 2: Training Intervention")
        print("="*60)
        run_training_intervention(
            num_train_samples=args.num_samples * 4,
            num_eval_samples=args.num_samples,
            num_epochs=args.num_epochs,
            alpha=args.alpha,
            entropy_sign=args.entropy_sign,
            output_dir=args.output_dir,
        )

    if args.experiment in ["ablation", "all"]:
        print("\n" + "="*60)
        print("EXPERIMENT 3: Entropy Sign Ablation")
        print("="*60)

        # 両モードで比較
        for entropy_sign in [1, -1]:
            print(f"\n--- entropy_sign={entropy_sign} ---")
            run_training_intervention(
                num_train_samples=args.num_samples * 4,
                num_eval_samples=args.num_samples,
                num_epochs=args.num_epochs,
                alpha=args.alpha,
                entropy_sign=entropy_sign,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
