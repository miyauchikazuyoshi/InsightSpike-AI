#!/usr/bin/env python3
"""
geDIG Fの予測力検証実験

仮説: Fine-tuning前のF値が低いサンプルほど、学習されやすい（早く正解する）

実験手順:
1. 事前学習済みモデルで各サンプルのF値を計算
2. Fine-tuningしながら、各サンプルの正解タイミングを記録
3. 初期F値と学習しやすさの相関を分析
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from gedig_v2 import GeDIGv2


def compute_initial_f_values(model, tokenizer, dataset, gedig, device):
    """Fine-tuning前の各サンプルのF値を計算"""

    model.eval()
    f_values = []

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if (i + 1) % 100 == 0:
                print(f"  Computing F for sample {i + 1}/{len(dataset)}")

            inputs = tokenizer(
                sample["sentence"],
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            ).to(device)

            outputs = model(**inputs)
            attentions = outputs.attentions

            # 参照attention（一様分布）
            B, H, S, _ = attentions[0].shape
            ref_attn = gedig.compute_reference_attention(B, H, S, inputs["attention_mask"], device)

            # 全層のF値を平均
            layer_f_values = []
            for attn in attentions:
                result = gedig(ref_attn, attn, inputs["attention_mask"])
                layer_f_values.append(result.F_mean)

            f_values.append({
                "sample_idx": i,
                "initial_F": float(np.mean(layer_f_values)),
                "initial_F_std": float(np.std(layer_f_values)),
                "label": sample["label"],
            })

    return f_values


def fine_tune_with_tracking(model, tokenizer, train_dataset, eval_dataset,
                            device, num_epochs=5, lr=2e-5):
    """Fine-tuningしながら各サンプルの正解タイミングを追跡"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 各サンプルの学習履歴を追跡
    sample_history = defaultdict(lambda: {
        "first_correct_epoch": None,
        "correct_epochs": [],
        "predictions": [],
    })

    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        total_loss = 0
        for i, sample in enumerate(train_dataset):
            inputs = tokenizer(
                sample["sentence"],
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            ).to(device)

            labels = torch.tensor([sample["label"]]).to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 200 == 0:
                print(f"  Step {i + 1}, Loss: {total_loss / (i + 1):.4f}")

        # Evaluation - 各サンプルの正解を記録
        model.eval()
        correct = 0
        with torch.no_grad():
            for i, sample in enumerate(eval_dataset):
                inputs = tokenizer(
                    sample["sentence"],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                ).to(device)

                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
                is_correct = (pred == sample["label"])

                sample_history[i]["predictions"].append(pred)

                if is_correct:
                    correct += 1
                    sample_history[i]["correct_epochs"].append(epoch)
                    if sample_history[i]["first_correct_epoch"] is None:
                        sample_history[i]["first_correct_epoch"] = epoch

        acc = correct / len(eval_dataset)
        print(f"  Epoch {epoch + 1} Accuracy: {acc:.4f}")

        model.train()

    return dict(sample_history)


def analyze_correlation(f_values, sample_history, num_epochs):
    """F値と学習しやすさの相関を分析"""

    results = []

    for f_data in f_values:
        idx = f_data["sample_idx"]
        history = sample_history.get(idx, {})

        first_correct = history.get("first_correct_epoch")
        num_correct = len(history.get("correct_epochs", []))

        # 学習しやすさスコア（早く正解するほど高い）
        if first_correct is not None:
            learnability = (num_epochs - first_correct) / num_epochs
        else:
            learnability = 0.0  # 一度も正解しなかった

        results.append({
            "sample_idx": idx,
            "initial_F": f_data["initial_F"],
            "first_correct_epoch": first_correct,
            "num_correct_epochs": num_correct,
            "learnability": learnability,
            "label": f_data["label"],
        })

    return results


def plot_results(results, output_dir):
    """結果を可視化"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    initial_f = [r["initial_F"] for r in results]
    learnability = [r["learnability"] for r in results]
    first_correct = [r["first_correct_epoch"] if r["first_correct_epoch"] is not None else -1
                     for r in results]

    # 相関係数を計算
    valid_mask = [r["first_correct_epoch"] is not None for r in results]
    valid_f = [f for f, v in zip(initial_f, valid_mask) if v]
    valid_first = [fc for fc, v in zip(first_correct, valid_mask) if v]

    if len(valid_f) > 2:
        corr_first, p_first = stats.pearsonr(valid_f, valid_first)
    else:
        corr_first, p_first = 0, 1

    corr_learn, p_learn = stats.pearsonr(initial_f, learnability)

    print(f"\n=== 相関分析結果 ===")
    print(f"初期F vs 初回正解epoch: r={corr_first:.3f}, p={p_first:.4f}")
    print(f"初期F vs 学習しやすさ: r={corr_learn:.3f}, p={p_learn:.4f}")

    # Figure 1: F vs First Correct Epoch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    learned = [r for r in results if r["first_correct_epoch"] is not None]
    not_learned = [r for r in results if r["first_correct_epoch"] is None]

    if learned:
        ax1.scatter([r["initial_F"] for r in learned],
                   [r["first_correct_epoch"] for r in learned],
                   alpha=0.5, label=f"Learned (n={len(learned)})", color='blue')
    if not_learned:
        ax1.scatter([r["initial_F"] for r in not_learned],
                   [5.5] * len(not_learned),  # 5エポック後も正解しなかった
                   alpha=0.5, label=f"Not learned (n={len(not_learned)})",
                   color='red', marker='x')

    ax1.set_xlabel("Initial F (before fine-tuning)", fontsize=12)
    ax1.set_ylabel("First Correct Epoch", fontsize=12)
    ax1.set_title(f"Initial F vs Learning Speed\n(r={corr_first:.3f}, p={p_first:.4f})", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: F distribution by learnability
    ax2 = axes[1]

    # 学習しやすさでグループ分け
    easy = [r["initial_F"] for r in results if r["learnability"] >= 0.8]
    medium = [r["initial_F"] for r in results if 0.4 <= r["learnability"] < 0.8]
    hard = [r["initial_F"] for r in results if r["learnability"] < 0.4]

    data = [easy, medium, hard]
    labels = [f"Easy\n(n={len(easy)})", f"Medium\n(n={len(medium)})", f"Hard\n(n={len(hard)})"]

    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel("Initial F", fontsize=12)
    ax2.set_title("Initial F Distribution by Learnability", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_path / "f_predictive_power.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")
    plt.close()

    # 統計サマリーをJSON保存
    summary = {
        "correlation": {
            "F_vs_first_correct": {"r": corr_first, "p": p_first},
            "F_vs_learnability": {"r": corr_learn, "p": p_learn},
        },
        "sample_counts": {
            "total": len(results),
            "learned": len(learned),
            "not_learned": len(not_learned),
            "easy": len(easy),
            "medium": len(medium),
            "hard": len(hard),
        },
        "f_stats_by_learnability": {
            "easy": {"mean": np.mean(easy) if easy else 0, "std": np.std(easy) if easy else 0},
            "medium": {"mean": np.mean(medium) if medium else 0, "std": np.std(medium) if medium else 0},
            "hard": {"mean": np.mean(hard) if hard else 0, "std": np.std(hard) if hard else 0},
        },
    }

    summary_path = output_path / "f_predictive_power_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return summary


def main(
    model_name: str = "bert-base-uncased",
    num_train: int = 500,
    num_eval: int = 200,
    num_epochs: int = 5,
    output_dir: str = "results/f_predictive",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")

    # モデルとトークナイザー
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=True,
    ).to(device)

    # データセット
    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(num_train))
    eval_dataset = dataset.select(range(num_train, num_train + num_eval))

    # geDIG v2
    gedig = GeDIGv2(
        lambda_param=1.0,
        gamma=0.5,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=[0],
        entropy_sign=-1,
    )

    # Step 1: 初期F値を計算
    print("\n=== Step 1: Computing initial F values ===")
    f_values = compute_initial_f_values(model, tokenizer, eval_dataset, gedig, device)

    # Step 2: Fine-tuningしながら追跡
    print("\n=== Step 2: Fine-tuning with tracking ===")
    sample_history = fine_tune_with_tracking(
        model, tokenizer, train_dataset, eval_dataset,
        device, num_epochs=num_epochs
    )

    # Step 3: 相関分析
    print("\n=== Step 3: Analyzing correlation ===")
    results = analyze_correlation(f_values, sample_history, num_epochs)

    # Step 4: 可視化
    print("\n=== Step 4: Plotting results ===")
    summary = plot_results(results, output_dir)

    # 詳細データを保存
    output_path = Path(output_dir)
    with open(output_path / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_eval", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    main(
        model_name=args.model,
        num_train=args.num_train,
        num_eval=args.num_eval,
        num_epochs=args.epochs,
    )
