#!/usr/bin/env python3
"""
層別geDIG v2分析

BERTの各層でgeDIG v2成分（F, H, SP）を計算し、
層方向の「相転移」を可視化する。

v2の改良点:
- SP = CLSへの経路集中度（アンカーベース）
- entropy_sign対応
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# geDIG v2をインポート
sys.path.insert(0, str(Path(__file__).parent))
from gedig_v2 import GeDIGv2


def analyze_layer_wise_gedig(
    model_name: str = "bert-base-uncased",
    dataset_name: str = "sst2",
    num_samples: int = 200,
    entropy_sign: int = -1,  # Fine-tuning用
    output_dir: str = "results/gedig_v2",
):
    """層別geDIG v2分析を実行"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"entropy_sign: {entropy_sign}")

    # モデルとトークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        output_attentions=True,
    ).to(device)
    model.eval()

    # データセット読み込み
    dataset = load_dataset("glue", dataset_name, split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    # geDIG v2インスタンス
    gedig = GeDIGv2(
        lambda_param=1.0,
        gamma=0.5,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=[0],  # CLS
        entropy_sign=entropy_sign,
    )

    num_layers = model.config.num_hidden_layers

    # 層ごとの統計を格納
    layer_stats = {
        layer: {"F": [], "H": [], "SP": []}
        for layer in range(num_layers)
    }

    print(f"\nAnalyzing {num_samples} samples...")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{num_samples}")

            # トークナイズ
            inputs = tokenizer(
                sample["sentence"],
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            ).to(device)

            # 推論
            outputs = model(**inputs)
            attentions = outputs.attentions  # tuple of (B, H, S, S) for each layer

            # 参照attention（一様分布）
            B, H, S, _ = attentions[0].shape
            ref_attn = gedig.compute_reference_attention(B, H, S, inputs["attention_mask"], device)

            # 各層を分析
            for layer_idx, attn in enumerate(attentions):
                # geDIG計算（一様分布との比較）
                result = gedig(ref_attn, attn, inputs["attention_mask"])

                layer_stats[layer_idx]["F"].append(result.F_mean)
                layer_stats[layer_idx]["H"].append(result.h_after)
                layer_stats[layer_idx]["SP"].append(result.sp_after)

    # 統計をまとめる
    summary = {}
    for layer in range(num_layers):
        summary[layer] = {
            "F_mean": float(np.mean(layer_stats[layer]["F"])),
            "F_std": float(np.std(layer_stats[layer]["F"])),
            "H_mean": float(np.mean(layer_stats[layer]["H"])),
            "H_std": float(np.std(layer_stats[layer]["H"])),
            "SP_mean": float(np.mean(layer_stats[layer]["SP"])),
            "SP_std": float(np.std(layer_stats[layer]["SP"])),
        }

    # 結果を保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"layer_analysis_entropy_sign_{entropy_sign}.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": {
                "model_name": model_name,
                "dataset": dataset_name,
                "num_samples": num_samples,
                "entropy_sign": entropy_sign,
            },
            "layer_stats": summary,
        }, f, indent=2)
    print(f"\nSaved to {result_file}")

    # 生データも保存（箱ひげ図用）
    raw_file = output_path / f"layer_analysis_raw_entropy_sign_{entropy_sign}.json"
    with open(raw_file, "w") as f:
        json.dump({
            "config": {
                "model_name": model_name,
                "dataset": dataset_name,
                "num_samples": num_samples,
                "entropy_sign": entropy_sign,
            },
            "layer_stats_raw": {str(k): v for k, v in layer_stats.items()},
        }, f, indent=2)
    print(f"Saved raw data to {raw_file}")

    # 可視化
    plot_layer_analysis(summary, output_path, entropy_sign, model_name)
    plot_layer_boxplot(layer_stats, output_path, entropy_sign, model_name)

    return summary


def plot_layer_analysis(summary, output_path, entropy_sign, model_name):
    """層別分析を可視化"""

    layers = sorted(summary.keys())
    F_means = [summary[l]["F_mean"] for l in layers]
    F_stds = [summary[l]["F_std"] for l in layers]
    H_means = [summary[l]["H_mean"] for l in layers]
    H_stds = [summary[l]["H_std"] for l in layers]
    SP_means = [summary[l]["SP_mean"] for l in layers]
    SP_stds = [summary[l]["SP_std"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # F値
    ax1 = axes[0]
    ax1.errorbar(layers, F_means, yerr=F_stds, marker='o', capsize=3, color='blue')
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("F (geDIG)")
    ax1.set_title(f"Layer-wise F (entropy_sign={entropy_sign})")
    ax1.grid(True, alpha=0.3)

    # エントロピー
    ax2 = axes[1]
    ax2.errorbar(layers, H_means, yerr=H_stds, marker='s', capsize=3, color='green')
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("H (Entropy)")
    ax2.set_title("Layer-wise Entropy")
    ax2.grid(True, alpha=0.3)

    # SP（ショートカット純度）
    ax3 = axes[2]
    ax3.errorbar(layers, SP_means, yerr=SP_stds, marker='^', capsize=3, color='red')
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("SP (Shortcut Purity)")
    ax3.set_title("Layer-wise SP (CLS concentration)")
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} - geDIG v2 Layer Analysis", fontsize=12)
    plt.tight_layout()

    fig_path = output_path / f"layer_analysis_entropy_sign_{entropy_sign}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

    # 論文用の簡潔な図（Fのみ）
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(layers, F_means, yerr=F_stds, marker='o', capsize=3,
                color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("F (geDIG v2)", fontsize=12)
    ax.set_title("Layer-wise geDIG F", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 相転移の領域をハイライト
    ax.axvspan(-0.5, 1.5, alpha=0.1, color='blue', label='Exploration phase')
    ax.axvspan(len(layers)-2.5, len(layers)-0.5, alpha=0.1, color='red', label='Structure phase')
    ax.legend(loc='lower right', fontsize=10)

    fig2_path = output_path / f"layer_f_paper_entropy_sign_{entropy_sign}.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved paper figure to {fig2_path}")
    plt.close()


def plot_layer_boxplot(layer_stats, output_path, entropy_sign, model_name):
    """箱ひげ図で層別F値を可視化"""

    layers = sorted(layer_stats.keys())
    F_data = [layer_stats[l]["F"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 箱ひげ図を作成
    bp = ax.boxplot(F_data, positions=layers, widths=0.6, patch_artist=True)

    # 色のグラデーション（浅層→深層で青→緑）
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # ランダムベースライン（参考）
    random_baseline = np.mean([np.mean(layer_stats[l]["F"]) for l in layers[:2]])
    ax.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7, label='Shallow layer avg')

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Free Energy (F)", fontsize=12)
    ax.set_title(f"Distribution of geDIG F-Score by Layer ({model_name})", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    fig_path = output_path / f"layer_f_boxplot_entropy_sign_{entropy_sign}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved boxplot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--entropy_sign", type=int, default=-1)
    args = parser.parse_args()

    analyze_layer_wise_gedig(
        model_name=args.model,
        num_samples=args.samples,
        entropy_sign=args.entropy_sign,
    )
