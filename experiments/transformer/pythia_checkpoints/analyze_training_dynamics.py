#!/usr/bin/env python3
"""
Pythiaチェックポイントを用いたgeDIG学習ダイナミクス分析

仮説: 学習が進むにつれて
- SP（ショートカット純度）が上昇する（効率的な経路が形成される）
- H（エントロピー）が適切な値に収束する
- F値が変化する

Pythiaは学習途中のチェックポイントを公開しているため、
自分で学習せずに学習過程全体を観察できる。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# geDIG v2をインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from gedig_v2 import GeDIGv2


# Pythiaのチェックポイント（step数）
# 全部: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 131000, 143000
# 代表的なものを選択
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 131000, 143000]

# 軽量版（デバッグ用）
CHECKPOINTS_LIGHT = [0, 64, 512, 2000, 8000, 33000, 143000]


def analyze_checkpoint(
    model_name: str,
    revision: str,
    tokenizer,
    gedig: GeDIGv2,
    test_texts: list[str],
    device: torch.device,
) -> dict:
    """特定のチェックポイントでgeDIG成分を計算"""

    print(f"  Loading {revision}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        output_attentions=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    f_values = []
    h_values = []
    sp_values = []
    epc_values = []

    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=64,
                truncation=True,
                padding="max_length",
            ).to(device)

            outputs = model(**inputs)
            attentions = outputs.attentions  # tuple of (B, H, S, S)

            # 参照attention（一様分布）
            B, num_heads, S, _ = attentions[0].shape
            ref_attn = gedig.compute_reference_attention(
                B, num_heads, S, inputs["attention_mask"], device
            )

            # 各層のgeDIG成分を計算
            layer_f = []
            layer_h = []
            layer_sp = []
            layer_epc = []

            for attn in attentions:
                result = gedig(ref_attn, attn, inputs["attention_mask"])
                layer_f.append(result.F_mean)
                layer_h.append(result.h_after)
                layer_sp.append(result.sp_after)
                layer_epc.append(result.delta_epc)

            f_values.append(np.mean(layer_f))
            h_values.append(np.mean(layer_h))
            sp_values.append(np.mean(layer_sp))
            epc_values.append(np.mean(layer_epc))

    # モデルをメモリから解放
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "F_mean": float(np.mean(f_values)),
        "F_std": float(np.std(f_values)),
        "H_mean": float(np.mean(h_values)),
        "H_std": float(np.std(h_values)),
        "SP_mean": float(np.mean(sp_values)),
        "SP_std": float(np.std(sp_values)),
        "EPC_mean": float(np.mean(epc_values)),
        "EPC_std": float(np.std(epc_values)),
    }


def run_analysis(
    model_name: str = "EleutherAI/pythia-70m",
    checkpoints: list[int] = None,
    num_samples: int = 20,
    output_dir: str = "results",
):
    """全チェックポイントで分析を実行"""

    if checkpoints is None:
        checkpoints = CHECKPOINTS_LIGHT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")

    # トークナイザー（全チェックポイントで共通）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # テスト文章（シンプルな英文）
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "The weather today is sunny and warm.",
        "Scientists discovered a new species in the rainforest.",
        "The stock market showed significant gains yesterday.",
        "Artificial intelligence is advancing rapidly.",
        "The concert was absolutely amazing last night.",
        "Climate change is affecting ecosystems worldwide.",
        "The new smartphone has impressive features.",
        "Education is the foundation of a successful society.",
        "The restaurant served delicious Italian cuisine.",
        "Space exploration continues to reveal new mysteries.",
        "The athlete broke the world record in swimming.",
        "Technology is reshaping how we communicate.",
        "The museum exhibition attracted thousands of visitors.",
        "Renewable energy is becoming more affordable.",
        "The novel received critical acclaim from reviewers.",
        "Healthcare innovations are improving patient outcomes.",
        "The city skyline looks beautiful at sunset.",
    ][:num_samples]

    # geDIG v2（Causal LMなので最後のトークンをアンカーに）
    # ただしCausal LMにはCLSがないので、全トークンの平均的なパターンを見る
    gedig = GeDIGv2(
        lambda_param=1.0,
        gamma=0.5,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=None,  # アンカーなし（全体の統計）
        entropy_sign=-1,
    )

    results = {}

    print(f"\nAnalyzing {len(checkpoints)} checkpoints...")

    for i, step in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] Step {step}")

        revision = f"step{step}"

        try:
            stats = analyze_checkpoint(
                model_name, revision, tokenizer, gedig, test_texts, device
            )
            results[step] = stats
            print(f"    F={stats['F_mean']:.3f}, EPC={stats['EPC_mean']:.3f}, H={stats['H_mean']:.3f}, SP={stats['SP_mean']:.3f}")
        except Exception as e:
            print(f"    Error: {e}")
            results[step] = None

    # 結果を保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / "training_dynamics.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": {
                "model_name": model_name,
                "checkpoints": checkpoints,
                "num_samples": num_samples,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results to {result_file}")

    # 可視化
    plot_training_dynamics(results, checkpoints, output_path, model_name)

    return results


def plot_training_dynamics(results, checkpoints, output_path, model_name):
    """学習ダイナミクスを可視化"""

    # 有効な結果のみ抽出
    valid_steps = [s for s in checkpoints if results.get(s) is not None]

    F_vals = [results[s]["F_mean"] for s in valid_steps]
    H_vals = [results[s]["H_mean"] for s in valid_steps]
    SP_vals = [results[s]["SP_mean"] for s in valid_steps]
    EPC_vals = [results[s]["EPC_mean"] for s in valid_steps]

    F_stds = [results[s]["F_std"] for s in valid_steps]
    H_stds = [results[s]["H_std"] for s in valid_steps]
    SP_stds = [results[s]["SP_std"] for s in valid_steps]
    EPC_stds = [results[s]["EPC_std"] for s in valid_steps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # F値
    ax1 = axes[0, 0]
    ax1.errorbar(valid_steps, F_vals, yerr=F_stds, marker='o', capsize=3, color='blue')
    ax1.set_xscale('symlog')
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("F (geDIG)", fontsize=12)
    ax1.set_title("F value during training", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # EPC（構造変化コスト）
    ax2 = axes[0, 1]
    ax2.errorbar(valid_steps, EPC_vals, yerr=EPC_stds, marker='d', capsize=3, color='purple')
    ax2.set_xscale('symlog')
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("EPC (Structure Change)", fontsize=12)
    ax2.set_title("EPC: Structure Change Cost", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # エントロピー H
    ax3 = axes[1, 0]
    ax3.errorbar(valid_steps, H_vals, yerr=H_stds, marker='s', capsize=3, color='green')
    ax3.set_xscale('symlog')
    ax3.set_xlabel("Training Step", fontsize=12)
    ax3.set_ylabel("H (Entropy)", fontsize=12)
    ax3.set_title("Entropy during training", fontsize=14)
    ax3.grid(True, alpha=0.3)

    # ショートカット純度 SP
    ax4 = axes[1, 1]
    ax4.errorbar(valid_steps, SP_vals, yerr=SP_stds, marker='^', capsize=3, color='red')
    ax4.set_xscale('symlog')
    ax4.set_xlabel("Training Step", fontsize=12)
    ax4.set_ylabel("SP (Shortcut Purity)", fontsize=12)
    ax4.set_title("Shortcut Purity during training", fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} - geDIG Training Dynamics", fontsize=14)
    plt.tight_layout()

    fig_path = output_path / "training_dynamics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

    # 論文用：SPの変化に焦点を当てた図
    fig2, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(valid_steps, SP_vals, yerr=SP_stds, marker='o', capsize=3,
                color='steelblue', linewidth=2, markersize=8)
    ax.set_xscale('symlog')
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Shortcut Purity (SP)", fontsize=14)
    ax.set_title("Formation of Structural Shortcuts during Training", fontsize=16)
    ax.grid(True, alpha=0.3)

    # 相転移の領域をマーク
    ax.axhline(y=SP_vals[0], color='gray', linestyle='--', alpha=0.5, label='Initial SP')
    ax.axhline(y=SP_vals[-1], color='green', linestyle='--', alpha=0.5, label='Final SP')
    ax.legend(fontsize=12)

    fig2_path = output_path / "sp_formation.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved SP figure to {fig2_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m",
                       help="Pythia model name (70m, 160m, 410m, etc.)")
    parser.add_argument("--light", action="store_true",
                       help="Use fewer checkpoints for faster testing")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of test samples")
    args = parser.parse_args()

    checkpoints = CHECKPOINTS_LIGHT if args.light else CHECKPOINTS

    run_analysis(
        model_name=args.model,
        checkpoints=checkpoints,
        num_samples=args.samples,
        output_dir="results",
    )
