#!/usr/bin/env python3
"""Pythia checkpoint training-dynamics analysis (attention-based, with B definition).

This keeps attention as the primary observable and supports two structural terms:
- `sp`: original shortcut-purity delta from geDIGv2
- `betti1`: B term defined as delta of attention-graph Betti-1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# geDIG v2 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from gedig_v2 import GeDIGv2


CHECKPOINTS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    33000,
    66000,
    131000,
    143000,
]
CHECKPOINTS_LIGHT = [0, 64, 512, 2000, 8000, 33000, 143000]


def _parse_checkpoints_arg(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("no checkpoints parsed from --checkpoints")
    return items


def _count_components(adj: np.ndarray) -> int:
    n = int(adj.shape[0])
    if n == 0:
        return 0
    visited = np.zeros((n,), dtype=bool)
    comps = 0
    for start in range(n):
        if visited[start]:
            continue
        comps += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adj[node])
            for nb in neighbors:
                if visited[nb]:
                    continue
                visited[nb] = True
                stack.append(int(nb))
    return comps


def _betti1_from_attention_2d(
    attn_2d: torch.Tensor,
    k_neighbors: int = 5,
    threshold: Optional[float] = None,
) -> float:
    """Betti-1 on symmetric graph derived from attention matrix."""
    n = int(attn_2d.shape[0])
    if n < 2:
        return 0.0

    # Use undirected weight from mean of both directions.
    w = 0.5 * (attn_2d + attn_2d.t())
    arr = w.detach().cpu().numpy().astype(np.float64, copy=False)
    adj = np.zeros((n, n), dtype=bool)

    if k_neighbors > 0:
        k_eff = min(int(k_neighbors), n - 1)
        for i in range(n):
            row = arr[i]
            # descending by weight
            order = np.argsort(-row, kind="mergesort")
            picked = 0
            for j in order:
                j = int(j)
                if j == i:
                    continue
                adj[i, j] = True
                picked += 1
                if picked >= k_eff:
                    break
        adj = np.logical_or(adj, adj.T)
    else:
        tri_u = np.triu_indices(n, k=1)
        vals = arr[tri_u]
        if vals.size == 0:
            return 0.0
        thr = float(np.median(vals)) if threshold is None else float(threshold)
        adj = arr > thr
        np.fill_diagonal(adj, False)
        adj = np.logical_or(adj, adj.T)

    edges = int(np.count_nonzero(np.triu(adj, k=1)))
    comps = _count_components(adj)
    beta1 = edges - n + comps
    return float(max(beta1, 0))


def _batch_betti1(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    k_neighbors: int = 5,
    threshold: Optional[float] = None,
) -> float:
    """Mean Betti-1 across batch x heads with valid-token masking."""
    bsz, n_heads, seq_len, _ = attention.shape
    vals: List[float] = []
    for b in range(bsz):
        if attention_mask is None:
            valid = torch.arange(seq_len)
        else:
            valid = torch.nonzero(attention_mask[b] > 0, as_tuple=False).squeeze(-1)
        if valid.numel() < 2:
            continue
        for h in range(n_heads):
            mat = attention[b, h][valid][:, valid]
            vals.append(
                _betti1_from_attention_2d(
                    attn_2d=mat,
                    k_neighbors=k_neighbors,
                    threshold=threshold,
                )
            )
    if not vals:
        return 0.0
    return float(np.mean(vals))


def analyze_checkpoint(
    model_name: str,
    revision: str,
    tokenizer,
    gedig: GeDIGv2,
    test_texts: List[str],
    device: torch.device,
    structural_term: str = "betti1",
    betti_k_neighbors: int = 5,
    betti_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute checkpoint statistics with selectable structural term."""
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
    b1_values = []
    epc_values = []
    delta_b1_values = []

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
            attentions = outputs.attentions  # tuple (B, H, S, S)
            if not attentions:
                continue

            bsz, n_heads, seq_len, _ = attentions[0].shape
            ref_attn = gedig.compute_reference_attention(
                bsz,
                n_heads,
                seq_len,
                inputs["attention_mask"],
                device,
            )

            layer_f = []
            layer_h = []
            layer_sp = []
            layer_b1 = []
            layer_epc = []
            layer_delta_b1 = []

            for attn in attentions:
                result = gedig(ref_attn, attn, inputs["attention_mask"])

                b1_before = _batch_betti1(
                    attention=ref_attn,
                    attention_mask=inputs["attention_mask"],
                    k_neighbors=betti_k_neighbors,
                    threshold=betti_threshold,
                )
                b1_after = _batch_betti1(
                    attention=attn,
                    attention_mask=inputs["attention_mask"],
                    k_neighbors=betti_k_neighbors,
                    threshold=betti_threshold,
                )
                delta_b1 = float(b1_after - b1_before)

                if structural_term == "betti1":
                    delta_struct = delta_b1
                else:
                    delta_struct = float(result.delta_sp)

                # Keep geDIG sign convention.
                effective_delta_h = gedig.entropy_sign * float(result.delta_h)
                f_val = float(result.delta_epc) - gedig.lambda_param * (
                    effective_delta_h + gedig.gamma * delta_struct
                )

                layer_f.append(f_val)
                layer_h.append(float(result.h_after))
                layer_sp.append(float(result.sp_after))
                layer_b1.append(b1_after)
                layer_epc.append(float(result.delta_epc))
                layer_delta_b1.append(delta_b1)

            if not layer_f:
                continue
            f_values.append(float(np.mean(layer_f)))
            h_values.append(float(np.mean(layer_h)))
            sp_values.append(float(np.mean(layer_sp)))
            b1_values.append(float(np.mean(layer_b1)))
            epc_values.append(float(np.mean(layer_epc)))
            delta_b1_values.append(float(np.mean(layer_delta_b1)))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not f_values:
        return {
            "F_mean": float("nan"),
            "F_std": float("nan"),
            "H_mean": float("nan"),
            "H_std": float("nan"),
            "SP_mean": float("nan"),
            "SP_std": float("nan"),
            "B1_mean": float("nan"),
            "B1_std": float("nan"),
            "delta_B1_mean": float("nan"),
            "delta_B1_std": float("nan"),
            "EPC_mean": float("nan"),
            "EPC_std": float("nan"),
        }

    return {
        "F_mean": float(np.mean(f_values)),
        "F_std": float(np.std(f_values)),
        "H_mean": float(np.mean(h_values)),
        "H_std": float(np.std(h_values)),
        "SP_mean": float(np.mean(sp_values)),
        "SP_std": float(np.std(sp_values)),
        "B1_mean": float(np.mean(b1_values)),
        "B1_std": float(np.std(b1_values)),
        "delta_B1_mean": float(np.mean(delta_b1_values)),
        "delta_B1_std": float(np.std(delta_b1_values)),
        "EPC_mean": float(np.mean(epc_values)),
        "EPC_std": float(np.std(epc_values)),
    }


def run_analysis(
    model_name: str = "EleutherAI/pythia-70m",
    checkpoints: Optional[List[int]] = None,
    num_samples: int = 20,
    output_dir: str = "results",
    lambda_param: float = 1.0,
    gamma: float = 0.5,
    entropy_sign: int = -1,
    structural_term: str = "betti1",
    betti_k_neighbors: int = 5,
    betti_threshold: Optional[float] = None,
) -> Dict[int, Optional[Dict[str, float]]]:
    """Run checkpoint series analysis."""
    if checkpoints is None:
        checkpoints = CHECKPOINTS_LIGHT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")
    print(f"structural_term: {structural_term}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    gedig = GeDIGv2(
        lambda_param=lambda_param,
        gamma=gamma,
        epc_threshold=0.05,
        sp_k_ratio=0.2,
        anchor_indices=None,
        entropy_sign=entropy_sign,
    )

    results: Dict[int, Optional[Dict[str, float]]] = {}
    print(f"\nAnalyzing {len(checkpoints)} checkpoints...")
    for i, step in enumerate(checkpoints):
        print(f"\n[{i + 1}/{len(checkpoints)}] Step {step}")
        revision = f"step{step}"
        try:
            stats = analyze_checkpoint(
                model_name=model_name,
                revision=revision,
                tokenizer=tokenizer,
                gedig=gedig,
                test_texts=test_texts,
                device=device,
                structural_term=structural_term,
                betti_k_neighbors=betti_k_neighbors,
                betti_threshold=betti_threshold,
            )
            results[step] = stats
            print(
                f"    F={stats['F_mean']:.3f}, EPC={stats['EPC_mean']:.3f}, "
                f"H={stats['H_mean']:.3f}, SP={stats['SP_mean']:.3f}, B1={stats['B1_mean']:.3f}"
            )
        except Exception as exc:
            print(f"    Error: {exc}")
            results[step] = None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / "training_dynamics.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_name": model_name,
                    "checkpoints": checkpoints,
                    "num_samples": num_samples,
                    "lambda_param": lambda_param,
                    "gamma": gamma,
                    "entropy_sign": entropy_sign,
                    "structural_term": structural_term,
                    "betti_k_neighbors": betti_k_neighbors,
                    "betti_threshold": betti_threshold,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {result_file}")

    plot_training_dynamics(
        results=results,
        checkpoints=checkpoints,
        output_path=output_path,
        model_name=model_name,
        structural_term=structural_term,
    )
    return results


def _series(results: Dict[int, Optional[Dict[str, float]]], steps: List[int], key: str) -> List[float]:
    vals: List[float] = []
    for s in steps:
        item = results.get(s)
        if item is None:
            vals.append(float("nan"))
        else:
            vals.append(float(item.get(key, float("nan"))))
    return vals


def plot_training_dynamics(
    results: Dict[int, Optional[Dict[str, float]]],
    checkpoints: List[int],
    output_path: Path,
    model_name: str,
    structural_term: str,
) -> None:
    """Visualize checkpoint dynamics."""
    valid_steps = [s for s in checkpoints if results.get(s) is not None]
    if not valid_steps:
        print("No valid checkpoints to plot.")
        return

    f_vals = _series(results, valid_steps, "F_mean")
    h_vals = _series(results, valid_steps, "H_mean")
    epc_vals = _series(results, valid_steps, "EPC_mean")
    struct_key = "B1_mean" if structural_term == "betti1" else "SP_mean"
    struct_label = "B1 (Betti-1)" if structural_term == "betti1" else "SP (Shortcut Purity)"
    struct_vals = _series(results, valid_steps, struct_key)

    f_stds = _series(results, valid_steps, "F_std")
    h_stds = _series(results, valid_steps, "H_std")
    epc_stds = _series(results, valid_steps, "EPC_std")
    struct_std_key = "B1_std" if structural_term == "betti1" else "SP_std"
    struct_stds = _series(results, valid_steps, struct_std_key)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.errorbar(valid_steps, f_vals, yerr=f_stds, marker="o", capsize=3, color="blue")
    ax1.set_xscale("symlog")
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("F", fontsize=12)
    ax1.set_title("F value during training", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.errorbar(valid_steps, epc_vals, yerr=epc_stds, marker="d", capsize=3, color="purple")
    ax2.set_xscale("symlog")
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("EPC", fontsize=12)
    ax2.set_title("EPC during training", fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.errorbar(valid_steps, h_vals, yerr=h_stds, marker="s", capsize=3, color="green")
    ax3.set_xscale("symlog")
    ax3.set_xlabel("Training Step", fontsize=12)
    ax3.set_ylabel("H", fontsize=12)
    ax3.set_title("Entropy during training", fontsize=14)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.errorbar(valid_steps, struct_vals, yerr=struct_stds, marker="^", capsize=3, color="red")
    ax4.set_xscale("symlog")
    ax4.set_xlabel("Training Step", fontsize=12)
    ax4.set_ylabel(struct_label, fontsize=12)
    ax4.set_title(f"{struct_label} during training", fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} - geDIG Training Dynamics ({structural_term})", fontsize=14)
    plt.tight_layout()
    fig_path = output_path / "training_dynamics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    plt.close()

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(valid_steps, struct_vals, yerr=struct_stds, marker="o", capsize=3, color="steelblue")
    ax.set_xscale("symlog")
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel(struct_label, fontsize=14)
    ax.set_title(f"Formation of {struct_label} during Training", fontsize=16)
    ax.grid(True, alpha=0.3)
    fig2_path = output_path / "structure_formation.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"Saved structure figure to {fig2_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="EleutherAI/pythia-70m",
        help="Pythia model name (70m, 160m, 410m, etc.)",
    )
    parser.add_argument("--light", action="store_true", help="Use fewer checkpoints")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help='Explicit comma-separated checkpoint steps (overrides --light), e.g. "0,64,512"',
    )
    parser.add_argument("--samples", type=int, default=20, help="Number of text samples")
    parser.add_argument("--lambda-param", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--entropy-sign", type=int, default=-1, choices=[-1, 1])
    parser.add_argument("--structural-term", type=str, default="betti1", choices=["sp", "betti1"])
    parser.add_argument("--betti-k-neighbors", type=int, default=5)
    parser.add_argument("--betti-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if args.checkpoints.strip():
        checkpoints = _parse_checkpoints_arg(args.checkpoints)
    else:
        checkpoints = CHECKPOINTS_LIGHT if args.light else CHECKPOINTS
    run_analysis(
        model_name=args.model,
        checkpoints=checkpoints,
        num_samples=args.samples,
        output_dir=args.output_dir,
        lambda_param=args.lambda_param,
        gamma=args.gamma,
        entropy_sign=args.entropy_sign,
        structural_term=args.structural_term,
        betti_k_neighbors=args.betti_k_neighbors,
        betti_threshold=args.betti_threshold,
    )
