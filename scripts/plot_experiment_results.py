"""
Generate summary figures for recent experiments.
"""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from insightspike.gedig import compute_f_score
from insightspike.rag.reranker import StructureReranker


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _device_from_arg(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _flash_metrics(device: str, batch: int, heads: int, seq: int, temperature: float, percentile: float, seed: int):
    if seed is not None:
        torch.manual_seed(seed)
    attention_raw = torch.rand(batch, heads, seq, seq, device=device, requires_grad=True)
    attention = torch.softmax(attention_raw, dim=-1)
    attention.retain_grad()

    f_val, _ = compute_f_score(attention, temperature=temperature, percentile=percentile)
    loss = -f_val.mean()
    loss.backward()
    grad_norm = attention_raw.grad.norm().item() if attention_raw.grad is not None else 0.0

    if seed is not None:
        torch.manual_seed(seed)
    attention_profile = torch.softmax(torch.rand(batch, heads, seq, seq, device=device), dim=-1)

    def step():
        compute_f_score(attention_profile, temperature=temperature, percentile=percentile)

    warmup = 3
    iters = 10
    for _ in range(warmup):
        step()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    end = time.perf_counter()
    profile_ms = (end - start) * 1000.0 / iters

    return f_val.mean().item(), grad_norm, profile_ms


def plot_flash_gedig(device: str, seed: int, batch: int, heads: int, seq: int, temperature: float, percentile: float):
    device = _device_from_arg(device)
    f_mean, grad_norm, profile_ms = _flash_metrics(
        device=device,
        batch=batch,
        heads=heads,
        seq=seq,
        temperature=temperature,
        percentile=percentile,
        seed=seed,
    )

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].bar(["f_mean"], [f_mean], color="#4c78a8")
    axes[0].set_title("F mean")
    axes[0].axhline(0.0, color="#999999", linewidth=0.8)

    axes[1].bar(["grad_norm"], [grad_norm], color="#f58518")
    axes[1].set_title("Grad norm")

    axes[2].bar(["profile_ms"], [profile_ms], color="#54a24b")
    axes[2].set_title("Profile ms")

    fig.suptitle(f"Flash-geDIG (device={device})")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(REPO_ROOT, "experiments", "preliminary", "flash_gedig", "flash_gedig_results.png")
    _ensure_dir(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"[flash] device={device} f_mean={f_mean:.6f} grad_norm={grad_norm:.6f} profile_ms={profile_ms:.3f}")
    print(f"[flash] wrote {path}")


def plot_neuro_pruning(pattern: str):
    runs = sorted(glob.glob(pattern))
    if not runs:
        print(f"[neuro] no runs found for pattern: {pattern}")
        return

    seeds = []
    pre_acc = []
    post_acc = []
    layer_counts = {}

    for path in runs:
        with open(path, "r") as fh:
            entry = json.load(fh)
        seed = entry.get("seed")
        pre = entry.get("pre_eval", {})
        post = entry.get("post_eval", {})
        seeds.append(seed)
        pre_acc.append(pre.get("accuracy", 0.0))
        post_acc.append(post.get("accuracy", 0.0))
        for layer, count in entry.get("pruned_heads_per_layer", {}).items():
            layer = int(layer)
            layer_counts[layer] = layer_counts.get(layer, 0) + count

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = list(range(len(seeds)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pre_acc, width, label="pre", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], post_acc, width, label="post", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SST2 accuracy (sampled)")
    ax.legend(frameon=False)
    fig.tight_layout()

    acc_path = os.path.join(REPO_ROOT, "experiments", "preliminary", "neuro_pruning", "neuro_pruning_acc.png")
    _ensure_dir(acc_path)
    fig.savefig(acc_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    layers = sorted(layer_counts)
    counts = [layer_counts[l] for l in layers]
    ax.bar([str(l) for l in layers], counts, color="#54a24b")
    ax.set_title("Pruned heads per layer (aggregate)")
    ax.set_xlabel("layer")
    ax.set_ylabel("count")
    fig.tight_layout()

    layer_path = os.path.join(REPO_ROOT, "experiments", "preliminary", "neuro_pruning", "neuro_pruning_layers.png")
    _ensure_dir(layer_path)
    fig.savefig(layer_path, dpi=140)
    plt.close(fig)

    print(f"[neuro] runs={len(runs)}")
    print(f"[neuro] wrote {acc_path}")
    print(f"[neuro] wrote {layer_path}")


def _load_rag_eval_helpers():
    rag_path = os.path.join(REPO_ROOT, "experiments", "rag_reranking")
    sys.path.append(rag_path)
    import eval_rerank as rag_eval
    return rag_eval


def plot_rag_reranking(mix_weight: float, gate_min_norm: float, gate_penalty: float):
    rag_eval = _load_rag_eval_helpers()
    cases = rag_eval.build_cases()

    reranker = StructureReranker(model_name="bert-base-uncased", mix_weight=mix_weight)

    base_correct = 0
    struct_correct = 0
    mixed_correct = 0

    for case in cases:
        query = case["query"]
        docs = case["docs"]
        answer_idx = case["answer_idx"]

        base_scores = rag_eval.compute_base_scores(query, docs)
        results = reranker.rerank(
            query,
            docs,
            base_scores=base_scores,
            mix_weight=mix_weight,
            gate_min_norm=gate_min_norm,
            gate_penalty=gate_penalty,
        )
        struct_results = sorted(results, key=lambda x: x["rank_score"], reverse=True)

        base_top_idx = max(range(len(docs)), key=lambda i: base_scores[i])
        struct_top_idx = rag_eval.doc_index(docs, struct_results[0]["doc"])
        mixed_top_idx = rag_eval.doc_index(docs, results[0]["doc"])

        if base_top_idx == answer_idx:
            base_correct += 1
        if struct_top_idx == answer_idx:
            struct_correct += 1
        if mixed_top_idx == answer_idx:
            mixed_correct += 1

    total = len(cases)
    base_acc = base_correct / total if total else 0.0
    struct_acc = struct_correct / total if total else 0.0
    mixed_acc = mixed_correct / total if total else 0.0

    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["base", "struct", "mixed"]
    values = [base_acc, struct_acc, mixed_acc]
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Top-1 accuracy (toy set)")
    fig.tight_layout()

    path = os.path.join(REPO_ROOT, "experiments", "preliminary", "rag_reranking", "rag_reranking_acc.png")
    _ensure_dir(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"[rag] total_cases={total} base={base_acc:.3f} struct={struct_acc:.3f} mixed={mixed_acc:.3f}")
    print(f"[rag] wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment summary figures")
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--neuro", action="store_true")
    parser.add_argument("--rag", action="store_true")

    parser.add_argument("--flash-device", type=str, default="auto")
    parser.add_argument("--flash-seed", type=int, default=0)
    parser.add_argument("--flash-batch", type=int, default=2)
    parser.add_argument("--flash-heads", type=int, default=2)
    parser.add_argument("--flash-seq", type=int, default=16)
    parser.add_argument("--flash-temperature", type=float, default=0.1)
    parser.add_argument("--flash-percentile", type=float, default=0.9)

    parser.add_argument("--neuro-pattern", type=str, default="results/neuro_pruning/*/pruning_info.json")

    parser.add_argument("--rag-mix-weight", type=float, default=0.2)
    parser.add_argument("--rag-gate-min-norm", type=float, default=0.1)
    parser.add_argument("--rag-gate-penalty", type=float, default=1.0)

    args = parser.parse_args()

    run_flash = args.flash or not (args.flash or args.neuro or args.rag)
    run_neuro = args.neuro or not (args.flash or args.neuro or args.rag)
    run_rag = args.rag or not (args.flash or args.neuro or args.rag)

    if run_flash:
        plot_flash_gedig(
            device=args.flash_device,
            seed=args.flash_seed,
            batch=args.flash_batch,
            heads=args.flash_heads,
            seq=args.flash_seq,
            temperature=args.flash_temperature,
            percentile=args.flash_percentile,
        )
    if run_neuro:
        plot_neuro_pruning(args.neuro_pattern)
    if run_rag:
        plot_rag_reranking(
            mix_weight=args.rag_mix_weight,
            gate_min_norm=args.rag_gate_min_norm,
            gate_penalty=args.rag_gate_penalty,
        )


if __name__ == "__main__":
    main()
