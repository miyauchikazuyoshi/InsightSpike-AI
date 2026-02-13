#!/usr/bin/env python3
"""Fit lightweight probe matrices (B_dist, B_depth) from dependency trees.

This script builds a local approximation of structural probe matrices:
  - B_dist: projects hidden states so token-pair distances track tree distance.
  - B_depth: projects hidden states so squared norm tracks dependency depth.

Outputs are saved as .npy and can be passed to:
  run_inference_gedig_v2.py --require-probe-b --b-dist ... --b-depth ...
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

try:
    import spacy
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "spaCy is required. Install with: pip install spacy && python -m spacy download en_core_web_sm"
    ) from exc


DEFAULT_TEXT_FILE = Path(
    "experiments/transformer/inference_gedig_v3_layerwise_redefine/data/hotpotqa_questions_128.txt"
)

CAUSAL_NAME_HINTS = (
    "gpt",
    "llama",
    "mistral",
    "gemma",
    "qwen",
    "phi",
    "pythia",
    "falcon",
    "opt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument(
        "--text-file",
        type=Path,
        default=DEFAULT_TEXT_FILE,
        help="One sentence per line",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=-1, help="-1 means last layer")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--epochs-dist", type=int, default=120)
    parser.add_argument("--epochs-depth", type=int, default=120)
    parser.add_argument("--lr-dist", type=float, default=5e-3)
    parser.add_argument("--lr-depth", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size-dist", type=int, default=2048)
    parser.add_argument("--batch-size-depth", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefer-safetensors", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/transformer/inference_gedig_v3_layerwise_redefine/artifacts/probe_b_lightweight"
        ),
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_texts(path: Path, max_samples: int) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if max_samples > 0:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError(f"no texts loaded from {path}")
    return texts


def infer_model_kind(model_name: str, config: AutoConfig) -> str:
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("encoder-decoder models are not supported")
    if getattr(config, "is_decoder", False):
        return "causal"
    lowered = model_name.lower()
    if any(hint in lowered for hint in CAUSAL_NAME_HINTS):
        return "causal"
    return "mlm"


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    local_files_only: bool,
    prefer_safetensors: bool,
) -> Tuple[torch.nn.Module, object, str]:
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    model_kind = infer_model_kind(model_name=model_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if hasattr(tokenizer, "add_prefix_space"):
        # Required for GPT2-like tokenizers when using is_split_into_words=True.
        tokenizer.add_prefix_space = True

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _load(auto_cls):
        if not prefer_safetensors:
            return auto_cls.from_pretrained(
                model_name,
                use_safetensors=False,
                local_files_only=local_files_only,
            )
        try:
            return auto_cls.from_pretrained(
                model_name,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
        except Exception:
            return auto_cls.from_pretrained(
                model_name,
                use_safetensors=False,
                local_files_only=local_files_only,
            )

    if model_kind == "causal":
        model = _load(AutoModelForCausalLM)
    else:
        model = _load(AutoModelForMaskedLM)

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device.type == "mps":
        model = model.to(dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer, model_kind


def _dependency_depths(heads: Sequence[int]) -> List[int]:
    depths: List[int] = []
    n = len(heads)
    for idx in range(n):
        d = 0
        cur = idx
        seen = set()
        while heads[cur] != cur:
            if cur in seen:
                d = 0
                break
            seen.add(cur)
            cur = int(heads[cur])
            d += 1
            if d > n:
                d = 0
                break
        depths.append(d)
    return depths


def _tree_distance_matrix(heads: Sequence[int]) -> np.ndarray:
    n = len(heads)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        p = int(heads[i])
        if p == i:
            continue
        neighbors[i].append(p)
        neighbors[p].append(i)

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for i in range(n):
        dist[i, i] = 0.0
        queue = [i]
        front = 0
        while front < len(queue):
            u = queue[front]
            front += 1
            du = dist[i, u]
            for v in neighbors[u]:
                if not np.isfinite(dist[i, v]):
                    dist[i, v] = du + 1.0
                    queue.append(v)
    if not np.isfinite(dist).all():
        raise ValueError("dependency graph is disconnected")
    return dist


@dataclass
class ProbeDataset:
    pair_diffs: torch.Tensor
    pair_targets: torch.Tensor
    depth_vectors: torch.Tensor
    depth_targets: torch.Tensor
    hidden_dim: int
    used_samples: int
    skipped_samples: int


def build_probe_dataset(
    texts: Sequence[str],
    nlp,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    layer_index: int,
    max_length: int,
) -> ProbeDataset:
    pair_diffs: List[torch.Tensor] = []
    pair_targets: List[float] = []
    depth_vectors: List[torch.Tensor] = []
    depth_targets: List[float] = []

    used_samples = 0
    skipped_samples = 0
    hidden_dim: Optional[int] = None

    for idx, text in enumerate(texts):
        doc = nlp(text)
        words = [tok.text for tok in doc]
        if len(words) < 2:
            skipped_samples += 1
            continue

        heads = [int(tok.head.i) for tok in doc]
        depths = _dependency_depths(heads)
        try:
            tree_dist = _tree_distance_matrix(heads)
        except ValueError:
            skipped_samples += 1
            continue

        encoded = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        word_ids = encoded.word_ids(batch_index=0)
        encoded_inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            skipped_samples += 1
            continue

        layer_pos = layer_index
        if layer_pos < 0:
            layer_pos = len(hidden_states) + layer_pos
        if layer_pos < 0 or layer_pos >= len(hidden_states):
            raise ValueError(f"invalid layer-index {layer_index} for {len(hidden_states)} hidden-state tensors")

        layer_hidden = hidden_states[layer_pos][0].detach().cpu().to(torch.float32)
        hidden_dim = int(layer_hidden.shape[-1])

        word_to_positions: Dict[int, List[int]] = {}
        for tok_pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid < 0 or wid >= len(words):
                continue
            word_to_positions.setdefault(int(wid), []).append(tok_pos)

        used_word_ids = sorted(word_to_positions.keys())
        if len(used_word_ids) < 2:
            skipped_samples += 1
            continue

        vecs: List[torch.Tensor] = []
        dep_depths: List[float] = []
        for wid in used_word_ids:
            positions = word_to_positions[wid]
            vec = layer_hidden[positions].mean(dim=0)
            vecs.append(vec)
            dep_depths.append(float(depths[wid]))

        sent_vecs = torch.stack(vecs, dim=0)
        sent_depth = torch.tensor(dep_depths, dtype=torch.float32)
        depth_vectors.append(sent_vecs)
        depth_targets.append(sent_depth)

        dist_sub = tree_dist[np.ix_(used_word_ids, used_word_ids)]
        n_words = sent_vecs.shape[0]
        for i in range(n_words):
            for j in range(i + 1, n_words):
                pair_diffs.append(sent_vecs[i] - sent_vecs[j])
                pair_targets.append(float(dist_sub[i, j]))

        used_samples += 1
        if (idx + 1) % 16 == 0:
            print(
                f"[build] {idx + 1}/{len(texts)} texts processed "
                f"(used={used_samples}, skipped={skipped_samples})"
            )

    if hidden_dim is None:
        raise RuntimeError("failed to build dataset (no usable samples)")
    if not pair_diffs:
        raise RuntimeError("no pairwise samples were generated")
    if not depth_vectors:
        raise RuntimeError("no depth samples were generated")

    pair_diffs_t = torch.stack(pair_diffs, dim=0)
    pair_targets_t = torch.tensor(pair_targets, dtype=torch.float32)
    depth_vectors_t = torch.cat(depth_vectors, dim=0)
    depth_targets_t = torch.cat(depth_targets, dim=0)

    return ProbeDataset(
        pair_diffs=pair_diffs_t,
        pair_targets=pair_targets_t,
        depth_vectors=depth_vectors_t,
        depth_targets=depth_targets_t,
        hidden_dim=hidden_dim,
        used_samples=used_samples,
        skipped_samples=skipped_samples,
    )


def _train_matrix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    proj_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode not in {"dist", "depth"}:
        raise ValueError(f"unsupported mode: {mode}")
    if proj_dim <= 0:
        raise ValueError("proj_dim must be > 0")

    x = inputs.to(device)
    y = targets.to(device)
    n, hidden_dim = x.shape

    weight = torch.nn.Parameter(0.01 * torch.randn((proj_dim, hidden_dim), device=device))
    optimizer = torch.optim.Adam([weight], lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        seen = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            z = xb @ weight.t()

            if mode == "dist":
                pred = torch.linalg.norm(z, dim=-1)
            else:
                pred = torch.sum(z * z, dim=-1)

            loss = torch.mean((pred - yb) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = int(yb.shape[0])
            running_loss += float(loss.item()) * bs
            seen += bs

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            avg = running_loss / max(seen, 1)
            print(f"[train:{mode}] epoch={epoch:03d}/{epochs} mse={avg:.6f}")

    return weight.detach().cpu()


def _fit_with_input_scaling(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    proj_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit probe matrix on per-feature scaled inputs, then map back to raw space."""
    scale = torch.std(inputs.to(torch.float32), dim=0).clamp_min(1e-6)
    scaled_inputs = inputs.to(torch.float32) / scale.unsqueeze(0)
    weight_scaled = _train_matrix(
        inputs=scaled_inputs,
        targets=targets,
        proj_dim=proj_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        mode=mode,
        device=device,
    )
    # pred = ||(x/scale) @ W_scaled^T||  == ||x @ (W_scaled/scale)^T||
    weight_raw = weight_scaled / scale.unsqueeze(0)
    return weight_raw, scale


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or b.size < 2:
        return None
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_dist(weight: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Optional[float]]:
    z = x @ weight.t()
    pred = torch.linalg.norm(z, dim=-1)
    mse = float(torch.mean((pred - y) ** 2).item())
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    corr = _safe_corr(pred_np, y_np)
    return {"mse": mse, "pearson": corr}


def evaluate_depth(weight: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Optional[float]]:
    z = x @ weight.t()
    pred = torch.sum(z * z, dim=-1)
    mse = float(torch.mean((pred - y) ** 2).item())
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    corr = _safe_corr(pred_np, y_np)
    return {"mse": mse, "pearson": corr}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    texts = load_texts(args.text_file, max_samples=args.max_samples)
    print(f"[info] model={args.model}")
    print(f"[info] texts={len(texts)} from {args.text_file}")
    print(f"[info] device={device}")

    nlp = spacy.load(args.spacy_model, disable=["ner", "textcat", "lemmatizer"])
    model, tokenizer, model_kind = load_model_and_tokenizer(
        model_name=args.model,
        device=device,
        local_files_only=args.local_files_only,
        prefer_safetensors=args.prefer_safetensors,
    )
    print(f"[info] model_kind={model_kind}")

    dataset = build_probe_dataset(
        texts=texts,
        nlp=nlp,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_index=args.layer_index,
        max_length=args.max_length,
    )
    print(
        "[info] dataset "
        f"used={dataset.used_samples} skipped={dataset.skipped_samples} "
        f"pairs={dataset.pair_diffs.shape[0]} depth_tokens={dataset.depth_vectors.shape[0]}"
    )

    proj_dim = min(int(args.proj_dim), int(dataset.hidden_dim))
    b_dist, dist_scale = _fit_with_input_scaling(
        inputs=dataset.pair_diffs,
        targets=dataset.pair_targets,
        proj_dim=proj_dim,
        epochs=int(args.epochs_dist),
        lr=float(args.lr_dist),
        batch_size=int(args.batch_size_dist),
        weight_decay=float(args.weight_decay),
        mode="dist",
        device=device,
    )
    b_depth, depth_scale = _fit_with_input_scaling(
        inputs=dataset.depth_vectors,
        targets=dataset.depth_targets,
        proj_dim=proj_dim,
        epochs=int(args.epochs_depth),
        lr=float(args.lr_depth),
        batch_size=int(args.batch_size_depth),
        weight_decay=float(args.weight_decay),
        mode="depth",
        device=device,
    )

    # Eval on training split (quick quality signal).
    dist_eval = evaluate_dist(
        weight=b_dist,
        x=dataset.pair_diffs.to(torch.float32),
        y=dataset.pair_targets.to(torch.float32),
    )
    depth_eval = evaluate_depth(
        weight=b_depth,
        x=dataset.depth_vectors.to(torch.float32),
        y=dataset.depth_targets.to(torch.float32),
    )

    out_dir = (
        args.output_dir
        / model_slug(args.model)
        / f"layer_{args.layer_index}"
        / f"proj_{proj_dim}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    b_dist_path = out_dir / "B_dist.npy"
    b_depth_path = out_dir / "B_depth.npy"
    np.save(b_dist_path, b_dist.numpy())
    np.save(b_depth_path, b_depth.numpy())

    report = {
        "model": args.model,
        "model_kind": model_kind,
        "text_file": str(args.text_file),
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "layer_index": args.layer_index,
        "proj_dim": proj_dim,
        "seed": args.seed,
        "device": str(device),
        "dataset": {
            "used_samples": dataset.used_samples,
            "skipped_samples": dataset.skipped_samples,
            "n_pair_samples": int(dataset.pair_diffs.shape[0]),
            "n_depth_samples": int(dataset.depth_vectors.shape[0]),
            "hidden_dim": dataset.hidden_dim,
        },
        "fit": {
            "dist": dist_eval,
            "depth": depth_eval,
        },
        "input_scaling": {
            "dist_scale_mean": float(dist_scale.mean().item()),
            "dist_scale_min": float(dist_scale.min().item()),
            "dist_scale_max": float(dist_scale.max().item()),
            "depth_scale_mean": float(depth_scale.mean().item()),
            "depth_scale_min": float(depth_scale.min().item()),
            "depth_scale_max": float(depth_scale.max().item()),
        },
        "artifacts": {
            "b_dist": str(b_dist_path),
            "b_depth": str(b_depth_path),
        },
    }
    report_path = out_dir / "fit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote artifacts:")
    print(f"  - {b_dist_path}")
    print(f"  - {b_depth_path}")
    print(f"  - {report_path}")
    print("[done] fit metrics:")
    print(f"  - dist mse={dist_eval['mse']:.6f}, pearson={dist_eval['pearson']}")
    print(f"  - depth mse={depth_eval['mse']:.6f}, pearson={depth_eval['pearson']}")


if __name__ == "__main__":
    main()
