#!/usr/bin/env python3
"""Compute LLV-lite for causal models and compare with F metrics.

LLV-lite here means:
  - fixed text set
  - per-model vector of average token log-probabilities per text
  - pairwise squared L2 distance as a practical proxy to model-space distance
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_texts(text_file: Path, max_samples: int) -> List[str]:
    lines = text_file.read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if max_samples > 0:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError("no text samples loaded")
    return texts


def infer_is_causal(model_name: str, config: AutoConfig) -> bool:
    if getattr(config, "is_decoder", False):
        return True
    lowered = model_name.lower()
    return any(hint in lowered for hint in CAUSAL_NAME_HINTS)


def _normalize_model_name(raw_name: str) -> str:
    marker = "/models--"
    snapshot_marker = "/snapshots/"
    if marker in raw_name and snapshot_marker in raw_name:
        fragment = raw_name.split(marker, 1)[1]
        repo_fragment = fragment.split(snapshot_marker, 1)[0]
        if repo_fragment:
            return repo_fragment.replace("--", "/")
    return raw_name


def per_text_logprob_vector(
    model_name: str,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
    prefer_safetensors: bool,
) -> List[float]:
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    if not infer_is_causal(model_name=model_name, config=config):
        raise ValueError(f"non-causal model is not supported in LLV-lite: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if prefer_safetensors:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=False,
            local_files_only=local_files_only,
        )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device.type == "mps":
        model = model.to(dtype=torch.float32)
    model.to(device)
    model.eval()

    values: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits  # [B, T, V]

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].to(dtype=torch.float32)

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp * shift_mask

            denom = torch.clamp_min(shift_mask.sum(dim=1), 1.0)
            sent_ll = token_logp.sum(dim=1) / denom
            values.extend(float(x) for x in sent_ll.detach().cpu().tolist())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return values


def _read_metrics_map(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = _normalize_model_name(str(row.get("model_name") or row.get("model_dir") or ""))
            if not name:
                continue
            item: Dict[str, float] = {}
            for key in ("delta_r2_struct", "delta_r2_learn", "baseline_r2"):
                try:
                    item[key] = float(row[key]) if row.get(key) not in ("", None, "None") else float("nan")
                except Exception:
                    item[key] = float("nan")
            out[name] = item
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or y.size < 3:
        return None
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)
    if denom <= 1e-12:
        return None
    return float(np.dot(x0, y0) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or y.size < 3:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LLV-lite and compare with F metrics.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated causal model names.")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None, help="Optional CSV from summarize/analyze for F metrics.")
    parser.add_argument("--reference-model", type=str, default=None, help="Reference model for distance-vs-metric table.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefer-safetensors", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    texts = load_texts(text_file=args.text_file, max_samples=args.max_samples)
    model_names = [_normalize_model_name(x.strip()) for x in args.models.split(",") if x.strip()]
    if not model_names:
        raise ValueError("no models specified")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    llv_rows: List[Dict[str, object]] = []
    vectors: Dict[str, np.ndarray] = {}
    for model_name in model_names:
        print(f"[run] llv-lite {model_name}")
        vec = per_text_logprob_vector(
            model_name=model_name,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            prefer_safetensors=args.prefer_safetensors,
        )
        vectors[model_name] = np.array(vec, dtype=np.float64)
        llv_rows.append(
            {
                "model_name": model_name,
                "num_texts": len(vec),
                "llv_mean": float(np.mean(vectors[model_name])),
                "llv_std": float(np.std(vectors[model_name])),
            }
        )

    # Pairwise squared L2 distances in LLV space.
    pair_rows: List[Dict[str, object]] = []
    names = list(vectors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            dist2 = float(np.sum((vectors[a] - vectors[b]) ** 2))
            pair_rows.append({"model_a": a, "model_b": b, "llv_dist2": dist2})

    # Optional distance-vs-F metric table against reference model.
    metrics_map = _read_metrics_map(args.metrics_csv)
    ref_model = args.reference_model or names[0]
    ref_vec = vectors.get(ref_model)
    ref_rows: List[Dict[str, object]] = []
    corr_x: List[float] = []
    corr_y: List[float] = []
    if ref_vec is not None:
        for name in names:
            dist2 = float(np.sum((vectors[name] - ref_vec) ** 2))
            item = metrics_map.get(name, {})
            delta_struct = item.get("delta_r2_struct", float("nan"))
            baseline_r2 = item.get("baseline_r2", float("nan"))
            ref_rows.append(
                {
                    "model_name": name,
                    "reference_model": ref_model,
                    "llv_dist2_to_ref": dist2,
                    "delta_r2_struct": delta_struct,
                    "baseline_r2": baseline_r2,
                }
            )
            if np.isfinite(delta_struct):
                corr_x.append(dist2)
                corr_y.append(delta_struct)

    llv_csv = args.output_dir / "llv_lite_summary.csv"
    with llv_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model_name", "num_texts", "llv_mean", "llv_std"])
        writer.writeheader()
        for row in llv_rows:
            writer.writerow(row)

    pair_csv = args.output_dir / "llv_lite_pairwise.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model_a", "model_b", "llv_dist2"])
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)

    ref_csv = args.output_dir / "llv_lite_vs_f.csv"
    with ref_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model_name", "reference_model", "llv_dist2_to_ref", "delta_r2_struct", "baseline_r2"],
        )
        writer.writeheader()
        for row in ref_rows:
            writer.writerow(row)

    pearson = _pearson(np.array(corr_x, dtype=np.float64), np.array(corr_y, dtype=np.float64)) if corr_x else None
    spearman = _spearman(np.array(corr_x, dtype=np.float64), np.array(corr_y, dtype=np.float64)) if corr_x else None

    md_lines = [
        "# LLV-lite Summary",
        "",
        f"- models: {len(names)}",
        f"- texts: {len(texts)}",
        f"- reference_model: {ref_model}",
        f"- corr(LLV_dist2_to_ref, delta_r2_struct) pearson: {pearson}",
        f"- corr(LLV_dist2_to_ref, delta_r2_struct) spearman: {spearman}",
        "",
        "## Per-model",
        "| model | llv_mean | llv_std |",
        "|---|---:|---:|",
    ]
    for row in llv_rows:
        md_lines.append(f"| {row['model_name']} | {row['llv_mean']:.6f} | {row['llv_std']:.6f} |")
    md_lines.append("")
    md_lines.append("## Pairwise LLV Distance")
    md_lines.append("| model_a | model_b | llv_dist2 |")
    md_lines.append("|---|---|---:|")
    for row in pair_rows:
        md_lines.append(f"| {row['model_a']} | {row['model_b']} | {float(row['llv_dist2']):.6f} |")

    md_path = args.output_dir / "llv_lite_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[out] {llv_csv}")
    print(f"[out] {pair_csv}")
    print(f"[out] {ref_csv}")
    print(f"[out] {md_path}")


if __name__ == "__main__":
    main()
