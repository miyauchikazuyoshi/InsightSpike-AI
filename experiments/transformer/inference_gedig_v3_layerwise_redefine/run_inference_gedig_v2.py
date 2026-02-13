#!/usr/bin/env python3
"""Run fixed-model Transformer inference experiment for geDIG v2."""

from __future__ import annotations

import argparse
import gc
import json
import random
from datetime import datetime, timezone
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

from metrics import (
    LayerCurves,
    compute_f_curve,
    compute_layer_curves,
    grid_search_f,
    linear_fit_r2,
    load_projection_matrix,
    mean_optional_curves,
    monotonic_nonincreasing,
)

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning helps us model complex patterns in data.",
    "What causes thunderstorms to develop so rapidly in summer?",
    "A researcher reads the paper, writes notes, and compares hypotheses.",
    "The chef who won the contest opened a small restaurant downtown.",
    "If the signal is noisy, the estimate can drift from the true value.",
]

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


def parse_float_list(csv_values: str) -> List[float]:
    values: List[float] = []
    for part in csv_values.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("at least one float value is required")
    return values


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_texts(text_file: Optional[Path], max_samples: int) -> List[str]:
    if text_file is None:
        texts = list(DEFAULT_TEXTS)
    else:
        lines = text_file.read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    if max_samples > 0:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError("no text samples loaded")
    return texts


def infer_model_kind(model_name: str, config: AutoConfig, explicit: str) -> str:
    if explicit in {"mlm", "causal"}:
        return explicit
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("encoder-decoder models are not supported in this script")
    if getattr(config, "is_decoder", False):
        return "causal"
    lowered = model_name.lower()
    if any(hint in lowered for hint in CAUSAL_NAME_HINTS):
        return "causal"
    return "mlm"


def load_model_and_tokenizer(
    model_name: str,
    model_kind: str,
    random_init: bool,
    device: torch.device,
    local_files_only: bool,
    prefer_safetensors: bool,
) -> Tuple[torch.nn.Module, object, str]:
    def _from_pretrained_preferring_safetensors(auto_cls, name: str):
        """Prefer safetensors to avoid legacy .bin loading paths."""
        try:
            return auto_cls.from_pretrained(
                name,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            # Transformers may try Hub API conversion checks when safetensors
            # is unavailable. Force a direct .bin load as fallback.
            try:
                return auto_cls.from_pretrained(
                    name,
                    use_safetensors=False,
                    local_files_only=local_files_only,
                )
            except Exception:
                raise exc

    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    resolved_kind = infer_model_kind(model_name=model_name, config=config, explicit=model_kind)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if resolved_kind == "causal":
        model = (
            AutoModelForCausalLM.from_config(config)
            if random_init
            else (
                _from_pretrained_preferring_safetensors(AutoModelForCausalLM, model_name)
                if prefer_safetensors
                else AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_safetensors=False,
                    local_files_only=local_files_only,
                )
            )
        )
    else:
        model = (
            AutoModelForMaskedLM.from_config(config)
            if random_init
            else (
                _from_pretrained_preferring_safetensors(AutoModelForMaskedLM, model_name)
                if prefer_safetensors
                else AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    use_safetensors=False,
                    local_files_only=local_files_only,
                )
            )
        )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device.type == "mps":
        # Some model configs default to bf16, which MPS cannot accept.
        model = model.to(dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer, resolved_kind


def extract_hidden_states(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> Tuple[List[torch.Tensor], int, torch.Tensor]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model did not return hidden_states")

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        valid_len = int(attention_mask[0].sum().item())
    else:
        valid_len = int(hidden_states[0].shape[1])

    layers = [layer[0, :valid_len, :].detach().cpu().to(torch.float32) for layer in hidden_states]
    token_ids = encoded["input_ids"][0, :valid_len].detach().cpu().to(torch.long)
    return layers, valid_len, token_ids


def select_entropy_indices(
    token_ids: torch.Tensor,
    tokenizer,
    model_kind: str,
    mode: str,
    tail_k: int,
) -> List[int]:
    """Select token positions used for entropy observation."""
    seq_len = int(token_ids.shape[0])
    if seq_len == 0:
        return []

    all_indices = list(range(seq_len))
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    content_indices = [i for i, tid in enumerate(token_ids.tolist()) if int(tid) not in special_ids]
    if not content_indices:
        content_indices = all_indices

    if mode == "all":
        return all_indices
    if mode == "content":
        return content_indices
    if mode == "tail_k":
        k = max(1, int(tail_k))
        base = content_indices if content_indices else all_indices
        return base[-k:] if len(base) >= k else base
    if mode == "auto":
        # Causal LM: prefer prediction-near tail tokens.
        if model_kind == "causal":
            k = max(1, int(tail_k))
            return content_indices[-k:] if len(content_indices) >= k else content_indices
        # MLM/encoder: exclude special tokens by default.
        return content_indices
    raise ValueError(f"unsupported entropy-token-mode: {mode}")


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    shuffled = words[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def resolve_structural_term(sp_mode: str, requested_term: str) -> str:
    """Resolve which structural delta curve should be used in F."""
    if requested_term not in {"sp", "betti1"}:
        raise ValueError(f"unsupported structural term: {requested_term}")
    if sp_mode not in {"spearman", "betti1", "both"}:
        raise ValueError(f"unsupported sp_mode: {sp_mode}")

    if sp_mode == "spearman" and requested_term != "sp":
        raise ValueError("f-structural-term=betti1 requires --sp-mode betti1 or both")
    if sp_mode == "betti1" and requested_term != "betti1":
        raise ValueError("f-structural-term=sp requires --sp-mode spearman or both")
    return requested_term


def run_condition(
    condition_name: str,
    model: torch.nn.Module,
    tokenizer,
    model_kind: str,
    texts: Sequence[str],
    args: argparse.Namespace,
    shuffle_input: bool,
    rng_seed: int,
) -> Dict[str, object]:
    output_head = model.get_output_embeddings()
    if output_head is None or getattr(output_head, "weight", None) is None:
        raise RuntimeError("model.get_output_embeddings().weight is required")

    unembed_weight = output_head.weight.detach().cpu().to(torch.float32)

    curves_by_metric: Dict[str, List[List[Optional[float]]]] = {
        "H": [],
        "EPC": [],
        "SP": [],
        "B1": [],
        "delta_H": [],
        "delta_EPC": [],
        "delta_SP": [],
        "delta_B1": [],
        "F": [],
    }
    sample_records: List[Dict[str, object]] = []
    monotonic_flags: List[bool] = []

    b_dist: Optional[torch.Tensor] = None
    b_depth: Optional[torch.Tensor] = None
    rng = random.Random(rng_seed)

    for idx, original_text in enumerate(texts):
        input_text = shuffle_words(original_text, rng) if shuffle_input else original_text
        hidden_states, token_count, token_ids = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            text=input_text,
            device=args.device,
            max_length=args.max_length,
        )
        hidden_dim = int(hidden_states[0].shape[-1])

        if b_dist is None:
            b_dist = load_projection_matrix(
                path=args.b_dist,
                hidden_dim=hidden_dim,
                default_proj_dim=args.proj_dim,
            )
        if b_depth is None:
            b_depth = load_projection_matrix(
                path=args.b_depth,
                hidden_dim=hidden_dim,
                default_proj_dim=args.proj_dim,
            )

        entropy_indices = select_entropy_indices(
            token_ids=token_ids,
            tokenizer=tokenizer,
            model_kind=model_kind,
            mode=args.entropy_token_mode,
            tail_k=args.entropy_tail_k,
        )

        curves: LayerCurves = compute_layer_curves(
            hidden_states=hidden_states,
            unembed_weight=unembed_weight,
            b_dist=b_dist,
            b_depth=b_depth,
            temperature=args.temperature,
            vocab_chunk_tokens=args.vocab_chunk_tokens,
            sp_mode=args.sp_mode,
            betti_k_neighbors=args.betti_k_neighbors,
            betti_threshold=args.betti_threshold,
            entropy_token_indices=entropy_indices,
            distance_norm=args.epc_distance_norm,
        )
        structural_term = resolve_structural_term(args.sp_mode, args.f_structural_term)
        structural_curve = curves.delta_SP if structural_term == "sp" else curves.delta_B1

        f_curve = compute_f_curve(
            delta_epc=curves.delta_EPC,
            delta_h=curves.delta_H,
            delta_sp=structural_curve,
            lambda_param=args.lambda_param,
            gamma=args.gamma,
        )
        fit = linear_fit_r2(f_curve)
        monotonic = monotonic_nonincreasing(f_curve)
        if monotonic is not None:
            monotonic_flags.append(bool(monotonic))

        curves_by_metric["H"].append(curves.H)
        curves_by_metric["EPC"].append(curves.EPC)
        curves_by_metric["SP"].append(curves.SP)
        curves_by_metric["B1"].append(curves.B1)
        curves_by_metric["delta_H"].append(curves.delta_H)
        curves_by_metric["delta_EPC"].append(curves.delta_EPC)
        curves_by_metric["delta_SP"].append(curves.delta_SP)
        curves_by_metric["delta_B1"].append(curves.delta_B1)
        curves_by_metric["F"].append(f_curve)

        record: Dict[str, object] = {
            "sample_index": idx,
            "text": original_text,
            "input_text": input_text,
            "token_count": token_count,
            "num_layers": len(hidden_states) - 1,
            "trajectory": {
                **curves.as_dict(),
                "F": f_curve,
            },
            "fit": fit,
            "is_monotonic_nonincreasing_F": monotonic,
        }
        sample_records.append(record)

    mean_curves = {name: mean_optional_curves(values) for name, values in curves_by_metric.items()}
    mean_fit = linear_fit_r2(mean_curves["F"])

    grid_result = None
    if args.grid_search:
        lambda_values = parse_float_list(args.grid_lambda)
        gamma_values = parse_float_list(args.grid_gamma)
        structural_term = resolve_structural_term(args.sp_mode, args.f_structural_term)
        structural_key = "delta_SP" if structural_term == "sp" else "delta_B1"
        grid_result = grid_search_f(
            delta_epc=mean_curves["delta_EPC"],
            delta_h=mean_curves["delta_H"],
            delta_sp=mean_curves[structural_key],
            lambda_values=lambda_values,
            gamma_values=gamma_values,
        )

    summary: Dict[str, object] = {
        "condition": condition_name,
        "num_samples": len(sample_records),
        "mean_curves": mean_curves,
        "mean_fit": mean_fit,
        "monotonic_nonincreasing_rate": (
            float(np.mean(monotonic_flags)) if monotonic_flags else None
        ),
        "mean_sample_r2": _safe_mean([record["fit"]["r2"] for record in sample_records]),  # type: ignore[index]
        "mean_sample_slope": _safe_mean([record["fit"]["slope"] for record in sample_records]),  # type: ignore[index]
        "grid_search_best": grid_result,
        "f_structural_term": resolve_structural_term(args.sp_mode, args.f_structural_term),
    }

    if args.save_samples:
        summary["samples"] = sample_records
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-model geDIG inference experiment")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--model-kind", type=str, default="auto", choices=["auto", "mlm", "causal"])
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--lambda-param", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vocab-chunk-tokens", type=int, default=8)
    parser.add_argument(
        "--entropy-token-mode",
        type=str,
        default="auto",
        choices=["auto", "all", "content", "tail_k"],
        help=(
            "Token positions used for H(l): "
            "auto=(causal:tail_k, mlm:content), all, content, tail_k."
        ),
    )
    parser.add_argument(
        "--entropy-tail-k",
        type=int,
        default=8,
        help="Tail token count for entropy-token-mode=tail_k/auto(causal).",
    )
    parser.add_argument(
        "--sp-mode",
        type=str,
        default="both",
        choices=["spearman", "betti1", "both"],
        help="Structural metrics to compute: spearman, betti1, or both",
    )
    parser.add_argument(
        "--f-structural-term",
        type=str,
        default="betti1",
        choices=["sp", "betti1"],
        help="Which structural delta term to use in F (delta_SP or delta_B1)",
    )
    parser.add_argument(
        "--betti-k-neighbors",
        type=int,
        default=5,
        help="k for k-NN graph when computing Betti-1 (<=0 uses threshold graph)",
    )
    parser.add_argument(
        "--betti-threshold",
        type=float,
        default=None,
        help="Distance threshold for Betti graph when --betti-k-neighbors <= 0 (default: median)",
    )

    parser.add_argument("--b-dist", type=str, default=None, help="Path to B_dist .npy")
    parser.add_argument("--b-depth", type=str, default=None, help="Path to B_depth .npy")
    parser.add_argument("--proj-dim", type=int, default=128, help="Fallback projection dim when B is absent")
    parser.add_argument(
        "--require-probe-b",
        action="store_true",
        help="Require explicit B matrices (--b-dist and --b-depth) for claim-grade runs.",
    )
    parser.add_argument(
        "--epc-distance-norm",
        type=str,
        default="none",
        choices=["none", "median"],
        help="Normalize pairwise distance matrix scale per layer before EPC computation.",
    )

    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--grid-lambda", type=str, default="0.01,0.1,0.5,1,2,5,10")
    parser.add_argument("--grid-gamma", type=str, default="0.01,0.1,0.5,1,2,5,10")

    parser.add_argument("--shuffle-control", action="store_true")
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local cache only (skip remote fetch).",
    )
    parser.add_argument(
        "--prefer-safetensors",
        action="store_true",
        help="Try safetensors first (falls back to .bin if needed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/transformer/inference_gedig_v3_layerwise_redefine/results"),
    )
    parser.add_argument("--save-samples", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.device = choose_device(args.device)
    resolve_structural_term(args.sp_mode, args.f_structural_term)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    texts = load_texts(text_file=args.text_file, max_samples=args.max_samples)
    if args.require_probe_b and (not args.b_dist or not args.b_depth):
        raise ValueError("--require-probe-b is set; both --b-dist and --b-depth are required")

    model, tokenizer, resolved_kind = load_model_and_tokenizer(
        model_name=args.model,
        model_kind=args.model_kind,
        random_init=False,
        device=args.device,
        local_files_only=args.local_files_only,
        prefer_safetensors=args.prefer_safetensors,
    )

    print(f"[info] device={args.device}")
    print(f"[info] model={args.model} kind={resolved_kind}")
    print(f"[info] local_files_only={args.local_files_only}")
    print(f"[info] prefer_safetensors={args.prefer_safetensors}")
    print(f"[info] samples={len(texts)}")
    print(f"[info] entropy_token_mode={args.entropy_token_mode} entropy_tail_k={args.entropy_tail_k}")
    print(f"[info] epc_distance_norm={args.epc_distance_norm}")
    print(f"[info] require_probe_b={args.require_probe_b}")
    print(
        f"[info] sp_mode={args.sp_mode} "
        f"f_structural_term={resolve_structural_term(args.sp_mode, args.f_structural_term)}"
    )

    conditions: Dict[str, object] = {}
    conditions["baseline"] = run_condition(
        condition_name="baseline",
        model=model,
        tokenizer=tokenizer,
        model_kind=resolved_kind,
        texts=texts,
        args=args,
        shuffle_input=False,
        rng_seed=args.seed,
    )

    if args.shuffle_control:
        conditions["shuffle_input"] = run_condition(
            condition_name="shuffle_input",
            model=model,
            tokenizer=tokenizer,
            model_kind=resolved_kind,
            texts=texts,
            args=args,
            shuffle_input=True,
            rng_seed=args.seed + 17,
        )

    if args.random_control:
        # Free pretrained model before building random-init model to reduce peak memory.
        del model
        gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        elif args.device.type == "mps":
            torch.mps.empty_cache()

        random_model, random_tokenizer, random_kind = load_model_and_tokenizer(
            model_name=args.model,
            model_kind=args.model_kind,
            random_init=True,
            device=args.device,
            local_files_only=args.local_files_only,
            prefer_safetensors=args.prefer_safetensors,
        )
        conditions["random_init"] = run_condition(
            condition_name="random_init",
            model=random_model,
            tokenizer=random_tokenizer,
            model_kind=random_kind,
            texts=texts,
            args=args,
            shuffle_input=False,
            rng_seed=args.seed + 31,
        )
        del random_model
        if random_kind != resolved_kind:
            print(
                f"[warn] resolved kind differs between pretrained and random model: "
                f"{resolved_kind} vs {random_kind}"
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"run_{timestamp}.json"

    payload = {
        "metadata": {
            "timestamp_utc": timestamp,
            "model": args.model,
            "model_kind": resolved_kind,
            "num_texts": len(texts),
            "definition": {
                "H": "vocab entropy from hidden->unembedding logits",
                "EPC": "normalized Frobenius distance-matrix change",
                "SP": "Spearman correlation of depth predictions across layers",
                "B1": "first Betti number from layer-wise distance graph",
                "F": "delta_EPC - lambda*(delta_H + gamma*delta_structural)",
            },
        },
        "config": {
            "lambda_param": args.lambda_param,
            "gamma": args.gamma,
            "temperature": args.temperature,
            "b_dist": args.b_dist,
            "b_depth": args.b_depth,
            "proj_dim": args.proj_dim,
            "sp_mode": args.sp_mode,
            "f_structural_term": resolve_structural_term(args.sp_mode, args.f_structural_term),
            "betti_k_neighbors": args.betti_k_neighbors,
            "betti_threshold": args.betti_threshold,
            "grid_search": args.grid_search,
            "grid_lambda": args.grid_lambda,
            "grid_gamma": args.grid_gamma,
            "shuffle_control": args.shuffle_control,
            "random_control": args.random_control,
            "seed": args.seed,
            "max_length": args.max_length,
            "vocab_chunk_tokens": args.vocab_chunk_tokens,
            "entropy_token_mode": args.entropy_token_mode,
            "entropy_tail_k": args.entropy_tail_k,
            "epc_distance_norm": args.epc_distance_norm,
            "require_probe_b": args.require_probe_b,
        },
        "texts": texts,
        "conditions": conditions,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
