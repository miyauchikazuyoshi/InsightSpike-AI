#!/usr/bin/env python3
"""
Extract attentions and compute geDIG F per head/layer on a small text set.
Supports multiple models (including local checkpoints) and percentile/absolute thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import networkx as nx
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import entropy
from transformers import AutoModel, AutoTokenizer


@dataclass
class GeDIGCalculator:
    lambda_param: float = 1.0
    gamma: float = 0.5
    threshold: float = 0.01
    use_percentile: bool = False
    percentile: float = 0.9
    undirected_sp: bool = True

    def compute_F(self, attn: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
        idx = np.where(valid_mask)[0]
        if len(idx) == 0:
            return {"F": 0.0, "E_eff": 0.0, "delta_epc": 0.0, "delta_sp": 0.0, "delta_h": 0.0, "num_edges": 0, "density": 0.0}
        attn = attn[np.ix_(idx, idx)]
        L = attn.shape[0]
        G = self._build_graph(attn)
        max_edges = L * L
        delta_epc = G.number_of_edges() / max_edges if max_edges > 0 else 0.0
        delta_sp = self._compute_path_efficiency(G)
        delta_h = self._compute_entropy(attn)
        E_eff = delta_epc - self.lambda_param * self.gamma * delta_sp
        F = E_eff - self.lambda_param * delta_h
        return {
            "F": float(F),
            "E_eff": float(E_eff),
            "delta_epc": float(delta_epc),
            "delta_sp": float(delta_sp),
            "delta_h": float(delta_h),
            "num_edges": int(G.number_of_edges()),
            "density": float(nx.density(G)) if G.number_of_nodes() > 0 else 0.0,
        }

    def _build_graph(self, attn: np.ndarray) -> nx.DiGraph:
        G = nx.DiGraph()
        L = attn.shape[0]
        G.add_nodes_from(range(L))
        thresh = float(np.quantile(attn, self.percentile)) if self.use_percentile else float(self.threshold)
        for i in range(L):
            for j in range(L):
                if attn[i, j] > thresh:
                    G.add_edge(i, j, weight=float(attn[i, j]))
        return G

    def _compute_path_efficiency(self, G: nx.DiGraph) -> float:
        if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
            return 0.0
        try:
            if self.undirected_sp:
                G2 = G.to_undirected()
                if nx.is_connected(G2):
                    avg_path = nx.average_shortest_path_length(G2)
                    return 1.0 / avg_path if avg_path > 0 else 0.0
                comp = max(nx.connected_components(G2), key=len)
                sub = G2.subgraph(comp).copy()
            else:
                if nx.is_weakly_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                    return 1.0 / avg_path if avg_path > 0 else 0.0
                comp = max(nx.weakly_connected_components(G), key=len)
                sub = G.subgraph(comp).copy()
            if sub.number_of_nodes() < 2:
                return 0.0
            avg_path = nx.average_shortest_path_length(sub)
            return 1.0 / avg_path if avg_path > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_entropy(self, attn: np.ndarray) -> float:
        flat = attn.flatten()
        flat = flat[flat > 1e-10]
        if flat.size == 0:
            return 0.0
        flat = flat / flat.sum()
        H = entropy(flat)
        max_H = np.log(flat.size)
        return float(H / max_H) if max_H > 0 else 0.0


def _resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if not dtype_str or dtype_str == "auto":
        return None
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def extract_attentions(
    model_name: str,
    texts: List[str],
    max_len: int = 512,
    device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_kwargs = {"output_attentions": True, "trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except TypeError as exc:
        if "attn_implementation" in str(exc):
            model_kwargs.pop("attn_implementation", None)
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            raise
    if not device_map:
        model.to(device)
    model.eval()
    layers: List[List[np.ndarray]] = []
    masks: List[np.ndarray] = []
    tokens: List[List[str]] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        attns = outputs.attentions  # tuple of (batch, heads, seq, seq)
        valid = inputs["attention_mask"][0].bool().cpu().numpy()
        layers.append([a[0].cpu().numpy() for a in attns])
        masks.append(valid)
        tokens.append(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    return layers, masks, tokens


def load_texts(max_count: int = 8, max_len: int = 120) -> List[str]:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:200]")
        texts = [t.strip() for t in ds["text"] if t.strip() and len(t) < max_len]
        return texts[:max_count]
    except Exception:
        return [
            "Transformers are powerful sequence models.",
            "The geDIG gauge balances structure and entropy in a graph.",
            "Lambda sweeps may reveal phase transitions in attention patterns.",
            "Causal decoders focus attention on the past context.",
            "Encoder-only models tend to spread attention across tokens.",
            "Entropy rewards diffuse attention maps.",
        ][:max_count]


def build_baselines(seq_len: int) -> Dict[str, np.ndarray]:
    rand_raw = np.random.rand(seq_len, seq_len)
    base_random = rand_raw / (rand_raw.sum(axis=1, keepdims=True) + 1e-10)
    base_uniform = np.ones((seq_len, seq_len)) / seq_len
    base_local = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        count = end - start
        base_local[i, start:end] = 1.0 / count
    base_diagonal = np.eye(seq_len)
    return {
        "random": base_random,
        "uniform": base_uniform,
        "local_w5": base_local,
        "diagonal": base_diagonal,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute geDIG F for transformer attentions (smoke run).")
    p.add_argument("--text-count", type=int, default=16, help="Number of short texts to sample.")
    p.add_argument("--text-max-len", type=int, default=120, help="Max characters of input text.")
    p.add_argument("--model", action="append", dest="models", help="Model names or checkpoint paths; repeatable.")
    p.add_argument("--layer-cap", type=int, default=4, help="Max layers per model (prefix).")
    p.add_argument("--attn-max-len", type=int, default=256, help="Max tokens for tokenizer truncation.")
    p.add_argument("--percentile", type=float, default=0.90, help="Percentile threshold when use_percentile.")
    p.add_argument("--threshold", type=float, default=0.01, help="Fixed threshold when not using percentile.")
    p.add_argument("--include-fixed-threshold", action="store_true", help="Also run a fixed threshold config (otherwise percentile only).")
    p.add_argument("--out", type=Path, default=Path("results/transformer_gedig/score_smoke.json"), help="Output JSON path.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for model execution.")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype override.")
    p.add_argument("--device-map", default=None, help="Device map for large models (e.g., auto).")
    p.add_argument("--trust-remote-code", action="store_true", help="Allow models that require trust_remote_code.")
    p.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention implementation override for compatible models.",
    )
    # subgraph options
    p.add_argument("--enable-subgraph", action="store_true", help="Compute geDIG on anchor ego subgraphs.")
    p.add_argument("--subgraph-hops", type=str, default="0,1,2,3,4", help="Comma-separated hop distances for ego graphs.")
    p.add_argument("--anchor-token", action="append", dest="anchor_tokens", help="Token substring to anchor on (repeatable).")
    p.add_argument("--anchor-cls-only", action="store_true", help="If set, use only position 0 as anchor (ignore anchor_token).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts(max_count=args.text_count, max_len=args.text_max_len)
    model_names = args.models or ["bert-base-uncased", "gpt2"]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _resolve_dtype(args.dtype)
    device_map = args.device_map
    if device_map in ("", "none", "null"):
        device_map = None
    attn_impl = args.attn_implementation
    configs = [
        {"lambda_param": 0.5, "gamma": 0.5, "threshold": 0.0, "use_percentile": True, "pct": args.percentile},
    ]
    if args.include_fixed_threshold:
        configs.append({"lambda_param": 0.5, "gamma": 0.5, "threshold": args.threshold, "use_percentile": False, "pct": None})

    records: List[Dict[str, object]] = []
    for model_name in model_names:
        model_label = Path(model_name).name
        try:
            layers, masks, tokens = extract_attentions(
                model_name,
                texts,
                max_len=args.attn_max_len,
                device=device,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=attn_impl,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] failed to load/extract for {model_name}: {exc}")
            continue

        for cfg in configs:
            gedig = GeDIGCalculator(
                lambda_param=cfg["lambda_param"],
                gamma=cfg["gamma"],
                threshold=cfg["threshold"],
                use_percentile=cfg["use_percentile"],
                percentile=cfg["pct"] if cfg["pct"] is not None else 0.9,
            )
            for doc_id, (layer_attns, mask) in enumerate(zip(layers, masks)):
                for layer_idx, attn_layer in enumerate(layer_attns[: args.layer_cap]):  # limit layers for speed
                    heads = attn_layer.shape[0]
                    for head_idx in range(heads):
                        metrics = gedig.compute_F(attn_layer[head_idx], mask)
                        seq_len = attn_layer.shape[-1]
                        baselines = build_baselines(seq_len)
                        baseline_F = {k: gedig.compute_F(v, mask)["F"] for k, v in baselines.items()}
                        rec = {
                            "doc_id": doc_id,
                            "text": texts[doc_id],
                            "model": model_label,
                            "model_path": model_name,
                            "layer": layer_idx,
                            "head": head_idx,
                            "lambda_param": cfg["lambda_param"],
                            "gamma": cfg["gamma"],
                            "threshold": cfg["threshold"],
                            "use_percentile": cfg["use_percentile"],
                            "percentile": cfg["pct"],
                            **metrics,
                            **{f"baseline_F_{k}": v for k, v in baseline_F.items()},
                        }
                        records.append(rec)

                        # subgraph mode: anchor ego graphs per hop
                        if args.enable_subgraph:
                            anchor_idxs: List[int] = []
                            if args.anchor_cls_only:
                                anchor_idxs = [0]
                            else:
                                anchor_idxs.append(0)
                                if args.anchor_tokens:
                                    toks = tokens[doc_id]
                                    for i, tok in enumerate(toks):
                                        for pat in args.anchor_tokens:
                                            if pat.lower() in tok.lower():
                                                anchor_idxs.append(i)
                            anchor_idxs = sorted(set([i for i in anchor_idxs if i < mask.shape[0] and mask[i]]))
                            G = gedig._build_graph(attn_layer[head_idx][np.ix_(mask, mask)])
                            hops = [int(x) for x in args.subgraph_hops.split(",") if x != ""]
                            for anchor in anchor_idxs:
                                for hop in hops:
                                    # ego graph on undirected to avoid reachability issues
                                    try:
                                        ego = nx.ego_graph(G.to_undirected(), anchor, radius=hop)
                                    except Exception:
                                        continue
                                    if ego.number_of_nodes() < 2:
                                        continue
                                    nodes = sorted(ego.nodes())
                                    sub_attn = attn_layer[head_idx][np.ix_(nodes, nodes)]
                                    sub_mask = np.ones(len(nodes), dtype=bool)
                                    sub_metrics = gedig.compute_F(sub_attn, sub_mask)
                                    rec_sub = {
                                        "doc_id": doc_id,
                                        "text": texts[doc_id],
                                        "model": model_label,
                                        "model_path": model_name,
                                        "layer": layer_idx,
                                        "head": head_idx,
                                        "anchor_idx": anchor,
                                        "anchor_token": tokens[doc_id][anchor] if anchor < len(tokens[doc_id]) else "",
                                        "hop": hop,
                                        "subgraph": True,
                                        "lambda_param": cfg["lambda_param"],
                                        "gamma": cfg["gamma"],
                                        "threshold": cfg["threshold"],
                                        "use_percentile": cfg["use_percentile"],
                                        "percentile": cfg["pct"],
                                        **sub_metrics,
                                    }
                                    records.append(rec_sub)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
