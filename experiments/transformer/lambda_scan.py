#!/usr/bin/env python3
"""
Lambda sweep for geDIG on Transformer attention.
Supports percentile/absolute thresholds and optional lambda-dependent threshold scaling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import torch
from scipy.stats import entropy
from transformers import AutoModel, AutoTokenizer


@dataclass
class GeDIGCalculator:
    lambda_param: float = 1.0
    gamma: float = 0.5
    threshold: float = 0.01
    use_percentile: bool = True
    percentile: float = 0.9

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


def extract_attentions(model_name: str, texts: List[str], max_len: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    layers = []
    masks = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            outputs = model(**inputs)
        attns = outputs.attentions  # tuple of (batch, heads, seq, seq)
        valid = inputs["attention_mask"][0].bool().cpu().numpy()
        layers.append([a[0].cpu().numpy() for a in attns])
        masks.append(valid)
    return layers, masks


def run_lambda_scan(model_name: str, texts: List[str], lambdas: List[float], layer_cap: int, attn_max_len: int, cfgs: Sequence[Dict]) -> Dict:
    layers, masks = extract_attentions(model_name, texts, max_len=attn_max_len)
    rows = []
    for cfg in cfgs:
        for lam in lambdas:
            # derive effective threshold driven by lambda if requested
            eff_thresh = cfg.get("threshold", 0.01)
            eff_pct = cfg.get("percentile", 0.9)
            if cfg.get("scale_abs_by_lambda"):
                eff_thresh = eff_thresh * lam
            if cfg.get("pct_offset_per_lambda"):
                eff_pct = max(0.0, min(0.99, eff_pct + cfg["pct_offset_per_lambda"] * lam))

            gedig = GeDIGCalculator(
                lambda_param=lam,
                gamma=0.5,
                threshold=eff_thresh,
                use_percentile=cfg.get("use_percentile", True),
                percentile=eff_pct,
            )
            for doc_id, (layer_attns, mask) in enumerate(zip(layers, masks)):
                for layer_idx, attn_layer in enumerate(layer_attns[:layer_cap]):  # limit layers for speed
                    heads = attn_layer.shape[0]
                    for head_idx in range(heads):
                        metrics = gedig.compute_F(attn_layer[head_idx], mask)
                        rows.append(
                            {
                                "lambda": lam,
                                "layer": layer_idx,
                                "head": head_idx,
                                "thresh_mode": cfg["mode"],
                                "thresh_value": cfg["value"],
                                "use_percentile": cfg["use_percentile"],
                                "thresh_effective": eff_pct if cfg.get("use_percentile") else eff_thresh,
                                "pct_offset_per_lambda": cfg.get("pct_offset_per_lambda", 0.0),
                                "scale_abs_by_lambda": cfg.get("scale_abs_by_lambda", False),
                                **metrics,
                            }
                        )
    agg: List[Dict] = []
    for cfg in cfgs:
        for lam in lambdas:
            subset = [r for r in rows if r["lambda"] == lam and r["thresh_mode"] == cfg["mode"] and r["thresh_value"] == cfg["value"]]
            if not subset:
                continue
            agg.append(
                {
                    "lambda": lam,
                    "thresh_mode": cfg["mode"],
                    "thresh_value": cfg["value"],
                    "pct_offset_per_lambda": cfg.get("pct_offset_per_lambda", 0.0),
                    "scale_abs_by_lambda": cfg.get("scale_abs_by_lambda", False),
                    "F_mean": float(np.mean([r["F"] for r in subset])),
                    "E_eff_mean": float(np.mean([r["E_eff"] for r in subset])),
                    "sparsity_mean": float(np.mean([1 - r["density"] for r in subset])),
                    "delta_sp_mean": float(np.mean([r["delta_sp"] for r in subset])),
                }
            )
    # derivatives per config
    for cfg in cfgs:
        cfg_agg = [a for a in agg if a["thresh_mode"] == cfg["mode"] and a["thresh_value"] == cfg["value"]]
        cfg_agg = sorted(cfg_agg, key=lambda d: d["lambda"])
        lam_vals = np.array([a["lambda"] for a in cfg_agg])
        if lam_vals.size == 0:
            continue
        F_mean = np.array([a["F_mean"] for a in cfg_agg])
        dF = np.gradient(F_mean, lam_vals)
        d2F = np.gradient(dF, lam_vals)
        for i, a in enumerate(cfg_agg):
            a["dF_dlambda"] = float(dF[i])
            a["d2F_dlambda2"] = float(d2F[i])
    return {"rows": rows, "agg": agg}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lambda sweep for transformer geDIG (percentile/absolute thresholds).")
    ap.add_argument("--text-count", type=int, default=12, help="Number of short texts to sample.")
    ap.add_argument("--text-max-len", type=int, default=120, help="Max characters of input text.")
    ap.add_argument("--layer-cap", type=int, default=6, help="Max layers to include.")
    ap.add_argument("--percentiles", type=str, default="0.90", help="Comma-separated percentile thresholds (e.g., 0.8,0.9).")
    ap.add_argument("--percentile-offset-per-lambda", type=float, default=0.0, help="Add this * lambda to percentile (clamped to [0,0.99]).")
    ap.add_argument("--thresholds", type=str, default="", help="Comma-separated absolute thresholds (e.g., 0.01,0.05).")
    ap.add_argument("--scale-abs-by-lambda", action="store_true", help="Multiply absolute thresholds by lambda.")
    ap.add_argument("--model", type=str, default="bert-base-uncased", help="HF model name.")
    ap.add_argument("--attn-max-len", type=int, default=256, help="Max tokens for tokenizer truncation.")
    ap.add_argument("--lambdas", type=str, default="0.05,0.1,0.2,0.5,1.0,2.0,3.0,5.0", help="Comma-separated lambda values.")
    ap.add_argument("--out-prefix", type=Path, default=Path("results/transformer_gedig/lambda_scan"), help="Output prefix (without extension).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Use a small text sample (wikitext-2) to keep runtime bounded
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:400]")
        texts = [t.strip() for t in ds["text"] if t.strip() and len(t) < args.text_max_len][: args.text_count]
    except Exception:
        texts = [
            "Transformers are powerful sequence models.",
            "The geDIG gauge balances structure and entropy in a graph.",
            "Lambda sweeps may reveal phase transitions in attention patterns.",
        ][: args.text_count]

    lambdas = [float(x) for x in args.lambdas.split(",") if x]
    pct_vals = [float(x) for x in args.percentiles.split(",") if x]
    thr_vals = [float(x) for x in args.thresholds.split(",") if x]
    cfgs: List[Dict] = []
    for p in pct_vals:
        cfgs.append(
            {
                "mode": "percentile",
                "value": p,
                "use_percentile": True,
                "percentile": p,
                "threshold": 0.0,
                "pct_offset_per_lambda": args.percentile_offset_per_lambda,
            }
        )
    for t in thr_vals:
        cfgs.append(
            {
                "mode": "absolute",
                "value": t,
                "use_percentile": False,
                "percentile": 0.0,
                "threshold": t,
                "scale_abs_by_lambda": args.scale_abs_by_lambda,
            }
        )
    if not cfgs:
        cfgs.append({"mode": "percentile", "value": 0.9, "use_percentile": True, "percentile": 0.9, "threshold": 0.0})

    res = run_lambda_scan(args.model, texts, lambdas, args.layer_cap, args.attn_max_len, cfgs)
    out_path = args.out_prefix.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_path} (rows={len(res['rows'])}, agg={len(res['agg'])})")


if __name__ == "__main__":
    main()
