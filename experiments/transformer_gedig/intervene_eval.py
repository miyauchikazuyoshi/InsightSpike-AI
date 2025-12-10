#!/usr/bin/env python3
"""
Offline-friendly attention interventions:
- sparsify: keep top-k tokens per query (lower F)
- noise: add Gaussian noise (raise F)
- identity: no change

Logs F on the last layer and a placeholder accuracy (None).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Callable, Tuple

import numpy as np
import torch
from scipy.stats import entropy
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


@dataclass
class GeDIGCalculator:
    lambda_param: float = 1.0
    gamma: float = 0.5
    threshold: float = 0.01
    use_percentile: bool = True
    percentile: float = 0.9

    def compute_F(self, attn: np.ndarray, mask: np.ndarray) -> float:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        attn = attn[np.ix_(idx, idx)]
        L = attn.shape[0]
        # density
        thresh = float(np.quantile(attn, self.percentile)) if self.use_percentile else float(self.threshold)
        edges = (attn > thresh).sum()
        delta_epc = edges / (L * L) if L > 0 else 0.0
        # entropy
        flat = attn.flatten()
        flat = flat[flat > 1e-10]
        if flat.size == 0:
            delta_h = 0.0
        else:
            flat = flat / flat.sum()
            delta_h = float(entropy(flat) / np.log(flat.size))
        # SP (simple proxy: 1/avg distance on largest component, undirected)
        delta_sp = 0.0
        # geDIG
        E_eff = delta_epc - self.lambda_param * self.gamma * delta_sp
        F = E_eff - self.lambda_param * delta_h
        return float(F)


def sparsify_topk(attn: torch.Tensor, k: int = 8) -> torch.Tensor:
    # attn: [heads, seq, seq]
    out = []
    for head in attn:
        vals, idx = torch.topk(head, k=min(k, head.size(-1)), dim=-1)
        mask = torch.zeros_like(head)
        mask.scatter_(-1, idx, 1.0)
        out.append(head * mask)
    return torch.stack(out, dim=0)


def add_noise(attn: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    return torch.clamp(attn + scale * torch.randn_like(attn), min=0.0)


def zero_out(attn: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(attn)


def apply_intervention_to_attn(
    attns: List[torch.Tensor], name: str, fn: Callable[[torch.Tensor], torch.Tensor]
) -> List[torch.Tensor]:
    """Apply intervention to each layer's attention."""
    out = []
    for layer_attn in attns:
        # layer_attn: [batch, heads, seq, seq]; apply per sample
        modified = []
        for sample_attn in layer_attn:
            modified.append(fn(sample_attn))
        out.append(torch.stack(modified, dim=0))
    return out


def run_last_layer_with_custom_attn(layer, hidden: torch.Tensor, attention_mask: torch.Tensor, intervention_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Re-implements DistilBertLayer forward with custom attention weights on the last layer only.
    """
    bs, seq_len, dim = hidden.size()
    n_heads = layer.attention.n_heads
    dim_per_head = dim // n_heads
    mask = attention_mask.unsqueeze(1).unsqueeze(2)
    mask = (1 - mask) * -10000.0

    q = layer.attention.q_lin(hidden)
    k = layer.attention.k_lin(hidden)
    v = layer.attention.v_lin(hidden)

    def shape(x):
        return x.view(bs, seq_len, n_heads, dim_per_head).transpose(1, 2)

    q = shape(q)
    k = shape(k)
    v = shape(v)

    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim_per_head, device=hidden.device, dtype=hidden.dtype))
    scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    weights_mod = []
    for b in range(bs):
        weights_mod.append(intervention_fn(weights[b]))
    weights = torch.stack(weights_mod, dim=0)
    weights = layer.attention.dropout(weights)

    context = torch.matmul(weights, v)
    context = context.transpose(1, 2).contiguous().view(bs, seq_len, dim)
    out = layer.attention.out_lin(context)
    out = layer.attention.dropout(out)
    out = layer.attention.layer_norm(out + hidden)

    ff = layer.ffn.lin1(out)
    ff = layer.ffn.activation(ff)
    ff = layer.ffn.lin2(ff)
    ff = layer.ffn.dropout(ff)
    ff = layer.ffn.layer_norm(ff + out)
    return ff


def eval_with_intervention(
    model_name: str,
    dataset_split: str,
    intervention_name: str,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    max_samples: int = 64,
    apply_all_layers: bool = False,
    target_layer: int | None = None,
) -> Dict:
    """
    Evaluate a sequence classifier (SST-2) with attention intervention.
    """
    clf_tok = AutoTokenizer.from_pretrained(model_name)
    clf_model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    clf_model.eval()
    ds = load_dataset("glue", "sst2", split=dataset_split)
    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    gedig = GeDIGCalculator(lambda_param=0.5, gamma=0.5, threshold=0.01, use_percentile=True, percentile=0.9)
    records: List[Dict[str, float]] = []
    corr_pairs: List[Tuple[float, float]] = []

    for item in ds:
        inputs = clf_tok(item["sentence"], return_tensors="pt", truncation=True, max_length=128)
        # baseline forward
        outputs = clf_model(**inputs, output_attentions=True, output_hidden_states=True)
        base_logits = outputs.logits
        base_probs = torch.softmax(base_logits, dim=-1)[0]
        base_pred = int(torch.argmax(base_probs, dim=-1).item())

        attn_mask = inputs["attention_mask"]
        layers = clf_model.distilbert.transformer.layer
        # start from embeddings (hidden_states[0])
        hidden = outputs.hidden_states[0]

        for idx, layer in enumerate(layers):
            bs, seq_len, dim = hidden.size()
            n_heads = layer.attention.n_heads
            dim_per_head = dim // n_heads
            mask_add = (1 - attn_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            q = layer.attention.q_lin(hidden)
            k = layer.attention.k_lin(hidden)
            v = layer.attention.v_lin(hidden)

            def shape(x):
                return x.view(bs, seq_len, n_heads, dim_per_head).transpose(1, 2)

            q = shape(q); k = shape(k); v = shape(v)
            scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim_per_head, device=hidden.device, dtype=hidden.dtype))
            scores = scores + mask_add
            weights = torch.softmax(scores, dim=-1)
            if apply_all_layers or (target_layer is not None and idx == target_layer) or (target_layer is None and idx == len(layers) - 1):
                weights = torch.stack([intervention_fn(weights[b]) for b in range(weights.size(0))], dim=0)
            weights = layer.attention.dropout(weights)

            context = torch.matmul(weights, v)
            context = context.transpose(1, 2).contiguous().view(bs, seq_len, dim)
            out = layer.attention.out_lin(context)
            out = layer.attention.dropout(out)
            out = layer.output_layer_norm(out + hidden)

            ff = layer.ffn.lin1(out)
            ff = layer.ffn.activation(ff)
            ff = layer.ffn.lin2(ff)
            ff = layer.ffn.dropout(ff)
            ff = layer.output_layer_norm(ff + out)
            hidden = ff

        pooled = hidden[:, 0]
        pooled = clf_model.pre_classifier(pooled)
        pooled = torch.relu(pooled)
        pooled = clf_model.dropout(pooled)
        logits = clf_model.classifier(pooled)

        probs = torch.softmax(logits, dim=-1)[0]
        conf = float(torch.max(probs).item())
        pred = int(torch.argmax(probs, dim=-1).item())
        correct += int(pred == item["label"])

        # F on modified last layer weights
        mask_np = inputs["attention_mask"][0].bool().cpu().numpy()
        F_vals = [gedig.compute_F(weights[0, h].detach().cpu().numpy(), mask_np) for h in range(weights.shape[1])]
        F_mean = float(np.mean(F_vals))
        corr_pairs.append((F_mean, conf))
        records.append(
            {
                "text": item["sentence"],
                "label": int(item["label"]),
                "pred": pred,
                "confidence": conf,
                "F_mean_last_layer": F_mean,
                "base_pred": base_pred,
            }
        )

    acc = correct / max(1, len(ds))
    corr = summarize_corr(corr_pairs)
    return {"accuracy": acc, "conf_vs_F_corr": corr, "samples": records}


def summarize_corr(pairs: List[Tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return 0.0
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> None:
    texts = [
        "I love this movie, it was great!",
        "The product broke after one day.",
        "Absolutely fantastic performance.",
        "Terrible service, not recommended.",
    ]
    labels = [1, 0, 1, 0]  # placeholder

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    interventions: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "identity": lambda x: x,
        "sparsify_top8": lambda x: sparsify_topk(x, k=8),
        "sparsify_top4": lambda x: sparsify_topk(x, k=4),
        "sparsify_top2": lambda x: sparsify_topk(x, k=2),
        "sparsify_top1": lambda x: sparsify_topk(x, k=1),
        "noise_scale0.1": lambda x: add_noise(x, scale=0.1),
        "noise_scale0.3": lambda x: add_noise(x, scale=0.3),
        "noise_scale0.5": lambda x: add_noise(x, scale=0.5),
        "noise_scale1.0": lambda x: add_noise(x, scale=1.0),
        "zero_out": zero_out,
    }

    gedig = GeDIGCalculator(lambda_param=0.5, gamma=0.5, threshold=0.01, use_percentile=True, percentile=0.9)
    records: List[Dict[str, float]] = []

    for name, fn in interventions.items():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        attns = outputs.attentions  # tuple len=layers, each [batch, heads, seq, seq]
        # Apply intervention to last layer attentions only (diag; light)
        attn_last = attns[-1].clone()  # [b,h,seq,seq]
        attn_last = fn(attn_last)
        mask = inputs["attention_mask"][0].bool().cpu().numpy()
        # Compute F on first sample, per head
        F_vals = []
        for head in attn_last[0]:
            F_vals.append(gedig.compute_F(head.cpu().numpy(), mask))
        F_mean = float(np.mean(F_vals))

        # Placeholder accuracy (no classification head here)
        records.append({"intervention": name, "F_mean_last_layer": F_mean, "accuracy": None})

    # Downstream proxy: SST2 small slice, record accuracy, confidence, and F on last layer
    def eval_sst2(max_samples: int = 32) -> Dict:
        ds = load_dataset("glue", "sst2", split=f"train[:{max_samples}]")
        clf_name = "distilbert-base-uncased-finetuned-sst-2-english"
        clf_tok = AutoTokenizer.from_pretrained(clf_name)
        clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name, output_attentions=True)
        clf_model.eval()

        samples: List[Dict[str, float]] = []
        correct = 0
        gedig_clf = GeDIGCalculator(lambda_param=0.5, gamma=0.5, threshold=0.01, use_percentile=True, percentile=0.9)
        corr_pairs: List[Tuple[float, float]] = []

        for item in ds:
            inputs = clf_tok(item["sentence"], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = clf_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            conf = float(torch.max(probs).item())
            pred = int(torch.argmax(probs, dim=-1).item())
            correct += int(pred == item["label"])
            attn_last = outputs.attentions[-1][0]  # [heads, seq, seq]
            mask = inputs["attention_mask"][0].bool().cpu().numpy()
            F_vals = [gedig_clf.compute_F(head.cpu().numpy(), mask) for head in attn_last]
            F_mean = float(np.mean(F_vals))
            corr_pairs.append((F_mean, conf))
            samples.append({"text": item["sentence"], "label": int(item["label"]), "pred": pred, "confidence": conf, "F_mean_last_layer": F_mean})

        acc = correct / max(1, len(ds))
        corr = summarize_corr(corr_pairs)
        return {"accuracy": acc, "conf_vs_F_corr": corr, "samples": samples}

    sst2_eval = eval_sst2(max_samples=32)

    # Lightweight intervention eval (re-using SST2) — note logits are original; F is under modified attentions
    sst2_interventions = {}
    clf_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # mid-strength: single-layer (last) interventions
    sst2_interventions["sparsify_top16_last"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "sparsify",
        lambda x: sparsify_topk(x, k=16),
        max_samples=32,
        apply_all_layers=False,
        target_layer=-1,
    )
    sst2_interventions["sparsify_top32_last"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "sparsify",
        lambda x: sparsify_topk(x, k=32),
        max_samples=32,
        apply_all_layers=False,
        target_layer=-1,
    )
    sst2_interventions["noise_scale0.05_last"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "noise",
        lambda x: add_noise(x, scale=0.05),
        max_samples=32,
        apply_all_layers=False,
        target_layer=-1,
    )
    sst2_interventions["noise_scale0.1_last"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "noise",
        lambda x: add_noise(x, scale=0.1),
        max_samples=32,
        apply_all_layers=False,
        target_layer=-1,
    )

    # strong: all-layer (keep for reference)
    sst2_interventions["sparsify_top4_all"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "sparsify",
        lambda x: sparsify_topk(x, k=4),
        max_samples=32,
        apply_all_layers=True,
    )
    sst2_interventions["noise_scale0.5_all"] = eval_with_intervention(
        clf_model_name,
        "train[:32]",
        "noise",
        lambda x: add_noise(x, scale=0.5),
        max_samples=32,
        apply_all_layers=True,
    )

    out_dir = Path("results/transformer_gedig")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intervene_sketch.json"
    payload = {"interventions": records, "sst2_eval": sst2_eval, "sst2_interventions": sst2_interventions}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[done] wrote {out_path} (interventions={len(records)}, sst2_samples={len(sst2_eval['samples'])}, "
        f"sst2_interventions={list(sst2_interventions.keys())})"
    )


if __name__ == "__main__":
    main()
