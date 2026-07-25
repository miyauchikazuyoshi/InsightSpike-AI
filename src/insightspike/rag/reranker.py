
"""
Structure-Guided RAG Reranker
=============================

Reranks retrieved documents using a single-state Flash structural profile.
This diagnostic is distinct from canonical before/after geDIG delta F.
"""

from typing import List, Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import logging

from insightspike.gedig import compute_structural_profile

logger = logging.getLogger(__name__)

def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if abs(max_s - min_s) < 1e-12:
        return [0.5 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

def _select_bottom_indices(scores: List[float], fraction: float) -> List[int]:
    if not scores:
        return []
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return list(range(len(scores)))
    count = int(math.ceil(len(scores) * fraction))
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    return sorted_idx[:count]


class StructureReranker:
    """
    Reranks chunks with an experiment-specific structural-profile heuristic.
    """
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        device: Optional[str] = None,
        mix_weight: float = 0.7,
        gate_percentile: Optional[float] = None,
        gate_min_norm: Optional[float] = None,
        gate_penalty: float = 1.0
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing StructureReranker with {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.mix_weight = mix_weight
        self.gate_percentile = gate_percentile
        self.gate_min_norm = gate_min_norm
        self.gate_penalty = gate_penalty

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None,
        base_scores: Optional[List[float]] = None,
        mix_weight: Optional[float] = None,
        gate_percentile: Optional[float] = None,
        gate_min_norm: Optional[float] = None,
        gate_penalty: Optional[float] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Rerank documents by a single-state structural profile.
        
        Args:
            query: The user query.
            documents: List of candidate document strings.
            top_k: Number of top documents to return (default: all).
            base_scores: Optional relevance scores from retriever (same length as documents).
            mix_weight: Weight for structural score when combining with base_scores.
            gate_percentile: Optional bottom fraction (0-1) to demote by low structure.
            gate_min_norm: Optional min normalized structure score required to avoid demotion.
            gate_penalty: Penalty subtracted from scores when gated.
            
        Returns:
            List of dicts: [{"doc": str, "score": float, "metrics": dict}, ...]
        """
        results = []
        if base_scores is not None and len(base_scores) != len(documents):
            raise ValueError("base_scores length must match documents length")
        
        for doc in documents:
            # Construct input: "[CLS] Query [SEP] Document [SEP]"
            # This allows the model to "attend" from query to doc and vice-versa.
            inputs = self.tokenizer(
                query, 
                doc, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                
            # Compute the single-state profile for this query-doc pair.
            # Average heads/layers to retain the historical score contract.
            # Note: We could weigh deeper layers more, but uniform average is a good start.
            
            # attentions: tuple of (batch=1, heads, seq, seq)
            total_profile = 0.0
            metrics_sum = {"epc": 0.0, "entropy": 0.0, "sp": 0.0, "clustering": 0.0}
            num_layers = len(outputs.attentions)
            
            for layer_attn in outputs.attentions:
                profile_value, metrics = compute_structural_profile(
                    layer_attn,
                    attention_mask=inputs.get("attention_mask")
                )
                # profile_value: (1, heads)
                total_profile += profile_value.mean().item()
                metrics_sum["epc"] += metrics["epc"].mean().item()
                metrics_sum["entropy"] += metrics["h"].mean().item()
                metrics_sum["sp"] += metrics["sp"].mean().item()
                metrics_sum["clustering"] += metrics.get(
                    "clustering",
                    torch.tensor(0.0),
                ).mean().item()

            avg_profile = total_profile / num_layers
            avg_metrics = {k: v / num_layers for k, v in metrics_sum.items()}
            
            results.append({
                "doc": doc,
                # ``score`` is retained for compatibility; its value is the
                # historical single-state profile, not canonical delta F.
                "score": avg_profile,
                "metrics": avg_metrics
            })

        for res in results:
            m = res["metrics"]
            # Experiment-specific ranking heuristic derived from profile terms.
            # New Formula: Clustering is KING. Random graphs have ~0 clustering. Semantic graphs have >0.
            # SP is also good. Entropy is bad. EPC is ambiguous but we prefer sparse.
            # Weighting: Clustering (x5.0) > SP (x2.0) > Entropy (x-1.0) > EPC (x-0.5)
            # The high weight on clustering is to brutally punish Random Graphs.
            rank_score = m["clustering"] * 5.0 + m["sp"] * 2.0 - m["entropy"] - m["epc"] * 0.5
            res["rank_score"] = rank_score

        struct_scores = [r["rank_score"] for r in results]
        struct_norm = _min_max_normalize(struct_scores)
        w = self.mix_weight if mix_weight is None else mix_weight
        gate_percentile = self.gate_percentile if gate_percentile is None else gate_percentile
        gate_min_norm = self.gate_min_norm if gate_min_norm is None else gate_min_norm
        gate_penalty = self.gate_penalty if gate_penalty is None else gate_penalty
        gating_active = gate_percentile is not None or gate_min_norm is not None

        gated_indices = set()
        if gate_percentile is not None:
            gated_indices.update(_select_bottom_indices(struct_norm, gate_percentile))
        if gate_min_norm is not None:
            for idx, score in enumerate(struct_norm):
                if score < gate_min_norm:
                    gated_indices.add(idx)

        if base_scores is not None:
            base_norm = _min_max_normalize(base_scores)
            for idx, res in enumerate(results):
                res["base_score"] = base_scores[idx]
                combined = (1.0 - w) * base_norm[idx] + w * struct_norm[idx]
                if gating_active and idx in gated_indices:
                    res["gated"] = True
                    combined -= gate_penalty
                elif gating_active:
                    res["gated"] = False
                res["combined_score"] = combined
            results.sort(key=lambda x: x["combined_score"], reverse=True)
        else:
            for idx, res in enumerate(results):
                if gating_active and idx in gated_indices:
                    res["gated"] = True
                    res["rank_score_gated"] = res["rank_score"] - gate_penalty
                elif gating_active:
                    res["gated"] = False
                    res["rank_score_gated"] = res["rank_score"]
            if gating_active:
                results.sort(key=lambda x: x["rank_score_gated"], reverse=True)
            else:
                results.sort(key=lambda x: x["rank_score"], reverse=True)
        
        if top_k:
            results = results[:top_k]
            
        return results
