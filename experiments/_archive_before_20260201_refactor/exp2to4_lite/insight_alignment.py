"""Experiment IV: Insight-vector alignment (lite).

Computes cosine alignment between:
- answer embeddings, and
- readout vectors from DG-confirmed subgraphs (approximated as the
  normalized mean of retrieved SUPPORT episode embeddings per query)

Compares against baselines (random subset, all retrieved) and reports:
  mean similarities, Δs, positive ratio, exact sign-test p-values, N.

Inputs:
- pipeline results JSON from v3-lite (per_samples + QA pairs for geDIG)
- dataset JSONL path (episodes with id/role/text)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys

import numpy as np


# Ensure repository root and local src are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SRC = Path(__file__).resolve().parent / "src"
for p in (REPO_ROOT, REPO_ROOT / "src", LOCAL_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)
from experiments.exp2to4_lite.src.dataset import load_dataset, QueryExample, DocumentExample  # type: ignore
from experiments.exp2to4_lite.src.embedder import Embedder  # type: ignore


def _env_cloud_safe() -> None:
    os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
    os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")


def _normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = float(np.linalg.norm(x) + 1e-12)
        return (x / n).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    return float(np.dot(a, b))


def _mean_vec(vs: np.ndarray) -> np.ndarray:
    if vs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    m = np.mean(vs, axis=0)
    return _normalize(m)


def _exact_sign_test_p(delta: Sequence[float]) -> float:
    """Two-sided exact sign test p-value for median=0.

    Counts positive vs negative deltas (zeros ignored). Returns exact binomial
    p-value (two-sided) using combinations.
    """
    pos = sum(1 for d in delta if d > 0)
    neg = sum(1 for d in delta if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    # p = 2 * sum_{i=k..n} C(n,i) / 2^n
    tot = 2 ** n
    tail = 0
    for i in range(k, n + 1):
        tail += math.comb(n, i)
    p = 2.0 * tail / tot
    return float(min(1.0, p))


def _embed_texts(embedder: Embedder, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    res = embedder.encode(texts)
    return res.vectors.astype(np.float32)


@dataclass
class AlignmentStats:
    n: int
    s_support_mean: float
    s_all_mean: float
    s_random_mean: float
    delta_sr_mean: float
    delta_sa_mean: float
    frac_delta_sr_pos: float
    frac_delta_sa_pos: float
    p_sign_sr: float
    p_sign_sa: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "s_support_mean": round(self.s_support_mean, 4),
            "s_all_mean": round(self.s_all_mean, 4),
            "s_random_mean": round(self.s_random_mean, 4),
            "delta_sr_mean": round(self.delta_sr_mean, 4),
            "delta_sa_mean": round(self.delta_sa_mean, 4),
            "frac_delta_sr_pos": round(self.frac_delta_sr_pos, 4),
            "frac_delta_sa_pos": round(self.frac_delta_sa_pos, 4),
            "p_sign_sr": float(self.p_sign_sr),
            "p_sign_sa": float(self.p_sign_sa),
        }


def compute_alignment(
    dataset_path: Path,
    results_path: Path,
    baseline_name: str,
    embedder_model: str | None,
    seed: int = 1234,
) -> Tuple[AlignmentStats, Path]:
    rng = np.random.default_rng(seed)
    ds = load_dataset(dataset_path)
    with results_path.open("r", encoding="utf-8") as fh:
        result_json = json.load(fh)

    per_samples = result_json.get("results", {}).get(baseline_name, {}).get("per_samples", [])
    if not per_samples:
        raise RuntimeError(f"No per_samples found for baseline '{baseline_name}' in {results_path}")

    embedder = Embedder(embedder_model, normalize=True, cache_dir=None)

    # Build query -> documents map for quick lookup
    query_docs: Dict[str, List[DocumentExample]] = {ex.query: list(ex.documents) for ex in ds}
    doc_text_cache: Dict[Tuple[str, str], str] = {}  # (query, doc_id) -> text
    doc_role_cache: Dict[Tuple[str, str], str] = {}
    for ex in ds:
        for d in ex.documents:
            doc_text_cache[(ex.query, d.doc_id)] = d.text
            role = d.metadata.get("role", "support")
            doc_role_cache[(ex.query, d.doc_id)] = role

    s_support: List[float] = []
    s_all: List[float] = []
    s_random: List[float] = []
    delta_sr: List[float] = []
    delta_sa: List[float] = []

    for sample in per_samples:
        q = sample.get("query", "")
        ans = sample.get("answer", "")
        ids: List[str] = list(sample.get("retrieved_doc_ids", []) or [])
        if not q or not ans or not ids:
            continue
        docs = query_docs.get(q, [])
        if not docs:
            continue

        # Prepare episode texts
        support_ids = [doc_id for doc_id in ids if doc_role_cache.get((q, doc_id), "distractor") == "support"]
        all_texts = [doc_text_cache.get((q, doc_id), "") for doc_id in ids]
        support_texts = [doc_text_cache.get((q, doc_id), "") for doc_id in support_ids]
        support_texts = [t for t in support_texts if t]
        all_texts = [t for t in all_texts if t]
        if not all_texts:
            continue
        k = max(1, len(support_texts))
        # Random baseline: sample k episodes from the available docs for this query
        pool_texts = [d.text for d in docs]
        if len(pool_texts) < k:
            rand_texts = pool_texts
        else:
            idx = rng.choice(len(pool_texts), size=k, replace=False)
            rand_texts = [pool_texts[i] for i in idx]

        # Embed
        vec_ans = _embed_texts(embedder, [ans])
        if vec_ans.size == 0:
            continue
        va = vec_ans[0]
        vec_support = _embed_texts(embedder, support_texts) if support_texts else _embed_texts(embedder, all_texts[:1])
        vec_all = _embed_texts(embedder, all_texts)
        vec_rand = _embed_texts(embedder, rand_texts)
        if vec_all.size == 0:
            continue

        ro_support = _mean_vec(vec_support) if vec_support.size else _mean_vec(vec_all)
        ro_all = _mean_vec(vec_all)
        ro_rand = _mean_vec(vec_rand)

        s_s = _cos(va, ro_support)
        s_a = _cos(va, ro_all)
        s_r = _cos(va, ro_rand)
        s_support.append(s_s)
        s_all.append(s_a)
        s_random.append(s_r)
        delta_sr.append(s_s - s_r)
        delta_sa.append(s_s - s_a)

    n = len(s_support)
    if n == 0:
        raise RuntimeError("Alignment: no valid samples after filtering")

    stats = AlignmentStats(
        n=n,
        s_support_mean=float(np.mean(s_support)),
        s_all_mean=float(np.mean(s_all)),
        s_random_mean=float(np.mean(s_random)),
        delta_sr_mean=float(np.mean(delta_sr)),
        delta_sa_mean=float(np.mean(delta_sa)),
        frac_delta_sr_pos=float(np.mean([1.0 if d > 0 else 0.0 for d in delta_sr])),
        frac_delta_sa_pos=float(np.mean([1.0 if d > 0 else 0.0 for d in delta_sa])),
        p_sign_sr=_exact_sign_test_p(delta_sr),
        p_sign_sa=_exact_sign_test_p(delta_sa),
    )

    # Save alongside results JSON
    out_path = results_path.with_name(results_path.stem + "_alignment.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"baseline": baseline_name, "stats": stats.to_dict()}, fh, ensure_ascii=False, indent=2)

    return stats, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Experiment IV alignment stats (lite)")
    parser.add_argument("--results", type=Path, required=True, help="Pipeline results JSON (v3-lite)")
    parser.add_argument("--dataset", type=Path, required=False, help="Dataset JSONL used by the pipeline")
    parser.add_argument("--baseline", type=str, default="gedig_ag_dg", help="Baseline name in results JSON")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name or null (random)")
    args = parser.parse_args()

    _env_cloud_safe()

    # If dataset not provided, try to read from results config
    dataset_path = args.dataset
    if dataset_path is None:
        with args.results.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
        ds_str = cfg.get("dataset")
        if not ds_str:
            raise SystemExit("Dataset path not provided and not found in results config")
        dataset_path = Path(ds_str)
        if not dataset_path.exists():
            # try relative to results location
            candidate = args.results.parent / ds_str
            if candidate.exists():
                dataset_path = candidate

    stats, out_path = compute_alignment(dataset_path, args.results, args.baseline, args.embedding_model)

    print("\n=== Experiment IV: Alignment Summary ===")
    for k, v in stats.to_dict().items():
        print(f"{k}: {v}")
    print(f"\nSaved alignment summary: {out_path}")


if __name__ == "__main__":
    main()
