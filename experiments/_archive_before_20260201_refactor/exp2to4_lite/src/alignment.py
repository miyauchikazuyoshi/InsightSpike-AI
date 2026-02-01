"""Experiment IV: Alignment (self-contained)."""

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

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src", PKG_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from .dataset import load_dataset, DocumentExample
from .embedder import Embedder


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
    pos = sum(1 for d in delta if d > 0)
    neg = sum(1 for d in delta if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
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

    # Query -> docs map
    query_docs = {ex.query: list(ex.documents) for ex in ds}
    doc_text_cache = {(ex.query, d.doc_id): d.text for ex in ds for d in ex.documents}
    doc_role_cache = {(ex.query, d.doc_id): d.metadata.get("role", "support") for ex in ds for d in ex.documents}

    s_support: List[float] = []
    s_all: List[float] = []
    s_random: List[float] = []
    s_topk: List[float] = []
    s_ag_pick: List[float] = []
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

        support_ids = [doc_id for doc_id in ids if doc_role_cache.get((q, doc_id), "distractor") == "support"]
        all_texts = [doc_text_cache.get((q, doc_id), "") for doc_id in ids]
        support_texts = [doc_text_cache.get((q, doc_id), "") for doc_id in support_ids]
        support_texts = [t for t in support_texts if t]
        all_texts = [t for t in all_texts if t]
        if not all_texts:
            continue
        k = max(1, len(support_texts))
        pool_texts = [d.text for d in docs]
        if len(pool_texts) < k:
            rand_texts = pool_texts
        else:
            idx = rng.choice(len(pool_texts), size=k, replace=False)
            rand_texts = [pool_texts[i] for i in idx]

        va = _embed_texts(embedder, [ans])[0]
        vec_support = _embed_texts(embedder, support_texts) if support_texts else _embed_texts(embedder, all_texts[:1])
        vec_all = _embed_texts(embedder, all_texts)
        vec_rand = _embed_texts(embedder, rand_texts)
        vec_topk = _embed_texts(embedder, all_texts[: max(1, k)])
        # AG-picked texts (union of doc_ids at iterations where AG fired)
        ag_pick_ids: List[str] = []
        meta = sample.get("metadata", {})
        iter_ids = meta.get("iter_retrieved_ids", []) or []
        iter_ag = meta.get("iter_ag", []) or []
        for flag, picked in zip(iter_ag, iter_ids):
            if int(flag) == 1:
                ag_pick_ids.extend(list(picked))
        ag_pick_ids = list(dict.fromkeys(ag_pick_ids))
        ag_pick_texts = [doc_text_cache.get((q, did), "") for did in ag_pick_ids]
        if vec_all.size == 0:
            continue

        ro_support = _mean_vec(vec_support) if vec_support.size else _mean_vec(vec_all)
        ro_all = _mean_vec(vec_all)
        ro_rand = _mean_vec(vec_rand)

        ro_topk = _mean_vec(vec_topk) if vec_topk.size else _mean_vec(vec_all)
        # Compose readouts
        ro_ag = _mean_vec(_embed_texts(embedder, ag_pick_texts)) if ag_pick_texts else _mean_vec(vec_all)
        s_s = _cos(va, ro_support)
        s_a = _cos(va, ro_all)
        s_r = _cos(va, ro_rand)
        s_t = _cos(va, ro_topk)
        s_ap = _cos(va, ro_ag)
        s_support.append(s_s)
        s_all.append(s_a)
        s_random.append(s_r)
        s_topk.append(s_t)
        s_ag_pick.append(s_ap)
        delta_sr.append(s_s - s_r)
        delta_sa.append(s_s - s_a)

    n = len(s_support)
    if n == 0:
        raise RuntimeError("Alignment: no valid samples after filtering")

    # Effect size (Cohen's d) and bootstrap CI for delta_sr
    def _cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        mean_diff = float(a.mean() - b.mean())
        sd_pooled = float(np.sqrt(((a.size - 1)*a.var(ddof=1) + (b.size - 1)*b.var(ddof=1)) / (a.size + b.size - 2))) if (a.size + b.size - 2) > 0 else 0.0
        return mean_diff / sd_pooled if sd_pooled > 1e-12 else 0.0

    d_sr = _cohen_d(s_support, s_random)

    def _bootstrap_ci(x: Sequence[float], iters: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        x = np.asarray(x)
        rng = np.random.default_rng(2025)
        vals = []
        n = x.size
        for _ in range(iters):
            idx = rng.integers(0, n, size=n)
            vals.append(float(x[idx].mean()))
        lo = float(np.percentile(vals, 100 * (alpha / 2)))
        hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        return lo, hi

    ci_lo, ci_hi = _bootstrap_ci(delta_sr)

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

    out_path = results_path.with_name(results_path.stem + "_alignment.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "baseline": baseline_name,
            "stats": stats.to_dict(),
            "extras": {
                "s_topk_mean": float(np.mean(s_topk) if s_topk else 0.0),
                "s_ag_pick_mean": float(np.mean(s_ag_pick) if s_ag_pick else 0.0),
                "cohen_d_sr": d_sr,
                "delta_sr_ci95": [ci_lo, ci_hi],
            }
        }, fh, ensure_ascii=False, indent=2)
    return stats, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Experiment IV alignment (self-contained)")
    parser.add_argument("--results", type=Path, required=True, help="Pipeline results JSON")
    parser.add_argument("--dataset", type=Path, required=False, help="Dataset JSONL")
    parser.add_argument("--baseline", type=str, default="gedig_ag_dg", help="Baseline name in results JSON")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name or null (random)")
    args = parser.parse_args()

    _env_cloud_safe()

    dataset_path = args.dataset
    if dataset_path is None:
        with args.results.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
        ds_str = cfg.get("dataset")
        if not ds_str:
            raise SystemExit("Dataset path not provided and not found in results config")
        dataset_path = Path(ds_str)
        if not dataset_path.exists():
            cand = args.results.parent / ds_str
            if cand.exists():
                dataset_path = cand

    stats, out_path = compute_alignment(dataset_path, args.results, args.baseline, args.embedding_model)
    print("\n=== Experiment IV: Alignment Summary ===")
    for k, v in stats.to_dict().items():
        print(f"{k}: {v}")
    print(f"\nSaved alignment summary: {out_path}")


if __name__ == "__main__":
    main()
