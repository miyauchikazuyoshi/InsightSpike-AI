#!/usr/bin/env python3
"""
Compute insight-vector vs LLM-answer alignment (Δs) for RAG supplemental experiment.

Pipeline (simple, dependency-light):
- Load KB texts from repo file: data/insight_store/knowledge_base/initial/insight_dataset.txt
- Embed KB texts and questions/answers via InsightSpike embedder (uses SBERT if available, falls back otherwise)
- Build initial weights w^(0) from cosine(q, e_i) (ReLU)
- Build adjacency A from cosine(e_i, e_j) >= tau (symmetric), row-normalized
- Propagate w^(h) = norm(gamma*A*w^(h-1) + (1-gamma)*w^(0)), H hops
- Insight vector z_ins = sum_h alpha_h * sum_i w_i^(h) e_i  (alpha_h ∝ gamma^h)
- Answer vector z_ans = SBERT(answer)  (optionally residual: ans - q)
- Control s_ctrl: H=0 (no propagation) baseline
- Report Δs = s - s_ctrl and save histogram figure

Usage:
  python3 docs/paper/run_insight_vector_alignment.py \
    --inputs docs/paper/_tmp/final_optimal_prompt_results.json \
    --outfig docs/paper/figures/rag_insight_alignment.pdf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

# Import embedder from project if available
try:
    from src.insightspike.processing.embedder import EmbeddingModelManager
except Exception:  # fallback to simple SBERT via sentence-transformers
    EmbeddingModelManager = None


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x) + 1e-9
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_adjacency(E: np.ndarray, tau: float = 0.35) -> np.ndarray:
    # Cosine similarity matrix
    S = (E @ E.T)
    np.fill_diagonal(S, 0.0)
    A = (S >= tau).astype(float)
    # Row-normalize
    rs = A.sum(axis=1, keepdims=True) + 1e-9
    return A / rs


def weighted_mp(E: np.ndarray, q: np.ndarray, A: np.ndarray, H: int = 3, gamma: float = 0.7) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # initial weights from ReLU(cos)
    w0 = np.maximum(E @ q, 0.0)
    if w0.sum() > 0:
        w0 = w0 / (w0.sum() + 1e-9)
    ws = [w0]
    for h in range(1, H + 1):
        w_prev = ws[-1]
        w = gamma * (A @ w_prev) + (1.0 - gamma) * w0
        s = w.sum() + 1e-9
        if s > 0:
            w = w / s
        ws.append(w)
    # Insight vectors per hop
    zs = [E.T @ w for w in ws]
    return ws, zs


def load_kb_texts(kb_path: Path) -> List[str]:
    texts = []
    if not kb_path.exists():
        return texts
    with kb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if len(t) >= 24:  # skip short fragments
                texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True, help="JSON file with question/response pairs")
    ap.add_argument("--outfig", type=str, default="docs/paper/figures/rag_insight_alignment.pdf")
    ap.add_argument("--kb", type=str, default="data/insight_store/knowledge_base/initial/insight_dataset.txt")
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--residual_ans", action="store_true", help="Use residual ans-ques embedding")
    # Negative controls (optional, multiple allowed)
    ap.add_argument(
        "--neg", nargs="*", default=[], choices=[
            "shuffle_answers",  # permute answers across questions
            "random_adj",       # random Erdos–Renyi adjacency with same density
            "degree_preserve",  # per-row degree-preserving random neighbors (approx.)
        ], help="Negative controls to compute additionally"
    )
    ap.add_argument("--outjson", type=str, default=None, help="Optional path to write JSON stats (including negatives)")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    kb_path = Path(args.kb)
    outfig = Path(args.outfig)
    outfig.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(inputs.read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    for rec in data.get("results", []):
        q = rec.get("question")
        a = rec.get("response")
        if q and a:
            pairs.append((q, a))

    if not pairs:
        print("[warn] No QA pairs found in inputs")
        return 0

    qs = [q for q, _ in pairs]
    ans_list = [a for _, a in pairs]

    kb_texts = load_kb_texts(kb_path)
    if not kb_texts:
        print(f"[warn] KB texts not found at {kb_path}")
        return 0

    # Initialize embedder (SBERT if available)
    if EmbeddingModelManager is not None:
        emm = EmbeddingModelManager()
        embed = lambda texts: emm.encode(texts, normalize_embeddings=True)
    else:
        # Minimal fallback using sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embed = lambda texts: l2norm(model.encode(texts))
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                all_texts = kb_texts + qs + ans_list
                if not all_texts:
                    all_texts = ["placeholder"]
                vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
                vectorizer.fit(all_texts)

                def embed(texts: List[str]) -> np.ndarray:
                    if not texts:
                        dim = vectorizer.max_features or len(vectorizer.get_feature_names_out())
                        return np.zeros((0, dim), dtype=np.float32)
                    mat = vectorizer.transform(texts)
                    return l2norm(mat.toarray().astype(np.float32))

            except Exception:
                rng = np.random.default_rng(42)

                def embed(texts: List[str]) -> np.ndarray:
                    if not texts:
                        return np.zeros((0, 384), dtype=np.float32)
                    return l2norm(rng.normal(size=(len(texts), 384)).astype(np.float32))

    E = embed(kb_texts)
    E = l2norm(E)
    A = build_adjacency(E, tau=args.tau)

    q_vecs = embed(qs) if qs else np.empty((0, E.shape[1]))
    ans_vecs = embed(ans_list) if ans_list else np.empty((0, E.shape[1]))

    def _compute_delta_with_adj(A_used: np.ndarray, use_residual: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _d, _s, _s0 = [], [], []
        for i in range(len(pairs)):
            qv = q_vecs[i]
            ws, zs = weighted_mp(E, qv, A_used, H=args.H, gamma=args.gamma)
            alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
            alphas = alphas / (alphas.sum() + 1e-9)
            z_ins = l2norm((np.vstack(zs).T @ alphas))
            av = ans_vecs[i]
            if use_residual:
                av = l2norm(av - qv)
            s = cosine(z_ins, av)
            z0 = l2norm(zs[0])
            s0 = cosine(z0, av)
            _d.append(s - s0); _s.append(s); _s0.append(s0)
        return np.array(_d), np.array(_s), np.array(_s0)

    def _sign_test(delta: np.ndarray) -> dict:
        n = int(len(delta))
        if n == 0:
            return {"pos_rate": 0.0, "p_one_sided": 1.0, "p_two_sided": 1.0}
        k = int((delta > 0).sum())
        from math import comb
        tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        p_two = min(1.0, 2.0 * min(tail, 1.0 - tail))
        return {"pos_rate": k / n, "p_one_sided": float(tail), "p_two_sided": float(p_two)}

    def _cohens_d(delta: np.ndarray) -> float:
        if len(delta) < 2:
            return 0.0
        m = float(delta.mean()); sd = float(delta.std(ddof=1) + 1e-9)
        return m / sd

    deltas, sims, sims_ctrl = _compute_delta_with_adj(A, args.residual_ans)

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = max(10, min(40, len(deltas)//2))
    ax.hist(deltas, bins=bins, alpha=0.8, color="#4C78A8")
    ax.axvline(deltas.mean(), color="red", linestyle="--", label=f"mean Δs={deltas.mean():+.3f}")
    ax.set_title("Insight vector alignment: Δs distribution (H vs H=0)")
    ax.set_xlabel("Δs = cos(z_ins, z_ans) - cos(z0, z_ans)")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfig, bbox_inches="tight")
    print(f"✅ Wrote {outfig}")

    # Stats and optional negative controls
    base_stats = {
        "n": int(len(deltas)),
        "mean_delta": float(deltas.mean()) if len(deltas) else 0.0,
        "p90_delta": float(np.percentile(deltas, 90)) if len(deltas) else 0.0,
        "mean_s": float(sims.mean()) if len(sims) else 0.0,
        "mean_s_ctrl": float(sims_ctrl.mean()) if len(sims_ctrl) else 0.0,
        "cohens_d": _cohens_d(deltas),
        "sign_test": _sign_test(deltas),
    }

    neg_stats = {}
    if args.neg:
        rng = np.random.default_rng(0)
        if "shuffle_answers" in args.neg and len(ans_vecs):
            perm = rng.permutation(len(ans_vecs))
            ans_perm = ans_vecs[perm]
            # compute using permuted answers
            d_neg, s_neg, s0_neg = [], [], []
            for i in range(len(pairs)):
                qv = q_vecs[i]
                ws, zs = weighted_mp(E, qv, A, H=args.H, gamma=args.gamma)
                alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
                alphas = alphas / (alphas.sum() + 1e-9)
                z_ins = l2norm((np.vstack(zs).T @ alphas))
                av = ans_perm[i]
                if args.residual_ans:
                    av = l2norm(av - qv)
                s = cosine(z_ins, av); z0 = l2norm(zs[0]); s0 = cosine(z0, av)
                d_neg.append(s - s0); s_neg.append(s); s0_neg.append(s0)
            d_neg = np.array(d_neg)
            neg_stats["shuffle_answers"] = {
                "n": int(len(d_neg)),
                "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
                "cohens_d": _cohens_d(d_neg),
                "sign_test": _sign_test(d_neg),
            }
            try:
                fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
                bins2 = max(10, min(40, len(d_neg)//2))
                ax2.hist(d_neg, bins=bins2, alpha=0.75, color="#bdbdbd")
                ax2.axvline(d_neg.mean(), color="black", linestyle=":", label=f"neg mean Δs={d_neg.mean():+.3f}")
                ax2.set_title("Negative control: shuffled answers Δs"); ax2.legend(frameon=False)
                ax2.set_xlabel("Δs (shuffled)"); ax2.set_ylabel("count")
                fig2.tight_layout(); fig2.savefig(outfig.with_name(outfig.stem + "_neg_shuffle.pdf"))
            except Exception:
                pass
        if "random_adj" in args.neg:
            p = float((A > 0).mean()); nN = A.shape[0]
            Ar = (rng.random((nN, nN)) < p).astype(float); np.fill_diagonal(Ar, 0.0)
            Ar = Ar / (Ar.sum(axis=1, keepdims=True) + 1e-9)
            d_neg, _, _ = _compute_delta_with_adj(Ar, args.residual_ans)
            neg_stats["random_adj"] = {
                "n": int(len(d_neg)),
                "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
                "cohens_d": _cohens_d(d_neg),
                "sign_test": _sign_test(d_neg),
            }
            try:
                fig3, ax3 = plt.subplots(figsize=(6.4, 4.2))
                bins3 = max(10, min(40, len(d_neg)//2))
                ax3.hist(d_neg, bins=bins3, alpha=0.75, color="#c7e9c0")
                ax3.axvline(d_neg.mean(), color="green", linestyle=":", label=f"neg mean Δs={d_neg.mean():+.3f}")
                ax3.set_title("Negative control: random adjacency Δs"); ax3.legend(frameon=False)
                ax3.set_xlabel("Δs (rand A)"); ax3.set_ylabel("count")
                fig3.tight_layout(); fig3.savefig(outfig.with_name(outfig.stem + "_neg_randA.pdf"))
            except Exception:
                pass
        if "degree_preserve" in args.neg:
            # Approximate per-row degree preserved sampling from thresholded similarity
            S = (E @ E.T)
            np.fill_diagonal(S, -1.0)
            Abin = (S >= args.tau).astype(int); np.fill_diagonal(Abin, 0)
            nN = Abin.shape[0]
            Br = np.zeros_like(Abin, dtype=float)
            for i in range(nN):
                deg = int(Abin[i].sum())
                if deg <= 0:
                    continue
                pool = [j for j in range(nN) if j != i]
                sel = rng.choice(pool, size=min(deg, len(pool)), replace=False)
                Br[i, sel] = 1.0
            Br = Br / (Br.sum(axis=1, keepdims=True) + 1e-9)
            d_neg, _, _ = _compute_delta_with_adj(Br, args.residual_ans)
            neg_stats["degree_preserve"] = {
                "n": int(len(d_neg)),
                "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
                "cohens_d": _cohens_d(d_neg),
                "sign_test": _sign_test(d_neg),
            }
            try:
                fig4, ax4 = plt.subplots(figsize=(6.4, 4.2))
                bins4 = max(10, min(40, len(d_neg)//2))
                ax4.hist(d_neg, bins=bins4, alpha=0.75, color="#fdd0a2")
                ax4.axvline(d_neg.mean(), color="orange", linestyle=":", label=f"neg mean Δs={d_neg.mean():+.3f}")
                ax4.set_title("Negative control: degree-preserved row Δs"); ax4.legend(frameon=False)
                ax4.set_xlabel("Δs (deg-preserve)"); ax4.set_ylabel("count")
                fig4.tight_layout(); fig4.savefig(outfig.with_name(outfig.stem + "_neg_degpres.pdf"))
            except Exception:
                pass

    all_stats = {"base": base_stats, "negatives": neg_stats, "args": {
        "H": args.H, "gamma": args.gamma, "tau": args.tau, "residual_ans": bool(args.residual_ans)
    }}

    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    if args.outjson:
        try:
            Path(args.outjson).parent.mkdir(parents=True, exist_ok=True)
            Path(args.outjson).write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Wrote {args.outjson}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
