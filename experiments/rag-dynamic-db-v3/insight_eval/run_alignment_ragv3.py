#!/usr/bin/env python3
"""
Run supplemental alignment using the RAG v3 experiment's own knowledge and queries.

Steps:
- Load multi-domain KB and queries from experiments/rag-dynamic-db-v3/src
- Build embeddings for KB (SBERT via repo embedder or sentence-transformers)
- Build adjacency by cosine threshold (τ)
- For each query: build a lightweight RAG prompt (via L4PromptBuilder) using top-K retrieved KB items
- Generate an LLM answer (ProviderFactory; env OPENAI_API_KEY / ANTHROPIC_API_KEY)
- Compute z_ins by multi-hop weighted message passing; z_ans by embedding answer
- Save Δs histogram and stats to results/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any, Dict
import time
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Reduce tokenizers threading to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reduce tokenizers threading to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_SRC = REPO_ROOT / "experiments/rag-dynamic-db-v3/src"

sys.path.insert(0, str(RAG_SRC))
sys.path.insert(0, str(REPO_ROOT / "src"))

from multidomain_knowledge_base import (
    create_multidomain_knowledge_base,
    create_multidomain_queries,
)
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.implementations.layers.layer4_prompt_builder import L4PromptBuilder

# Prefer repo's EmbeddingManager (has deterministic fallback if ST not installed)
try:
    from insightspike.processing.embedder import EmbeddingManager as _EmbeddingManager
except Exception:
    _EmbeddingManager = None


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_adjacency(E: np.ndarray, tau: float = 0.35) -> np.ndarray:
    S = (E @ E.T)
    np.fill_diagonal(S, 0.0)
    A = (S >= tau).astype(float)
    rs = A.sum(axis=1, keepdims=True) + 1e-9
    return A / rs


def weighted_mp(E: np.ndarray, q: np.ndarray, A: np.ndarray, H: int = 3, gamma: float = 0.7) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    w0 = np.maximum(E @ q, 0.0)
    if w0.sum() > 0:
        w0 = w0 / (w0.sum() + 1e-9)
    ws = [w0]
    for _ in range(1, H + 1):
        w_prev = ws[-1]
        w = gamma * (A @ w_prev) + (1.0 - gamma) * w0
        s = w.sum() + 1e-9
        if s > 0:
            w = w / s
        ws.append(w)
    zs = [E.T @ w for w in ws]
    return ws, zs


def build_embedder(embed_mode: str, kb_texts: List[str], queries: List[str]):
    """Return an embedding function f(list[str])->np.ndarray with L2 norm rows.

    embed_mode: 'auto' | 'tfidf' | 'fallback'
    """
    mode = (embed_mode or "auto").lower()
    if mode == "tfidf":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        vec.fit(kb_texts + queries)
        return lambda texts: l2norm(vec.transform(texts).astype(np.float64).toarray())

    if mode == "fallback":
        def _rand_embed(texts):
            vecs = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(seed)
                v = rng.normal(size=(384,))
                v = v / (np.linalg.norm(v) + 1e-9)
                vecs.append(v)
            return np.vstack(vecs)
        return lambda texts: l2norm(_rand_embed(texts))

    # auto: prefer sentence-transformers, then repo embedder, else tfidf
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return lambda texts: l2norm(model.encode(texts))
    except Exception:
        pass
    if _EmbeddingManager is not None:
        try:
            mgr = _EmbeddingManager()
            return lambda texts: l2norm(mgr.encode(texts, normalize_embeddings=True))
        except Exception:
            pass
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    vec.fit(kb_texts + queries)
    return lambda texts: l2norm(vec.transform(texts).astype(np.float64).toarray())


def choose_provider(explicit: str | None = None) -> str:
    """Pick an LLM provider with sensible fallbacks.

    Order of precedence:
    - explicit CLI arg if provided
    - Anthropic if any of {ANTHROPIC_API_KEY, CLAUDE_API_KEY, ANTH_API_KEY} is set
    - OpenAI if OPENAI_API_KEY is set
    - mock otherwise
    """
    if explicit:
        return explicit.lower()
    env = os.environ
    if env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_API_KEY") or env.get("ANTH_API_KEY"):
        return "anthropic"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def build_prompt(builder: L4PromptBuilder, query: str, kb_texts: List[str], E: np.ndarray, embed, topk: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    qv = embed([query])[0]
    sims = (E @ qv)
    idx = np.argsort(-sims)[:topk]
    docs = []
    for i in idx:
        docs.append({"text": kb_texts[i], "c_value": float(sims[i]), "similarity": float(sims[i])})
    context = {"retrieved_documents": docs}
    return builder.build_prompt(context, query), docs


def _generate_with_retry(
    provider,
    provider_name: str,
    prompt: str,
    model_name: str | None,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> str:
    last_err = None
    for i in range(max_retries):
        try:
            if provider_name in ("anthropic", "openai"):
                return provider.generate(prompt, model=model_name)
            return provider.generate(prompt)
        except Exception as e:
            last_err = e
            try:
                time.sleep(backoff ** i)
            except Exception:
                pass
    return f"(error after {max_retries} retries) {last_err}"


def _generate_with_retry(provider, provider_name: str, prompt: str, model_name: str | None, max_retries: int = 3, backoff: float = 1.5) -> str:
    last_err = None
    for i in range(max_retries):
        try:
            if provider_name in ("anthropic", "openai"):
                return provider.generate(prompt, model=model_name)  # kwargs supported by both providers
            return provider.generate(prompt)
        except Exception as e:
            last_err = e
            try:
                time.sleep(backoff ** i)
            except Exception:
                pass
    return f"(error after {max_retries} retries) {last_err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="experiments/rag-dynamic-db-v3/insight_eval/results")
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--residual_ans", action="store_true")
    ap.add_argument("--embed_mode", type=str, default="auto", choices=["auto","tfidf","fallback"], help="Embedding backend")
    ap.add_argument("--provider", type=str, default=None, choices=["anthropic","openai","mock"], help="Force provider (default: auto-detect)")
    ap.add_argument("--model", type=str, default=None, help="Override model name for provider (e.g., claude-3-5-sonnet-20240620)")
    ap.add_argument("--from_json", type=str, default=None, help="Reuse answers from a previous JSON instead of calling the provider")
    ap.add_argument("--plot_hops", action="store_true", help="Emit hop profile figure (mean s_h with error bars)")
    # Negative controls
    ap.add_argument("--neg", nargs="*", default=[], choices=["shuffle_answers","random_adj","degree_preserve"], help="Negative controls to compute additionally")
    # Logging options
    ap.add_argument("--log_prompt", action="store_true", help="Include built prompt in JSON records (truncated)")
    ap.add_argument("--prompt_chars", type=int, default=1200, help="Max characters of prompt to record")
    ap.add_argument("--log_docs", action="store_true", help="Include retrieved docs metadata (text/similarity)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outjson = outdir / "outputs" / "rag_insight_alignment_ragv3.json"
    figdir.mkdir(parents=True, exist_ok=True)
    outjson.parent.mkdir(parents=True, exist_ok=True)

    # Load .env if present (no logging of secrets)
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line or line.strip().startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # Do not override pre-set env vars
                    if k and (k not in os.environ):
                        os.environ[k] = v
        except Exception:
            pass

    # Load knowledge and queries from rag_v3
    kb_items = create_multidomain_knowledge_base()
    kb_texts = [k.text for k in kb_items]
    queries = [q for q, _lvl in create_multidomain_queries()]
    embed = build_embedder(args.embed_mode, kb_texts, queries)
    E = embed(kb_texts)
    A = build_adjacency(E, tau=args.tau)

    # Prompt builder and provider (unless --from_json)
    builder = L4PromptBuilder()
    provider_name = choose_provider(args.provider)
    model_name = args.model
    provider = None
    if args.from_json is None:
        # Pass API key explicitly if present in env/.env
        api_key = None
        if provider_name == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTH_API_KEY")
        elif provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        # Resolve model name (CLI > env default)
        if not model_name:
            if provider_name == "anthropic":
                model_name = os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL")
            elif provider_name == "openai":
                model_name = os.getenv("OPENAI_MODEL")
        provider = ProviderFactory.create(
            provider_name,
            config=dict(provider=provider_name, api_key=api_key, model_name=model_name),
        )
        try:
            print(f"[insight_eval] provider={provider_name}, model={model_name or '(default)'}")
        except Exception:
            pass
    else:
        provider_name = "from_json"
        try:
            print(f"[insight_eval] reuse answers from {args.from_json}")
        except Exception:
            pass
        with open(args.from_json, 'r', encoding='utf-8') as f:
            prev = json.load(f)
        # Map question -> answer
        prev_map = {r["question"]: r["answer"] for r in prev.get("records", [])}

    deltas = []
    sims = []
    sims_ctrl = []
    records: List[Dict[str, Any]] = []

    # Retry policy from env (optional)
    max_retries = int(os.getenv("INSIGHT_EVAL_MAX_RETRIES", "3"))
    backoff = float(os.getenv("INSIGHT_EVAL_RETRY_BACKOFF", "1.5"))

    for q in queries:
        # Build prompt and get answer
        prompt, used_docs = build_prompt(builder, q, kb_texts, E, embed, topk=args.topk)
        if provider_name == "mock":
            # Simple mock: echo query as answer
            ans = f"(mock) {q}"
        elif provider_name == "from_json":
            ans = prev_map.get(q, f"(missing answer) {q}")
        else:
            # With simple retry/backoff for overloads
            ans = _generate_with_retry(provider, provider_name, prompt, model_name, max_retries, backoff)

        # Insight vector via MP
        qv = embed([q])[0]
        ws, zs = weighted_mp(E, qv, A, H=args.H, gamma=args.gamma)
        alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
        alphas = alphas / (alphas.sum() + 1e-9)
        z_ins = l2norm((np.vstack(zs).T @ alphas))

        ans_vec = embed([ans])[0]
        if args.residual_ans:
            ans_vec = l2norm(ans_vec - qv)

        # Per-hop similarities
        s_h = [cosine(l2norm(z), ans_vec) for z in zs]
        s = cosine(z_ins, ans_vec)
        z0 = l2norm(zs[0])
        s0 = cosine(z0, ans_vec)
        deltas.append(s - s0)
        sims.append(s)
        sims_ctrl.append(s0)
        rec: Dict[str, Any] = {"question": q, "answer": ans, "s": s, "s0": s0, "delta_s": s - s0, "s_h": s_h}
        if args.log_prompt:
            rec["prompt"] = str(prompt)[: max(0, int(args.prompt_chars))]
        if args.log_docs:
            # Limit each doc text length to keep JSON small
            trimmed = []
            for d in used_docs:
                td = dict(d)
                if isinstance(td.get("text"), str):
                    td["text"] = td["text"][:400]
                trimmed.append(td)
            rec["docs"] = trimmed
        records.append(rec)

    deltas = np.array(deltas)
    sims = np.array(sims)
    sims_ctrl = np.array(sims_ctrl)

    # Save alignment histogram figure
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = max(10, min(40, len(deltas)//2))
    ax.hist(deltas, bins=bins, alpha=0.85, color="#4C78A8")
    ax.axvline(deltas.mean(), color="red", linestyle="--", label=f"mean Δs={deltas.mean():+.3f}")
    ax.set_title("RAG v3: Insight vector alignment Δs")
    ax.set_xlabel("Δs = cos(z_ins, z_ans) - cos(z0, z_ans)")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    fig.tight_layout()
    figpath = figdir / "rag_insight_alignment_ragv3.pdf"
    fig.savefig(figpath, bbox_inches="tight")

    # Absolute alignment s vs s0 (overlay + paired)
    try:
        # bootstrap 95% CIs for s and s0
        rng = np.random.default_rng(0)
        nbs = 10000 if len(sims) < 200 else 3000
        bs_s = [np.mean(rng.choice(sims, len(sims), replace=True)) for _ in range(nbs)]
        bs_s0 = [np.mean(rng.choice(sims_ctrl, len(sims_ctrl), replace=True)) for _ in range(nbs)]
        ci_s = (np.percentile(bs_s, 2.5), np.percentile(bs_s, 97.5))
        ci_s0 = (np.percentile(bs_s0, 2.5), np.percentile(bs_s0, 97.5))

        figA, axs = plt.subplots(1, 2, figsize=(10, 4))
        # (left) overlay histogram
        lo = float(min(sims.min(), sims_ctrl.min()))
        hi = float(max(sims.max(), sims_ctrl.max()))
        bins_abs = max(10, min(40, len(sims)//2))
        axs[0].hist(sims_ctrl, bins=bins_abs, alpha=0.5, color="#9ecae1", label=f"s0 mean={sims_ctrl.mean():+.3f} \n95%CI=[{ci_s0[0]:+.3f},{ci_s0[1]:+.3f}]")
        axs[0].hist(sims, bins=bins_abs, alpha=0.5, color="#3182bd", label=f"s mean={sims.mean():+.3f} \n95%CI=[{ci_s[0]:+.3f},{ci_s[1]:+.3f}]")
        axs[0].axvline(sims_ctrl.mean(), color="#6baed6", linestyle=":")
        axs[0].axvline(sims.mean(), color="#08519c", linestyle=":")
        axs[0].set_title("Absolute alignment: s vs s0")
        axs[0].set_xlabel("cosine similarity")
        axs[0].set_ylabel("count")
        axs[0].legend(frameon=False, fontsize=8)

        # (right) paired plot
        x0 = np.zeros_like(sims)
        x1 = np.ones_like(sims)
        for i in range(len(sims)):
            axs[1].plot([0, 1], [sims_ctrl[i], sims[i]], color="#636363", alpha=0.5)
        axs[1].scatter(x0, sims_ctrl, color="#9ecae1", edgecolor="none", alpha=0.8, s=18, label="s0")
        axs[1].scatter(x1, sims, color="#3182bd", edgecolor="none", alpha=0.8, s=18, label="s")
        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(["s0 (0-hop)", "s (H-hop)"])
        axs[1].set_xlim(-0.3, 1.3)
        axs[1].set_ylim(min(lo, -1.0), max(hi, 1.0))
        axs[1].set_title(f"Paired alignment per query (n={len(sims)})")
        axs[1].grid(True, axis="y", alpha=0.2)

        figA.tight_layout()
        figabs = figdir / "rag_insight_alignment_abs.pdf"
        figA.savefig(figabs, bbox_inches="tight")
    except Exception:
        pass

    # Triptych: s0, s, and Δs side-by-side for quick comparison
    try:
        figT, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
        # Common bins
        b0 = max(10, min(40, len(sims_ctrl)//2))
        b1 = max(10, min(40, len(sims)//2))
        b2 = max(10, min(40, len(deltas)//2))

        # Panel 1: s0
        rng = np.random.default_rng(0)
        nbs = 8000 if len(sims_ctrl) < 200 else 3000
        bs0 = [np.mean(rng.choice(sims_ctrl, len(sims_ctrl), replace=True)) for _ in range(nbs)]
        ci0 = (np.percentile(bs0, 2.5), np.percentile(bs0, 97.5))
        axes[0].hist(sims_ctrl, bins=b0, color="#9ecae1", alpha=0.85)
        axes[0].axvline(sims_ctrl.mean(), color="#08519c", linestyle=":", lw=1.2)
        axes[0].set_title(f"s0 (0-hop)\nmean={sims_ctrl.mean():+.3f} 95%CI=[{ci0[0]:+.3f},{ci0[1]:+.3f}]")
        axes[0].set_xlabel("cos(z0, z_ans)")
        axes[0].set_ylabel("count")

        # Panel 2: s
        bs1 = [np.mean(rng.choice(sims, len(sims), replace=True)) for _ in range(nbs)]
        ci1 = (np.percentile(bs1, 2.5), np.percentile(bs1, 97.5))
        axes[1].hist(sims, bins=b1, color="#3182bd", alpha=0.85)
        axes[1].axvline(sims.mean(), color="#08306b", linestyle=":", lw=1.2)
        axes[1].set_title(f"s (H-hop)\nmean={sims.mean():+.3f} 95%CI=[{ci1[0]:+.3f},{ci1[1]:+.3f}]")
        axes[1].set_xlabel("cos(z_ins, z_ans)")
        axes[1].set_ylabel("")

        # Panel 3: Δs
        bs2 = [np.mean(rng.choice(deltas, len(deltas), replace=True)) for _ in range(nbs)]
        ci2 = (np.percentile(bs2, 2.5), np.percentile(bs2, 97.5))
        pos_ratio = float((deltas > 0).mean())
        axes[2].hist(deltas, bins=b2, color="#4C78A8", alpha=0.85)
        axes[2].axvline(deltas.mean(), color="red", linestyle="--", lw=1.2)
        axes[2].set_title(f"Δs (H-hop − 0-hop)\nmean={deltas.mean():+.3f} 95%CI=[{ci2[0]:+.3f},{ci2[1]:+.3f}] pos={pos_ratio*100:.0f}%")
        axes[2].set_xlabel("Δs")
        axes[2].set_ylabel("")

        figT.tight_layout()
        trip = figdir / "rag_insight_alignment_triptych.pdf"
        figT.savefig(trip, bbox_inches="tight")
    except Exception:
        pass

    # Optional: Hop profile plot (mean s_h with error bars)
    if args.plot_hops and len(records) > 0 and "s_h" in records[0]:
        # Gather s_h matrix
        max_h = max(len(r["s_h"]) for r in records)
        mat = np.array([r["s_h"] + [np.nan]*(max_h-len(r["s_h"])) for r in records], dtype=float)
        means = np.nanmean(mat, axis=0)
        ses = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
        x = np.arange(max_h)
        fig2, ax2 = plt.subplots(figsize=(6.0, 3.6))
        ax2.plot(x, means, marker='o', color="#4C78A8", label="mean s_h")
        ax2.fill_between(x, means - 1.96*ses, means + 1.96*ses, color="#4C78A8", alpha=0.2, label="95% CI")
        ax2.set_xlabel("hop h")
        ax2.set_ylabel("cos(z_h, z_ans)")
        ax2.set_title("Hop-wise alignment (s_h)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)
        hop_fig = figdir / "rag_insight_hop_profile.pdf"
        fig2.tight_layout()
        fig2.savefig(hop_fig, bbox_inches="tight")

    # Stats helpers
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
        return float(delta.mean()) / float(delta.std(ddof=1) + 1e-9)

    base_stats = {
        "provider": provider_name,
        "model": model_name,
        "embed_mode": args.embed_mode,
        "H": args.H,
        "gamma": args.gamma,
        "tau": args.tau,
        "topk": args.topk,
        "n": len(deltas),
        "mean_delta": float(deltas.mean()) if len(deltas) else 0.0,
        "p90_delta": float(np.percentile(deltas, 90)) if len(deltas) else 0.0,
        "mean_s": float(sims.mean()) if len(sims) else 0.0,
        "mean_s_ctrl": float(sims_ctrl.mean()) if len(sims_ctrl) else 0.0,
        "sign_test": _sign_test(np.array(deltas)),
        "cohens_d": _cohens_d(np.array(deltas)),
    }

    neg_stats = {}
    # Negative: shuffled answers
    if "shuffle_answers" in (args.neg or []):
        rng = np.random.default_rng(0)
        ans_all = [r["answer"] for r in records]
        perm = rng.permutation(len(ans_all)) if len(ans_all) else np.array([], dtype=int)
        ans_perm = [ans_all[i] for i in perm]
        d_neg = []
        for i, q in enumerate(queries):
            qv = embed([q])[0]
            ws, zs = weighted_mp(E, qv, A, H=args.H, gamma=args.gamma)
            alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
            alphas = alphas / (alphas.sum() + 1e-9)
            z_ins = l2norm((np.vstack(zs).T @ alphas))
            av = embed([ans_perm[i]])[0]
            if args.residual_ans:
                av = l2norm(av - qv)
            s = cosine(z_ins, av); s0 = cosine(l2norm(zs[0]), av)
            d_neg.append(s - s0)
        d_neg = np.array(d_neg)
        neg_stats["shuffle_answers"] = {
            "n": int(len(d_neg)),
            "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
            "sign_test": _sign_test(d_neg),
            "cohens_d": _cohens_d(d_neg),
        }
        try:
            figN, axN = plt.subplots(figsize=(6.4, 4.2))
            bN = max(10, min(40, len(d_neg)//2))
            axN.hist(d_neg, bins=bN, alpha=0.8, color="#bdbdbd")
            axN.axvline(d_neg.mean(), color="black", linestyle=":")
            axN.set_title("Negative: shuffled answers Δs"); axN.set_xlabel("Δs"); axN.set_ylabel("count")
            figN.tight_layout(); figN.savefig(figdir / "rag_insight_alignment_neg_shuffle.pdf")
        except Exception:
            pass

    # Negative: random adjacency
    if "random_adj" in (args.neg or []):
        rng = np.random.default_rng(0)
        p = float((A > 0).mean()); nN = A.shape[0]
        Ar = (rng.random((nN, nN)) < p).astype(float); np.fill_diagonal(Ar, 0.0)
        Ar = Ar / (Ar.sum(axis=1, keepdims=True) + 1e-9)
        d_neg = []
        for q in queries:
            qv = embed([q])[0]
            ws, zs = weighted_mp(E, qv, Ar, H=args.H, gamma=args.gamma)
            alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
            alphas = alphas / (alphas.sum() + 1e-9)
            z_ins = l2norm((np.vstack(zs).T @ alphas))
            av = embed([" ".join(q.split()[::-1])])[0] if False else embed([q])[0]  # placeholder same-embed to avoid extra calls
            # Use original answers for comparability
            av = embed([records[queries.index(q)]["answer"]])[0] if len(records) == len(queries) else embed([q])[0]
            if args.residual_ans:
                av = l2norm(av - qv)
            s = cosine(z_ins, av); s0 = cosine(l2norm(zs[0]), av)
            d_neg.append(s - s0)
        d_neg = np.array(d_neg)
        neg_stats["random_adj"] = {
            "n": int(len(d_neg)), "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
            "sign_test": _sign_test(d_neg), "cohens_d": _cohens_d(d_neg)
        }
        try:
            figR, axR = plt.subplots(figsize=(6.4, 4.2))
            bR = max(10, min(40, len(d_neg)//2))
            axR.hist(d_neg, bins=bR, alpha=0.8, color="#c7e9c0")
            axR.axvline(d_neg.mean(), color="green", linestyle=":")
            axR.set_title("Negative: random adjacency Δs"); axR.set_xlabel("Δs"); axR.set_ylabel("count")
            figR.tight_layout(); figR.savefig(figdir / "rag_insight_alignment_neg_randA.pdf")
        except Exception:
            pass

    # Negative: degree-preserve (rowwise)
    if "degree_preserve" in (args.neg or []):
        rng = np.random.default_rng(0)
        S = (E @ E.T); np.fill_diagonal(S, -1.0)
        Abin = (S >= args.tau).astype(int); np.fill_diagonal(Abin, 0)
        nN = Abin.shape[0]; Br = np.zeros_like(Abin, dtype=float)
        for i in range(nN):
            deg = int(Abin[i].sum());
            if deg <= 0: continue
            pool = [j for j in range(nN) if j != i]
            sel = rng.choice(pool, size=min(deg, len(pool)), replace=False)
            Br[i, sel] = 1.0
        Br = Br / (Br.sum(axis=1, keepdims=True) + 1e-9)
        d_neg = []
        for q in queries:
            qv = embed([q])[0]
            ws, zs = weighted_mp(E, qv, Br, H=args.H, gamma=args.gamma)
            alphas = np.array([args.gamma**h for h in range(len(zs))], dtype=float)
            alphas = alphas / (alphas.sum() + 1e-9)
            z_ins = l2norm((np.vstack(zs).T @ alphas))
            av = embed([records[queries.index(q)]["answer"]])[0] if len(records) == len(queries) else embed([q])[0]
            if args.residual_ans:
                av = l2norm(av - qv)
            s = cosine(z_ins, av); s0 = cosine(l2norm(zs[0]), av)
            d_neg.append(s - s0)
        d_neg = np.array(d_neg)
        neg_stats["degree_preserve"] = {
            "n": int(len(d_neg)), "mean_delta": float(d_neg.mean()) if len(d_neg) else 0.0,
            "sign_test": _sign_test(d_neg), "cohens_d": _cohens_d(d_neg)
        }
        try:
            figD, axD = plt.subplots(figsize=(6.4, 4.2))
            bD = max(10, min(40, len(d_neg)//2))
            axD.hist(d_neg, bins=bD, alpha=0.8, color="#fdd0a2")
            axD.axvline(d_neg.mean(), color="orange", linestyle=":")
            axD.set_title("Negative: degree-preserve (row) Δs"); axD.set_xlabel("Δs"); axD.set_ylabel("count")
            figD.tight_layout(); figD.savefig(figdir / "rag_insight_alignment_neg_degpres.pdf")
        except Exception:
            pass

    # Save JSON summary
    outjson.write_text(
        json.dumps({"base": base_stats, "negatives": neg_stats, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ Wrote {figpath}")
    print(f"✅ Wrote {outjson}")


if __name__ == "__main__":
    sys.exit(main())
