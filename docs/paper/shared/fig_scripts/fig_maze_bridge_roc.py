#!/usr/bin/env python3
"""
Fig. M‑ROC: Bridge Detection ROC (A2)

Expected input CSV (default: data/maze_eval/bridge_scores.csv):
  columns: run_id, step, aggregator, score, y_true
    - aggregator ∈ {min, softmin:τ, sum}（例: 'min', 'softmin:0.5', 'sum'）
    - score: lower is better (negative spike more confident) → invert if needed
    - y_true: 1 if "true shortcut" (ΔSPL ≤ -ε), else 0

If the CSV is missing, the script synthesizes separable distributions to render
placeholder ROC curves and AUC.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        req = {"aggregator", "score", "y_true"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns in {path}: {miss}")
        return df
    # Synthetic placeholder
    rng = np.random.default_rng(123)
    n = 2000
    y = rng.integers(0, 2, size=n, endpoint=False)
    # Better separation for 'min', worse for 'sum', medium for softmin
    scores_min = (rng.normal(-1.5, 0.8, size=n) * (y == 1) + rng.normal(-0.1, 0.8, size=n) * (y == 0))
    scores_soft = (rng.normal(-1.0, 0.9, size=n) * (y == 1) + rng.normal(0.0, 0.9, size=n) * (y == 0))
    scores_sum = (rng.normal(-0.6, 1.1, size=n) * (y == 1) + rng.normal(0.2, 1.1, size=n) * (y == 0))
    df = pd.DataFrame({
        "aggregator": np.repeat(["min", "softmin:0.5", "sum"], repeats=n),
        "score": np.concatenate([scores_min, scores_soft, scores_sum]),
        "y_true": np.tile(y, 3),
    })
    return df


def roc_xy(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC (FPR, TPR) arrays and AUC; lower scores = more positive.
    """
    # Sort by score ascending (more negative → more confident positive)
    order = np.argsort(score)
    y = y_true[order]
    # Unique thresholds
    thresholds = np.r_[[-np.inf], (score[order][::50]), [np.inf]]  # subsample for speed
    P = y.sum()
    N = len(y) - P
    tprs, fprs = [], []
    for thr in thresholds:
        pred = (score <= thr)
        TP = (pred & (y == 1)).sum()
        FP = (pred & (y == 0)).sum()
        TPR = TP / P if P else 0.0
        FPR = FP / N if N else 0.0
        tprs.append(TPR)
        fprs.append(FPR)
    # Close curve (0,0) to (1,1)
    f = np.array([0.0, *fprs, 1.0])
    t = np.array([0.0, *tprs, 1.0])
    # AUC via trapezoid (np.trapz is deprecated in newer NumPy)
    try:
        auc = np.trapezoid(t, f)  # type: ignore[attr-defined]
    except Exception:
        auc = np.trapz(t, f)
    return f, t, float(auc)


def pr_xy(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve and AUPR; lower scores = more positive.
    """
    order = np.argsort(score)
    y = y_true[order]
    # thresholds along sorted scores
    thresholds = score[order][::50]
    ps, rs = [], []
    P = y.sum()
    for thr in thresholds:
        pred = (score <= thr)
        TP = (pred & (y_true == 1)).sum()
        FP = (pred & (y_true == 0)).sum()
        FN = P - TP
        prec = TP / (TP + FP) if (TP + FP) else 1.0
        rec = TP / P if P else 0.0
        ps.append(prec)
        rs.append(rec)
    # sort by recall
    rs = np.array(rs)
    ps = np.array(ps)
    idx = np.argsort(rs)
    rs = rs[idx]
    ps = ps[idx]
    # area under PR using trapezoid on recall axis
    try:
        aupr = np.trapezoid(ps, rs)  # type: ignore[attr-defined]
    except Exception:
        aupr = np.trapz(ps, rs)
    return rs, ps, float(aupr)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    in_path = Path(os.environ.get("BRIDGE_SCORES", repo_root / "data/maze_eval/bridge_scores.csv"))
    df = load_data(in_path)
    outdir = repo_root / "docs/paper/figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10.8, 4.8))
    aucs = []
    for agg, sub in df.groupby("aggregator"):
        y = sub["y_true"].to_numpy().astype(int)
        s = sub["score"].to_numpy().astype(float)
        fpr, tpr, auc = roc_xy(y, s)
        # partial AUC for low-FPR operating region
        mask_low = fpr <= 0.1
        try:
            pauc = np.trapezoid(tpr[mask_low], fpr[mask_low])  # type: ignore[attr-defined]
        except Exception:
            pauc = np.trapz(tpr[mask_low], fpr[mask_low])
        ax_roc.plot(fpr, tpr, lw=2, label=f"{agg} (AUC={auc:.3f}, pAUC@0.1={pauc:.3f})")
        aucs.append((agg, auc, pauc))
        # PR curve
        recall, prec, aupr = pr_xy(y, s)
        ax_pr.plot(recall, prec, lw=2, label=f"{agg} (AUPR={aupr:.3f})")
    # ROC panel formatting
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.axvspan(0, 0.1, color='orange', alpha=0.08, label='operating region (FPR≤0.1)')
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("FPR (False Positive Rate)")
    ax_roc.set_ylabel("TPR (True Positive Rate)")
    ax_roc.set_title("Bridge Detection ROC (M‑ROC)")
    ax_roc.legend(loc="lower right", frameon=False)
    ax_roc.grid(alpha=0.3)
    # PR panel formatting
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall (low‑FPR sensitivity)")
    ax_pr.legend(loc="lower left", frameon=False)
    ax_pr.grid(alpha=0.3)
    out = outdir / "fig_m_roc.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
