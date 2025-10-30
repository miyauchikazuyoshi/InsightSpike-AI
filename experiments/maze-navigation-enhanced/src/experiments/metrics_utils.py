"""Common metrics utilities for maze evidence experiments.

Functions:
- loop_erased(path)
- compute_path_metrics(path, gedig_history=None, gedig_threshold=0.3, clip_min_k=5)
- compute_backtrack_rate(path)
- effect_size_and_ci(a, b, n_boot=10000, seed=42)  -> diff & Cohen's d with bootstrap CIs
- roc_auc(labels, scores)
- label_phases(path, maze) -> phase labels (terminal / branch_entry / deepening / retreat)
- auc_suite(gedig, labels) -> {auc_raw, auc_neg_delta, auc_best_lin, best_a, best_b}

Redundancy metrics:
    loop_redundancy = path_len / loop_erased_len
    clipped_redundancy = path_len / max(loop_erased_len, clip_min_k)
    normalized_over_budget = (path_len - loop_erased_len)/path_len

Backtrack: a step returning to the cell visited two steps earlier (immediate reversal).
Phase labeling (maze-based degrees):
    terminal: degree==1 and not start
    branch_entry: degree>=3 and new unexplored branch (heuristic: first time at node with degree>=3)
    retreat: immediate reversal
    deepening: otherwise
"""
from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Any, Iterable, Tuple
import math, random
import statistics
import numpy as np

Coord = Tuple[int,int]

def loop_erased(path: Sequence[Coord]) -> List[Coord]:
    """Loop erase by retaining chronological first occurrence and removing cycles.
    Simpler alternative: maintain stack; when a node repeats, pop until removed.
    """
    pos_to_index = {}
    stack: List[Coord] = []
    for p in path:
        if p in pos_to_index:
            # remove cycle
            idx = pos_to_index[p]
            # purge indices >= idx
            for q in stack[idx+1:]:
                pos_to_index.pop(q, None)
            stack = stack[:idx+1]
        else:
            stack.append(p)
            pos_to_index[p] = len(stack)-1
    return stack

def compute_backtrack_rate(path: Sequence[Coord]) -> float:
    if len(path) < 3:
        return 0.0
    backtracks = 0
    for i in range(2, len(path)):
        if path[i] == path[i-2]:  # immediate reversal
            backtracks += 1
    return backtracks / (len(path) - 1)


def compute_path_metrics(path: Sequence[Coord], gedig_history: Optional[Sequence[float]] = None,
                         gedig_threshold: float = 0.3, clip_min_k: int = 5) -> Dict[str, Any]:
    path_len = len(path)
    le = loop_erased(path)
    le_len = len(le)
    redundancy = (path_len / le_len) if le_len > 0 else None
    clipped_redundancy = (path_len / max(le_len, clip_min_k)) if le_len > 0 else None
    normalized_over_budget = (path_len - le_len) / path_len if path_len > 0 and le_len > 0 else None
    unique_positions = len(set(path))
    unique_coverage = unique_positions / path_len if path_len > 0 else None
    backtrack_rate = compute_backtrack_rate(path)
    gedig_mean = None
    gedig_low_frac = None
    if gedig_history:
        arr = np.array(gedig_history, dtype=float)
        gedig_mean = float(arr.mean())
        gedig_low_frac = float((arr < gedig_threshold).sum() / len(arr)) if len(arr) else None
    return {
        'path_length': path_len,
        'loop_erased_length': le_len,
        'loop_redundancy': redundancy,
        'clipped_redundancy': clipped_redundancy,
        'normalized_over_budget': normalized_over_budget,
        'unique_positions': unique_positions,
        'unique_coverage': unique_coverage,
        'backtrack_rate': backtrack_rate,
        'mean_geDIG': gedig_mean,
        'geDIG_low_frac': gedig_low_frac,
    }

# --- Statistics helpers ---

def effect_size_and_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Compute Cohen's d and bootstrap CIs for mean difference & d.

    Returns:
        diff_mean, diff_ci_low/high: bootstrap percentile CI of mean difference
        cohens_d, d_ci_low/high: bootstrap percentile CI of Cohen's d
    """
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return {'d': None, 'mean_a': None, 'mean_b': None}
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.pvariance(a)
    var_b = statistics.pvariance(b)
    spooled = math.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2)) if (len(a)+len(b)-2) > 0 else float('nan')
    d = (mean_a - mean_b)/spooled if spooled > 0 else float('nan')
    # Bootstrap CI for mean difference
    rng = random.Random(seed)
    diffs = []
    d_samples = []
    B = min(n_boot, 3000)
    for _ in range(B):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        mra = statistics.fmean(ra); mrb = statistics.fmean(rb)
        diffs.append(mra - mrb)
        var_ra = statistics.pvariance(ra); var_rb = statistics.pvariance(rb)
        sp = math.sqrt(((len(ra)-1)*var_ra + (len(rb)-1)*var_rb) / max((len(ra)+len(rb)-2),1))
        d_samples.append((mra - mrb)/sp if sp>0 else 0.0)
    diffs.sort(); d_samples.sort()
    def pct(arr, q): return arr[int(q*len(arr)) if int(q*len(arr)) < len(arr) else -1]
    lo = pct(diffs, 0.025); hi = pct(diffs, 0.975)
    d_lo = pct(d_samples, 0.025); d_hi = pct(d_samples, 0.975)
    return {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'diff_mean': mean_a - mean_b,
        'diff_ci_low': lo,
        'diff_ci_high': hi,
        'cohens_d': d,
        'd_ci_low': d_lo,
        'd_ci_high': d_hi,
        'n_a': len(a),
        'n_b': len(b)
    }

# --- ROC/AUC ---

def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(s, l) for s,l in zip(scores, labels) if l in (0,1) and s is not None]
    if not pairs:
        return None
    # Sort descending by score
    pairs.sort(key=lambda x: x[0], reverse=True)
    P = sum(1 for _,l in pairs if l==1)
    N = sum(1 for _,l in pairs if l==0)
    if P==0 or N==0:
        return None
    tp=0; fp=0; prev_s=None
    auc=0.0; prev_tp_rate=0.0; prev_fp_rate=0.0
    for s,l in pairs:
        if prev_s is None or s != prev_s:
            # trapezoid from prev point
            auc += (prev_fp_rate - fp/ N) * (prev_tp_rate + tp/ P) / 2.0 if prev_s is not None else 0.0
            prev_s = s
        if l==1: tp+=1
        else: fp+=1
        prev_tp_rate = tp / P
        prev_fp_rate = fp / N
    # final segment
    auc += (prev_fp_rate) * (prev_tp_rate) / 2.0
    return float(auc)

def label_phases(path: Sequence[Coord], maze: np.ndarray) -> List[str]:
    """Context-aware phase labeling for each step.

    Improvements over the initial heuristic:
      - Distinguish first-time junction entries vs later junction revisits.
      - Mark true dead-end (degree==1) only on first arrival as 'terminal' (positive label for AUC).
      - Treat immediate reversal (A,B,A) as 'backtrack'.
      - Differentiate corridor progression vs corridor revisit to avoid inflating positives.
      - Provide more granular revisit states that downstream analyses can optionally collapse.

    Returned labels (subset may appear depending on maze):
      terminal            : first arrival at a degree-1 cell (potential dead-end)
      backtrack           : immediate reversal (A,B,A pattern)
      junction_entry      : first arrival at degree>=3 node
      junction_revisit    : subsequent visit to degree>=3 node
      corridor_progress   : first-time or forward move in degree==2 corridor with at least one unvisited neighbor
      corridor_revisit    : revisiting degree==2 cell (all neighbors visited)
      dead_end_revisit    : revisiting a degree==1 cell (after marking terminal earlier)
      revisit             : generic revisit fallback
      isolated            : degree==0 (should be rare / start artifact)

    NOTE: AUC routines continue to treat only 'terminal' as positive, making the
    signal sparser and (empirically) improving discrimination.
    """
    if not path:
        return []
    h, w = maze.shape
    # Precompute degree for passable cells encountered (lazy evaluation)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def degree(cell: Coord) -> int:
        x,y = cell
        deg=0
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny,nx]==0:
                deg += 1
        return deg

    def unvisited_neighbor_count(cell: Coord, visited: set[Coord]) -> int:
        x,y = cell
        c=0
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny,nx]==0 and (nx,ny) not in visited:
                c+=1
        return c

    labels: List[str] = []
    visited: set[Coord] = set()
    terminal_marked: set[Coord] = set()  # record which dead-end cells already emitted 'terminal'

    for i, p in enumerate(path):
        d = degree(p)
        # Immediate reversal detection
        if i >= 2 and path[i] == path[i-2]:
            labels.append('backtrack')
        else:
            first_visit = p not in visited
            if d == 0:
                labels.append('isolated')
            elif d == 1:
                if first_visit and p not in terminal_marked and p != path[0]:
                    labels.append('terminal')  # positive event
                    terminal_marked.add(p)
                else:
                    labels.append('dead_end_revisit')
            elif d >= 3:
                if first_visit:
                    labels.append('junction_entry')
                else:
                    labels.append('junction_revisit')
            else:  # d == 2 corridor
                unv = unvisited_neighbor_count(p, visited)
                if first_visit or unv > 0:
                    labels.append('corridor_progress')
                else:
                    labels.append('corridor_revisit')
        visited.add(p)
    return labels

def auc_suite(gedig_history: Sequence[float], phases: Sequence[str]) -> Dict[str, Any]:
    """Compute AUC variants for terminal detection given geDIG history.
    Aligns length to min(len(gedig_history), len(phases))."""
    L = min(len(gedig_history), len(phases))
    if L == 0:
        return {}
    g = np.array(gedig_history[:L])
    labels = [1 if ph=='terminal' else 0 for ph in phases[:L]]
    # Raw AUC
    auc_raw = roc_auc(labels, g)
    # Negative delta
    dg = np.diff(g, prepend=g[0]) * -1.0
    auc_neg = roc_auc(labels, dg)
    # Linear combo grid search a*g + b*dg
    best_auc = None; best_a=0; best_b=0
    for a in [0.0,0.25,0.5,0.75,1.0]:
        for b in [0.0,0.25,0.5,0.75,1.0]:
            combo = a*g + b*dg
            auc_c = roc_auc(labels, combo)
            if auc_c is not None and (best_auc is None or auc_c > best_auc):
                best_auc = auc_c; best_a=a; best_b=b
    return {
        'auc_raw': auc_raw,
        'auc_neg_delta': auc_neg,
        'auc_best_linear': best_auc,
        'best_a': best_a,
        'best_b': best_b
    }

__all__ = [
    'compute_path_metrics','compute_backtrack_rate','loop_erased','effect_size_and_ci','roc_auc',
    'label_phases','auc_suite'
]
