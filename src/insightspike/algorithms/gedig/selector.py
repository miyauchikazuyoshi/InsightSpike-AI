from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable


def _extract_score(item: Dict[str, Any], key: str = "similarity") -> float:
    try:
        val = item.get(key, 0.0)
        return float(val)
    except Exception:
        return 0.0


@dataclass
class TwoThresholdSelection:
    """Result container for two-threshold candidate selection."""

    candidates: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    forced_links: List[Dict[str, Any]]
    k_star: int
    theta_cand: float
    theta_link: float
    k_cap: int
    top_m: Optional[int] = None

    def to_summary(self, score_key: str = "similarity", max_samples: int = 5) -> Dict[str, Any]:
        """Return a lightweight summary suitable for telemetry/context passing."""

        def _indices(items: Sequence[Dict[str, Any]]) -> List[Any]:
            out: List[Any] = []
            for item in items:
                if isinstance(item, dict) and "index" in item:
                    out.append(item["index"])
                else:
                    out.append(None)
            return out

        sample_scores = [
            _extract_score(item, score_key)
            for item in self.candidates[: max(0, max_samples)]
        ]
        forced_indices = []
        for item in self.forced_links:
            if isinstance(item, dict) and "index" in item:
                forced_indices.append(item["index"])
            else:
                forced_indices.append(None)
        return {
            "mode": "two_threshold",
            "theta_cand": self.theta_cand,
            "theta_link": self.theta_link,
            "k_cap": self.k_cap,
            "top_m": self.top_m,
            "candidate_count": len(self.candidates),
            "link_count": len(self.links),
            "k_star": self.k_star,
            "l1_candidates": min(self.k_star, len(self.candidates)),
            "candidate_indices": _indices(self.candidates[: self.k_cap]),
            "link_indices": _indices(self.links),
            "forced_link_indices": forced_indices,
            "top_scores": sample_scores,
            "log_k_star": float(_safe_log(self.k_star)) if self.k_star >= 1 else None,
        }


def _safe_log(value: int) -> float:
    import math

    if value <= 0:
        return 0.0
    try:
        return math.log(float(value))
    except Exception:
        return 0.0


class TwoThresholdCandidateSelector:
    """Selects candidates using θ_cand and θ_link thresholds.

    θ_cand > θ_link を前提とし、候補集合 (S_cand) と接続コミット集合 (S_link)
    を返す。S_link は常に S_cand の部分集合に丸め込み、|S_cand| の上限を
    k_cap で制限する。結果の k⋆ は min(|S_cand|, k_cap) で定義される。
    """

    def __init__(
        self,
        theta_cand: float,
        theta_link: float,
        k_cap: int,
        *,
        top_m: Optional[int] = None,
        score_key: str = "similarity",
        radius_cand: Optional[float] = None,
        radius_link: Optional[float] = None,
        higher_is_better: bool = True,
        prefilter_fn: Optional[Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]] = None,
    ) -> None:
        self.theta_cand = float(theta_cand)
        self.theta_link = float(theta_link)
        self.k_cap = max(1, int(k_cap))
        self.top_m = int(top_m) if top_m is not None else None
        self.score_key = score_key
        self.radius_cand = float(radius_cand) if radius_cand is not None else None
        self.radius_link = float(radius_link) if radius_link is not None else None
        self.higher_is_better = bool(higher_is_better)
        self.prefilter_fn = prefilter_fn

        # For distanceモードでは cand >= link を期待する
        if self.higher_is_better:
            if self.theta_cand < self.theta_link:
                self.theta_cand, self.theta_link = self.theta_link, self.theta_cand
        else:
            if self.theta_cand < self.theta_link:
                # r_cand は r_link 以上であるべきなので、足りない場合は交換
                self.theta_cand, self.theta_link = self.theta_link, self.theta_cand

    def select(self, items: Iterable[Dict[str, Any]]) -> TwoThresholdSelection:
        # Optional prefilter hook (no-op by default)
        try:
            if self.prefilter_fn is not None:
                items = self.prefilter_fn(items)
        except Exception:
            # Fail-safe: ignore prefilter errors
            pass
        ordered: List[Dict[str, Any]] = sorted(
            (dict(item) for item in items),
            key=lambda x: _extract_score(x, self.score_key),
            reverse=self.higher_is_better,
        )

        if self.top_m is not None and self.top_m > 0:
            ordered = ordered[: self.top_m]

        def _within_radius(item: Dict[str, Any], key: str, limit: Optional[float]) -> bool:
            if limit is None:
                return True
            value = item.get(key)
            if value is None:
                return True
            try:
                return float(value) <= float(limit)
            except Exception:
                return True

        candidates: List[Dict[str, Any]] = []
        raw_linkables: List[Dict[str, Any]] = []
        for item in ordered:
            if _within_radius(item, "radius_cand", self.radius_cand):
                score = _extract_score(item, self.score_key)
                if self.higher_is_better:
                    if score >= self.theta_cand:
                        candidates.append(item)
                else:
                    if score <= self.theta_cand:
                        candidates.append(item)
            if _within_radius(item, "radius_link", self.radius_link):
                score = _extract_score(item, self.score_key)
                if self.higher_is_better:
                    if score >= self.theta_link:
                        raw_linkables.append(item)
                else:
                    if score <= self.theta_link:
                        raw_linkables.append(item)

        linkables: List[Dict[str, Any]] = []

        candidate_ids = {self._item_identity(item): item for item in candidates}
        seen_candidates = set(candidate_ids.keys())

        for item in raw_linkables:
            ident = self._item_identity(item)
            if ident in candidate_ids:
                linkables.append(candidate_ids[ident])
                continue
            if item.get("origin") == "mem":
                if ident not in seen_candidates:
                    candidates.append(item)
                    candidate_ids[ident] = item
                    seen_candidates.add(ident)
                linkables.append(candidate_ids[ident])

        if not linkables:
            linkables = []

        fallback_pool = [
            dict(item)
            for item in ordered
            if _within_radius(item, "radius_cand", self.radius_cand)
        ]

        forced_links: List[Dict[str, Any]] = []
        if not raw_linkables and fallback_pool:
            best_link = fallback_pool[0]
            best_link.setdefault("forced", True)
            forced_links.append(best_link)

        k_star = min(len(candidates), self.k_cap)
        return TwoThresholdSelection(
            candidates=candidates,
            links=linkables,
            forced_links=forced_links,
            k_star=k_star,
            theta_cand=self.theta_cand,
            theta_link=self.theta_link,
            k_cap=self.k_cap,
            top_m=self.top_m,
        )

    @staticmethod
    def _item_identity(item: Dict[str, Any]) -> Tuple[Any, Any]:
        idx = item.get("index") if isinstance(item, dict) else None
        sim = _extract_score(item) if isinstance(item, dict) else 0.0
        return idx, sim


def compute_gedig(
    G_prev: Any,
    G_curr: Any,
    *,
    mode: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical geDIG entry.

    Returns a dict with keys: mode, gedig, ged, ig. For mode=='ab', returns
    an object with embedded 'pure' and 'full' results and 'ab.variant'.

    This selector stays light and side-effect free (no CSV/logging). Callers
    can compose with A/B loggers or monitors as needed.
    """
    m = (mode or "full").lower()
    if m not in ("pure", "full", "ab"):
        m = "full"

    if m == "pure":
        from ..gedig_pure import PureGeDIGCalculator  # local import to keep import-time light

        calc = PureGeDIGCalculator()
        res = calc.calculate(G_prev, G_curr)
        return {"mode": "pure", "gedig": float(res.gedig), "ged": float(res.ged), "ig": float(res.ig)}

    if m == "full":
        from ..gedig_core import GeDIGCore  # uses normalized GED + entropy IG orchestration
        try:
            from ..linkset_adapter import build_linkset_info as _build_ls  # type: ignore
        except Exception:
            _build_ls = None  # type: ignore

        core = GeDIGCore()
        _ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
        if _ls is not None:
            out = core.calculate(g_prev=G_prev, g_now=G_curr, linkset_info=_ls)
        else:
            out = core.calculate(g_prev=G_prev, g_now=G_curr)
        # Map to canonical shape
        return {"mode": "full", "gedig": float(out.gedig_value), "ged": float(out.ged_value), "ig": float(out.ig_value)}

    # m == 'ab'
    pure = compute_gedig(G_prev, G_curr, mode="pure")
    full = compute_gedig(G_prev, G_curr, mode="full")
    return {"mode": "ab", "ab": {"variant": (variant or "A")}, "pure": pure, "full": full, "gedig": float(full["gedig"]), "ged": float(full["ged"]), "ig": float(full["ig"]) }


__all__ = [
    "compute_gedig",
    "TwoThresholdCandidateSelector",
    "TwoThresholdSelection",
]
