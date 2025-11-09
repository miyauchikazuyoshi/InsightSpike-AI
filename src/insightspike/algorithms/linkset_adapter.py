from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def build_linkset_info(
    *,
    s_link: Optional[Sequence[Dict[str, Any]]] = None,
    candidate_pool: Optional[Sequence[Dict[str, Any]]] = None,
    decision: Optional[Dict[str, Any]] = None,
    query_vector: Optional[Sequence[float]] = None,
    base_mode: str = "link",
) -> Dict[str, Any]:
    """Construct a linkset_info payload from selection data.

    - s_link: sequence of dicts with at least {'index', 'similarity'}
    - candidate_pool: full candidate pool entries (optional)
    - decision: {'index', 'similarity', 'origin', ...}
    - query_vector: optional numeric vector for the query entry
    - base_mode: 'link'|'mem'|'pool' (choose baseline set)
    """
    base_mode = str(base_mode or "link").lower()
    s_link = list(s_link or [])
    candidate_pool = list(candidate_pool or [])
    decision = dict(decision or {})
    qsim = decision.get("similarity")
    try:
        qsim = float(qsim) if isinstance(qsim, (int, float)) else 1.0
        if qsim <= 0:
            qsim = 1.0
    except Exception:
        qsim = 1.0
    q_entry: Dict[str, Any] = {
        "index": "query",
        "origin": "query",
        "similarity": float(qsim),
        "distance": 0.0,
        "weighted_distance": 0.0,
    }
    if query_vector is not None:
        try:
            q_entry["vector"] = list(query_vector)
            q_entry["abs_vector"] = list(query_vector)
        except Exception:
            pass
    info: Dict[str, Any] = {
        "s_link": [dict(it) for it in s_link],
        "candidate_pool": [dict(it) for it in candidate_pool],
        "decision": {
            "origin": decision.get("origin"),
            "index": decision.get("index"),
            "action": decision.get("action"),
            "distance": decision.get("distance"),
            "similarity": decision.get("similarity"),
        },
        "query_entry": q_entry,
        "base_mode": base_mode,
    }
    return info

