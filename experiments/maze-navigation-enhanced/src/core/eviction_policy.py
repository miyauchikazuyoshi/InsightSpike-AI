"""Eviction policy abstraction (Phase3 of geDIG refactor plan).

Provides a pluggable interface so that maze navigator memory guard can
swap strategies without modifying core orchestration logic.

Policies implemented:
  - HeuristicPolicy (current blended score: recency, inverse visits, distance)
  - LRUVisitPolicy (pure LRU with visit-count tie‑breaker)

Future extension candidates (not implemented yet):
  - DistanceWeightedPolicy
  - ImportanceBasedPolicy (using geDIG / structural improvement deltas)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Protocol, Any, Sequence
import math

try:
    # Episode type used inside experiments (light duck typing sufficient)
    from core.episode_manager import Episode  # type: ignore
except Exception:  # pragma: no cover - fallback for type checking in isolation
    class Episode:  # type: ignore
        episode_id: int
        timestamp: float
        visit_count: int
        position: tuple[int,int]

class IEvictionPolicy(Protocol):
    name: str
    def select(self, episodes: Sequence[Episode], over_by: int, *, context: dict[str, Any]) -> List[int]:
        """Return list of episode_ids to evict (length == over_by or fewer if insufficient).

        Implementations should avoid mutating the episodes.
        """

@dataclass
class HeuristicPolicy:
    name: str = "heuristic"
    recency_weight: float = 0.6
    inverse_visit_weight: float = 1.2
    distance_weight: float = 0.2

    def select(self, episodes: Sequence[Episode], over_by: int, *, context: dict[str, Any]) -> List[int]:
        if over_by <= 0 or not episodes:
            return []
        # Rank episodes by timestamp (older = smaller rank)
        timestamps = sorted(((ep.timestamp, ep.episode_id) for ep in episodes), key=lambda x: x[0])
        rank_map = {eid: i for i, (_ts, eid) in enumerate(timestamps)}
        cx, cy = context.get("current_pos", (0,0))
        scored = []
        for ep in episodes:
            rec_rank = rank_map.get(ep.episode_id, 0)
            inv_visit = 1.0 / (1.0 + max(0, getattr(ep, 'visit_count', 0)))
            ex, ey = getattr(ep, 'position', (0,0))
            dist = abs(ex - cx) + abs(ey - cy)
            score = (self.recency_weight * rec_rank +
                     self.inverse_visit_weight * inv_visit +
                     self.distance_weight * dist)
            scored.append((score, ep.episode_id))
        scored.sort(key=lambda x: x[0])
        return [eid for _s, eid in scored[:over_by]]

@dataclass
class LRUVisitPolicy:
    name: str = "lru_visit"
    visit_bias: float = 0.05  # small bias to prefer evicting low-visit episodes among similar age

    def select(self, episodes: Sequence[Episode], over_by: int, *, context: dict[str, Any]) -> List[int]:
        if over_by <= 0 or not episodes:
            return []
        scored = []
        for ep in episodes:
            age = getattr(ep, 'timestamp', 0.0)
            visits = getattr(ep, 'visit_count', 0)
            # Older first (timestamp ascending). We store score with slight visit penalty.
            score = age + self.visit_bias * visits
            scored.append((score, ep.episode_id))
        # We want oldest (lowest timestamp) first.
        scored.sort(key=lambda x: x[0])
        return [eid for _s, eid in scored[:over_by]]

# Registry helper
_POLICIES: dict[str, IEvictionPolicy] = {
    "heuristic": HeuristicPolicy(),
    "lru": LRUVisitPolicy(),
    "lru_visit": LRUVisitPolicy(),
}

def get_policy(name: str | None) -> IEvictionPolicy:
    if not name:
        return _POLICIES["heuristic"]
    key = name.lower()
    return _POLICIES.get(key, _POLICIES["heuristic"])
