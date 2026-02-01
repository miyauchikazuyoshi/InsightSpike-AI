"""Query-hub specific utility functions and constants.

Contains:
- Node/direction conversion helpers
- Feature vector computation
- Distance metrics
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from insightspike.environments.maze import MazeObservation

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

QUERY_MARKER = -1  # third coordinate for query nodes
QUERY_LABEL = "Q"
DIR_TO_DELTA = {
    0: (-1, 0),  # north
    1: (0, 1),   # east
    2: (1, 0),   # south
    3: (0, -1),  # west
}
DELTA_TO_DIR = {delta: direction for direction, delta in DIR_TO_DELTA.items()}
DIR_LABELS = {0: "N", 1: "E", 2: "S", 3: "W", QUERY_MARKER: QUERY_LABEL}

WEIGHT_VECTOR = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0], dtype=float)
QUERY_TEMPERATURE = 0.1
RADIUS_BLOCK = 1e6


# --------------------------------------------------------------------------------------
# Direction/node conversion helpers
# --------------------------------------------------------------------------------------

def direction_from_delta(delta: Tuple[int, int]) -> Optional[int]:
    if delta is None:
        return None
    dr, dc = int(delta[0]), int(delta[1])
    return DELTA_TO_DIR.get((dr, dc))


def delta_from_direction(direction: int) -> Tuple[int, int]:
    if direction in DIR_TO_DELTA:
        return DIR_TO_DELTA[direction]
    return (0, 0)


def make_query_node(position: Tuple[int, int]) -> Tuple[int, int, int]:
    """Return canonical query node id for the given position."""

    return (int(position[0]), int(position[1]), QUERY_MARKER)


def make_direction_node(anchor: Tuple[int, int], direction: int) -> Tuple[int, int, int]:
    return (int(anchor[0]), int(anchor[1]), int(direction))


def canonical_node_id(node: Any) -> Tuple[int, int, int]:
    if isinstance(node, (list, tuple)):
        if len(node) == 3:
            return (int(node[0]), int(node[1]), int(node[2]))
        if len(node) == 2:
            return (int(node[0]), int(node[1]), QUERY_MARKER)
    if hasattr(node, "tolist"):
        seq = list(node.tolist())
        return canonical_node_id(seq)
    return (int(node), 0, QUERY_MARKER)


def canonical_edge_id(a: Any, b: Any) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    na = canonical_node_id(a)
    nb = canonical_node_id(b)
    return tuple(sorted([na, nb]))


# --------------------------------------------------------------------------------------
# Feature vector computation
# --------------------------------------------------------------------------------------

def compute_episode_vector(
    base_position: Tuple[int, int],
    maze_shape: Tuple[int, int],
    action_delta: Tuple[int, int] | None,
    *,
    is_passable: bool,
    visits: int,
    success: bool,
    is_goal: bool,
    target_position: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Return the 8D feature vector used across the maze experiments."""

    row, col = base_position
    height, width = maze_shape
    dr, dc = action_delta if action_delta is not None else (0, 0)
    if action_delta is None and target_position is not None:
        dr = target_position[0] - base_position[0]
        dc = target_position[1] - base_position[1]
    dx = float(dc)
    dy = float(-dr)

    vector = np.zeros(8, dtype=float)
    vector[0] = row / max(height, 1)
    vector[1] = col / max(width, 1)
    vector[2] = dx
    vector[3] = dy
    vector[4] = 1.0 if is_passable else -1.0
    vector[5] = math.log1p(max(0, visits))
    vector[6] = 1.0 if success else 0.0
    vector[7] = 1.0 if is_goal else 0.0
    return vector


def compute_query_vector(position: Tuple[int, int], maze_shape: Tuple[int, int]) -> np.ndarray:
    vector = np.zeros(8, dtype=float)
    row, col = position
    height, width = maze_shape
    vector[0] = row / max(height, 1)
    vector[1] = col / max(width, 1)
    vector[4] = 1.0
    return vector


def weighted_distance(query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
    diff = WEIGHT_VECTOR * (query_vec - candidate_vec)
    return float(np.linalg.norm(diff))


def gather_node_features(graph: nx.Graph, default_dim: int = 8) -> np.ndarray:
    """Collect node feature vectors for geDIG IG calculation."""

    features: List[np.ndarray] = []
    for node in graph.nodes():
        data = graph.nodes[node]
        vec = data.get("abs_vector")
        if vec is None:
            vec = data.get("vector")
        arr = np.asarray(vec, dtype=np.float32) if vec is not None else None
        if arr is None or arr.size == 0:
            arr = np.zeros(default_dim, dtype=np.float32)
        else:
            arr = arr.flatten()
            if arr.size < default_dim:
                arr = np.pad(arr, (0, default_dim - arr.size))
            elif arr.size > default_dim:
                arr = arr[:default_dim]
        features.append(arr.astype(np.float32))
    if not features:
        return np.zeros((0, default_dim), dtype=np.float32)
    return np.vstack(features)


def build_feature_matrix(
    graph: nx.Graph,
    candidate_nodes: Set[Tuple[int, int, int]],
    query_node: Tuple[int, int, int],
    *,
    default_dim: int = 8,
    zero_candidates: bool = False,
) -> np.ndarray:
    """Build feature matrix focusing on candidate nodes around the query."""

    features: List[np.ndarray] = []
    for node in graph.nodes():
        data = graph.nodes[node]
        vec = data.get("abs_vector")
        if vec is None:
            vec = data.get("vector")
        arr = np.asarray(vec, dtype=np.float32) if vec is not None else None
        if arr is None or arr.size == 0:
            arr = np.zeros(default_dim, dtype=np.float32)
        else:
            arr = arr.flatten()
            if arr.size < default_dim:
                arr = np.pad(arr, (0, default_dim - arr.size))
            elif arr.size > default_dim:
                arr = arr[:default_dim]
        if node == query_node:
            pass  # keep query vector as-is
        elif node in candidate_nodes:
            if zero_candidates:
                arr = np.zeros(default_dim, dtype=np.float32)
        else:
            arr = np.zeros(default_dim, dtype=np.float32)
        features.append(arr.astype(np.float32))
    if not features:
        return np.zeros((0, default_dim), dtype=np.float32)
    return np.vstack(features)


def encode_observation(obs: MazeObservation) -> np.ndarray:
    return np.array(
        [
            obs.cell_type.value,
            obs.num_paths / 4.0,
            1.0 if obs.is_goal else 0.0,
            1.0 if obs.hit_wall else 0.0,
            1.0 if obs.is_dead_end else 0.0,
            1.0 if obs.is_junction else 0.0,
        ],
        dtype=float,
    )


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --------------------------------------------------------------------------------------
# Snapshot / serialization helpers
# --------------------------------------------------------------------------------------

def candidate_index(anchor: Tuple[int, int], dir_idx: int) -> str:
    """Create a unique string index for a candidate (anchor position + direction)."""
    return f"{anchor[0]},{anchor[1]},{dir_idx}"


def edge_set(g) -> Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Extract a set of normalized edges from a graph."""
    return set(
        tuple(sorted((
            (int(a[0]), int(a[1]), int(a[2])),
            (int(b[0]), int(b[1]), int(b[2]))
        )))
        for a, b in g.edges()
    )


def norm_edge(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Normalize an edge to a canonical (sorted) form."""
    ua = (int(a[0]), int(a[1]), int(a[2]))
    vb = (int(b[0]), int(b[1]), int(b[2]))
    return tuple(sorted((ua, vb)))


def normalise_container(
    items: Any,
    direction_from_delta_fn: Optional[Any] = None,
    dir_labels: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Normalize a container of candidate items for JSON serialization.

    Args:
        items: Iterable of dictionaries to normalize
        direction_from_delta_fn: Function to derive direction from delta (optional)
        dir_labels: Direction label mapping (optional)

    Returns:
        List of normalized dictionaries
    """
    if direction_from_delta_fn is None:
        direction_from_delta_fn = direction_from_delta
    if dir_labels is None:
        dir_labels = DIR_LABELS

    normalised: List[Dict[str, Any]] = []
    for entry in items:
        item = dict(entry)
        for key in ("vector", "abs_vector"):
            if isinstance(item.get(key), np.ndarray):
                item[key] = item[key].tolist()
        for key in ("anchor_position", "target_position", "relative_delta"):
            value = item.get(key)
            if value is not None and hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, list) and len(value) == 2:
                item[key] = [int(value[0]), int(value[1])]
        dir_idx = item.get("direction")
        if dir_idx is None and item.get("relative_delta") is not None:
            rd = item["relative_delta"]
            if isinstance(rd, (list, tuple)) and len(rd) == 2:
                derived = direction_from_delta_fn((int(rd[0]), int(rd[1])))
                if derived is not None:
                    item["direction"] = int(derived)
                    dir_idx = int(derived)
        if isinstance(dir_idx, (int, np.integer)):
            item["direction_label"] = dir_labels.get(int(dir_idx), "?")
        if "meta_visits" in item and isinstance(item["meta_visits"], (int, float)):
            item["visit"] = int(item["meta_visits"])
        elif "visit_count" in item and isinstance(item["visit_count"], (int, float)):
            item["visit"] = int(item["visit_count"])
        normalised.append(item)
    return normalised


def item_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """Extract a unique key from a candidate item for deduplication."""
    target = item.get("target_position") or item.get("targetPosition") or []
    if isinstance(target, list):
        target = tuple(int(v) for v in target)
    return (
        item.get("origin"),
        item.get("index"),
        item.get("direction"),
        target,
    )


def merge_unique(items: Any) -> List[Dict[str, Any]]:
    """Merge items, keeping unique entries based on item_key."""
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for raw in items:
        item = dict(raw)
        key = item_key(item)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
        else:
            if item.get("forced"):
                existing["forced"] = True
            if "visit" in item and "visit" not in existing:
                existing["visit"] = item["visit"]
            if "direction_label" in item and "direction_label" not in existing:
                existing["direction_label"] = item["direction_label"]
    return list(merged.values())
