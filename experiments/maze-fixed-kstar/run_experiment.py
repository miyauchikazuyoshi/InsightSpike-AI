"""Fixed-K★ geDIG maze experiment driver.

This script replays the paper's online maze study with the updated
geDIG formulation (fixed denominator log K★ and local Cmax). It relies
only on the main InsightSpike codebase (no legacy experiment modules).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import os

os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")
# Ensure IG non-negativity for maze (clip negative IG to 0)
os.environ.setdefault("MAZE_GEDIG_IG_NONNEG", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import networkx as nx
import numpy as np

from insightspike.algorithms.gedig.selector import TwoThresholdCandidateSelector
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.environments.maze import MazeObservation, SimpleMaze


@dataclass
class StepRecord:
    seed: int
    step: int
    position: Tuple[int, int]
    action: str
    candidate_selection: Dict[str, float]
    delta_ged: float
    delta_ig: float
    delta_ged_min: float
    delta_ig_min: float
    g0: float
    gmin: float
    best_hop: int
    is_dead_end: bool
    reward: float
    done: bool
    possible_moves: List[int]
    candidate_pool: List[Dict[str, Any]]
    selected_links: List[Dict[str, Any]]
    ranked_candidates: List[Dict[str, Any]]
    graph_nodes: List[List[int]]
    graph_edges: List[List[List[int]]]
    forced_edges: List[List[List[int]]]
    new_edge: List[List[int]]
    episode_vector: List[float]
    query_vector: List[float]


# Weight vector for similarity and IG, aligned with paper/README
# w = [1, 1, 0, 0, 3, 2, 0, 0]
WEIGHT_VECTOR = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0], dtype=float)
QUERY_TEMPERATURE = 0.1
RADIUS_BLOCK = 1e6

# Direction indices (matching SimpleMaze deltas)
DIR_TO_DELTA = {
    0: (-1, 0),  # north / up
    1: (0, 1),   # east / right
    2: (1, 0),   # south / down
    3: (0, -1),  # west / left
}
DELTA_TO_DIR = {v: k for k, v in DIR_TO_DELTA.items()}
CENTER_DIR = 4
# For UI: human-readable direction labels
DIR_LABELS = {0: "N", 1: "E", 2: "S", 3: "W", 4: "C"}


def direction_from_delta(delta: Tuple[int, int]) -> Optional[int]:
    dr, dc = int(delta[0]), int(delta[1])
    return DELTA_TO_DIR.get((dr, dc))


def delta_from_direction(direction: int) -> Tuple[int, int]:
    if direction in DIR_TO_DELTA:
        return DIR_TO_DELTA[direction]
    return (0, 0)


def make_center_node(position: Tuple[int, int]) -> Tuple[int, int, int]:
    return (int(position[0]), int(position[1]), CENTER_DIR)


def make_direction_node(anchor: Tuple[int, int], direction: int) -> Tuple[int, int, int]:
    return (int(anchor[0]), int(anchor[1]), int(direction))


def canonical_node_id(node: Any) -> Tuple[int, int, int]:
    if isinstance(node, (list, tuple)):
        if len(node) == 3:
            return (int(node[0]), int(node[1]), int(node[2]))
        if len(node) == 2:
            return (int(node[0]), int(node[1]), CENTER_DIR)
    if hasattr(node, "tolist"):
        seq = list(node.tolist())
        return canonical_node_id(seq)
    return (int(node), 0, CENTER_DIR)


def canonical_edge_id(a: Any, b: Any) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    na = canonical_node_id(a)
    nb = canonical_node_id(b)
    return tuple(sorted([na, nb]))


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
    """Return 8D episode vector described in the paper."""

    row, col = base_position
    height, width = maze_shape
    dr, dc = action_delta if action_delta is not None else (0, 0)
    if action_delta is None and target_position is not None:
        dr = target_position[0] - base_position[0]
        dc = target_position[1] - base_position[1]
    # Normalise so that positive dy means "north/up", positive dx means "east/right"
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
    """Direction-neutral query vector favouring open passages."""

    row, col = position
    height, width = maze_shape
    vector = np.zeros(8, dtype=float)
    vector[0] = row / max(height, 1)
    vector[1] = col / max(width, 1)
    vector[4] = 1.0  # prefer open passages
    return vector


def weighted_distance(query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
    """Weighted L2 distance according to the paper's diagonal weights."""

    diff = WEIGHT_VECTOR * (query_vec - candidate_vec)
    return float(np.linalg.norm(diff))


def encode_observation(obs: MazeObservation) -> np.ndarray:
    """Convert MazeObservation to a numeric feature vector."""

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


def run_episode(
    seed: int,
    maze_size: int,
    max_steps: int,
    maze_type: str,
    selector_params: Dict[str, float],
    gedig_params: Dict[str, float],
) -> Tuple[Dict[str, float], List[StepRecord], Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)

    env = SimpleMaze(size=(maze_size, maze_size), maze_type=maze_type)
    selector = TwoThresholdCandidateSelector(
        theta_cand=selector_params["theta_cand"],
        theta_link=selector_params["theta_link"],
        k_cap=int(selector_params["candidate_cap"]),
        top_m=int(selector_params["top_m"]),
        radius_cand=selector_params.get("cand_radius"),
        radius_link=selector_params.get("link_radius"),
        score_key="distance",
        higher_is_better=False,
    )
    core = GeDIGCore(
        enable_multihop=True,
        max_hops=int(gedig_params["max_hops"]),
        decay_factor=gedig_params["decay_factor"],
        lambda_weight=gedig_params["lambda_weight"],
        use_local_normalization=True,
        adaptive_hops=bool(gedig_params["adaptive_hops"]),
        feature_weights=WEIGHT_VECTOR,
    )

    graph = nx.Graph()
    maze_shape = (maze_size, maze_size)
    obs = env.reset()
    current_position = (int(obs.position[0]), int(obs.position[1]))
    visit_counts: Dict[Tuple[int, int], int] = {current_position: 1}
    prev_action_delta: Optional[Tuple[int, int]] = None
    prev_success = False

    anchor_position = (int(current_position[0]), int(current_position[1]))
    start_vec = compute_episode_vector(
        base_position=anchor_position,
        maze_shape=maze_shape,
        action_delta=prev_action_delta,
        is_passable=True,
        visits=visit_counts[current_position],
        success=prev_success,
        is_goal=obs.is_goal,
        target_position=anchor_position,
    )
    current_center_node = make_center_node(anchor_position)
    graph.add_node(
        current_center_node,
        abs_vector=start_vec,
        visit_count=visit_counts[current_position],
        last_action_delta=prev_action_delta or (0, 0),
        success=prev_success,
        is_goal=obs.is_goal,
        is_passable=True,
        anchor_positions=[[anchor_position[0], anchor_position[1]]],
        target_position=[anchor_position[0], anchor_position[1]],
        direction=CENTER_DIR,
    )

    step_records: List[StepRecord] = []
    visited = {current_position}
    success = False
    maze_snapshot = {
        "layout": env.grid.astype(int).tolist(),
        "start_pos": list(env.start_pos),
        "goal_pos": list(env.goal_pos),
        "size": [env.height, env.width],
        "maze_type": maze_type,
    }
    forced_edge_store: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

    def _parse_position(value: Any) -> Optional[Tuple[int, int]]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        if hasattr(value, "tolist"):
            seq = value.tolist()
            if isinstance(seq, (list, tuple)) and len(seq) == 2:
                return int(seq[0]), int(seq[1])
        return None

    for step in range(max_steps):
        possible_moves = obs.possible_moves
        if not possible_moves:
            break

        anchor_position = (int(current_position[0]), int(current_position[1]))
        node_entry = graph.nodes[current_center_node]
        current_vector = node_entry.get("abs_vector")
        if current_vector is None:
            current_vector = compute_episode_vector(
                base_position=anchor_position,
                maze_shape=maze_shape,
                action_delta=prev_action_delta,
                is_passable=True,
                visits=visit_counts.get(anchor_position, 0),
                success=prev_success,
                is_goal=obs.is_goal,
                target_position=anchor_position,
            )
            node_entry["abs_vector"] = current_vector

        query_vec = compute_query_vector(anchor_position, maze_shape)
        query_vec_list = query_vec.tolist()

        observation_candidates: List[Dict[str, Any]] = []
        memory_candidates: List[Dict[str, Any]] = []
        seen_positions: set[str] = set()
        possible_moves_set = set(possible_moves)

        anchor_center_node = current_center_node

        def register_observation_node(
            anchor_tuple: Tuple[int, int],
            dir_idx: int,
            delta: Tuple[int, int],
            candidate_vec_abs: np.ndarray,
            *,
            target_visits: int,
            is_passable: bool,
            is_goal: bool,
            target_pos: Tuple[int, int],
        ) -> None:
            node_id = make_direction_node(anchor_tuple, dir_idx)
            if node_id not in graph:
                graph.add_node(node_id)
            node_entry = graph.nodes[node_id]
            node_entry["abs_vector"] = np.asarray(candidate_vec_abs, dtype=float)
            node_entry["visit_count"] = int(target_visits)
            node_entry["last_action_delta"] = (int(delta[0]), int(delta[1]))
            node_entry["success"] = False
            node_entry["is_goal"] = bool(is_goal)
            node_entry["is_passable"] = bool(is_passable)
            anchor_list = node_entry.get("anchor_positions")
            anchor_entry = [int(anchor_tuple[0]), int(anchor_tuple[1])]
            if anchor_list is None:
                node_entry["anchor_positions"] = [anchor_entry]
            elif anchor_entry not in anchor_list:
                anchor_list.append(anchor_entry)
            node_entry["target_position"] = [int(target_pos[0]), int(target_pos[1])]
            node_entry["relative_delta"] = [int(delta[0]), int(delta[1])]
            node_entry["direction"] = int(dir_idx)

        for action in SimpleMaze.ACTIONS.keys():
            delta = SimpleMaze.ACTIONS[action]
            dir_idx = direction_from_delta(delta)
            if dir_idx is None:
                continue
            next_pos = (anchor_position[0] + delta[0], anchor_position[1] + delta[1])
            pos_key = f"{anchor_position[0]},{anchor_position[1]},{dir_idx}"
            target_visits = visit_counts.get(next_pos, 0)
            is_passable = action in possible_moves_set and not env._is_wall(next_pos)
            candidate_vec_abs = compute_episode_vector(
                base_position=next_pos,
                maze_shape=maze_shape,
                action_delta=delta,
                is_passable=is_passable,
                visits=target_visits,
                success=False,
                is_goal=(next_pos == env.goal_pos),
                target_position=next_pos,
            )
            candidate_vec_abs = np.asarray(candidate_vec_abs, dtype=float)
            candidate_vec_rel = compute_episode_vector(
                base_position=anchor_position,
                maze_shape=maze_shape,
                action_delta=None,
                is_passable=is_passable,
                visits=target_visits,
                success=False,
                is_goal=(next_pos == env.goal_pos),
                target_position=next_pos,
            )
            candidate_vec_rel = np.asarray(candidate_vec_rel, dtype=float)
            w_distance_rel = weighted_distance(query_vec, candidate_vec_rel)
            w_distance_abs = weighted_distance(query_vec, candidate_vec_abs)
            similarity = math.exp(-w_distance_rel / QUERY_TEMPERATURE)
            radius = w_distance_abs
            observation_candidates.append(
                {
                    "index": f"obs:{dir_idx}",
                    "action": int(action),
                    "action_label": SimpleMaze.ACTION_NAMES.get(action, str(action)),
                    "position": [anchor_position[0], anchor_position[1]],
                    "target_position": [next_pos[0], next_pos[1]],
                    "similarity": similarity,
                    "distance": float(w_distance_rel),
                    "weighted_distance": w_distance_rel,
                    "origin": "obs",
                    "pos_key": pos_key,
                    "radius_cand": radius,
                    "radius_link": radius,
                    "vector": candidate_vec_rel.tolist(),
                    "abs_vector": candidate_vec_abs.tolist(),
                    "passable": bool(is_passable),
                    "meta_delta": delta,
                    "meta_visits": target_visits,
                    "meta_success": False,
                    "meta_passable": bool(is_passable),
                    "anchor_position": [anchor_position[0], anchor_position[1]],
                    "relative_delta": [int(delta[0]), int(delta[1])],
                    "direction": dir_idx,
                }
            )
            register_observation_node(
                anchor_tuple=anchor_position,
                dir_idx=dir_idx,
                delta=delta,
                candidate_vec_abs=candidate_vec_abs,
                target_visits=target_visits,
                is_passable=is_passable,
                is_goal=(next_pos == env.goal_pos),
                target_pos=next_pos,
            )
            seen_positions.add(pos_key)

        for node in graph.nodes():
            nr, nc, ndir = canonical_node_id(node)
            if ndir == CENTER_DIR:
                continue
            anchor = (nr, nc)
            pos_key = f"{anchor[0]},{anchor[1]},{ndir}"
            if pos_key in seen_positions:
                continue
            node_data = graph.nodes[node]
            stored_vec_abs = node_data.get("abs_vector")
            if stored_vec_abs is None:
                continue
            stored_vec_abs = np.asarray(stored_vec_abs, dtype=float)
            stored_vec_rel = stored_vec_abs.copy()
            stored_visits = int(node_data.get("visit_count", 0))
            stored_success = bool(node_data.get("success", False))
            stored_passable = bool(node_data.get("is_passable", True))
            stored_anchors = node_data.get("anchor_positions") or []
            if isinstance(stored_anchors, tuple):
                stored_anchors = [list(stored_anchors)]
            if stored_anchors:
                anchor_list = list(stored_anchors[0])
            else:
                anchor_list = [anchor[0], anchor[1]]
            rel_delta_tuple = tuple(node_data.get("relative_delta") or delta_from_direction(ndir))
            target_position = node_data.get("target_position")
            if target_position is None:
                tpos = (anchor_list[0] + rel_delta_tuple[0], anchor_list[1] + rel_delta_tuple[1])
                target_position = [int(tpos[0]), int(tpos[1])]
            else:
                target_position = [int(target_position[0]), int(target_position[1])]
            w_distance_rel = weighted_distance(query_vec, stored_vec_rel)
            w_distance_abs = weighted_distance(query_vec, stored_vec_abs)
            similarity = math.exp(-w_distance_rel / QUERY_TEMPERATURE)
            radius = w_distance_abs
            memory_candidates.append(
                {
                    "index": f"mem:{pos_key}",
                    "position": [anchor_list[0], anchor_list[1]],
                    "target_position": target_position,
                    "similarity": similarity,
                    "distance": float(w_distance_rel),
                    "weighted_distance": w_distance_rel,
                    "origin": "mem",
                    "action": None,
                    "action_label": "memory",
                    "pos_key": pos_key,
                    "radius_cand": radius,
                    "radius_link": radius,
                    "vector": stored_vec_rel.tolist(),
                    "abs_vector": stored_vec_abs.tolist(),
                    "meta_delta": rel_delta_tuple,
                    "meta_visits": stored_visits,
                    "meta_success": stored_success,
                    "meta_passable": stored_passable,
                    "anchor_position": [anchor_list[0], anchor_list[1]],
                    "relative_delta": [int(rel_delta_tuple[0]), int(rel_delta_tuple[1])],
                    "direction": ndir,
                }
            )
            seen_positions.add(pos_key)

        candidates: List[Dict[str, Any]] = observation_candidates + memory_candidates
        ranked_all_candidates = sorted(
            (dict(item) for item in candidates),
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )

        selection = selector.select(candidates)
        forced_links = list(getattr(selection, "forced_links", []) or [])
        forced_count = len(forced_links)
        k_star = selection.k_star
        effective_link_count = len(selection.links) + forced_count
        if k_star < effective_link_count:
            k_star = effective_link_count
        if k_star < 1 and effective_link_count > 0:
            k_star = 1
        ig_fixed_den = math.log(k_star + 1.0) if k_star >= 1 else None
        l1_candidates = k_star if k_star >= 1 else None

        def choose_observation_candidate(collections: List[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
            for items in collections:
                if not items:
                    continue
                obs_items = [
                    item
                    for item in items
                    if item.get("origin") == "obs"
                    and item.get("action") in possible_moves_set
                ]
                if obs_items:
                    return max(obs_items, key=lambda entry: float(entry.get("similarity", 0.0)))
            return None

        chosen_obs = choose_observation_candidate(
            [selection.links, selection.candidates, ranked_all_candidates]
        )

        prev_graph = graph.copy()
        registered_link_positions: Set[Tuple[int, int, int]] = set()

        for link_item in list(selection.links) + forced_links:
            origin = link_item.get("origin")
            if origin not in ("obs", "mem"):
                continue
            anchor_source = link_item.get("anchor_position")
            if anchor_source is None:
                if origin == "obs":
                    anchor_source = [anchor_position[0], anchor_position[1]]
                else:
                    continue
            anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
            rel_delta = tuple(link_item.get("relative_delta") or link_item.get("meta_delta") or (0, 0))
            dir_idx = link_item.get("direction")
            if dir_idx is None:
                dir_idx = direction_from_delta(rel_delta)
            if dir_idx is None:
                continue
            node_id = make_direction_node(anchor_tuple, dir_idx)
            target_pos = link_item.get("target_position")
            if target_pos is None:
                target_tuple = (anchor_tuple[0] + rel_delta[0], anchor_tuple[1] + rel_delta[1])
            else:
                target_tuple = (int(target_pos[0]), int(target_pos[1]))
            target_pos = [int(target_tuple[0]), int(target_tuple[1])]
            if node_id not in graph:
                graph.add_node(node_id)
            node_entry = graph.nodes[node_id]
            anchor_center_node = make_center_node(anchor_tuple)
            if anchor_center_node not in graph:
                anchor_center_vec = compute_episode_vector(
                    base_position=anchor_tuple,
                    maze_shape=maze_shape,
                    action_delta=None,
                    is_passable=True,
                    visits=visit_counts.get(anchor_tuple, 0),
                    success=False,
                    is_goal=(anchor_tuple == env.goal_pos),
                    target_position=anchor_tuple,
                )
                graph.add_node(
                    anchor_center_node,
                    abs_vector=anchor_center_vec,
                    visit_count=visit_counts.get(anchor_tuple, 0),
                    last_action_delta=(0, 0),
                    success=False,
                    is_goal=(anchor_tuple == env.goal_pos),
                    is_passable=True,
                    anchor_positions=[[anchor_tuple[0], anchor_tuple[1]]],
                    target_position=[anchor_tuple[0], anchor_tuple[1]],
                    direction=CENTER_DIR,
                    relative_delta=[0, 0],
                )
            meta_delta = rel_delta
            meta_visits = int(link_item.get("meta_visits", visit_counts.get(target_tuple, 0)))
            meta_success = bool(link_item.get("meta_success", False))
            meta_passable = bool(link_item.get("meta_passable", True))
            link_vec_abs = link_item.get("abs_vector")
            if link_vec_abs is not None:
                node_entry["abs_vector"] = np.asarray(link_vec_abs, dtype=float)
            else:
                node_entry["abs_vector"] = compute_episode_vector(
                    base_position=anchor_tuple,
                    maze_shape=maze_shape,
                    action_delta=meta_delta,
                    is_passable=meta_passable,
                    visits=meta_visits,
                    success=meta_success,
                    is_goal=(tuple(target_pos) == env.goal_pos),
                    target_position=anchor_tuple,
                )
            node_entry["visit_count"] = meta_visits
            node_entry["last_action_delta"] = meta_delta
            node_entry["success"] = meta_success
            node_entry["is_goal"] = (target_tuple == env.goal_pos)
            node_entry["is_passable"] = meta_passable
            anchor_list = node_entry.get("anchor_positions")
            if anchor_list is None:
                node_entry["anchor_positions"] = [[anchor_tuple[0], anchor_tuple[1]]]
            else:
                anchor_entry = [anchor_tuple[0], anchor_tuple[1]]
                if anchor_entry not in anchor_list:
                    anchor_list.append(anchor_entry)
            node_entry["target_position"] = target_pos
            node_entry["relative_delta"] = [int(rel_delta[0]), int(rel_delta[1])]
            node_entry["direction"] = dir_idx
            if not graph.has_edge(anchor_center_node, node_id):
                graph.add_edge(anchor_center_node, node_id)
            if not graph.has_edge(current_center_node, node_id):
                graph.add_edge(current_center_node, node_id)
            registered_link_positions.add(node_id)
            if link_item.get("forced"):
                edge_key = canonical_edge_id(current_center_node, node_id)
                forced_edge_store.add(edge_key)

        if chosen_obs is not None and chosen_obs.get("action") is not None:
            action = int(chosen_obs["action"])
        else:
            action = random.choice(possible_moves)

        result = core.calculate(
            g_prev=prev_graph,
            g_now=graph,
            k_star=k_star if k_star >= 1 else None,
            l1_candidates=l1_candidates,
            ig_fixed_den=ig_fixed_den,
        )

        hop0 = result.hop_results.get(0) if result.hop_results else None
        g0 = hop0.gedig if hop0 else result.gedig_value
        gmin = result.gedig_value
        delta_ged = hop0.ged if hop0 else result.ged_value
        delta_ig = hop0.ig if hop0 else result.ig_value
        best_hop = (
            int(min(result.hop_results, key=lambda h: result.hop_results[h].gedig))
            if result.hop_results
            else 0
        )
        hop_min = result.hop_results.get(best_hop) if result.hop_results else None
        delta_ged_min = hop_min.ged if hop_min else result.ged_value
        delta_ig_min = hop_min.ig if hop_min else result.ig_value

        last_center_node = current_center_node
        last_position = (int(current_position[0]), int(current_position[1]))
        action_delta = SimpleMaze.ACTIONS[action]
        action_dir_idx = direction_from_delta(action_delta)

        # Execute action in environment
        obs, reward, done, _ = env.step(action)
        next_position_raw = obs.position
        anchor_position = (int(next_position_raw[0]), int(next_position_raw[1]))
        moved = (anchor_position != last_position)
        current_position = anchor_position
        current_center_node = make_center_node(anchor_position)

        visit_counts[anchor_position] = visit_counts.get(anchor_position, 0) + 1

        if current_center_node not in graph:
            graph.add_node(current_center_node)

        if not graph.has_edge(last_center_node, current_center_node):
            graph.add_edge(last_center_node, current_center_node)

        visited.add(last_position)
        visited.add(anchor_position)
        prev_action_delta = action_delta
        prev_success = True

        if last_center_node in graph:
            last_entry = graph.nodes[last_center_node]
            stored_delta = last_entry.get("last_action_delta") or (0, 0)
            stored_delta = tuple(int(val) for val in stored_delta)
            last_vec = compute_episode_vector(
                base_position=last_position,
                maze_shape=maze_shape,
                action_delta=stored_delta,
                is_passable=last_entry.get("is_passable", True),
                visits=visit_counts[last_position],
                success=last_entry.get("success", True),
                is_goal=(last_position == env.goal_pos),
                target_position=last_position,
            )
            last_entry["abs_vector"] = last_vec
            last_entry["visit_count"] = visit_counts[last_position]

        if action_dir_idx is not None:
            direction_node = make_direction_node(last_position, action_dir_idx)
            if direction_node not in graph:
                graph.add_node(direction_node)
            dir_entry = graph.nodes[direction_node]
            dir_vec = compute_episode_vector(
                base_position=last_position,
                maze_shape=maze_shape,
                action_delta=action_delta,
                is_passable=bool(moved),
                visits=visit_counts[anchor_position],
                success=bool(moved),
                is_goal=(anchor_position == env.goal_pos),
                target_position=anchor_position,
            )
            dir_entry["abs_vector"] = dir_vec
            dir_entry["visit_count"] = visit_counts[anchor_position]
            dir_entry["last_action_delta"] = action_delta
            dir_entry["success"] = bool(moved)
            dir_entry["is_passable"] = bool(moved)
            dir_entry["target_position"] = [anchor_position[0], anchor_position[1]]
            dir_entry["relative_delta"] = [int(action_delta[0]), int(action_delta[1])]
            dir_entry["direction"] = action_dir_idx
            anchor_list = dir_entry.get("anchor_positions")
            anchor_entry = [last_position[0], last_position[1]]
            if anchor_list is None:
                dir_entry["anchor_positions"] = [anchor_entry]
            elif anchor_entry not in anchor_list:
                anchor_list.append(anchor_entry)
            if not graph.has_edge(last_center_node, direction_node):
                graph.add_edge(last_center_node, direction_node)
            if not graph.has_edge(current_center_node, direction_node):
                graph.add_edge(current_center_node, direction_node)

        current_episode_vec = compute_episode_vector(
            base_position=anchor_position,
            maze_shape=maze_shape,
            action_delta=prev_action_delta,
            is_passable=True,  # current tile exists; passable as a node
            visits=visit_counts[anchor_position],
            success=bool(moved),
            is_goal=obs.is_goal,
            target_position=anchor_position,
        )
        node_entry = graph.nodes[current_center_node]
        node_entry["abs_vector"] = current_episode_vec.copy()
        node_entry["visit_count"] = visit_counts[anchor_position]
        node_entry["last_action_delta"] = prev_action_delta
        node_entry["success"] = bool(moved)
        node_entry["is_goal"] = obs.is_goal
        node_entry["is_passable"] = True
        anchor_list = node_entry.get("anchor_positions")
        curr_anchor = [anchor_position[0], anchor_position[1]]
        if anchor_list is None:
            node_entry["anchor_positions"] = [curr_anchor]
        elif curr_anchor not in anchor_list:
            anchor_list.append(curr_anchor)
        node_entry["direction"] = CENTER_DIR
        node_entry["relative_delta"] = [0, 0]
        node_entry["target_position"] = [anchor_position[0], anchor_position[1]]

        counts = {
            "obs_total": len(observation_candidates),
            "mem_total": len(memory_candidates),
            "cand_obs": sum(1 for item in selection.candidates if item.get("origin") == "obs"),
            "cand_mem": sum(1 for item in selection.candidates if item.get("origin") == "mem"),
            "link_obs": sum(1 for item in selection.links if item.get("origin") == "obs"),
            "link_mem": sum(1 for item in selection.links if item.get("origin") == "mem"),
            "forced_total": len(forced_links),
            "forced_obs": sum(1 for item in forced_links if item.get("origin") == "obs"),
            "forced_mem": sum(1 for item in forced_links if item.get("origin") == "mem"),
        }
        counts["cand_obs"] += counts["forced_obs"]
        counts["cand_mem"] += counts["forced_mem"]

        decision_info = {
            "origin": chosen_obs.get("origin") if chosen_obs else "fallback",
            "index": chosen_obs.get("index") if chosen_obs else None,
            "action": chosen_obs.get("action") if chosen_obs else action,
            "distance": chosen_obs.get("distance") if chosen_obs else None,
            "similarity": chosen_obs.get("similarity") if chosen_obs else None,
        }

        candidate_selection = {
            "k_star": float(k_star),
            "theta_cand": selector_params["theta_cand"],
            "theta_link": selector_params["theta_link"],
            "k_cap": selector_params["candidate_cap"],
            "top_m": selector_params["top_m"],
            "log_k_star": float(math.log(k_star + 1.0)) if k_star >= 1 else None,
            "r_cand": selector_params.get("theta_cand"),
            "r_link": selector_params.get("theta_link"),
            "radius_cand": selector_params.get("cand_radius"),
            "radius_link": selector_params.get("link_radius"),
            "counts": counts,
            "decision": decision_info,
            "link_registered_positions": [
                [int(pos[0]), int(pos[1]), int(pos[2])] for pos in sorted(registered_link_positions)
            ],
            "forced_links": [
                dict(item) for item in forced_links
            ],
        }
        for entry in candidate_selection["forced_links"]:
            vector_val = entry.get("vector")
            if isinstance(vector_val, np.ndarray):
                entry["vector"] = vector_val.tolist()
            abs_val = entry.get("abs_vector")
            if isinstance(abs_val, np.ndarray):
                entry["abs_vector"] = abs_val.tolist()
            if "relative_delta" in entry and not isinstance(entry["relative_delta"], list):
                entry["relative_delta"] = list(entry["relative_delta"])
            if "anchor_position" in entry and not isinstance(entry["anchor_position"], list):
                entry["anchor_position"] = list(entry["anchor_position"])
            if "target_position" in entry and not isinstance(entry["target_position"], list):
                entry["target_position"] = list(entry["target_position"])

        def _norm_pos(val: Any) -> List[int]:
            node_id = canonical_node_id(val)
            return [int(node_id[0]), int(node_id[1]), int(node_id[2])]

        graph_nodes_snapshot = [_norm_pos(node) for node in graph.nodes()]
        graph_edges_snapshot = [[_norm_pos(u), _norm_pos(v)] for u, v in graph.edges()]
        forced_edges_snapshot = [
            [_norm_pos(edge[0]), _norm_pos(edge[1])] for edge in sorted(forced_edge_store)
        ]
        ranked_candidates = [dict(item) for item in ranked_all_candidates]
        selected_links = [dict(item) for item in selection.links]
        forced_links_log = [dict(item) for item in forced_links]
        for entry in forced_links_log:
            entry.setdefault("forced", True)
        candidate_pool = [dict(item) for item in selection.candidates]
        for entry in forced_links_log:
            entry.setdefault("forced", True)
        candidate_pool.extend(forced_links_log)
        new_edge = [_norm_pos(last_center_node), _norm_pos(current_center_node)]
        for container in (candidate_pool, selected_links, ranked_candidates, forced_links_log):
            for entry in container:
                vector_val = entry.get("vector")
                if isinstance(vector_val, np.ndarray):
                    entry["vector"] = vector_val.tolist()
                abs_val = entry.get("abs_vector")
                if isinstance(abs_val, np.ndarray):
                    entry["abs_vector"] = abs_val.tolist()
                anchor_val = entry.get("anchor_position")
                if anchor_val is not None:
                    if hasattr(anchor_val, "tolist"):
                        anchor_val = anchor_val.tolist()
                    anchor_list = list(anchor_val)
                    if len(anchor_list) == 2:
                        entry["anchor_position"] = [int(anchor_list[0]), int(anchor_list[1])]
                rel_val = entry.get("relative_delta")
                if rel_val is not None:
                    if hasattr(rel_val, "tolist"):
                        rel_val = rel_val.tolist()
                    rel_list = list(rel_val)
                    if len(rel_list) == 2:
                        entry["relative_delta"] = [int(rel_list[0]), int(rel_list[1])]
                target_val = entry.get("target_position")
                if target_val is not None:
                    if hasattr(target_val, "tolist"):
                        target_val = target_val.tolist()
                    target_list = list(target_val)
                    if len(target_list) == 2:
                        entry["target_position"] = [int(target_list[0]), int(target_list[1])]
                # Augment candidate snapshot with direction label and visit alias
                dir_idx = entry.get("direction")
                if dir_idx is None and entry.get("relative_delta") is not None:
                    rd = entry["relative_delta"]
                    if isinstance(rd, (list, tuple)) and len(rd) == 2:
                        derv = direction_from_delta((int(rd[0]), int(rd[1])))
                        if derv is not None:
                            entry["direction"] = int(derv)
                            dir_idx = int(derv)
                if isinstance(dir_idx, (int, np.integer)):
                    entry["direction_label"] = DIR_LABELS.get(int(dir_idx), "?")
                if "meta_visits" in entry and isinstance(entry["meta_visits"], (int, float)):
                    entry["visit"] = int(entry["meta_visits"])  # alias for UI
                elif "visit_count" in entry and isinstance(entry["visit_count"], (int, float)):
                    entry["visit"] = int(entry["visit_count"])  # alias for UI

        step_records.append(
            StepRecord(
                seed=seed,
                step=step,
                position=current_position,
                action=SimpleMaze.ACTION_NAMES.get(action, str(action)),
                candidate_selection=candidate_selection,
                delta_ged=float(delta_ged),
                delta_ig=float(delta_ig),
                delta_ged_min=float(delta_ged_min),
                delta_ig_min=float(delta_ig_min),
                g0=float(g0),
                gmin=float(gmin),
                best_hop=best_hop,
                is_dead_end=bool(obs.is_dead_end),
                reward=float(reward),
                done=bool(done),
                possible_moves=list(possible_moves),
                candidate_pool=candidate_pool,
                selected_links=selected_links + forced_links_log,
                ranked_candidates=ranked_candidates,
                graph_nodes=graph_nodes_snapshot,
                graph_edges=graph_edges_snapshot,
                forced_edges=forced_edges_snapshot,
                new_edge=new_edge,
                episode_vector=current_episode_vec.tolist(),
                query_vector=query_vec_list,
            )
        )

        if obs.is_goal and done:
            success = True
            break

    dead_end_steps = sum(1 for rec in step_records if rec.is_dead_end)
    dead_end_escape = 0
    for idx, rec in enumerate(step_records):
        if not rec.is_dead_end:
            continue
        if idx + 1 < len(step_records):
            next_pos = step_records[idx + 1].position
            if next_pos != rec.position:
                dead_end_escape += 1
        elif success:
            dead_end_escape += 1

    episode_summary = {
        "seed": seed,
        "success": success,
        "steps": len(step_records),
        "edges": graph.number_of_edges(),
        "k_star_series": [rec.candidate_selection["k_star"] for rec in step_records],
        "g0_series": [rec.g0 for rec in step_records],
        "gmin_series": [rec.gmin for rec in step_records],
        "multihop_best_hop": [rec.best_hop for rec in step_records],
        "dead_end_steps": dead_end_steps,
        "dead_end_escape_rate": (dead_end_escape / dead_end_steps) if dead_end_steps else 1.0,
    }

    return episode_summary, step_records, maze_snapshot


def aggregate(runs: List[Dict[str, float]]) -> Dict[str, float]:
    successes = [1.0 if run["success"] else 0.0 for run in runs]
    steps = [run["steps"] for run in runs]
    edges = [run["edges"] for run in runs]
    k_star = [value for run in runs for value in run["k_star_series"]]
    g0_all = [value for run in runs for value in run["g0_series"]]
    gmin_all = [value for run in runs for value in run["gmin_series"]]
    multihop_flags = [1.0 if value > 0 else 0.0 for run in runs for value in run["multihop_best_hop"]]
    dead_end_steps = [run.get("dead_end_steps", 0) for run in runs]
    dead_end_escape = [
        run.get("dead_end_steps", 0) * run.get("dead_end_escape_rate", 0.0)
        for run in runs
    ]


    summary = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_edges": float(np.mean(edges)) if edges else 0.0,
        "g0_mean": float(np.mean(g0_all)) if g0_all else 0.0,
        "g0_std": float(np.std(g0_all)) if g0_all else 0.0,
        "gmin_mean": float(np.mean(gmin_all)) if gmin_all else 0.0,
        "k_star_mean": float(np.mean(k_star)) if k_star else 0.0,
        "multihop_usage": float(np.mean(multihop_flags)) if multihop_flags else 0.0,
        "dead_end_steps_avg": float(np.mean(dead_end_steps)) if dead_end_steps else 0.0,
        "dead_end_escape_rate_avg": (
            float(sum(dead_end_escape)) / float(sum(dead_end_steps))
            if sum(dead_end_steps) > 0
            else 1.0
        ),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-K★ geDIG maze experiment")
    parser.add_argument("--maze-size", type=int, default=15, help="Maze grid size (square)")
    parser.add_argument("--maze-type", type=str, default="dfs", help="Maze generator type (dfs, prim, rooms, etc.)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Step limit per episode")
    parser.add_argument("--seeds", type=int, default=20, help="Number of sequential seeds to run")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed index (inclusive)")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="geDIG lambda weight")
    parser.add_argument("--max-hops", type=int, default=5, help="Max hops for multihop evaluation")
    parser.add_argument("--decay-factor", type=float, default=0.85, help="Hop decay factor")
    parser.add_argument(
        "--adaptive-hops",
        action="store_true",
        help="Enable adaptive hop early-exit (disabled by default for maze runs)",
    )
    parser.add_argument("--theta-cand", type=float, default=1.0, help="Distance threshold r_cand")
    parser.add_argument("--theta-link", type=float, default=0.1, help="Distance threshold r_link")
    parser.add_argument("--candidate-cap", type=int, default=32, help="K★ cap")
    parser.add_argument("--top-m", type=int, default=32, help="Top-M cutoff before gating")
    parser.add_argument("--cand-radius", type=float, default=1.0, help="Max distance r_cand")
    parser.add_argument("--link-radius", type=float, default=0.1, help="Max distance r_link")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON output path")
    parser.add_argument("--step-log", type=Path, help="Optional per-step JSON log path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selector_params = {
        "theta_cand": args.theta_cand,
        "theta_link": args.theta_link,
        "candidate_cap": args.candidate_cap,
        "top_m": args.top_m,
        "cand_radius": args.cand_radius,
        "link_radius": args.link_radius,
    }
    selector_params["r_cand"] = selector_params["theta_cand"]
    selector_params["r_link"] = selector_params["theta_link"]
    gedig_params = {
        "lambda_weight": args.lambda_weight,
        "max_hops": args.max_hops,
        "decay_factor": args.decay_factor,
        "adaptive_hops": args.adaptive_hops,
    }

    runs: List[Dict[str, float]] = []
    all_steps: List[Dict[str, float]] = []
    maze_data: Dict[str, Any] = {}

    for offset in range(args.seeds):
        seed = args.seed_start + offset
        episode_summary, step_records, maze_snapshot = run_episode(
            seed=seed,
            maze_size=args.maze_size,
            max_steps=args.max_steps,
            maze_type=args.maze_type,
            selector_params=selector_params,
            gedig_params=gedig_params,
        )
        runs.append(episode_summary)
        maze_data[str(seed)] = maze_snapshot
        for record in step_records:
            all_steps.append(
                {
                    "seed": record.seed,
                    "step": record.step,
                    "position": list(record.position),
                    "action": record.action,
                    "candidate_selection": record.candidate_selection,
                    "delta_ged": record.delta_ged,
                    "delta_ig": record.delta_ig,
                    "delta_ged_min": record.delta_ged_min,
                    "delta_ig_min": record.delta_ig_min,
                    "g0": record.g0,
                    "gmin": record.gmin,
                    "best_hop": record.best_hop,
                    "is_dead_end": record.is_dead_end,
                    "reward": record.reward,
                    "done": record.done,
                    "possible_moves": record.possible_moves,
                    "candidate_pool": record.candidate_pool,
                    "selected_links": record.selected_links,
                    "ranked_candidates": record.ranked_candidates,
                    "graph_nodes": record.graph_nodes,
                    "graph_edges": record.graph_edges,
                    "forced_edges": record.forced_edges,
                    "new_edge": record.new_edge,
                    "episode_vector": record.episode_vector,
                    "query_vector": record.query_vector,
                }
            )

    summary = aggregate(runs)
    output_payload = {
        "config": {
            "maze_size": args.maze_size,
            "maze_type": args.maze_type,
            "max_steps": args.max_steps,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "selector": selector_params,
            "gedig": gedig_params,
        },
        "summary": summary,
        "runs": runs,
        "maze_data": maze_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.step_log:
        args.step_log.parent.mkdir(parents=True, exist_ok=True)
        args.step_log.write_text(json.dumps(all_steps, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
