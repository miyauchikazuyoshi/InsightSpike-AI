"""Sleep-related functions for maze experiments.

Contains:
- _build_sleep_action_plan: Derive 1-step action plan from warmup transitions
- _build_sleep_q_table: Derive Q(s,a) table from warmup transitions
- _build_sleep_edge_weights: Build edge weight priors from warmup transitions
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from insightspike.environments.maze import SimpleMaze

from .models import StepRecord


def build_sleep_action_plan(
    steps: Sequence[StepRecord],
    *,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, Any]]:
    """Derive a 1-step action plan from experienced transitions (Sleep path compression).

    The plan maps (row,col) -> action_id (0-3) for the next step along the shortest
    path to the goal, restricted to edges actually observed during the warmup run.
    """
    adjacency: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
    for rec in steps:
        try:
            pre_node = (getattr(rec, "query_node_pre", None) or [])[:2]
            post_node = (getattr(rec, "query_node_post", None) or [])[:2]
            pre = (int(pre_node[0]), int(pre_node[1]))
            post = (int(post_node[0]), int(post_node[1]))
        except Exception:
            continue
        if pre == post:
            continue
        adjacency[pre].add(post)
        adjacency[post].add(pre)

    frontier: deque[Tuple[int, int]] = deque([start_pos])
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen: Set[Tuple[int, int]] = {start_pos}
    found = False
    while frontier:
        u = frontier.popleft()
        if u == goal_pos:
            found = True
            break
        for v in adjacency.get(u, set()):
            if v in seen:
                continue
            seen.add(v)
            prev[v] = u
            frontier.append(v)

    explored_edges_undirected = int(sum(len(v) for v in adjacency.values()) // 2)
    if not found:
        meta = {
            "found": False,
            "start_pos": [int(start_pos[0]), int(start_pos[1])],
            "goal_pos": [int(goal_pos[0]), int(goal_pos[1])],
            "explored_nodes": int(len(adjacency)),
            "explored_edges_undirected": explored_edges_undirected,
            "path_len": None,
        }
        return {}, meta

    # Reconstruct path positions from goal -> start
    path: List[Tuple[int, int]] = [goal_pos]
    cur = goal_pos
    while cur != start_pos:
        cur = prev.get(cur)
        if cur is None:
            break
        path.append(cur)
    if not path or path[-1] != start_pos:
        meta = {
            "found": False,
            "start_pos": [int(start_pos[0]), int(start_pos[1])],
            "goal_pos": [int(goal_pos[0]), int(goal_pos[1])],
            "explored_nodes": int(len(adjacency)),
            "explored_edges_undirected": explored_edges_undirected,
            "path_len": None,
        }
        return {}, meta
    path.reverse()

    delta_to_action = {tuple(v): int(k) for k, v in SimpleMaze.ACTIONS.items()}
    plan: Dict[Tuple[int, int], int] = {}
    for a, b in zip(path[:-1], path[1:]):
        dr = int(b[0] - a[0])
        dc = int(b[1] - a[1])
        act = delta_to_action.get((dr, dc))
        if act is None:
            continue
        plan[a] = int(act)

    meta = {
        "found": True,
        "start_pos": [int(start_pos[0]), int(start_pos[1])],
        "goal_pos": [int(goal_pos[0]), int(goal_pos[1])],
        "explored_nodes": int(len(adjacency)),
        "explored_edges_undirected": explored_edges_undirected,
        "path_len": int(max(0, len(path) - 1)),
    }
    return plan, meta


def build_sleep_q_table(
    steps: Sequence[StepRecord],
    *,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    gamma: float = 0.99,
    alpha: float = 0.4,
    iters: int = 50,
    step_penalty: float = -0.01,
    goal_reward: float = 1.0,
    revisit_penalty: float = -0.2,
    deadend_penalty: float = 0.0,
    blocked_penalty: float = 0.0,
    revisit_threshold: int = 2,
    episode_boundaries: Optional[Sequence[int]] = None,
) -> Tuple[Dict[Tuple[int, int], Dict[int, float]], Dict[str, Any]]:
    """Derive a Q(s,a) table from warmup transitions (Sleep value propagation).

    This is a lightweight, tabular replay update (Q-learning style) over the
    experienced transitions only. It is intended to act as a *prior* for the
    next Wake episode (soft bias), not as a hard plan.

    episode_boundaries: indices into `steps` where a new episode begins
    (multi-cycle warmup concatenates episodes). At each boundary the
    revisit visit_counts reset, so a later episode's traversal of cells
    seen in an earlier episode is not mislabeled as a revisit. None (the
    default) keeps the historical behavior: one continuous count across
    the whole sequence.
    """
    # Collect transitions and per-state available actions.
    actions_by_state: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    transitions: List[Tuple[Tuple[int, int], int, Tuple[int, int], bool, bool, bool, bool]] = []

    goal_reached = False
    visit_counts: Dict[Tuple[int, int], int] = {tuple(start_pos): 1}
    threshold = max(2, int(revisit_threshold or 2))
    action_name_to_id = {str(v): int(k) for k, v in SimpleMaze.ACTION_NAMES.items()}
    _boundary_set = set(int(b) for b in (episode_boundaries or []))

    for _rec_idx, rec in enumerate(steps):
        if _rec_idx in _boundary_set:
            visit_counts = {tuple(start_pos): 1}
        try:
            pre_node = (getattr(rec, "query_node_pre", None) or [])[:2]
            post_node = (getattr(rec, "query_node_post", None) or [])[:2]
            if len(pre_node) < 2 or len(post_node) < 2:
                continue
            s = (int(pre_node[0]), int(pre_node[1]))
        except Exception:
            continue
        try:
            s2 = (int(post_node[0]), int(post_node[1]))
        except Exception:
            continue
        try:
            a_raw = getattr(rec, "action", None)
            if isinstance(a_raw, int):
                a = int(a_raw)
            else:
                a_str = str(a_raw)
                if a_str.lstrip("-").isdigit():
                    a = int(a_str)
                else:
                    a = action_name_to_id.get(a_str)
            if a is None:
                continue
            a = int(a)
        except Exception:
            continue

        # Record action availability for s and s2 when provided.
        try:
            for act in (getattr(rec, "possible_moves", []) or []):
                actions_by_state[s].add(int(act))
        except Exception:
            pass
        try:
            for act in (getattr(rec, "possible_moves_post", []) or []):
                actions_by_state[s2].add(int(act))
        except Exception:
            pass

        # Revisit label: prefer logged label; otherwise recompute from visit counts.
        revisit = False
        try:
            if "revisit" in str(getattr(rec, "cortisol_reason", "") or ""):
                revisit = True
        except Exception:
            revisit = False
        try:
            visit_counts[s2] = int(visit_counts.get(s2, 0)) + 1
            if visit_counts[s2] >= threshold:
                revisit = True
        except Exception:
            pass

        done = False
        try:
            done = bool(getattr(rec, "done", False))
        except Exception:
            done = False
        if s2 == tuple(goal_pos):
            done = True
            goal_reached = True

        moved = (s2 != s)
        blocked = (not moved)
        deadend = False
        try:
            deadend = bool(getattr(rec, "is_dead_end", False))
        except Exception:
            deadend = False

        transitions.append((s, a, s2, bool(done), bool(revisit), bool(blocked), bool(deadend)))

    # Initialize Q-table for seen states/actions (missing entries default to 0.0).
    q: Dict[Tuple[int, int], Dict[int, float]] = {}
    for s, acts in actions_by_state.items():
        q[s] = {int(a): 0.0 for a in sorted(acts)}
    # Ensure start/goal exist as keys for diagnostics
    q.setdefault(tuple(start_pos), {})
    q.setdefault(tuple(goal_pos), {})

    if not transitions:
        meta = {
            "goal_reached": False,
            "states": int(len(q)),
            "transitions": 0,
            "q_min": 0.0,
            "q_max": 0.0,
            "episode_boundaries_applied": sorted(_boundary_set),
            "params": {
                "gamma": float(gamma),
                "alpha": float(alpha),
                "iters": int(iters),
                "step_penalty": float(step_penalty),
                "goal_reward": float(goal_reward),
                "revisit_penalty": float(revisit_penalty),
                "deadend_penalty": float(deadend_penalty),
                "blocked_penalty": float(blocked_penalty),
                "revisit_threshold": int(threshold),
            },
        }
        return q, meta

    g = float(gamma)
    a_lr = float(alpha)
    n_iter = max(1, int(iters or 1))
    r_step = float(step_penalty)
    r_goal = float(goal_reward)
    r_revisit = float(revisit_penalty)
    r_deadend = float(deadend_penalty)
    r_blocked = float(blocked_penalty)

    # Replay updates
    for _ in range(n_iter):
        for s, act, s2, done, revisit, blocked, deadend in transitions:
            r = r_step
            if revisit:
                r += r_revisit
            if blocked:
                r += r_blocked
            if deadend:
                r += r_deadend
            if s2 == tuple(goal_pos):
                r += r_goal
                done = True
            if done:
                target = r
            else:
                next_q = q.get(s2, {})
                max_next = max(next_q.values()) if next_q else 0.0
                target = r + g * float(max_next)
            cur = q.setdefault(s, {}).get(int(act), 0.0)
            q[s][int(act)] = float(cur + a_lr * (float(target) - float(cur)))

    q_vals = [float(v) for m in q.values() for v in m.values()]
    meta = {
        "goal_reached": bool(goal_reached),
        "states": int(len(q)),
        "transitions": int(len(transitions)),
        "q_min": float(min(q_vals)) if q_vals else 0.0,
        "q_max": float(max(q_vals)) if q_vals else 0.0,
        "episode_boundaries_applied": sorted(_boundary_set),
        "params": {
            "gamma": float(gamma),
            "alpha": float(alpha),
            "iters": int(iters),
            "step_penalty": float(step_penalty),
            "goal_reward": float(goal_reward),
            "revisit_penalty": float(revisit_penalty),
            "deadend_penalty": float(deadend_penalty),
            "blocked_penalty": float(blocked_penalty),
            "revisit_threshold": int(threshold),
        },
    }
    return q, meta


def build_sleep_edge_weights(
    steps: Sequence[StepRecord],
    *,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    alpha: float = 0.4,
    gamma: float = 0.95,
    alpha_explore: float = 0.05,
    revisit_penalty: float = 0.2,
    deadend_penalty: float = 0.2,
    blocked_penalty: float = 0.2,
    revisit_threshold: int = 2,
) -> Tuple[Dict[Tuple[int, int], Dict[int, float]], Dict[str, Any]]:
    """Build edge weight priors from warmup transitions."""
    edge_weights: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    visit_counts: Dict[Tuple[int, int], int] = {tuple(start_pos): 1}
    action_name_to_id = {str(v): int(k) for k, v in SimpleMaze.ACTION_NAMES.items()}
    threshold = max(2, int(revisit_threshold or 2))

    path_edges: List[Tuple[Tuple[int, int], int, Tuple[int, int]]] = []
    goal_reached = False

    revisit_hits = 0
    deadend_hits = 0
    blocked_hits = 0
    explore_hits = 0

    for rec in steps:
        try:
            pre_node = (getattr(rec, "query_node_pre", None) or [])[:2]
            post_node = (getattr(rec, "query_node_post", None) or [])[:2]
            if len(pre_node) < 2 or len(post_node) < 2:
                continue
            current_state = (int(pre_node[0]), int(pre_node[1]))
            next_state = (int(post_node[0]), int(post_node[1]))
        except Exception:
            continue
        try:
            action_raw = getattr(rec, "action", None)
            if isinstance(action_raw, int):
                action_id = int(action_raw)
            else:
                action_str = str(action_raw)
                if action_str.lstrip("-").isdigit():
                    action_id = int(action_str)
                else:
                    action_id = action_name_to_id.get(action_str)
            if action_id is None:
                continue
            action_id = int(action_id)
        except Exception:
            continue

        moved = next_state != current_state
        blocked = not moved
        if blocked and blocked_penalty:
            edge_weights[current_state][action_id] -= float(blocked_penalty)
            blocked_hits += 1

        if moved:
            prev_visits = int(visit_counts.get(next_state, 0))
            new_visits = prev_visits + 1
            visit_counts[next_state] = new_visits

            if prev_visits == 0 and alpha_explore:
                edge_weights[current_state][action_id] += float(alpha_explore)
                explore_hits += 1
            if new_visits >= threshold and revisit_penalty:
                edge_weights[current_state][action_id] -= float(revisit_penalty)
                revisit_hits += 1
            try:
                if bool(getattr(rec, "is_dead_end", False)) and deadend_penalty:
                    edge_weights[current_state][action_id] -= float(deadend_penalty)
                    deadend_hits += 1
            except Exception:
                pass

            path_edges.append((current_state, action_id, next_state))

        if next_state == tuple(goal_pos):
            goal_reached = True
            if bool(getattr(rec, "done", False)):
                break

    if goal_reached and alpha:
        for depth, (state, action_id, _next_state) in enumerate(reversed(path_edges)):
            edge_weights[state][action_id] += float(alpha) * (float(gamma) ** float(depth))

    clean_weights: Dict[Tuple[int, int], Dict[int, float]] = {}
    all_values: List[float] = []
    for state, action_map in edge_weights.items():
        cleaned = {
            int(action_id): float(weight)
            for action_id, weight in action_map.items()
            if math.isfinite(weight) and abs(float(weight)) > 1e-12
        }
        if cleaned:
            clean_weights[state] = cleaned
            all_values.extend(cleaned.values())

    meta = {
        "goal_reached": bool(goal_reached),
        "states": int(len(clean_weights)),
        "edges": int(sum(len(v) for v in clean_weights.values())),
        "weight_min": float(min(all_values)) if all_values else 0.0,
        "weight_max": float(max(all_values)) if all_values else 0.0,
        "path_len": int(len(path_edges)) if goal_reached else 0,
        "stats": {
            "revisit_hits": int(revisit_hits),
            "deadend_hits": int(deadend_hits),
            "blocked_hits": int(blocked_hits),
            "explore_hits": int(explore_hits),
        },
        "params": {
            "alpha": float(alpha),
            "gamma": float(gamma),
            "alpha_explore": float(alpha_explore),
            "revisit_penalty": float(revisit_penalty),
            "deadend_penalty": float(deadend_penalty),
            "blocked_penalty": float(blocked_penalty),
            "revisit_threshold": int(threshold),
        },
    }
    return clean_weights, meta
