"""
Maze query-hub experiment driver (prototype).

This module mirrors the CLI surface of `maze-fixed-kstar/run_experiment.py`
while swapping the graph semantics so that *query nodes* become the hub
structure instead of legacy CENTER_DIR nodes. The implementation is kept
close to the reference code for ease of diffing and validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")
# Allow ΔH (entropy差分) が負に振れるケースでも geDIG に反映されるよう、
# IG 非負クランプはデフォルト無効化する。
os.environ.setdefault("MAZE_GEDIG_IG_NONNEG", "0")
# IG 符号規約: 論文の新定義（ΔIG = ΔH + γ·ΔSP）に合わせ、
# ΔH は after−before（秩序化で負）を既定に固定する。

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import networkx as nx
import numpy as np

from insightspike.algorithms.gedig.selector import TwoThresholdCandidateSelector
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.environments.maze import MazeObservation, SimpleMaze
from insightspike.algorithms.core.metrics import normalized_ged as _norm_ged
from insightspike.algorithms.core.metrics import entropy_ig as _entropy_ig
from insightspike.algorithms.sp_distcache import DistanceCache
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
from qhlib.timeline import build_current_step_timeline
from qhlib.commit import apply_commit_policy
from qhlib.evaluator import evaluate_multihop
from qhlib.edges import build_ecand
from qhlib.store import SQLiteStore
from qhlib.edges import build_ecand
from qhlib.spatial_index import SpatialGridIndex
from qhlib.l1index import WeightedL2Index
from qhlib.sp_pairsets import SQLitePairsetService, InMemoryPairsetService, SignatureBuilder

MazeSummary = Dict[str, Any]
StepLog = Dict[str, Any]

# --------------------------------------------------------------------------------------
# Query-hub specific helpers
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


@dataclass
class StepRecord:
    seed: int
    step: int
    position: Tuple[int, int]
    action: str
    candidate_selection: Dict[str, Any]
    delta_ged: float
    delta_ig: float
    delta_ged_min: float
    delta_ig_min: float
    delta_sp: float
    delta_sp_min: float
    delta_h: float
    delta_h_min: float
    linkset_delta_ged: float = 0.0
    linkset_delta_h: float = 0.0
    linkset_delta_sp: float = 0.0
    linkset_g: float = 0.0
    # For display: linkset entropies and SP levels
    linkset_entropy_before: float = 0.0
    linkset_entropy_after: float = 0.0
    linkset_pos_w_before: int = 0
    linkset_pos_w_after: int = 0
    linkset_topw_before: List[float] = field(default_factory=list)
    linkset_topw_after: List[float] = field(default_factory=list)
    sp_before: float = 0.0
    sp_after: float = 0.0
    structural_cost: float = 0.0
    structural_improvement: float = 0.0
    g0: float = 0.0
    gmin: float = 0.0
    best_hop: int = 0
    # Dead-end flags: pre = before env.step(action), post = after step
    is_dead_end: bool = False  # post-step (kept for backward compat)
    is_dead_end_pre: bool = False
    reward: float = 0.0
    done: bool = False
    # possible_moves: pre-step (before action). For clarity, also store post-step.
    possible_moves: List[int] = field(default_factory=list)
    possible_moves_post: List[int] = field(default_factory=list)
    candidate_pool: List[Dict[str, Any]] = field(default_factory=list)
    selected_links: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    graph_nodes: List[List[int]] = field(default_factory=list)
    graph_edges: List[List[List[int]]] = field(default_factory=list)
    cand_edges: List[Dict[str, Any]] = field(default_factory=list)
    new_edge: List[List[int]] = field(default_factory=list)
    episode_vector: List[float] = field(default_factory=list)
    query_vector: List[float] = field(default_factory=list)
    query_node: List[int] = field(default_factory=list)
    query_node_pre: List[int] = field(default_factory=list)
    query_node_post: List[int] = field(default_factory=list)
    ag_fire: bool = False
    dg_fire: bool = False
    # Dynamic AG threshold diagnostics
    theta_ag: float = 0.0
    ag_auto: bool = False
    ag_quantile: float = 0.0
    g0_history_len: int = 0
    debug_hop0: Dict[str, Any] = field(default_factory=dict)
    hop_series: List[Dict[str, Any]] = field(default_factory=list)
    # Post-step diagnostics (after env.step including executed action edges)
    hop_series_post: List[Dict[str, Any]] = field(default_factory=list)
    # Timeline edges (explicit edges added this step, e.g., Q_prev↔Q_next)
    timeline_edges: List[List[List[int]]] = field(default_factory=list)
    # Only the edges newly committed this step (prev→now difference)
    committed_only_edges: List[List[List[int]]] = field(default_factory=list)
    # Only the nodes newly created this step (prev→now difference)
    committed_only_nodes: List[List[int]] = field(default_factory=list)
    # With metadata
    committed_only_edges_meta: List[Dict[str, Any]] = field(default_factory=list)
    committed_only_nodes_meta: List[Dict[str, Any]] = field(default_factory=list)
    # Forced link candidates (edges Q↔dir that were forced but not necessarily committed)
    forced_edges: List[List[List[int]]] = field(default_factory=list)
    forced_edges_meta: List[Dict[str, Any]] = field(default_factory=list)
    # Full sequence snapshots (per step)
    graph_nodes_preselect: List[List[int]] = field(default_factory=list)
    graph_edges_preselect: List[List[List[int]]] = field(default_factory=list)
    graph_nodes_pre: List[List[int]] = field(default_factory=list)
    graph_edges_pre: List[List[List[int]]] = field(default_factory=list)
    graph_nodes_eval: List[List[int]] = field(default_factory=list)   # after commit, before env.step
    graph_edges_eval: List[List[List[int]]] = field(default_factory=list)
    graph_nodes_post: List[List[int]] = field(default_factory=list)   # after env.step
    graph_edges_post: List[List[List[int]]] = field(default_factory=list)
    # Datastore snapshot (if enabled)
    ds_nodes_total: int = 0
    ds_edges_total: int = 0
    ds_nodes_saved: List[Dict[str, Any]] = field(default_factory=list)
    ds_edges_saved: List[Dict[str, Any]] = field(default_factory=list)
    ds_graph_nodes: List[List[int]] = field(default_factory=list)
    ds_graph_edges: List[List[List[int]]] = field(default_factory=list)
    # Debug: candidate and hop stats
    ecand_count: int = 0
    ecand_mem_count: int = 0
    ecand_qpast_count: int = 0
    hop_series_len: int = 0
    # SP diagnostics per hop: pair_count/Lb and top δSP candidates
    sp_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    # Debug: recomputed min and best to verify JSON vs calc
    dbg_gmin_calc: float = 0.0
    dbg_best_hop_calc: int = 0
    # Multi-hop only minima (exclude hop0)
    gmin_mh: float = 0.0
    delta_ged_min_mh: float = 0.0
    delta_ig_min_mh: float = 0.0
    delta_sp_min_mh: float = 0.0
    # DG commit snapshot
    dg_committed_edges: List[List[List[int]]] = field(default_factory=list)
    # Profiling & diagnostics
    ring_Rr: int = 0
    ring_Rc: int = 0
    ring_max_cells: int = 0
    ring_cells: int = 0
    ring_nodes: int = 0
    ring_fallback: bool = False
    obs_dist_evals: int = 0
    mem_dist_evals: int = 0
    total_dist_evals: int = 0
    time_ms_candidates: float = 0.0
    time_ms_eval: float = 0.0
    # SP diagnostics (perf): per-step counters
    sp_sssp_du: int = 0
    sp_sssp_dv: int = 0
    sp_dv_leaf_skips: int = 0
    sp_cycle_verifies: int = 0
    # SP pairset DS persistence (post-step) diagnostics
    sp_ds_saved: bool = False
    sp_ds_eff_saved: int = 0


@dataclass
class QueryHubConfig:
    maze_size: int
    maze_type: str
    max_steps: int
    selector: Dict[str, Any]
    gedig: Dict[str, Any]
    linkset_mode: bool = False
    linkset_base: str = "mem"  # 'link' (S_link) | 'mem' (memory pool) | 'pool' (all candidates)
    theta_ag: float = 0.0
    theta_dg: float = 0.0
    top_link: int = 1
    # 0-hopベースで S_link を全本自動配線する（Top‑Lを無視）
    link_autowire_all: bool = True
    commit_budget: int = 1
    commit_from: str = "cand"
    norm_base: str = "link"  # 'link' or 'cand'
    # Action selection
    action_policy: str = "softmax"  # default to softmax
    action_temp: float = 0.1       # default low temperature
    anti_backtrack: bool = True
    # Diagnostics/Eval scope
    anchor_recent_q: int = 12  # include recent Q nodes into anchors for SP eval
    # SP cache controls
    sp_cache: bool = False
    sp_cache_mode: str = "core"  # 'core' or 'cached'
    sp_cand_topk: int = 0  # 0 = no cap
    sp_eval_allpairs: bool = False
    sp_eval_allpairs_exact: bool = False
    # ALL-PAIRS-EXACT: keep SA node-set stable across steps to improve APSP reuse
    sp_exact_stable_nodes: bool = False
    # Treat ΔSP as signed (no clamp) for reporting and diagnostics
    sp_signed: bool = False
    # Report delta_sp as best-hop (min g) instead of hop0 when True
    sp_report_best_hop: bool = False
    sp_pair_samples: int = 400
    sp_verify_threshold: float = 0.05
    # Evaluation policy
    eval_all_hops: bool = False  # if True, add one candidate per hop even if δSP<=0 (diagnostic)
    # GED policy: treat hop>0 GED as hop0 (paper-style structural fixed)
    ged_hop0_const: bool = True
    # Sequence ablations
    gh_mode: str = "greedy"  # 'greedy' or 'radius' (no extra edges; radius-only eval)
    pre_eval: bool = True     # enable pre-eval (IG/SP before wiring) diagnostics
    snapshot_mode: str = "after_select"  # 'before_select' or 'after_select'
    snapshot_level: str = "standard"     # 'minimal', 'standard', 'full'
    # Timeline/next-Q graph policies
    timeline_to_graph: bool = False  # if True, add timeline edges to graph
    add_next_q: bool = False         # if True, add next-step Q node to graph at end of step
    # Persistence (optional)
    persist_sqlite_path: Optional[str] = None  # if set, persist diffs to SQLite
    persist_namespace: str = "maze_query_hub"
    # Observation guard for action feasibility (walls/backtracks)
    obs_guard: bool = True
    # Persist also forced candidate links (even if not committed)
    persist_forced_candidates: bool = False
    # Use forced fallback as linkset base when S_link is empty (ablation 'f')
    link_forced_as_base: bool = False
    # Persist timeline edges (Q_prev->dir->Q_next) into DS snapshots/SQLite
    persist_timeline_edges: bool = True
    # DG commit policy: 'threshold' (default), 'always', 'never'
    dg_commit_policy: str = "threshold"
    # On DG fire, commit all hop0 S_link edges (not only Top-L)
    dg_commit_all_linkset: bool = False
    # Skip multi-hop evaluation on dead-end/backtrack steps (evaluate hop0 only)
    skip_mh_on_deadend: bool = False
    # Spatial prefilter options
    ring_ellipse: bool = True
    # Layer1 vector prefilter options
    layer1_prefilter: bool = False
    l1_cap: int = 128
    # Dynamic AG threshold
    ag_auto: bool = False
    ag_window: int = 30
    ag_quantile: float = 0.9
    # Verbose step logging (debug)
    verbose: bool = False
    # DS-backed SP pairsets
    sp_ds_sqlite: Optional[str] = None
    sp_ds_namespace: str = "mq_sp"
    # Checkpointing (periodic JSON writes to avoid timeout loss)
    checkpoint_interval: int = 0  # 0 disables periodic checkpoints
    checkpoint_path: Optional[str] = None  # path to write partial steps JSON
    # Post-step SP diagnostics (hop_series_post)
    post_sp_diagnostics: bool = True
    # Ultra-light steps JSON (skip heavy snapshot arrays entirely)
    steps_ultra_light: bool = False
    # Maze snapshot explicit dump (paper reproducibility)
    maze_snapshot_out: Optional[str] = None
    # Force per-hop series via evaluator fallback even in L3-only mode
    force_per_hop: bool = False
    # Per-hop evaluator fallback only when AG fires (L3-only)
    eval_per_hop_on_ag: bool = False
    # DG fire: commit chosen multi-hop edges as a BFS-like shortcut in one step
    dg_bfs_shortcut: bool = False


@dataclass
class EpisodeArtifacts:
    summary: MazeSummary
    steps: List[StepRecord]
    maze_snapshot: Dict[str, Any]


# --------------------------------------------------------------------------------------
# Core query-hub episode runner
# --------------------------------------------------------------------------------------

def run_episode_query(seed: int, config: QueryHubConfig) -> EpisodeArtifacts:
    random.seed(seed)
    np.random.seed(seed)

    # Respect SP boundary mode via environment before Core init
    try:
        spb = str(config.gedig.get("sp_boundary_mode", "")).strip()
        if spb:
            os.environ['MAZE_GEDIG_SP_BOUNDARY'] = spb
    except Exception:
        pass
    env = SimpleMaze(size=(config.maze_size, config.maze_size), maze_type=config.maze_type)
    selector = TwoThresholdCandidateSelector(
        theta_cand=config.selector["theta_cand"],
        theta_link=config.selector["theta_link"],
        k_cap=int(config.selector["candidate_cap"]),
        top_m=int(config.selector["top_m"]),
        radius_cand=config.selector.get("cand_radius"),
        radius_link=config.selector.get("link_radius"),
        score_key="distance",
        higher_is_better=False,
    )
    core = GeDIGCore(
        enable_multihop=bool(config.gedig["max_hops"]),
        max_hops=int(config.gedig["max_hops"]),
        decay_factor=config.gedig["decay_factor"],
        lambda_weight=config.gedig["lambda_weight"],
        use_local_normalization=True,
        adaptive_hops=bool(config.gedig["adaptive_hops"]),
        feature_weights=WEIGHT_VECTOR,
        linkset_mode=config.linkset_mode,
        sp_beta=float(config.gedig.get("sp_beta", 0.2)),
        sp_pair_samples=int(getattr(config, 'sp_pair_samples', 400)),
        # 論文準拠寄せ: linkset IG + 候補台ベースGED正規化
        ig_source_mode=("linkset" if config.linkset_mode else "graph"),
        ig_hop_apply=str(config.gedig.get("ig_hop_apply", "all")),
        ged_norm_scheme=("candidate_base" if str(config.norm_base) in ("link","cand") else "edges_after"),
        sp_eval_mode="fixed_before_pairs",
        sp_scope_mode=str(config.gedig.get("sp_scope_mode", "auto")),
        sp_hop_expand=int(config.gedig.get("sp_hop_expand", 0)),
    )

    graph = nx.Graph()
    # Optional SP distance cache (core/cached only). DS-based refactorは撤回。
    from insightspike.algorithms.sp_distcache import DistanceCache
    sp_mode = str(getattr(config, 'sp_cache_mode', 'core'))
    distcache = DistanceCache(mode=sp_mode, pair_samples=int(getattr(config, 'sp_pair_samples', 400)))
    obs = env.reset()
    current_position = (int(obs.position[0]), int(obs.position[1]))
    maze_shape = (config.maze_size, config.maze_size)
    visit_counts: Dict[Tuple[int, int], int] = {current_position: 1}
    prev_action_delta: Optional[Tuple[int, int]] = None
    prev_success = False

    current_query_node = make_query_node(current_position)
    query_vec = compute_query_vector(current_position, maze_shape)
    graph.add_node(
        current_query_node,
        abs_vector=query_vec,
        visit_count=0,
        anchor_positions=[[current_position[0], current_position[1]]],
        direction=QUERY_MARKER,
        relative_delta=[0, 0],
        target_position=[current_position[0], current_position[1]],
        node_type="query",
    )

    step_records: List[StepRecord] = []
    # Accumulated datastore-like graph (episode-local snapshot)
    ds_nodes_accum: Set[Tuple[int,int,int]] = set()
    ds_edges_accum: Set[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = set()
    # Spatial index for fast neighborhood lookup
    sp_index = SpatialGridIndex(maze_shape)
    # Layer1 weighted-L2 index for vector prefilter
    l1_index = WeightedL2Index(dim=8, weights=WEIGHT_VECTOR)
    # Keep recent Q nodes for anchor expansion
    recent_q_nodes: deque[Tuple[int, int, int]] = deque(maxlen=max(1, int(getattr(config, 'anchor_recent_q', 12))))
    recent_q_nodes.append(current_query_node)
    success = False
    maze_snapshot = {
        "layout": env.grid.astype(int).tolist(),
        "start_pos": list(env.start_pos),
        "goal_pos": list(env.goal_pos),
        "size": [env.height, env.width],
        "maze_type": config.maze_type,
    }
    # Optional: write explicit maze snapshot for paper reproducibility
    try:
        outp = getattr(config, 'maze_snapshot_out', None)
        if outp:
            Path(outp).parent.mkdir(parents=True, exist_ok=True)
            Path(outp).write_text(json.dumps(maze_snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass

    def register_direction_node(
        anchor_tuple: Tuple[int, int],
        dir_idx: int,
        delta: Tuple[int, int],
        candidate_vec_abs: np.ndarray,
        *,
        target_visits: int,
        is_passable: bool,
        is_goal: bool,
        target_pos: Tuple[int, int],
        source: Optional[str],
        step_index: int,
    ) -> None:
        node_id = make_direction_node(anchor_tuple, dir_idx)
        if node_id not in graph:
            graph.add_node(node_id)
        node_entry = graph.nodes[node_id]
        if "birth_step" not in node_entry:
            node_entry["birth_step"] = int(step_index)
        if source and "source" not in node_entry:
            node_entry["source"] = source
        node_entry["abs_vector"] = np.asarray(candidate_vec_abs, dtype=float)
        node_entry["visit_count"] = int(target_visits)
        node_entry["last_action_delta"] = (int(delta[0]), int(delta[1]))
        node_entry["success"] = False
        node_entry["is_goal"] = bool(is_goal)
        node_entry["is_passable"] = bool(is_passable)
        node_entry["target_position"] = [int(target_pos[0]), int(target_pos[1])]
        node_entry["relative_delta"] = [int(delta[0]), int(delta[1])]
        node_entry["direction"] = int(dir_idx)
        node_entry["node_type"] = "direction"
        anchors = node_entry.setdefault("anchor_positions", [])
        anchor_entry = [int(anchor_tuple[0]), int(anchor_tuple[1])]
        if anchor_entry not in anchors:
            anchors.append(anchor_entry)
            # Update spatial index
            try:
                sp_index.add((int(anchor_tuple[0]), int(anchor_tuple[1])), node_id)
            except Exception:
                pass
        # Update L1 index (abs_vector weighted)
        try:
            l1_index.add(node_id, np.asarray(candidate_vec_abs, dtype=float))
        except Exception:
            pass

    # Dynamic AG g0 history (for online percentile threshold)
    g0_history: deque = deque(maxlen=int(getattr(config, 'ag_window', 30)))
    # Cross-step APSP carry for ALL-PAIRS-EXACT
    apsp_exact_state: Dict[int, Dict[str, Any]] = {} if bool(getattr(config, 'sp_eval_allpairs_exact', False)) else {}

    for step in range(config.max_steps):
        minimal_snap = str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal'
        cand_edge_store: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, int]] = []
        possible_moves = obs.possible_moves
        # Pre-step dead-end heuristic (before taking action)
        deadend_pre = False
        try:
            deadend_pre = (len(possible_moves) <= 1)
        except Exception:
            deadend_pre = False
        # Debug: trace possible moves only when verbose enabled (or env override)
        try:
            if bool(getattr(config, 'verbose', False)) or os.environ.get('QH_DEBUG_STEP'):
                sys.stdout.flush()
                print(f"[run_episode_query] seed={seed} step={step} pos={current_position} possible_moves={possible_moves}")
        except Exception:
            pass
        if not possible_moves:
            break

        # Sequence ablation: early snapshot (before candidate selection)
        prev_graph_preselect = graph.copy()

        anchor_position = current_position
        if current_query_node not in graph:
            graph.add_node(
                current_query_node,
                abs_vector=compute_query_vector(anchor_position, maze_shape),
                visit_count=visit_counts.get(anchor_position, 0),
                direction=QUERY_MARKER,
                node_type="query",
                anchor_positions=[[anchor_position[0], anchor_position[1]]],
                target_position=[anchor_position[0], anchor_position[1]],
            )
        current_query_entry = graph.nodes[current_query_node]
        if current_query_entry.get("abs_vector") is None:
            current_query_entry["abs_vector"] = compute_query_vector(anchor_position, maze_shape)

        # 0-hop定義の厳密化: 実行経路（Q_prev→dir→Q_now）の強制結線はデフォルトで行わない。
        # 可視化や緩和モードで必要な場合のみ、明示フラグでグラフへ注入する。
        try:
            if getattr(config, 'timeline_to_graph', False) and prev_action_delta is not None:
                dir_idx_prev = direction_from_delta(prev_action_delta)
                if dir_idx_prev is not None:
                    anchor_prev = (int(anchor_position[0] - prev_action_delta[0]), int(anchor_position[1] - prev_action_delta[1]))
                    candidate_vec_abs = compute_episode_vector(
                        base_position=anchor_position,
                        maze_shape=maze_shape,
                        action_delta=prev_action_delta,
                        is_passable=True,
                        visits=visit_counts.get(anchor_position, 0),
                        success=True,
                        is_goal=False,
                        target_position=anchor_position,
                    )
                    direction_node_id = make_direction_node(anchor_prev, int(dir_idx_prev))
                    register_direction_node(
                        anchor_tuple=anchor_prev,
                        dir_idx=int(dir_idx_prev),
                        delta=prev_action_delta,
                        candidate_vec_abs=candidate_vec_abs,
                        target_visits=visit_counts.get(anchor_position, 0),
                        is_passable=True,
                        is_goal=False,
                        target_pos=anchor_position,
                        source="exec",
                        step_index=step,
                    )
                    last_q_node_id = make_query_node(anchor_prev)
                    if last_q_node_id in graph and not graph.has_edge(last_q_node_id, direction_node_id):
                        graph.add_edge(last_q_node_id, direction_node_id)
                    if not graph.has_edge(direction_node_id, current_query_node):
                        graph.add_edge(direction_node_id, current_query_node)
        except Exception:
            pass

        query_vec = compute_query_vector(anchor_position, maze_shape)
        query_vec_list = query_vec.tolist()

        observation_candidates: List[Dict[str, Any]] = []
        memory_candidates: List[Dict[str, Any]] = []
        seen_positions: set[str] = set()
        possible_moves_set = set(possible_moves)

        def candidate_index(anchor: Tuple[int, int], dir_idx: int) -> str:
            return f"{anchor[0]},{anchor[1]},{dir_idx}"

        t_cand_start = time.perf_counter()

        obs_dist_evals_counter = 0
        mem_dist_evals_counter = 0
        ring_cells_counter = 0
        ring_nodes_counter = 0
        ring_max_cells_counter = 0
        ring_fallback_used = False

        for action in SimpleMaze.ACTIONS.keys():
            delta = SimpleMaze.ACTIONS[action]
            dir_idx = direction_from_delta(delta)
            if dir_idx is None:
                continue

            next_pos = (anchor_position[0] + delta[0], anchor_position[1] + delta[1])
            pos_key = candidate_index(anchor_position, dir_idx)
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

            w_distance_rel = weighted_distance(query_vec, candidate_vec_rel)
            w_distance_abs = weighted_distance(query_vec, candidate_vec_abs)
            similarity = math.exp(-w_distance_rel / QUERY_TEMPERATURE)
            obs_dist_evals_counter += 1

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
                    "d_w_rel": w_distance_rel,
                    "origin": "obs",
                    "pos_key": pos_key,
                    # 半径フィルタ用の距離は相対距離ベースに切替（クエリ基準）。
                    # 従来の絶対距離は診断用キーとして残す。
                    "r_abs_cand": w_distance_abs,
                    "r_abs_link": w_distance_abs,
                    "radius_cand": w_distance_rel,
                    "radius_link": w_distance_rel,
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

            register_direction_node(
                anchor_tuple=anchor_position,
                dir_idx=dir_idx,
                delta=delta,
                candidate_vec_abs=candidate_vec_abs,
                target_visits=target_visits,
                is_passable=is_passable,
                is_goal=(next_pos == env.goal_pos),
                target_pos=next_pos,
                source="obs",
                step_index=step,
            )

        # Memory candidates via prefilter
        try:
            r_link_norm = float(config.selector.get("link_radius", config.selector.get("r_link", 0.0)) or 0.0)
        except Exception:
            r_link_norm = 0.0
        Rr = int(math.ceil(r_link_norm * maze_shape[0]))
        Rc = int(math.ceil(r_link_norm * maze_shape[1]))
        Rr = max(0, Rr); Rc = max(0, Rc)

        def iter_indexed_mem_nodes(center: Tuple[int,int]):
            # Fallback to full scan when no index or ring too large
            max_cells = (2*Rr + 1) * (2*Rc + 1)
            nonlocal ring_max_cells_counter, ring_fallback_used, ring_cells_counter, ring_nodes_counter
            ring_max_cells_counter = max_cells
            if (not getattr(config, 'layer1_prefilter', False)) and (sp_index.empty or max_cells <= 0 or max_cells > 10000):
                ring_fallback_used = True
                for nid, data in graph.nodes(data=True):
                    if data.get("node_type") != "direction":
                        continue
                    if data.get("source") == "obs" and data.get("birth_step") == step:
                        continue
                    for anchor in (data.get("anchor_positions") or []):
                        yield canonical_node_id(nid), (int(anchor[0]), int(anchor[1]))
                return
            # Layer1 vector prefilter path (weighted L2 on abs vectors)
            if bool(getattr(config, 'layer1_prefilter', False)):
                try:
                    topk = int(max(1, int(getattr(config, 'l1_cap', 128))))
                except Exception:
                    topk = 128
                # Search by current query's abs vector
                q_abs = compute_query_vector(center, maze_shape)
                # Use candidate radius (relative distance) to bound L1 search
                try:
                    r_cand = float(config.selector.get("cand_radius", 1.0) or 1.0)
                except Exception:
                    r_cand = 1.0
                results = l1_index.search_radius(q_abs, radius=r_cand, top_k=topk)
                ring_cells_counter = 0
                ring_nodes_counter = len(results)
                for nid, dist in results:
                    data = graph.nodes.get(nid, {}) if hasattr(graph, "nodes") else {}
                    if data.get("node_type") != "direction":
                        continue
                    if data.get("source") == "obs" and data.get("birth_step") == step:
                        continue
                    anchor_tuple = (int(nid[0]), int(nid[1]))
                    yield canonical_node_id(nid), anchor_tuple
                return
            # Use spatial index window (ellipse or rectangle) when not using layer1 prefilter
            if bool(getattr(config, 'ring_ellipse', False)):
                collected = list(sp_index.iter_nodes_ellipse(center, Rr, Rc))
                ring_nodes_counter += len(collected)
                cell_set = set((int(n[0]), int(n[1])) for n in collected)
                ring_cells_counter += len(cell_set)
                for nid in collected:
                    data = graph.nodes.get(nid, {}) if hasattr(graph, "nodes") else {}
                    if data.get("node_type") != "direction":
                        continue
                    if data.get("source") == "obs" and data.get("birth_step") == step:
                        continue
                    anchor_tuple = (int(nid[0]), int(nid[1]))
                    yield canonical_node_id(nid), anchor_tuple
            else:
                for rr, cc2, bucket in sp_index.iter_cells_rect(center, Rr, Rc):
                    if not bucket:
                        continue
                    ring_cells_counter += 1
                    ring_nodes_counter += len(bucket)
                    for nid in bucket:
                        data = graph.nodes.get(nid, {}) if hasattr(graph, "nodes") else {}
                        if data.get("node_type") != "direction":
                            continue
                        if data.get("source") == "obs" and data.get("birth_step") == step:
                            continue
                        anchor_tuple = (int(nid[0]), int(nid[1]))
                        yield canonical_node_id(nid), anchor_tuple

        for nid, anchor_tuple in iter_indexed_mem_nodes(anchor_position):
            data = graph.nodes[nid]
            dir_idx = data.get("direction")
            if dir_idx not in DIR_TO_DELTA:
                continue
            pos_key = candidate_index(anchor_tuple, int(dir_idx))
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            rel_delta = tuple(data.get("relative_delta") or delta_from_direction(int(dir_idx)))
            target_tuple = (
                anchor_tuple[0] + int(rel_delta[0]),
                anchor_tuple[1] + int(rel_delta[1]),
            )
            stored_vec_rel = compute_episode_vector(
                base_position=anchor_tuple,
                maze_shape=maze_shape,
                action_delta=None,
                is_passable=data.get("is_passable", True),
                visits=data.get("visit_count", 0),
                success=data.get("success", False),
                is_goal=(target_tuple == env.goal_pos),
                target_position=target_tuple,
            )
            stored_vec_abs = np.asarray(data.get("abs_vector", stored_vec_rel), dtype=float)
            w_distance_rel = weighted_distance(query_vec, stored_vec_rel)
            w_distance_abs = weighted_distance(query_vec, stored_vec_abs)
            similarity = math.exp(-w_distance_rel / QUERY_TEMPERATURE)
            mem_dist_evals_counter += 1

            memory_candidates.append(
                {
                    "index": f"mem:{anchor_tuple[0]},{anchor_tuple[1]},{dir_idx}",
                    "position": [anchor_tuple[0], anchor_tuple[1]],
                    "target_position": [target_tuple[0], target_tuple[1]],
                    "similarity": similarity,
                    "distance": float(w_distance_rel),
                    "weighted_distance": w_distance_rel,
                    "d_w_rel": w_distance_rel,
                    "origin": "mem",
                    "action": None,
                    "action_label": "memory",
                    "pos_key": pos_key,
                    "r_abs_cand": w_distance_abs,
                    "r_abs_link": w_distance_abs,
                    "radius_cand": w_distance_rel,
                    "radius_link": w_distance_rel,
                    "vector": stored_vec_rel.tolist(),
                    "abs_vector": stored_vec_abs.tolist(),
                    "meta_delta": list(rel_delta),
                    "meta_visits": int(data.get("visit_count", 0)),
                    "meta_success": bool(data.get("success", False)),
                    "meta_passable": bool(data.get("is_passable", True)),
                    "anchor_position": [anchor_tuple[0], anchor_tuple[1]],
                    "relative_delta": [int(rel_delta[0]), int(rel_delta[1])],
                    "direction": int(dir_idx),
                }
            )

        candidates: List[Dict[str, Any]] = observation_candidates + memory_candidates
        t_cand_end = time.perf_counter()

        ranked_all_candidates = sorted(
            (dict(item) for item in candidates),
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )

        selection = selector.select(candidates)
        t_cand_done = time.perf_counter()
        cand_node_ids: Set[Tuple[int, int, int]] = set()
        for cand in selection.candidates:
            anchor_source = cand.get("anchor_position") or cand.get("position") or [anchor_position[0], anchor_position[1]]
            anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
            dir_idx = cand.get("direction")
            if dir_idx is None:
                rel_delta = tuple(cand.get("relative_delta") or cand.get("meta_delta") or (0, 0))
                dir_idx = direction_from_delta(rel_delta)
            if dir_idx is None:
                continue
            cand_node_ids.add(make_direction_node(anchor_tuple, int(dir_idx)))
        forced_links = [dict(item) for item in (getattr(selection, "forced_links", []) or [])]
        for forced in forced_links:
            anchor_source = forced.get("anchor_position") or forced.get("position") or [anchor_position[0], anchor_position[1]]
            anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
            dir_idx = forced.get("direction")
            if dir_idx is None:
                rel_delta = tuple(forced.get("relative_delta") or forced.get("meta_delta") or (0, 0))
                dir_idx = direction_from_delta(rel_delta)
            if dir_idx is None:
                continue
            cand_node_ids.add(make_direction_node(anchor_tuple, int(dir_idx)))
        for entry in forced_links:
            entry.setdefault("forced", True)
        links_all: List[Dict[str, Any]] = list(selection.links) + forced_links
        forced_count = len(forced_links)
        # linkset-size: optionally use forced count when links are empty (forced-as-base)
        if len(selection.links) > 0:
            effective_link_count = len(selection.links)
        else:
            effective_link_count = len(forced_links) if getattr(config, 'link_forced_as_base', False) else 0
        base_count = max(1, effective_link_count) if str(config.norm_base) == "link" else max(1, int(selection.k_star))
        ig_fixed_den = math.log(base_count + 1.0)
        l1_candidates = base_count

        def choose_observation_candidate(collections: Sequence[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
            """Prefer feasible observation candidates; if none, fallback to best overall (incl. mem).

            This avoids returning None at dead-ends (no passable obs), which would
            cause linkset before-set to be empty and ΔH≈0 at hop0.
            """
            # 1) Try passable observation candidates in each collection
            for items in collections:
                if not items:
                    continue
                if getattr(config, 'obs_guard', True):
                    obs_items = [
                        item for item in items
                        if item.get("origin") == "obs"
                        and item.get("action") in possible_moves_set
                        and bool(item.get("passable", True))
                    ]
                else:
                    obs_items = [item for item in items if item.get("origin") == "obs"]
                if not obs_items:
                    continue
                # Anti-backtrack: avoid immediately reversing previous action if alternatives exist
                if getattr(config, 'anti_backtrack', True) and prev_action_delta is not None and len(obs_items) > 1:
                    opp = (-int(prev_action_delta[0]), -int(prev_action_delta[1]))
                    filtered = []
                    for x in obs_items:
                        rd = x.get("relative_delta") or x.get("meta_delta")
                        if isinstance(rd, (list, tuple)) and len(rd) == 2:
                            if (int(rd[0]), int(rd[1])) == opp:
                                continue
                        filtered.append(x)
                    if filtered:
                        obs_items = filtered
                # Action policy: argmax or softmax over similarity
                policy = str(getattr(config, 'action_policy', 'argmax') or 'argmax').lower()
                if policy == 'softmax':
                    tau = float(getattr(config, 'action_temp', 1.0) or 1.0)
                    try:
                        sims = [max(0.0, float(x.get("similarity", 0.0) or 0.0)) for x in obs_items]
                        if all(s <= 1e-12 for s in sims) or not math.isfinite(tau) or tau <= 1e-9:
                            raise ValueError("degenerate sims or tau")
                        weights = [math.exp(s / tau) for s in sims]
                        total = sum(weights)
                        if not (total > 0 and math.isfinite(total)):
                            raise ValueError("invalid weights")
                        r = random.random() * total
                        acc = 0.0
                        for w, item in zip(weights, obs_items):
                            acc += w
                            if r <= acc:
                                return item
                    except Exception:
                        pass  # fallback to argmax
                return max(obs_items, key=lambda entry: float(entry.get("similarity", 0.0)))
            # 2) Fallback: choose the single best candidate (obs or mem) across all
            try:
                flat = [it for items in collections for it in items]
                if flat:
                    return max(flat, key=lambda entry: float(entry.get("similarity", 0.0) or 0.0))
            except Exception:
                pass
            return None

        chosen_obs = choose_observation_candidate(
            [selection.links, selection.candidates, ranked_all_candidates]
        )

        # Baseline snapshot before any provisional wiring (timing ablation)
        if str(getattr(config, 'snapshot_mode', 'after_select')).lower() == 'before_select':
            prev_graph = prev_graph_preselect
        else:
            prev_graph = graph.copy()
        registered_link_positions: Set[Tuple[int, int, int]] = set()
        # Snapshot preselect baseline
        graph_nodes_preselect_snapshot = [[int(n[0]), int(n[1]), int(n[2])] for n in prev_graph_preselect.nodes()]
        graph_edges_preselect_snapshot = [
            [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]] for u, v in prev_graph_preselect.edges()
        ]

        def _is_passable_obs(item: Dict[str, Any]) -> bool:
            if item.get("origin") != "obs":
                return True
            act = item.get("action")
            return (act in possible_moves_set) and bool(item.get("passable", True))

        temp_links_ordered: List[Dict[str, Any]] = [it for it in selection.links if _is_passable_obs(it)]
        temp_links: List[Dict[str, Any]] = []
        commit_only_links = str(getattr(config, 'commit_from', 'link')).lower() == 'link'
        if temp_links_ordered:
            # Allow: 全本自動配線 or Top‑L
            if bool(getattr(config, 'link_autowire_all', False)):
                temp_links = list(temp_links_ordered)
            else:
                # Allow sequence ablation: top_link=0 means "no pre-wire at hop0"
                temp_links = temp_links_ordered[: max(0, int(config.top_link))]
        else:
            # S_link が空のときの挙動を切替
            if not commit_only_links:
                # Fallback: forced_links の Top‑L を暫定『選択』（この段階では配線しない）
                fb = [it for it in forced_links if _is_passable_obs(it)]
                if fb:
                    # For top_link=0, avoid committing fallback; treat as candidates only
                    temp_links = fb[: max(0, int(config.top_link))]
            else:
                # commit_from=link の場合は fallback を配線対象にしない
                temp_links = []

        def _ensure_dir_node_on(target_graph: nx.Graph, anchor_tuple: Tuple[int, int], dir_idx: int, base_item: Dict[str, Any]) -> Tuple[int, int, int]:
            node_id = make_direction_node(anchor_tuple, int(dir_idx))
            if node_id not in target_graph:
                target_graph.add_node(node_id)
            entry = target_graph.nodes[node_id]
            vec_abs = np.asarray(base_item.get("abs_vector") or base_item.get("vector") or np.zeros(8), dtype=float)
            rel = base_item.get("relative_delta") or base_item.get("meta_delta") or [0, 0]
            tgt_pos = base_item.get("target_position") or base_item.get("position") or [anchor_tuple[0], anchor_tuple[1]]
            entry["abs_vector"] = vec_abs
            entry["visit_count"] = int(base_item.get("meta_visits", 0))
            entry["is_goal"] = bool(base_item.get("meta_goal", False))
            entry["is_passable"] = bool(base_item.get("meta_passable", True))
            entry["target_position"] = [int(tgt_pos[0]), int(tgt_pos[1])]
            if isinstance(rel, (list, tuple)) and len(rel) == 2:
                entry["relative_delta"] = [int(rel[0]), int(rel[1])]
            else:
                entry["relative_delta"] = [0, 0]
            entry["direction"] = int(dir_idx)
            entry["node_type"] = "direction"
            anchors = entry.setdefault("anchor_positions", [])
            anc = [int(anchor_tuple[0]), int(anchor_tuple[1])]
            if anc not in anchors:
                anchors.append(anc)
            return node_id

        # Commit rule: only S_link (Top‑L) is committed. No fallback auto-commit.
        cand_edge_store: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, int]] = []
        commit_items: List[Dict[str, Any]] = []
        if temp_links:
            commit_items = list(temp_links)

        # 『配線前』の評価を行うため、一旦コミットは保留し、
        # 評価用にグラフ（ノードのみ必要なら追加）を分岐する。
        # - graph_pre: 配線前（必要なら候補ノードを生成）
        # - graph_after: 配線後（Q↔dir を接続）
        graph_pre = graph.copy()
        # 候補方向ノードは focal_nodes に含める必要があるため、存在しなければ生成する（エッジは張らない）
        pre_anchor_nodes: Set[Tuple[int, int, int]] = {current_query_node}
        try:
            for it in commit_items:
                anchor_source = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
                anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
                dir_idx = it.get("direction")
                if dir_idx is None:
                    rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
                    dir_idx = direction_from_delta(rd)
                if dir_idx is None:
                    continue
                node_id = _ensure_dir_node_on(graph_pre, anchor_tuple, int(dir_idx), it)
                pre_anchor_nodes.add(node_id)
        except Exception:
            pass
        # Anchor expansion: include recent T past Q nodes present in prev_graph
        try:
            for qn in list(recent_q_nodes):
                if qn in prev_graph:
                    pre_anchor_nodes.add(qn)
        except Exception:
            pass

        # Pre-evaluate geDIG before env.step using 『配線前』の状態
        pre_linkset_info = None
        if config.linkset_mode:
            try:
                from qhlib.linkset_adapter import build_linkset_info as _build_ls
            except Exception:
                _build_ls = None  # type: ignore
            # forced-as-base: when S_link empty and forced exists, use forced as base
            if selection.links:
                pre_s_link: List[Dict[str, Any]] = [dict(item) for item in selection.links]
            else:
                pre_s_link = [dict(item) for item in forced_links] if getattr(config, 'link_forced_as_base', False) else []
            base_mode = str(getattr(config, 'linkset_base', 'link')).lower()
            if _build_ls is not None:
                pre_linkset_info = _build_ls(
                    s_link=pre_s_link,
                    candidate_pool=[dict(item) for item in selection.candidates],
                    decision={
                        "origin": chosen_obs.get("origin") if chosen_obs else "fallback",
                        "index": chosen_obs.get("index") if chosen_obs else None,
                        "action": chosen_obs.get("action") if chosen_obs else None,
                        "distance": chosen_obs.get("distance") if chosen_obs else None,
                        "similarity": chosen_obs.get("similarity") if chosen_obs else None,
                    },
                    query_vector=list(query_vec_list),
                    base_mode=base_mode,
                )
                # include positional tags for debug parity
                try:
                    pre_linkset_info["query_entry"]["position"] = [int(anchor_position[0]), int(anchor_position[1])]
                    pre_linkset_info["query_entry"]["target_position"] = [int(anchor_position[0]), int(anchor_position[1])]
                except Exception:
                    pass
            else:
                # Fallback: prior inline construction
                qsim = chosen_obs.get("similarity") if chosen_obs else None
                if not isinstance(qsim, (int, float)) or qsim <= 0:
                    qsim = 1.0
                pre_query_entry = {
                    "index": f"query:{anchor_position[0]},{anchor_position[1]}",
                    "origin": "query",
                    "position": [int(anchor_position[0]), int(anchor_position[1])],
                    "target_position": [int(anchor_position[0]), int(anchor_position[1])],
                    "similarity": float(qsim),
                    "distance": 0.0,
                    "weighted_distance": 0.0,
                    "vector": list(query_vec_list),
                    "abs_vector": list(query_vec_list),
                }
                pre_linkset_info = {
                    "s_link": pre_s_link,
                    "candidate_pool": [dict(item) for item in selection.candidates],
                    "decision": {
                        "origin": chosen_obs.get("origin") if chosen_obs else "fallback",
                        "index": chosen_obs.get("index") if chosen_obs else None,
                        "action": chosen_obs.get("action") if chosen_obs else None,
                        "distance": chosen_obs.get("distance") if chosen_obs else None,
                        "similarity": chosen_obs.get("similarity") if chosen_obs else None,
                    },
                    "query_entry": pre_query_entry,
                    "base_mode": base_mode,
                }

        debug_hop0_pre = {}
        pre_result = None
        pre_hop0 = None
        if bool(getattr(config, 'pre_eval', True)):
            pre_features_prev = build_feature_matrix(prev_graph, cand_node_ids, current_query_node, zero_candidates=True)
            pre_features_now = build_feature_matrix(graph_pre, cand_node_ids, current_query_node, zero_candidates=False)
            pre_result = core.calculate(
                g_prev=prev_graph,
                g_now=graph_pre,
                features_prev=pre_features_prev,
                features_now=pre_features_now,
                k_star=base_count,
                l1_candidates=l1_candidates,
                ig_fixed_den=ig_fixed_den,
                query_vector=query_vec_list,
                linkset_info=pre_linkset_info,
                focal_nodes=set(pre_anchor_nodes),
            )
            pre_hop0 = pre_result.hop_results.get(0) if pre_result.hop_results else None
            # 事前（配線前）評価のデバッグ情報
            try:
                dbg_nodes_pre = sorted([[int(n[0]), int(n[1]), int(n[2])] for n in pre_anchor_nodes])
                sub_b_pre = prev_graph.subgraph(pre_anchor_nodes).copy()
                sub_a_pre = graph_pre.subgraph(pre_anchor_nodes).copy()
                nb_pre = sub_b_pre.number_of_nodes(); eb_pre = sub_b_pre.number_of_edges()
                na_pre = sub_a_pre.number_of_nodes(); ea_pre = sub_a_pre.number_of_edges()
                common_e_pre = sum(1 for u, v in sub_b_pre.edges() if sub_a_pre.has_edge(u, v))
                node_ops_pre = abs(na_pre - nb_pre)
                edge_ops_pre = (eb_pre - common_e_pre) + (ea_pre - common_e_pre)
                raw_ged_pre = float(node_ops_pre + edge_ops_pre)
                den_pre = float(1.0 + ea_pre)
                norm_ged_pre = (raw_ged_pre / den_pre) if den_pre > 0 else 0.0
                # 配線前はGEDを評価対象から外す（ログ上は0として扱う）。IG/SPは記録。
                debug_hop0_pre = {
                    'anchors': dbg_nodes_pre,
                    'nodes_before': nb_pre, 'edges_before': eb_pre,
                    'nodes_after': na_pre, 'edges_after': ea_pre,
                    'raw_ged_est': raw_ged_pre, 'norm_den_est': den_pre,
                    'norm_ged_est': norm_ged_pre,
                    'ged': 0.0,
                    'ig': float(pre_hop0.ig) if pre_hop0 else 0.0,
                    'sp': float(pre_hop0.sp) if pre_hop0 else 0.0,
                    'gedig': float(pre_result.gedig_value) if pre_hop0 else 0.0,
                }
            except Exception:
                debug_hop0_pre = {}

        # 『配線評価用（未コミット）』: 評価専用に Q↔dir を接続して g_now を構成
        eval_after = graph_pre.copy()
        for link_item in commit_items:
            anchor_source = link_item.get("anchor_position") or link_item.get("position") or [anchor_position[0], anchor_position[1]]
            anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
            dir_idx = link_item.get("direction")
            if dir_idx is None:
                rel_delta = tuple(link_item.get("relative_delta") or link_item.get("meta_delta") or (0, 0))
                dir_idx = direction_from_delta(rel_delta)
            if dir_idx is None:
                continue
            node_id = _ensure_dir_node_on(eval_after, anchor_tuple, int(dir_idx), link_item)
            if not eval_after.has_edge(current_query_node, node_id):
                eval_after.add_edge(current_query_node, node_id)
            registered_link_positions.add(node_id)

        # eval-only forced fallback: if S_link commit is empty and forced exists, overlay Top-L forced Q↔dir on eval graph
        graph_eval = eval_after.copy()
        try:
            if not commit_items and forced_links:
                # pick Top-L forced by similarity
                forced_sorted = sorted(forced_links, key=lambda it: float(it.get('similarity', 0.0)), reverse=True)
                for it in forced_sorted[: max(0, int(getattr(config, 'top_link', 1)) )]:
                    anchor_src = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
                    at = (int(anchor_src[0]), int(anchor_src[1]))
                    d = it.get("direction")
                    if d is None:
                        rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
                        d = direction_from_delta(rd)
                    if d is None:
                        continue
                    nid = _ensure_dir_node_on(graph_eval, at, int(d), it)
                    if not graph_eval.has_edge(current_query_node, nid):
                        graph_eval.add_edge(current_query_node, nid)
        except Exception:
            pass

        # 『配線後』の評価
        t_eval_start = time.perf_counter()
        # Preserve pre-step query node for logging/visualization
        query_node_pre = current_query_node
        features_prev = build_feature_matrix(prev_graph, cand_node_ids, current_query_node, zero_candidates=True)
        features_now = build_feature_matrix(eval_after, cand_node_ids, current_query_node, zero_candidates=False)
        anchor_nodes = set(pre_anchor_nodes)
        # 段階的な仮想コミットで after_graph_h を構築し、g(h) を評価
        # 準備: base after_graph (h=0) は eval_after（Top‑L S_link を接続済み：評価用）
        stage_graph = eval_after.copy()
        stage_anchor_nodes = set(anchor_nodes)

        # Before/After の評価サブグラフを "二重アンカーのユニオン" で切り出す
        # A_core: 現在Q（配線済みの直前エピソード）
        anchors_core = {current_query_node}
        # A_topL: 0hopで選択したTop‑L候補（dir/過去Q）。
        # - Before(prev) 側には存在しない場合があるので、孤立ノードとして一時追加（案A）
        anchors_top_after: Set[Tuple[int,int,int]] = set()
        anchors_top_before: Set[Tuple[int,int,int]] = set()
        g_before_for_expansion = prev_graph.copy()
        for it in commit_items:
            anchor_source = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
            at = (int(anchor_source[0]), int(anchor_source[1]))
            d = it.get("direction")
            if d is None:
                rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
                d = direction_from_delta(rd)
            if d is None:
                continue
            # after側のアンカー
            nid_after = _ensure_dir_node_on(stage_graph, at, int(d), it)
            anchors_top_after.add(nid_after)
            # before側のアンカー（孤立で種だけ作る）
            nid_before = make_direction_node(at, int(d))
            if nid_before not in g_before_for_expansion:
                g_before_for_expansion.add_node(nid_before)
            anchors_top_before.add(nid_before)

        def _union_khop_subgraph(graph_obj: nx.Graph,
                                  anchors_a: Set[Tuple[int,int,int]],
                                  anchors_b: Set[Tuple[int,int,int]],
                                  hop_k: int) -> nx.Graph:
            g1, n1 = core._extract_k_hop_subgraph(graph_obj, anchors_a, hop_k)
            g2, n2 = core._extract_k_hop_subgraph(graph_obj, anchors_b, hop_k)
            all_nodes = set(n1) | set(n2)
            sub = graph_obj.subgraph(all_nodes).copy()
            return sub

        def _sp_gain_fixed_pairs_ex(sub_before: nx.Graph, sub_after: nx.Graph, eff_hop: int = 1) -> Tuple[float, float, float]:
            """Compute relative SP gain along with before/after ASPL levels using the SAME pair set
            as SP計算（fixed-before-pairs）で使われるもの。

            Returns: (delta_sp_rel, Lb, La)
            """
            try:
                # Align scope to core settings
                scope = str(core.sp_scope_mode).lower()
                bound = str(core.sp_boundary_mode).lower()
                g1 = sub_before
                g2 = sub_after
                if scope in ("union", "merge", "superset"):
                    all_nodes = set(g1.nodes()) | set(g2.nodes())
                    g1 = sub_before.subgraph(all_nodes).copy()
                    g2 = sub_after.subgraph(all_nodes).copy()
                if bound in ("trim", "terminal", "nodes"):
                    g1 = core._trim_terminal_edges(g1, anchors_core, max(1, int(eff_hop)))
                    g2 = core._trim_terminal_edges(g2, anchors_core, max(1, int(eff_hop)))

                # Use DistanceCache fixed-before pairs to get the SAME pair set as SP
                try:
                    sig = distcache.signature(g1, anchors_core, max(1, int(eff_hop)), str(core.sp_scope_mode), str(core.sp_boundary_mode))
                    pairs_obj = distcache.get_fixed_pairs(sig, g1)
                    pairs = list(pairs_obj.pairs)
                    # Lb as average baseline from pairs_obj
                    Lb = float(pairs_obj.lb_avg) if getattr(pairs_obj, 'lb_avg', None) is not None else 0.0
                except Exception:
                    pairs = []
                    Lb = 0.0
                if not pairs or Lb <= 0.0:
                    return 0.0, 0.0, 0.0

                # La on g2 measured over the same pairs
                dist2 = dict(nx.all_pairs_shortest_path_length(g2))
                total2 = 0.0
                count2 = 0
                for u, v, _ in pairs:
                    dm = dist2.get(u, {})
                    if v in dm:
                        total2 += float(dm[v])
                        count2 += 1
                La = (total2 / count2) if count2 > 0 else Lb
                gain = Lb - La
                rel = max(-1.0, min(1.0, gain / (Lb if Lb > 0 else 1.0)))
                return float(rel), float(Lb), float(La)
            except Exception:
                return 0.0, 0.0, 0.0

        def _sp_gain_fixed_pairs(sub_before: nx.Graph, sub_after: nx.Graph) -> float:
            rel, _, _ = _sp_gain_fixed_pairs_ex(sub_before, sub_after)
            return rel

        def _sp_gain_allpairs_ex(sub_before: nx.Graph, sub_after: nx.Graph, eff_hop: int = 1) -> Tuple[float, float, float]:
            """Compute ΔSP using ALL-PAIRS average shortest-path (fallback when fixed-pairs miss improvements).

            Returns: (delta_sp_rel, Lb, La)
            """
            try:
                scope = str(core.sp_scope_mode).lower()
                bound = str(core.sp_boundary_mode).lower()
                g1 = sub_before
                g2 = sub_after
                if scope in ("union", "merge", "superset"):
                    all_nodes = set(g1.nodes()) | set(g2.nodes())
                    g1 = sub_before.subgraph(all_nodes).copy()
                    g2 = sub_after.subgraph(all_nodes).copy()
                if bound in ("trim", "terminal", "nodes"):
                    g1 = core._trim_terminal_edges(g1, stage_anchor_nodes, max(1, int(eff_hop)))
                    g2 = core._trim_terminal_edges(g2, stage_anchor_nodes, max(1, int(eff_hop)))
                # Average shortest-path on all unordered pairs
                def _avg_sp(g: nx.Graph) -> float:
                    try:
                        n = g.number_of_nodes()
                        if n < 2:
                            return 0.0
                        total = 0.0
                        cnt = 0
                        for u, dmap in nx.all_pairs_shortest_path_length(g):
                            for v, d in dmap.items():
                                if v == u:
                                    continue
                                if v <= u:
                                    continue
                                total += float(d)
                                cnt += 1
                        return (total / cnt) if cnt > 0 else 0.0
                    except Exception:
                        return 0.0
                Lb = _avg_sp(g1)
                La = _avg_sp(g2)
                if Lb <= 0.0:
                    return 0.0, 0.0, 0.0
                rel = max(0.0, min(1.0, max(0.0, (Lb - La)) / Lb))
                return float(rel), float(Lb), float(La)
            except Exception:
                return 0.0, 0.0, 0.0

        # IG は linkset に基づき一定（paper-mode）。以降は SP で差をつける
        # linkset IG を pre_result から流用（なければ通常IG）。pre-eval無効時は0。
        base_ig = 0.0
        if pre_result is not None:
            if getattr(pre_result, 'linkset_metrics', None) is not None:
                base_ig = float(pre_result.linkset_metrics.delta_h_norm)
            else:
                base_ig = float(getattr(pre_result, 'delta_h_norm', 0.0))

        # GED の分母（候補台固定: 1 + |S_link| を採用）
        denom_cmax_base = float(core.node_cost + core.edge_cost * max(1, int(base_count)))

        # 候補エッジ集合（Ecand）: S_cand 由来を基本とし、方向ノードを保証
        def _edge_from_item(item: Dict[str, Any]) -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
            anchor_src = item.get("anchor_position") or item.get("position") or [anchor_position[0], anchor_position[1]]
            at = (int(anchor_src[0]), int(anchor_src[1]))
            d = item.get("direction")
            if d is None:
                rd = tuple(item.get("relative_delta") or item.get("meta_delta") or (0, 0))
                d = direction_from_delta(rd)
            if d is None:
                return None
            nid = _ensure_dir_node_on(stage_graph, at, int(d), item)
            return canonical_edge_id(current_query_node, nid)

        # Optional weighted-norm prefilter for Ecand (reuse Layer1 index results)
        prefilter_nodes: Optional[Set[Tuple[int,int,int]]] = None
        try:
            if bool(getattr(config, 'layer1_prefilter', False)):
                try:
                    topk_ec = int(max(1, int(getattr(config, 'l1_cap', 128))))
                except Exception:
                    topk_ec = 128
                try:
                    r_ec = float(config.selector.get("cand_radius", 1.0) or 1.0)
                except Exception:
                    r_ec = 1.0
                q_abs_vec = compute_query_vector(anchor_position, maze_shape)
                l1_hits = l1_index.search_radius(q_abs_vec, radius=r_ec, top_k=topk_ec)
                prefilter_nodes = set(nid for (nid, _d) in l1_hits)
        except Exception:
            prefilter_nodes = None

        ecand, ecand_mem_count, ecand_qpast_count = build_ecand(
            prev_graph=prev_graph,
            selection_candidates=selection.candidates,
            current_query_node=current_query_node,
            anchor_position=anchor_position,
            cap_topk=int(getattr(config, 'sp_cand_topk', 0)),
            include_qpast=True,
            ring_center=anchor_position,
            ring_size=(Rr, Rc),
            ellipse=bool(getattr(config, 'ring_ellipse', False)),
            prefilter_nodes=prefilter_nodes,
        )

        # Ensure forced fallback candidates are also evaluated for ΔSP when S_link is empty
        try:
            if (not selection.links) and getattr(config, 'link_forced_as_base', True) and forced_links:
                existing_e: Set[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = set(
                    canonical_edge_id(u, v) for (u, v, _m) in ecand
                )
                for forced in forced_links:
                    # Consider only mem-origin forced entries by default
                    if str(forced.get("origin", "")).lower() != "mem":
                        continue
                    anchor_src = forced.get("anchor_position") or forced.get("position") or [anchor_position[0], anchor_position[1]]
                    at = (int(anchor_src[0]), int(anchor_src[1]))
                    d = forced.get("direction")
                    if d is None:
                        rd = tuple(forced.get("relative_delta") or forced.get("meta_delta") or (0, 0))
                        d = direction_from_delta(rd)
                    if d is None:
                        continue
                    nid_prev = make_direction_node(at, int(d))
                    if nid_prev not in prev_graph:
                        # Require existence on prev_graph so Lb>0 is meaningful
                        continue
                    e_forced = canonical_edge_id(current_query_node, nid_prev)
                    if e_forced in existing_e:
                        continue
                    meta = dict(forced)
                    meta.setdefault("forced", True)
                    ecand.append((e_forced[0], e_forced[1], meta))
                    existing_e.add(e_forced)
        except Exception:
            pass

        # δe（SP利得の限界寄与）: DistanceCache 経由で計算（core|cached）。
        def _sp_gain_rel(g_before: nx.Graph, g_after: nx.Graph, eff_hop: int, e_u=None, e_v=None) -> float:
            sp_g1, nodes_sp1 = core._extract_k_hop_subgraph(g_before, stage_anchor_nodes, eff_hop)
            sp_g2, nodes_sp2 = core._extract_k_hop_subgraph(g_after, stage_anchor_nodes, eff_hop)
            if str(core.sp_scope_mode).lower() in ("union","merge","superset"):
                all_nodes = set(nodes_sp1) | set(nodes_sp2)
                if all_nodes:
                    sp_g1 = g_before.subgraph(all_nodes).copy()
                    sp_g2 = g_after.subgraph(all_nodes).copy()
            if str(core.sp_boundary_mode).lower() in ("trim","terminal","nodes"):
                sp_g1 = core._trim_terminal_edges(sp_g1, stage_anchor_nodes, eff_hop)
                sp_g2 = core._trim_terminal_edges(sp_g2, stage_anchor_nodes, eff_hop)

            # coreモード: Coreに委譲
            if not getattr(config, 'sp_cache', False) or str(getattr(config, 'sp_cache_mode', 'core')) == 'core' or e_u is None or e_v is None:
                try:
                    return float(core._compute_sp_gain_norm(sp_g1, sp_g2, mode=core.sp_norm_mode))
                except Exception:
                    return 0.0
            # cachedモード: before固定ペア＋SSSP合成
            try:
                sig = distcache.signature(sp_g1, stage_anchor_nodes, eff_hop, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                pairs = distcache.get_fixed_pairs(sig, sp_g1)
                return float(distcache.estimate_sp_cached(sig=sig, g_before=sp_g1, pairs=pairs, endpoint_u=e_u, endpoint_v=e_v))
            except Exception:
                return 0.0

        max_h = int(config.gedig["max_hops"]) if int(config.gedig["max_hops"]) > 0 else 0
        h_best = 0
        g_best = None
        records_h: List[Tuple[int,float,float,float,float]] = []  # (h, g, ged, ig, sp)
        hop_extras: Dict[int, Dict[str, float]] = {}
        chosen_edges_by_hop: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = []

        # h=0 評価（追加なし）: 二重アンカーのユニオンでサブグラフを切る
        sub_b0 = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, 0)
        sub_a0 = _union_khop_subgraph(stage_graph, anchors_core, anchors_top_after, 0)
        # DS距離辞書（ds/ds_exact）は撤回済み（辞書前準備は行わない）
        # GED は Cmax=c_node + |S_link|·c_edge（候補台固定, hop に依らず）で正規化し、1.0 にクリップ
        ged0 = float(_norm_ged(sub_b0, sub_a0, node_cost=core.node_cost, edge_cost=core.edge_cost,
                               normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                               enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                               norm_override=denom_cmax_base)["normalized_ged"]) if denom_cmax_base > 0 else 0.0
        ged0 = float(min(1.0, max(0.0, ged0)))
        sp0 = 0.0
        # Compute SP levels at hop0 neighborhood for display (Lb/La)
        try:
            eff_hop_eval0 = 1
            sub_b0 = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop_eval0))
            sub_a0 = _union_khop_subgraph(stage_graph, anchors_core, anchors_top_after, max(1, eff_hop_eval0))
            sp0_rel, sp0_before, sp0_after = _sp_gain_fixed_pairs_ex(sub_b0, sub_a0, eff_hop=eff_hop_eval0)
        except Exception:
            sp0_rel, sp0_before, sp0_after = (0.0, 0.0, 0.0)
        const_ig0 = base_ig + core.sp_beta * sp0_rel
        g0_stage = float(ged0 - core.lambda_weight * const_ig0)
        records_h.append((0, g0_stage, ged0, const_ig0, sp0_rel))
        # Ensure SP_after reflects the ΔSP reported: infer La_eff = Lb*(1-ΔSP)
        sp0_after_eff = float(max(0.0, sp0_before * (1.0 - sp0_rel)))
        hop_extras[0] = {"sp_before": float(sp0_before), "sp_after": float(sp0_after_eff), "h": float(base_ig)}
        g_best = g0_stage; h_best = 0

        # h>=1 の貪欲構築
        used_edges: Set[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = set()
        # 初期に既にあるエッジは使用済みに
        for u,v in stage_graph.edges():
            used_edges.add(canonical_edge_id(u,v))

        h_graph = stage_graph.copy()
        sp_diagnostics: List[Dict[str, Any]] = []
        gh_mode = str(getattr(config, 'gh_mode', 'greedy')).lower()
        if gh_mode == 'radius':
            # Radius-only: do not add edges per hop; just expand evaluation neighborhood
            for h in range(1, max_h + 1):
                eff_hop_eval = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                sub_b = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop_eval))
                sub_a = _union_khop_subgraph(h_graph, anchors_core, anchors_top_after, max(1, eff_hop_eval))
                ged_h = float(_norm_ged(sub_b, sub_a, node_cost=core.node_cost, edge_cost=core.edge_cost,
                                        normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                                        enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                                        norm_override=denom_cmax_base)["normalized_ged"]) if denom_cmax_base > 0 else 0.0
                ged_h = float(min(1.0, max(0.0, ged_h)))
                sp_h_rel, sp_bef, sp_aft = _sp_gain_fixed_pairs_ex(sub_b, sub_a, eff_hop=eff_hop_eval)
                # Fallback: if compressive structure (added edges exceed nodes) OR no fixed-pair gain, recompute by ALL-PAIRS
                add_nodes = max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes())
                add_edges = max(0, sub_a.number_of_edges() - sub_b.number_of_edges())
                if (add_edges > add_nodes) or (sp_h_rel <= 0.0):
                    sp_h_rel2, sp_bef2, sp_aft2 = _sp_gain_allpairs_ex(sub_b, sub_a, eff_hop=eff_hop_eval)
                    # Prefer the recomputed ALL-PAIRS result
                    sp_h_rel, sp_bef, sp_aft = sp_h_rel2, sp_bef2, sp_aft2
                # IG per-hop
                ig_h_val = base_ig
                try:
                    if str(core.ig_source_mode).lower() in ("linkset","paper","strict") and str(core.ig_hop_apply).lower() == "all" and pre_result is not None:
                        ls_h = core._compute_linkset_metrics(prev_graph, h_graph, pre_linkset_info, query_vector=query_vec_list, ig_fixed_den=ig_fixed_den)
                        ig_h_val = float(ls_h.delta_h_norm)
                except Exception:
                    ig_h_val = base_ig
                ig_h = ig_h_val + core.sp_beta * sp_h_rel
                g_h = float(ged_h - core.lambda_weight * ig_h)
                records_h.append((h, g_h, ged_h, ig_h, sp_h_rel))
                sp_aft_eff = float(max(0.0, sp_bef * (1.0 - sp_h_rel)))
                hop_extras[h] = {"sp_before": float(sp_bef), "sp_after": float(sp_aft_eff), "h": float(ig_h_val)}
                if g_best is None or g_h < g_best:
                    g_best = g_h; h_best = h
        else:
            for h in range(1, max_h + 1):
                # 最良 δe を探す
                best_delta = 0.0
                best_item = None
                # Build before subgraph for SP diagnostics (fixed-pair set)
                eff_h_diag = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                sp_before_g, sp_nodes = core._extract_k_hop_subgraph(prev_graph, stage_anchor_nodes, max(1, eff_h_diag))
                try:
                    sig_diag = distcache.signature(sp_before_g, stage_anchor_nodes, max(1, eff_h_diag), str(core.sp_scope_mode), str(core.sp_boundary_mode))
                    pairs_diag = distcache.get_fixed_pairs(sig_diag, sp_before_g)
                    pair_cnt = int(len(pairs_diag.pairs))
                    lb_avg = float(pairs_diag.lb_avg)
                except Exception:
                    sig_diag = ""
                    pair_cnt = 0
                    lb_avg = 0.0
                    pairs_diag = None
                cand_scores: List[Tuple[float, Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = []
                for e_u, e_v, meta in ecand:
                    if (e_u, e_v) in used_edges:
                        continue
                    g_try = h_graph.copy()
                    if not g_try.has_node(e_u): g_try.add_node(e_u)
                    if not g_try.has_node(e_v): g_try.add_node(e_v)
                    if not g_try.has_edge(e_u, e_v): g_try.add_edge(e_u, e_v)
                    # eff_hop を拡張（SP評価の近傍）
                    eff_hop = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                    de = _sp_gain_rel(prev_graph, g_try, max(1, eff_hop), e_u, e_v)
                    cand_scores.append((float(de), (e_u, e_v)))
                    if de > best_delta:
                        best_delta = de; best_item = (e_u, e_v, meta)
                # Diagnostics: top-5 by δSP
                try:
                    top = sorted(cand_scores, key=lambda x: x[0], reverse=True)[:5]
                    top_payload = [
                        {
                            "delta_sp": float(sc),
                            "u": [int(a[0]), int(a[1]), int(a[2])],
                            "v": [int(b[0]), int(b[1]), int(b[2])],
                        }
                        for sc, (a, b) in top
                    ]
                except Exception:
                    top_payload = []
                sp_diagnostics.append({
                    "hop": int(h),
                    "pair_count": int(pair_cnt),
                    "lb_avg": float(lb_avg),
                    "top": top_payload,
                })
                if best_item is None or best_delta <= 0.0:
                    # 無採用: 設定に応じて継続評価のために最良（δSP最大、0でも可）候補を1本だけ仮追加
                    if getattr(config, 'eval_all_hops', False) and cand_scores:
                        # pick top even if <=0
                        cand_scores.sort(key=lambda x: x[0], reverse=True)
                        (de_top, (eu_top, ev_top)) = cand_scores[0]
                        if (eu_top, ev_top) not in used_edges:
                            if not h_graph.has_node(eu_top): h_graph.add_node(eu_top)
                            if not h_graph.has_node(ev_top): h_graph.add_node(ev_top)
                            if not h_graph.has_edge(eu_top, ev_top): h_graph.add_edge(eu_top, ev_top)
                            used_edges.add((eu_top, ev_top))
                            stage_anchor_nodes.update([eu_top, ev_top])
                            chosen_edges_by_hop.append((eu_top, ev_top))
                    else:
                        # 無採用でもhopの評価レコードを入れて状況を可視化（ΔSPは0として）
                        eff_hop_eval = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                        sub_b = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop_eval))
                        sub_a = _union_khop_subgraph(h_graph, anchors_core, anchors_top_after, max(1, eff_hop_eval))
                        ged_h = float(_norm_ged(sub_b, sub_a, node_cost=core.node_cost, edge_cost=core.edge_cost,
                                                normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                                                enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                                                norm_override=denom_cmax_base)["normalized_ged"]) if denom_cmax_base > 0 else 0.0
                        ged_h = float(min(1.0, max(0.0, ged_h)))
                        sp_h_rel, sp_bef, sp_aft = _sp_gain_fixed_pairs_ex(sub_b, sub_a)
                        # per-hop linkset ΔH（可能なら）
                        ig_h_val = base_ig
                        try:
                            if str(core.ig_source_mode).lower() in ("linkset","paper","strict") and str(core.ig_hop_apply).lower() == "all" and pre_result is not None:
                                ls_h = core._compute_linkset_metrics(prev_graph, h_graph, pre_linkset_info, query_vector=query_vec_list, ig_fixed_den=ig_fixed_den)
                                ig_h_val = float(ls_h.delta_h_norm)
                        except Exception:
                            ig_h_val = base_ig
                        ig_h = ig_h_val + core.sp_beta * sp_h_rel
                        g_h = float(ged_h - core.lambda_weight * ig_h)
                        records_h.append((h, g_h, ged_h, ig_h, sp_h_rel))
                        hop_extras[h] = {"sp_before": float(sp_bef), "sp_after": float(sp_aft), "h": float(ig_h_val)}
                        # g_best/h_best の更新（無採用でも評価値が改善していれば反映）
                        if g_best is None or g_h < g_best:
                            g_best = g_h; h_best = h
                        # diagnostics-only: continue evaluating remaining hops if eval_all_hops is set; otherwise continue
                        if not bool(getattr(config, 'eval_all_hops', False)):
                            continue
                # 採用して h_graph を更新
                if best_item is not None:
                    e_u, e_v, meta = best_item
                    if not h_graph.has_edge(e_u, e_v):
                        h_graph.add_edge(e_u, e_v)
                    used_edges.add((e_u, e_v))
                    stage_anchor_nodes.update([e_u, e_v])
                    chosen_edges_by_hop.append((e_u, e_v))

                # g(h) を評価
                eff_hop_eval = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                sub_b = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop_eval))
                sub_a = _union_khop_subgraph(h_graph, anchors_core, anchors_top_after, max(1, eff_hop_eval))
                ged_h = float(_norm_ged(sub_b, sub_a, node_cost=core.node_cost, edge_cost=core.edge_cost,
                                        normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                                        enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                                        norm_override=denom_cmax_base)["normalized_ged"]) if denom_cmax_base > 0 else 0.0
                ged_h = float(min(1.0, max(0.0, ged_h)))
                sp_h_rel, sp_bef, sp_aft = _sp_gain_fixed_pairs_ex(sub_b, sub_a, eff_hop=eff_hop_eval)
                # IGをhopごとに再評価（linksetモード時は全hop適用）
                ig_h_val = base_ig
                try:
                    if str(core.ig_source_mode).lower() in ("linkset","paper","strict") and str(core.ig_hop_apply).lower() == "all":
                        # linkset情報はpreのものを使い、g_afterだけ差し替える
                        ls_h = core._compute_linkset_metrics(prev_graph, h_graph, pre_linkset_info, query_vector=query_vec_list, ig_fixed_den=ig_fixed_den)
                        ig_h_val = float(ls_h.delta_h_norm)
                except Exception:
                    ig_h_val = base_ig
                ig_h = ig_h_val + core.sp_beta * sp_h_rel
                g_h = float(ged_h - core.lambda_weight * ig_h)
                records_h.append((h, g_h, ged_h, ig_h, sp_h_rel))
                sp_aft_eff = float(max(0.0, sp_bef * (1.0 - sp_h_rel)))
                hop_extras[h] = {"sp_before": float(sp_bef), "sp_after": float(sp_aft_eff), "h": float(ig_h_val)}
                if g_best is None or g_h < g_best:
                    g_best = g_h; h_best = h

        # 外部評価器による置き換え（安定化のため上書き運用）
        eval_applied = False
        t_eval_ms = 0.0
        try:
            # Dynamic θAG: use percentile of past g0 if enabled and history present
            theta_ag_used = float(getattr(config, 'theta_ag', 0.0))
            if bool(getattr(config, 'ag_auto', False)) and len(g0_history) > 0:
                try:
                    theta_ag_used = float(np.quantile(np.array(g0_history, dtype=float), float(getattr(config, 'ag_quantile', 0.9))))
                except Exception:
                    theta_ag_used = float(getattr(config, 'theta_ag', 0.0))

            t_eval_run_start = time.perf_counter()
            # Optional DS-backed SP pairsets
            sp_svc = None
            sig_builder = None
            try:
                if getattr(config, 'sp_ds_sqlite', None):
                    sp_svc = SQLitePairsetService(str(getattr(config, 'sp_ds_sqlite')), str(getattr(config, 'sp_ds_namespace', 'maze_query_hub_sp')))
                    sig_builder = SignatureBuilder()
            except Exception:
                sp_svc = None; sig_builder = None

            # Optionally skip multi-hop when current state is a dead-end/backtrack (possible moves <=1)
            try:
                deadend_now = (len(possible_moves) <= 1)
            except Exception:
                deadend_now = False
            local_max_hops = int(config.gedig.get("max_hops", 0))
            if bool(getattr(config, 'skip_mh_on_deadend', False)) and deadend_now:
                local_max_hops = 0

            if bool(getattr(config, 'use_main_l3', False)) and str(getattr(config, 'sp_cache_mode', 'core')).lower() in ('cached','cached_incr'):
                # Lightweight path: query-centric eval via main L3 (hop0)
                try:
                    from qhlib.l3_adapter import eval_query_centric_via_l3
                    # Centers: anchors_core (Q nodes)
                    centers_nodes = list(anchors_core)
                    # Candidate edges: Ecand as (u,v,meta)
                    cand_edges = list(ecand)
                    sp_mode = str(getattr(config, 'sp_cache_mode', 'cached_incr'))
                    # Minimal selection summary for ΔH denominator (log k★)
                    _sel_summary = {
                        'k_star': int(base_count),
                        'log_k_star': float(math.log(base_count)) if base_count >= 1 else None,
                        'l1_candidates': int(base_count),
                    }
                    res_l3 = eval_query_centric_via_l3(
                        prev_graph=prev_graph,
                        curr_graph=stage_graph,
                        centers=centers_nodes,
                        cand_edges=cand_edges,
                        selection_summary=_sel_summary,
                        sp_engine=sp_mode,
                        pair_samples=int(getattr(config, 'sp_pair_samples', 200)),
                        budget=int(getattr(config, 'commit_budget', 1)),
                        cand_topk=int(getattr(config, 'sp_cand_topk', 0)),
                        default_dim=8,
                        lambda_weight=float(core.lambda_weight),
                    )
                    m = res_l3.get('metrics', {})
                    g0 = float(m.get('g0', 0.0))
                    gmin = float(m.get('gmin', g0))
                    best_hop = 0
                    delta_ged = float(m.get('delta_ged', 0.0))
                    delta_ig = float(m.get('delta_ig', 0.0))
                    delta_sp = float(m.get('delta_sp', 0.0))
                    # H uses delta_h (after-before; negative when entropy decreases)
                    hop_series = [{
                        "hop": 0,
                        "g": gmin,
                        "ged": float(m.get('delta_ged_norm', abs(delta_ged))),
                        "ig": delta_ig,
                        "h": float(m.get('delta_h', delta_ig)),
                        "sp": delta_sp
                    }]
                    # Enrich hop0 with SP_before/after if provided by L3 (global path)
                    try:
                        sp_b = float(m.get('sp_before', 0.0))
                    except Exception:
                        sp_b = 0.0
                    try:
                        # Prefer L3-provided sp_after (La). If missing, leave 0 and fallback later.
                        sp_a = float(m.get('sp_after', 0.0))
                    except Exception:
                        sp_a = 0.0
                    hop_series[0]['sp_before'] = float(sp_b)
                    hop_series[0]['sp_after'] = float(sp_a)
                    records_h = [(0, gmin, float(m.get('delta_ged_norm', abs(delta_ged))), delta_ig, delta_sp)]
                    chosen_edges_by_hop = []
                    # SP perf counters are not available from L3 in this mode
                    sp_sssp_du_ct = sp_sssp_dv_ct = sp_dv_leaf_skips_ct = sp_cycle_verifies_ct = 0
                    gmin_mh_val = gmin; ged_mh_val = float(m.get('delta_ged_norm', abs(delta_ged))); ig_mh_val = delta_ig; sp_mh_val = delta_sp
                    eval_applied = True
                    # When adaptive union-of-k-hop comparison is desired, fall back to evaluator for per-hop series
                    try:
                        if (os.getenv('INSIGHTSPIKE_SP_ADAPTIVE','').strip().lower() in ('1','true','on','yes')) or (str(core.sp_scope_mode).lower()=='union'):
                            eval_applied = False
                    except Exception:
                        pass
                except Exception as _lite_e:
                    print(f"[WARN] use_main_l3 path failed, falling back to evaluator: {_lite_e}")
                    eval_applied = False

            # Optionally fallback to evaluator when AG fires (per-hop only on-demand)
            try:
                theta_ag_cfg = float(getattr(config, 'theta_ag', 0.0))
            except Exception:
                theta_ag_cfg = 0.0
            ag_wants_hops = bool(getattr(config, 'eval_per_hop_on_ag', False)) and bool(g0 > theta_ag_cfg)
            if (not eval_applied) or bool(getattr(config, 'force_per_hop', False)) or ag_wants_hops:
                eval_res = evaluate_multihop(
            core=core,
            prev_graph=prev_graph,
            stage_graph=stage_graph,
            g_before_for_expansion=g_before_for_expansion,
            anchors_core=anchors_core,
            anchors_top_before=anchors_top_before,
            anchors_top_after=anchors_top_after,
            ecand=ecand,
            base_ig=float(base_ig),
            denom_cmax_base=float(denom_cmax_base),
            max_hops=local_max_hops,
            ged_hop0_const=bool(getattr(config, 'ged_hop0_const', True)),
            ig_recompute=(str(core.ig_source_mode).lower() in ("linkset","paper","strict") and str(core.ig_hop_apply).lower() == "all"),
            pre_linkset_info=pre_linkset_info,
            query_vec=query_vec_list,
            ig_fixed_den=ig_fixed_den,
                theta_ag=float(theta_ag_used),
                theta_dg=float(getattr(config, 'theta_dg', 0.0)),
                eval_all_hops=bool(getattr(config, 'eval_all_hops', False)),
                sp_early_stop=False,
                sp_cache=bool(getattr(config, 'sp_cache', False)),
                sp_cache_mode=str(getattr(config, 'sp_cache_mode', 'core')),
                sp_pair_samples=int(getattr(config, 'sp_pair_samples', 400)),
                sp_verify_threshold=float(getattr(config, 'sp_verify_threshold', 0.05)),
                sp_allpairs=bool(getattr(config, 'sp_eval_allpairs', False)),
                sp_allpairs_exact=bool(getattr(config, 'sp_eval_allpairs_exact', False)),
                apsp_carry=apsp_exact_state,
                sp_exact_stable_nodes=bool(getattr(config, 'sp_exact_stable_nodes', False)),
                sp_signed=bool(getattr(config, 'sp_signed', False)),
                pairset_service=sp_svc,
                signature_builder=sig_builder,
                )
                try:
                    if getattr(eval_res, 'apsp_carry', None) is not None:
                        apsp_exact_state = eval_res.apsp_carry  # type: ignore[assignment]
                except Exception:
                    pass
                hop_series = eval_res.hop_series
                # Optional: enrich hop_series with L3-union cand debug (cand_used/cand_total)
                try:
                    want_union_dbg = (os.getenv('INSIGHTSPIKE_SP_ADAPTIVE','').strip().lower() in ('1','true','on','yes')) or (str(core.sp_scope_mode).lower()=='union')
                except Exception:
                    want_union_dbg = False
                if want_union_dbg:
                    try:
                        from qhlib.l3_adapter import eval_query_centric_via_l3
                        centers_nodes = list(anchors_core)
                        cand_edges = list(ecand)
                        sp_mode = str(getattr(config, 'sp_cache_mode', 'cached_incr'))
                        dbg_res = eval_query_centric_via_l3(
                            prev_graph=prev_graph,
                            curr_graph=stage_graph,
                            centers=centers_nodes,
                            cand_edges=cand_edges,
                            sp_engine=sp_mode,
                            pair_samples=int(getattr(config, 'sp_pair_samples', 200)),
                            budget=int(getattr(config, 'commit_budget', 1)),
                            cand_topk=int(getattr(config, 'sp_cand_topk', 0)),
                            default_dim=8,
                            max_hops=int(core.max_hops) if hasattr(core, 'max_hops') else int(getattr(config.gedig, 'max_hops', 0)),
                        )
                        dbg_map = {}
                        try:
                            dbg_map = (dbg_res.get('metrics', {}) or {}).get('debug', {}).get('union_linkset', {}) or {}
                        except Exception:
                            dbg_map = {}
                        if isinstance(dbg_map, dict) and hop_series:
                            # merge per-hop cand counts if present
                            for rec in hop_series:
                                try:
                                    h = int(rec.get('hop', -1))
                                except Exception:
                                    h = -1
                                if h in dbg_map and isinstance(dbg_map[h], dict):
                                    dh = dbg_map[h]
                                    if 'cand_used' in dh:
                                        rec['cand_used'] = int(dh.get('cand_used', 0))
                                    if 'cand_count_total' in dh:
                                        rec['cand_count_total'] = int(dh.get('cand_count_total', 0))
                    except Exception:
                        pass
                records_h = [(int(rec["hop"]), float(rec["g"]), float(rec["ged"]), float(rec["ig"]), float(rec["sp"])) for rec in hop_series]
                chosen_edges_by_hop = list(eval_res.chosen_edges_by_hop)
                g0 = float(eval_res.g0)
                gmin = float(eval_res.gmin)
                best_hop = int(eval_res.best_hop)
                delta_ged = float(eval_res.delta_ged)
                delta_ig = float(eval_res.delta_ig)
                # Start with hop0 ΔSP; may override after computing best-hop row
                delta_sp = float(eval_res.delta_sp)
                # SP perf counters
                sp_sssp_du_ct = int(getattr(eval_res, 'sssp_calls_du', 0))
                sp_sssp_dv_ct = int(getattr(eval_res, 'sssp_calls_dv', 0))
                sp_dv_leaf_skips_ct = int(getattr(eval_res, 'dv_leaf_skips', 0))
                sp_cycle_verifies_ct = int(getattr(eval_res, 'cycle_verifies', 0))
                # mh-only minima
                gmin_mh_val = float(eval_res.gmin_mh)
                ged_mh_val = float(eval_res.delta_ged_min_mh)
                ig_mh_val = float(eval_res.delta_ig_min_mh)
                sp_mh_val = float(eval_res.delta_sp_min_mh)
                # Precompute cand_used/cand_total per hop from ecand and union-of-k-hop node sets
                cand_stats = {}
                try:
                    def _union_nodes(G, anchors_core, anchors_top, hop: int):
                        sub = _union_khop_subgraph(G, anchors_core, anchors_top, max(1, int(hop)))
                        return set(sub.nodes())
                    # Build quick mapping of chosen edges per hop (if available)
                    chosen_map = {}
                    try:
                        for (eu, ev) in chosen_edges_by_hop:
                            for rec_h in hop_series:
                                h = int(rec_h.get('hop', 0))
                                if h >= 1:
                                    chosen_map.setdefault(h, set()).add((eu, ev))
                    except Exception:
                        chosen_map = {}
                    for rec_h in hop_series:
                        h = int(rec_h.get('hop', 0))
                        try:
                            nodes_h = _union_nodes(g_before_for_expansion, anchors_core, anchors_top_before, h)
                            total = 0; used = 0
                            for (u, v, _meta) in (ecand or []):
                                if (u in nodes_h) and (v in nodes_h):
                                    total += 1
                                    if (h in chosen_map) and (((u, v) in chosen_map[h]) or ((v, u) in chosen_map[h])):
                                        used += 1
                            cand_stats[h] = { 'total': int(total), 'used': int(used) }
                        except Exception:
                            continue
                except Exception:
                    cand_stats = {}
            t_eval_ms = (time.perf_counter() - t_eval_run_start) * 1000.0
            # best-hop deltas for display
            try:
                hrec_obj = next((rec for rec in hop_series if int(rec.get("hop", -1)) == best_hop), None)
                if hrec_obj is not None:
                    delta_ged_min = float(hrec_obj.get("ged", delta_ged))
                    delta_ig_min = float(hrec_obj.get("ig", delta_ig))
                    delta_sp_min = float(hrec_obj.get("sp", delta_sp))
                    # Prefer best-hop ΔSP for reporting when enabled
                    if bool(getattr(config, 'sp_report_best_hop', False)) and best_hop > 0:
                        delta_sp = float(delta_sp_min)
            except Exception:
                pass
            eval_applied = True
        except Exception:
            # fall back to inline results if evaluator fails
            pass

        # 最終値の反映
        if not eval_applied:
            # hop0の delta は records_h[0]
            delta_ged = records_h[0][2]
            delta_ig = records_h[0][3]
            delta_sp = records_h[0][4]
            g0 = records_h[0][1]
            # best hop の値
            best_hop = int(h_best)
            gmin = g_best if g_best is not None else g0
        # 再計算（検証用）
        try:
            gvals = [(h, float(g)) for (h, g, _, _, _) in records_h]
            h_calc, gmin_calc = min(gvals, key=lambda t: t[1]) if gvals else (0, g0)
        except Exception:
            h_calc, gmin_calc = (best_hop, gmin)
        # multi-hop only minima（hop>=1）
        if not eval_applied:
            try:
                gvals_mh_full = [(h, float(g), float(ged), float(ig), float(sp)) for (h, g, ged, ig, sp) in records_h]
                gvals_mh = [t for t in gvals_mh_full if int(t[0]) >= 1]
                if gvals_mh:
                    h_mh, gmin_mh_val, ged_mh_val, ig_mh_val, sp_mh_val = min(gvals_mh, key=lambda t: t[1])
                else:
                    h_mh, gmin_mh_val, ged_mh_val, ig_mh_val, sp_mh_val = (0, g0, delta_ged, delta_ig, delta_sp)
            except Exception:
                h_mh, gmin_mh_val, ged_mh_val, ig_mh_val, sp_mh_val = (0, g0, delta_ged, delta_ig, delta_sp)
            hrec = next((rec for rec in records_h if rec[0]==best_hop), records_h[0])
            delta_ged_min = hrec[2]
            delta_ig_min = hrec[3]
            delta_sp_min = hrec[4]
        # H (after-before) を A/B でも entropy_ig ベースで統一（L3 と同源）
        # Prefer L3-provided ΔH when use_main_l3 経路が適用された場合（union/L3ベースの比較）
        try:
            if eval_applied and isinstance(m, dict) and ('delta_h' in m):
                delta_h = float(m.get('delta_h', 0.0))
                delta_h_min = delta_h
            else:
                feats_b = gather_node_features(graph_pre)
                feats_a = gather_node_features(graph_eval)
                if feats_b.size and feats_b.shape[1] == WEIGHT_VECTOR.size:
                    feats_b = feats_b * WEIGHT_VECTOR
                if feats_a.size and feats_a.shape[1] == WEIGHT_VECTOR.size:
                    feats_a = feats_a * WEIGHT_VECTOR
                ab_h_dict = _entropy_ig(graph_eval, feats_b, feats_a, delta_mode='after_before')
                ab_delta_h = float(ab_h_dict.get('ig_value', 0.0))
                delta_h = ab_delta_h
                delta_h_min = ab_delta_h
        except Exception:
            delta_h = float(base_ig)
            delta_h_min = delta_h
        # 序列をJSON化
        # 優先: evaluator が返した per-hop 'h'（ΔH: after-before）をそのまま使用。
        # フォールバック: hop_extras['h'] → delta_h（pre-evalのΔH）。
        _orig_hs_by_hop = {}
        try:
            if isinstance(hop_series, list):
                for _rec in hop_series:
                    try:
                        _orig_hs_by_hop[int(_rec.get('hop', -1))] = dict(_rec)
                    except Exception:
                        pass
        except Exception:
            pass
        hop_series = []
        for rec_h in records_h:
            h, g, ged, ig, sp = rec_h
            extra = hop_extras.get(int(h), {})
            if int(h) in _orig_hs_by_hop and isinstance(_orig_hs_by_hop[int(h)], dict):
                dh = float(_orig_hs_by_hop[int(h)].get('h', delta_h))
            else:
                dh = float(extra.get("h", delta_h))
            sp_bef = extra.get("sp_before")
            sp_aft = extra.get("sp_after")
            row = {"hop": int(h), "g": float(g), "ged": float(ged), "ig": float(ig), "h": float(dh), "sp": float(sp)}
            # Carry through H_before/H_after if provided by evaluator
            try:
                if int(h) in _orig_hs_by_hop:
                    o = _orig_hs_by_hop[int(h)]
                    if 'h_before' in o:
                        row['h_before'] = float(o.get('h_before'))
                    if 'h_after' in o:
                        row['h_after'] = float(o.get('h_after'))
                    # Also carry evaluator SP diagnostics when present
                    if 'sp_mode' in o:
                        row['sp_mode'] = str(o.get('sp_mode'))
                    if 'sp_Lb' in o:
                        row['sp_Lb'] = float(o.get('sp_Lb'))
                    if 'sp_La' in o:
                        row['sp_La'] = float(o.get('sp_La'))
                    if 'sp_pairs' in o:
                        try:
                            row['sp_pairs'] = int(o.get('sp_pairs'))
                        except Exception:
                            pass
                    if 'sp_improved' in o:
                        try:
                            row['sp_improved'] = int(o.get('sp_improved'))
                        except Exception:
                            pass
                    if 'sp_examples' in o:
                        try:
                            row['sp_examples'] = o.get('sp_examples')
                        except Exception:
                            pass
                    if 'add_nodes' in o:
                        try:
                            row['add_nodes'] = int(o.get('add_nodes'))
                        except Exception:
                            pass
                    if 'add_edges' in o:
                        try:
                            row['add_edges'] = int(o.get('add_edges'))
                        except Exception:
                            pass
                    if 'sp_improved' in o:
                        try:
                            row['sp_improved'] = int(o.get('sp_improved'))
                        except Exception:
                            pass
                    if 'sp_examples' in o:
                        try:
                            row['sp_examples'] = o.get('sp_examples')
                        except Exception:
                            pass
            except Exception:
                pass
            if sp_bef is not None:
                row["sp_before"] = float(sp_bef)
            if sp_aft is not None:
                # Show raw shortest-path averages: Lb (before), La (after)
                row["sp_after"] = float(sp_aft)
            try:
                if 'cand_stats' in locals():
                    cs = cand_stats.get(int(h))
                    if cs:
                        row['cand_count_total'] = int(cs.get('total', 0))
                        row['cand_used'] = int(cs.get('used', 0))
            except Exception:
                pass
            hop_series.append(row)
        # Ensure hop0 carries SP_before/after (evaluate_multihop 経由では未設定な場合がある)
        try:
            if hop_series and (hop_series[0].get('sp_before') is None or hop_series[0].get('sp_after') is None):
                eff_hop_eval0 = 1
                sub_b0 = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop_eval0))
                sub_a0 = _union_khop_subgraph(stage_graph, anchors_core, anchors_top_after, max(1, eff_hop_eval0))
                sp0_rel_fix, sp0_bef_fix, sp0_aft_fix = _sp_gain_fixed_pairs_ex(sub_b0, sub_a0, eff_hop=eff_hop_eval0)
                hop_series[0]['sp_before'] = float(sp0_bef_fix)
                hop_series[0]['sp_after'] = float(sp0_aft_fix)
        except Exception:
            pass

        # Snapshot the evaluation graph (pre-step, after commit; before env.step) — uses eval overlay graph
        if str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal':
            graph_nodes_eval_snapshot = []
            graph_edges_eval_snapshot = []
        else:
            graph_nodes_eval_snapshot = [[int(node[0]), int(node[1]), int(node[2])] for node in graph_eval.nodes()]
            graph_edges_eval_snapshot = [
                [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]] for u, v in graph_eval.edges()
            ]
        # Snapshot graph_pre (before commit)
        if str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal':
            graph_nodes_pre_snapshot = []
            graph_edges_pre_snapshot = []
        else:
            graph_nodes_pre_snapshot = [[int(node[0]), int(node[1]), int(node[2])] for node in graph_pre.nodes()]
            graph_edges_pre_snapshot = [
                [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]] for u, v in graph_pre.edges()
            ]
        structural_cost_val = float(delta_ged)
        structural_impr_val = float(-delta_ged)

        # Build debug payload for hop0 post-eval
        try:
            dbg_nodes_post = sorted([[int(n[0]), int(n[1]), int(n[2])] for n in anchor_nodes])
            sub_b_post = prev_graph.subgraph(anchor_nodes).copy()
            sub_a_post = graph_eval.subgraph(anchor_nodes).copy()
            nb_post = sub_b_post.number_of_nodes(); eb_post = sub_b_post.number_of_edges()
            na_post = sub_a_post.number_of_nodes(); ea_post = sub_a_post.number_of_edges()
            common_e_post = sum(1 for u, v in sub_b_post.edges() if sub_a_post.has_edge(u, v))
            node_ops_post = abs(na_post - nb_post)
            edge_ops_post = (eb_post - common_e_post) + (ea_post - common_e_post)
            raw_ged_post = float(node_ops_post + edge_ops_post)
            den_post = float(1.0 + ea_post)
            norm_ged_post = (raw_ged_post / den_post) if den_post > 0 else 0.0
            debug_hop0_post = {
                'anchors': dbg_nodes_post,
                'nodes_before': nb_post, 'edges_before': eb_post,
                'nodes_after': na_post, 'edges_after': ea_post,
                'raw_ged_est': raw_ged_post, 'norm_den_est': den_post,
                'norm_ged_est': norm_ged_post,
                'ged': float(hop0.ged) if hop0 else 0.0,
                'ig': float(hop0.ig) if hop0 else 0.0,
                'sp': float(hop0.sp) if hop0 else 0.0,
                'gedig': float(hop0.gedig) if hop0 else float(result.gedig_value),
            }
        except Exception:
            debug_hop0_post = {}
        if str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal':
            debug_hop0 = {}
            hop_series = []
            sp_diagnostics = []

        # DG実コミット（貪欲）: geDIG計算終了後に、best_hop までの仮想採用セットを実コミット
        ag_fire = bool(g0 > float(locals().get('theta_ag_used', getattr(config, 'theta_ag', 0.0))))
        # DG判定は multi-hop のみ: best_hop>=1 かつ gmin_mh < θDG
        dg_fire = bool(int(best_hop) >= 1 and float(gmin_mh_val) < float(config.theta_dg))
        dg_commit_policy = str(getattr(config, 'dg_commit_policy', 'threshold')).lower()

        # Option: when DG fires, expand hop0 commit set to all S_link items (passable only).
        if dg_fire and bool(getattr(config, 'dg_commit_all_linkset', False)):
            try:
                def _is_passable_obs(item: Dict[str, Any]) -> bool:
                    if item.get("origin") != "obs":
                        return True
                    act = item.get("action")
                    return (act in possible_moves_set) and bool(item.get("passable", True))
                # Replace commit_items (Top-L) with all S_link (passable)
                commit_items = [it for it in selection.links if _is_passable_obs(it)]
            except Exception:
                pass
        graph_commit, dg_committed_edges_snapshot = apply_commit_policy(
            base_graph=graph,
            current_query_node=current_query_node,
            hop0_items=commit_items,
            chosen_edges_by_hop=chosen_edges_by_hop,
            best_hop=int(best_hop),
            policy=dg_commit_policy,
            fire_dg=bool(dg_fire),
            commit_budget=int(getattr(config, 'commit_budget', 0)),
            bfs_shortcut=bool(getattr(config, 'dg_bfs_shortcut', False)),
            cand_edge_store=cand_edge_store,
        )
        # 選択された配線を実体の graph にコミット（評価専用のeval_afterとは分離）
        graph = graph_commit
        graph_temp = graph_commit

        if chosen_obs is not None and chosen_obs.get("action") is not None:
            action = int(chosen_obs["action"]) 
        else:
            # Fallback policy with anti-backtrack masking if possible
            pm = list(possible_moves)
            if getattr(config, 'anti_backtrack', True) and prev_action_delta is not None and len(pm) > 1:
                opp = (-int(prev_action_delta[0]), -int(prev_action_delta[1]))
                # Map deltas back to action ids
                delta_to_action = {v:k for k,v in SimpleMaze.ACTIONS.items()}
                opp_action = delta_to_action.get(opp)
                if opp_action in pm:
                    pm2 = [x for x in pm if x != opp_action]
                    if pm2:
                        pm = pm2
            action = random.choice(pm)

        last_query_node = current_query_node
        last_position = current_position
        action_delta = SimpleMaze.ACTIONS[action]
        action_dir_idx = direction_from_delta(action_delta)

        obs, reward, done, _ = env.step(action)
        next_position_raw = obs.position
        current_position = (int(next_position_raw[0]), int(next_position_raw[1]))
        moved = (current_position != last_position)
        current_query_node = make_query_node(current_position)
        visit_counts[current_position] = visit_counts.get(current_position, 0) + 1
        # Post-step possible moves (for consistent UI/debug)
        try:
            possible_moves_post = list(getattr(obs, 'possible_moves', []))
        except Exception:
            possible_moves_post = []

        # Build timeline edges for DS/visualization
        timeline_edges_now: List[List[List[int]]] = []
        if getattr(config, 'add_next_q', False):
            if current_query_node not in graph:
                graph.add_node(
                    current_query_node,
                    abs_vector=compute_query_vector(current_position, maze_shape),
                    visit_count=0,
                    direction=QUERY_MARKER,
                    node_type="query",
                    anchor_positions=[[current_position[0], current_position[1]]],
                    target_position=[current_position[0], current_position[1]],
                )
        if action_dir_idx is not None:
            direction_node = make_direction_node(last_position, action_dir_idx)
            register_direction_node(
                anchor_tuple=last_position,
                dir_idx=action_dir_idx,
                delta=action_delta,
                candidate_vec_abs=compute_episode_vector(
                    base_position=current_position,
                    maze_shape=maze_shape,
                    action_delta=action_delta,
                    is_passable=bool(moved),
                    visits=visit_counts[current_position],
                    success=bool(moved),
                    is_goal=obs.is_goal,
                    target_position=current_position,
                ),
                target_visits=visit_counts[current_position],
                is_passable=bool(moved),
                is_goal=obs.is_goal,
                target_pos=current_position,
                source="exec",
                step_index=step,
            )
            if getattr(config, 'timeline_to_graph', False):
                if not graph.has_edge(last_query_node, direction_node):
                    graph.add_edge(last_query_node, direction_node)
                if getattr(config, 'add_next_q', False):
                    if not graph.has_edge(direction_node, current_query_node):
                        graph.add_edge(direction_node, current_query_node)
            timeline_edges_now.extend(
                build_current_step_timeline(
                    last_query_node,
                    direction_node,
                    current_query_node,
                    include_dir_to_next=True,
                    include_pair=False,
                )
            )
        # 明示的に Q↔次Q を時系列エッジとして接続（可視化と追跡用）
        if getattr(config, 'timeline_pair', False):
            # optional Q_prev <-> Q_next pair edge (disabled by default)
            if getattr(config, 'timeline_to_graph', False) and getattr(config, 'add_next_q', False):
                if last_query_node in graph and current_query_node in graph and not graph.has_edge(last_query_node, current_query_node):
                    graph.add_edge(last_query_node, current_query_node)
            # Suppress Q_prev <-> Q_next timeline pair in this configuration

        # After env.step: compute post-step per-hop diagnostics (SP with executed action)
        hop_series_post = []
        _need_post = bool(getattr(config, 'post_sp_diagnostics', True)) and (
            (len(locals().get('dg_committed_edges_snapshot', []) or []) > 0) or bool(getattr(config, 'timeline_to_graph', False))
        )
        if _need_post:
            try:
                post_stage_anchors: Set[Tuple[int,int,int]] = set()
                if last_query_node in graph:
                    post_stage_anchors.add(last_query_node)
                if current_query_node in graph:
                    post_stage_anchors.add(current_query_node)
                # also include committed dir nodes around current Q
                for it in commit_items:
                    anchor_source = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
                    anchor_tuple = (int(anchor_source[0]), int(anchor_source[1]))
                    d = it.get("direction")
                    if d is None:
                        rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
                        d = direction_from_delta(rd)
                    if d is not None:
                        nid = make_direction_node(anchor_tuple, int(d))
                        if nid in graph:
                            post_stage_anchors.add(nid)
                # include recent Q anchors
                for qn in list(recent_q_nodes):
                    if qn in graph:
                        post_stage_anchors.add(qn)
                denom_cmax_post = float(core.node_cost + core.edge_cost * max(1, int(base_count)))
                max_h_post = int(config.gedig["max_hops"]) if int(config.gedig["max_hops"]) > 0 else 0
                hop_series_post = []
                for h in range(0, max_h_post + 1):
                    eff_h = h + int(max(0, int(config.gedig.get("sp_hop_expand", 0))))
                    # Compute subgraphs and SP stats under core scope/boundary rules
                    sub_b = prev_graph.subgraph(post_stage_anchors).copy()
                    sub_a = graph.subgraph(post_stage_anchors).copy()
                    sp_g1, nodes_sp1 = core._extract_k_hop_subgraph(sub_b, post_stage_anchors, max(1, eff_h))
                    sp_g2, nodes_sp2 = core._extract_k_hop_subgraph(sub_a, post_stage_anchors, max(1, eff_h))
                    if str(core.sp_scope_mode).lower() in ("union","merge","superset"):
                        all_nodes = set(nodes_sp1) | set(nodes_sp2)
                        if all_nodes:
                            sp_g1 = sub_b.subgraph(all_nodes).copy()
                            sp_g2 = sub_a.subgraph(all_nodes).copy()
                    if str(core.sp_boundary_mode).lower() in ("trim","terminal","nodes"):
                        sp_g1 = core._trim_terminal_edges(sp_g1, post_stage_anchors, max(1, eff_h))
                        sp_g2 = core._trim_terminal_edges(sp_g2, post_stage_anchors, max(1, eff_h))
                    # Use DistanceCache fixed-before pairs to evaluate SP efficiently
                    try:
                        sigp = distcache.signature(sp_g1, post_stage_anchors, max(1, eff_h), str(core.sp_scope_mode), str(core.sp_boundary_mode))
                        pairs_obj = distcache.get_fixed_pairs(sigp, sp_g1)
                        pairs = list(getattr(pairs_obj, 'pairs', []) or [])
                        Lb = float(getattr(pairs_obj, 'lb_avg', 0.0) or 0.0)
                    except Exception:
                        pairs = []
                        Lb = 0.0
                    La = 0.0
                    nb_pairs = len(pairs)
                    na_pairs = 0
                    if Lb > 0.0 and pairs:
                        total2 = 0.0
                        count2 = 0
                        for u, v, _ in pairs:
                            try:
                                d2 = nx.shortest_path_length(sp_g2, source=u, target=v)
                                total2 += float(d2)
                                count2 += 1
                            except Exception:
                                pass
                        na_pairs = count2
                        La = (total2 / count2) if count2 > 0 else Lb
                    sp_rel = 0.0
                    if Lb > 0.0:
                        sp_rel = max(0.0, min(1.0, max(0.0, Lb - La) / Lb))
                    ged_h = float(_norm_ged(sub_b, sub_a, node_cost=core.node_cost, edge_cost=core.edge_cost,
                                            normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                                            enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                                            norm_override=denom_cmax_post)["normalized_ged"]) if denom_cmax_post > 0 else 0.0
                    ig_h = base_ig + core.sp_beta * sp_rel
                    g_h = float(ged_h - core.lambda_weight * ig_h)
                    hop_series_post.append({
                        "hop": int(h), "g": float(g_h), "ged": float(ged_h), "ig": float(ig_h), "h": float(base_ig), "sp": float(sp_rel),
                        "Lb": float(Lb), "La": float(La), "pairs_before": int(nb_pairs), "pairs_after": int(na_pairs)
                    })
            except Exception:
                hop_series_post = []

        if last_query_node in graph:
            graph.nodes[last_query_node]["visit_count"] = visit_counts[last_position]
        if current_query_node in graph:
            graph.nodes[current_query_node]["visit_count"] = visit_counts[current_position]

        # Persist SP pairsets for "next-step BEFORE" reuse (includes dir->Q_now timeline edge)
        sp_ds_eff_saved_ct = 0
        try:
            if getattr(config, 'sp_ds_sqlite', None):
                # Build DS services on demand
                sp_svc_post = SQLitePairsetService(str(getattr(config, 'sp_ds_sqlite')), str(getattr(config, 'sp_ds_namespace', 'maze_query_hub_sp')))
                sig_builder_post = SignatureBuilder()
                # Anchors for next step: current Q only (matches evaluator signature)
                anchors_next: Set[Tuple[int,int,int]] = {current_query_node}
                # Evaluate limited eff-hops for persistence to keep cost bounded
                try:
                    h_max_conf = int(config.gedig.get("max_hops", 0))
                except Exception:
                    h_max_conf = 0
                try:
                    sp_expand = int(config.gedig.get("sp_hop_expand", 0))
                except Exception:
                    sp_expand = 0
                H_eval_save = max(1, min(h_max_conf + max(0, sp_expand), 4))  # persist up to 4 eff-hops by default
                # Use a local DistanceCache sampler
                from insightspike.algorithms.sp_distcache import DistanceCache as _DC
                _dc = _DC(mode="cached", pair_samples=int(getattr(config, 'sp_pair_samples', 400)))
                for he in range(1, H_eval_save + 1):
                    # k-hop neighborhood around current Q on post-step graph
                    sp_g2, nodes2 = core._extract_k_hop_subgraph(graph, anchors_next, max(1, he))
                    sig2, meta2 = sig_builder_post.for_subgraph(sp_g2, list(anchors_next), he, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                    ps2 = _dc.get_fixed_pairs(sig2, sp_g2)
                    pr2 = [
                        {
                            'u_id': f"{u[0]},{u[1]},{u[2]}",
                            'v_id': f"{v[0]},{v[1]},{v[2]}",
                            'd_before': float(d)
                        }
                        for (u, v, d) in ps2.pairs
                    ]
                    # Upsert pairset into DS
                    sp_svc_post.upsert(
                        Pairset(
                            signature=sig2,
                            lb_avg=float(ps2.lb_avg),
                            pairs=[PairRecord(**rec) for rec in pr2],
                            node_count=int(sp_g2.number_of_nodes()),
                            edge_count=int(sp_g2.number_of_edges()),
                            scope=str(core.sp_scope_mode),
                            boundary=str(core.sp_boundary_mode),
                            eff_hop=int(he),
                            meta=meta2,
                        )
                    )
                    sp_ds_eff_saved_ct += 1
        except Exception:
            # Pairset DS persistence is best-effort; ignore failures
            pass

        prev_action_delta = action_delta
        prev_success = moved

        if obs.is_goal and done:
            success = True

        counts = {
            "obs_total": len(observation_candidates),
            "mem_total": len(memory_candidates),
            "cand_obs": sum(1 for item in selection.candidates if item.get("origin") == "obs"),
            "cand_mem": sum(1 for item in selection.candidates if item.get("origin") == "mem"),
            "link_obs": sum(1 for item in links_all if item.get("origin") == "obs"),
            "link_mem": sum(1 for item in links_all if item.get("origin") == "mem"),
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

        linkset_info = None
        if config.linkset_mode:
            query_similarity = decision_info.get("similarity")
            if not isinstance(query_similarity, (int, float)) or query_similarity <= 0:
                query_similarity = 1.0
            query_entry = {
                "index": f"query:{last_position[0]},{last_position[1]}",
                "origin": "query",
                "position": [int(last_position[0]), int(last_position[1])],
                "target_position": [int(last_position[0]), int(last_position[1])],
                "similarity": float(query_similarity),
                "distance": 0.0,
                "weighted_distance": 0.0,
                "vector": list(query_vec_list),
                "abs_vector": list(query_vec_list),
            }
            if selection.links:
                s_link_payload: List[Dict[str, Any]] = [dict(item) for item in selection.links]
            else:
                s_link_payload = [dict(item) for item in forced_links] if getattr(config, 'link_forced_as_base', False) else []
            linkset_info = {
                "s_link": s_link_payload,
                "candidate_pool": [dict(item) for item in selection.candidates],
                "decision": dict(decision_info),
                "query_entry": query_entry,
                "base_mode": str(getattr(config, 'linkset_base', 'link')).lower(),
            }
        # Compute linkset metrics against the post-commit graphs for logging
        linkset_delta_ged = 0.0
        linkset_delta_h = 0.0
        linkset_delta_sp = 0.0
        linkset_g = 0.0
        if config.linkset_mode and linkset_info is not None:
            try:
                ls = core._compute_linkset_metrics(prev_graph, graph_eval, linkset_info, query_vector=query_vec_list, ig_fixed_den=ig_fixed_den)
                linkset_delta_ged = float(ls.delta_ged_norm)
                linkset_delta_h = float(ls.delta_h_norm)
                linkset_delta_sp = float(ls.delta_sp_rel)
                linkset_g = float(ls.gedig_value)
                try:
                    linkset_entropy_before = float(ls.entropy_before)
                    linkset_entropy_after = float(ls.entropy_after)
                except Exception:
                    linkset_entropy_before = 0.0
                    linkset_entropy_after = 0.0
                try:
                    linkset_pos_w_before = int(getattr(ls, 'pos_w_before', 0))
                    linkset_pos_w_after = int(getattr(ls, 'pos_w_after', 0))
                    linkset_topw_before = list(getattr(ls, 'topw_before', []) or [])[:5]
                    linkset_topw_after = list(getattr(ls, 'topw_after', []) or [])[:5]
                except Exception:
                    linkset_pos_w_before = 0
                    linkset_pos_w_after = 0
                    linkset_topw_before = []
                    linkset_topw_after = []
            except Exception:
                linkset_delta_ged = linkset_delta_h = linkset_delta_sp = linkset_g = 0.0
                linkset_entropy_before = 0.0
                linkset_entropy_after = 0.0
                linkset_pos_w_before = 0
                linkset_pos_w_after = 0
                linkset_topw_before = []
                linkset_topw_after = []
        else:
            linkset_entropy_before = 0.0
            linkset_entropy_after = 0.0
            linkset_pos_w_before = 0
            linkset_pos_w_after = 0
            linkset_topw_before = []
            linkset_topw_after = []

        if False:
            features_prev = build_feature_matrix(prev_graph, cand_node_ids, current_query_node, zero_candidates=True)
            features_now = build_feature_matrix(graph_temp, cand_node_ids, current_query_node, zero_candidates=False)
            anchor_nodes = {current_query_node}
            result = core.calculate(
                g_prev=prev_graph,
                g_now=graph_temp,
                features_prev=features_prev,
                features_now=features_now,
                k_star=base_count,
                l1_candidates=l1_candidates,
                ig_fixed_den=ig_fixed_den,
                query_vector=query_vec_list,
                linkset_info=linkset_info,
                focal_nodes=set(anchor_nodes),
            )

        # Gate states (logged for analysis). Commit is not performed here;
        # selected_links reflects current link set unless a later commit stage is added.
        ag_fire = bool(g0 > float(config.theta_ag))
        dg_fire = bool(gmin < float(config.theta_dg))
        committed_links: List[Dict[str, Any]] = list(links_all)

        link_count = effective_link_count
        candidate_selection = {
            "k_star": float(base_count),
            "linkset_size": float(link_count),
            "linked": float(link_count),  # backward compatibility
            "theta_cand": config.selector["theta_cand"],
            "theta_link": config.selector["theta_link"],
            "k_cap": config.selector["candidate_cap"],
            "top_m": config.selector["top_m"],
            "log_k_star": float(math.log(base_count + 1.0)) if base_count >= 1 else None,
            "r_cand": config.selector.get("theta_cand"),
            "r_link": config.selector.get("theta_link"),
            "radius_cand": config.selector.get("cand_radius"),
            "radius_link": config.selector.get("link_radius"),
            "counts": counts,
            "decision": decision_info,
            "linkset_registered_positions": [
                [int(pos[0]), int(pos[1]), int(pos[2])] for pos in sorted(registered_link_positions)
            ],
            "link_registered_positions": [
                [int(pos[0]), int(pos[1]), int(pos[2])] for pos in sorted(registered_link_positions)
            ],
            "forced_links": [dict(item) for item in forced_links],
        }

        for entry in candidate_selection["forced_links"]:
            if isinstance(entry.get("vector"), np.ndarray):
                entry["vector"] = entry["vector"].tolist()
            if isinstance(entry.get("abs_vector"), np.ndarray):
                entry["abs_vector"] = entry["abs_vector"].tolist()

        if str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal':
            graph_nodes_snapshot = []
            graph_edges_snapshot = []
        else:
            graph_nodes_snapshot = [[int(node[0]), int(node[1]), int(node[2])] for node in graph.nodes()]
            graph_edges_snapshot = [
                [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]] for u, v in graph.edges()
            ]
        graph_nodes_post_snapshot = graph_nodes_snapshot
        graph_edges_post_snapshot = graph_edges_snapshot
        # Compute committed-only edges (new edges compared to baseline at step start)
        def _edge_set(g):
            return set(tuple(sorted(((int(a[0]),int(a[1]),int(a[2])),(int(b[0]),int(b[1]),int(b[2]))))) for a,b in g.edges())
        # Use preselect snapshot as baseline so that early-step edges (e.g., lastQ->dir->currentQ) are included
        diff_baseline_graph = prev_graph_preselect if 'prev_graph_preselect' in locals() else prev_graph
        prev_set = _edge_set(diff_baseline_graph)
        now_set = _edge_set(graph)
        new_edges = sorted(now_set - prev_set)
        committed_only_edges_snapshot = [
            [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]] for (u, v) in new_edges
        ]
        # Build metadata for committed-only edges using cand_edge_store when available
        def _norm_edge(a, b):
            ua = (int(a[0]), int(a[1]), int(a[2])); vb = (int(b[0]), int(b[1]), int(b[2]))
            return tuple(sorted((ua, vb)))
        cand_edge_meta: Dict[Tuple[Tuple[int,int,int], Tuple[Tuple[int,int,int]]], Dict[str, Any]] = {}
        try:
            for ea, eb, forced_flag, bridge_flag in cand_edge_store:
                key = _norm_edge(ea, eb)
                cand_edge_meta[key] = {
                    "forced": bool(forced_flag),
                    "stage": ("dg" if int(bridge_flag) == 1 else "base"),
                }
        except Exception:
            cand_edge_meta = {}
        committed_only_edges_meta_snapshot: List[Dict[str, Any]] = []
        for (u, v) in new_edges:
            key = _norm_edge(u, v)
            meta = cand_edge_meta.get(key, {"forced": False, "stage": "unknown"})
            committed_only_edges_meta_snapshot.append({
                "nodes": [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]],
                "forced": bool(meta.get("forced", False)),
                "stage": str(meta.get("stage", "unknown")),
                "step": int(step),
            })
        # Compute committed-only nodes (new nodes compared to baseline at step start)
        prev_nodes_set = set((int(n[0]), int(n[1]), int(n[2])) for n in diff_baseline_graph.nodes())
        now_nodes_set = set((int(n[0]), int(n[1]), int(n[2])) for n in graph.nodes())
        new_nodes = sorted(now_nodes_set - prev_nodes_set)
        if str(getattr(config, 'snapshot_level', 'standard')).lower() == 'minimal':
            committed_only_nodes_snapshot = []
            committed_only_nodes_meta_snapshot = []
        else:
            committed_only_nodes_snapshot = [[int(n[0]), int(n[1]), int(n[2])] for n in new_nodes]
            committed_only_nodes_meta_snapshot: List[Dict[str, Any]] = []
            for n in new_nodes:
                try:
                    data = graph.nodes[n]
                except Exception:
                    data = {}
                committed_only_nodes_meta_snapshot.append({
                    "node": [int(n[0]), int(n[1]), int(n[2])],
                    "node_type": data.get("node_type"),
                    "direction": int(data.get("direction", -99)) if data.get("direction") is not None else None,
                    "source": data.get("source"),
                    "visit_count": int(data.get("visit_count", 0)),
                    "anchor_positions": data.get("anchor_positions"),
                    "target_position": data.get("target_position"),
                    "birth_step": int(data.get("birth_step", step)),
                    "step": int(step),
                })
        cand_edges_snapshot = []
        if str(getattr(config, 'snapshot_level', 'standard')).lower() != 'minimal':
            seen_edges: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, int]] = set()
            for edge_a, edge_b, forced_flag, bridge_flag in cand_edge_store:
                key = (edge_a, edge_b, bool(forced_flag), int(bridge_flag))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                cand_edges_snapshot.append({
                    "nodes": [
                        [int(edge_a[0]), int(edge_a[1]), int(edge_a[2])],
                        [int(edge_b[0]), int(edge_b[1]), int(edge_b[2])],
                    ],
                    "forced": bool(forced_flag),
                    "bridge": bool(bridge_flag),
                })

        def _normalise_container(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                        derived = direction_from_delta((int(rd[0]), int(rd[1])))
                        if derived is not None:
                            item["direction"] = int(derived)
                            dir_idx = int(derived)
                if isinstance(dir_idx, (int, np.integer)):
                    item["direction_label"] = DIR_LABELS.get(int(dir_idx), "?")
                if "meta_visits" in item and isinstance(item["meta_visits"], (int, float)):
                    item["visit"] = int(item["meta_visits"])
                elif "visit_count" in item and isinstance(item["visit_count"], (int, float)):
                    item["visit"] = int(item["visit_count"])
                normalised.append(item)
            return normalised

        def _item_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
            target = item.get("target_position") or item.get("targetPosition") or []
            if isinstance(target, list):
                target = tuple(int(v) for v in target)
            return (
                item.get("origin"),
                item.get("index"),
                item.get("direction"),
                target,
            )

        def _merge_unique(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            for raw in items:
                item = dict(raw)
                key = _item_key(item)
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

        normalised_candidates = _normalise_container(selection.candidates)
        normalised_links = _normalise_container(committed_links)
        normalised_forced = _normalise_container(candidate_selection["forced_links"])

        forced_marked = [dict(entry, forced=True) for entry in normalised_forced]
        candidate_pool = _merge_unique(normalised_candidates + forced_marked)
        selected_links = _merge_unique(normalised_links + forced_marked)
        ranked_candidates = _merge_unique(_normalise_container(ranked_all_candidates))
        for entry in normalised_forced:
            entry["forced"] = True
        candidate_selection["forced_links"] = normalised_forced

        new_edge: List[List[int]] = []

        # Post-step SP diagnostics: compute after env.step with action edges included later in the loop
        hop_series_post: List[Dict[str, Any]] = []

        current_episode_vec = compute_episode_vector(
            base_position=current_position,
            maze_shape=maze_shape,
            action_delta=prev_action_delta,
            is_passable=True,
            visits=visit_counts[current_position],
            success=bool(moved),
            is_goal=obs.is_goal,
            target_position=current_position,
        )

        # Debug payload: both pre and post evaluations captured above
        debug_hop0 = {"pre": debug_hop0_pre, "post": debug_hop0_post}

        # Build forced edges (Q↔dir for forced_links) for logging/persistence
        forced_edges_snapshot: List[List[List[int]]] = []
        forced_edges_meta_snapshot: List[Dict[str, Any]] = []
        try:
            for it in candidate_selection["forced_links"]:
                anchor_source = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
                at = (int(anchor_source[0]), int(anchor_source[1]))
                d = it.get("direction")
                if d is None:
                    rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
                    d = direction_from_delta(rd)
                if d is None:
                    continue
                dir_node = make_direction_node(at, int(d))
                forced_edges_snapshot.append([[int(current_query_node[0]), int(current_query_node[1]), int(current_query_node[2])], [int(dir_node[0]), int(dir_node[1]), int(dir_node[2])]])
                forced_edges_meta_snapshot.append({
                    "nodes": forced_edges_snapshot[-1],
                    "forced": True,
                    "stage": "forced_candidate",
                    "step": int(step),
                })
        except Exception:
            pass

        # Optional: persist diffs (nodes/edges) to SQLite for later replay/analysis
        ds_nodes_saved_snapshot: List[Dict[str, Any]] = []
        ds_edges_saved_snapshot: List[Dict[str, Any]] = []
        ds_nodes_total_snapshot = 0
        ds_edges_total_snapshot = 0
        try:
            if getattr(config, 'persist_sqlite_path', None):
                store = SQLiteStore(db_path=str(config.persist_sqlite_path), namespace=str(getattr(config, 'persist_namespace', 'maze_query_hub')))
                # Nodes and edges committed this step
                ds_nodes_saved_snapshot = store.insert_nodes(committed_only_nodes_meta_snapshot)
                ds_edges_saved_snapshot = store.insert_edges(committed_only_edges_meta_snapshot, edge_type_fallback='auto')
                # Forced candidates and timeline edges
                if getattr(config, 'persist_forced_candidates', False):
                    ds_edges_saved_snapshot += store.insert_forced_edges(forced_edges_meta_snapshot)
                if getattr(config, 'persist_timeline_edges', True):
                    # Ensure nodes for timeline endpoints are present in DS
                    timeline_nodes_meta: List[Dict[str, Any]] = []
                    seen_ids = set()
                    for e in (timeline_edges_now or []):
                        for n in e:
                            nid = f"{int(n[0])},{int(n[1])},{int(n[2])}"
                            if nid in seen_ids:
                                continue
                            seen_ids.add(nid)
                            node_type = 'query' if int(n[2]) == -1 else 'direction'
                            # next-step queryノードは step+1 で登録し、現ステップでは出さない
                            n_step = int(step + 1) if (int(n[2]) == -1 and int(n[0]) == int(current_query_node[0]) and int(n[1]) == int(current_query_node[1])) else int(step)
                            timeline_nodes_meta.append({
                                'node': [int(n[0]), int(n[1]), int(n[2])],
                                'node_type': node_type,
                                'stage': 'timeline',
                                'step': n_step,
                            })
                    if timeline_nodes_meta:
                        store.insert_nodes(timeline_nodes_meta)
                    # Save timeline edges with per-edge step: dir->Q_now は step+1 に
                    timeline_edges_meta: List[Dict[str, Any]] = []
                    for e in (timeline_edges_now or []):
                        u, v = e
                        # if target is next Q (dir->Q_now), shift to next step
                        e_step = int(step + 1) if int(v[2]) == -1 else int(step)
                        timeline_edges_meta.append({
                            'nodes': [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]],
                            'stage': 'timeline',
                            'step': e_step,
                        })
                    if timeline_edges_meta:
                        ds_edges_saved_snapshot += store.insert_edges(timeline_edges_meta, edge_type_fallback='timeline')
                # Totals
                ds_nodes_total_snapshot, ds_edges_total_snapshot = store.totals()
        except Exception:
            pass

        # Update episode-local DS accumulators regardless of SQLite persistence
        try:
            for n in committed_only_nodes_snapshot:
                ds_nodes_accum.add((int(n[0]), int(n[1]), int(n[2])))
            for e in committed_only_edges_snapshot:
                u = (int(e[0][0]), int(e[0][1]), int(e[0][2])); v = (int(e[1][0]), int(e[1][1]), int(e[1][2]))
                key = tuple(sorted((u, v)))
                ds_edges_accum.add(key)
            if getattr(config, 'persist_forced_candidates', False):
                for e in forced_edges_snapshot:
                    u = (int(e[0][0]), int(e[0][1]), int(e[0][2])); v = (int(e[1][0]), int(e[1][1]), int(e[1][2]))
                    key = tuple(sorted((u, v)))
                    ds_edges_accum.add(key)
                    ds_nodes_accum.add(u); ds_nodes_accum.add(v)
            if getattr(config, 'persist_timeline_edges', True):
                for e in (timeline_edges_now or []):
                    if not (isinstance(e, list) and len(e) == 2):
                        continue
                    u = (int(e[0][0]), int(e[0][1]), int(e[0][2])); v = (int(e[1][0]), int(e[1][1]), int(e[1][2]))
                    key = tuple(sorted((u, v)))
                    ds_edges_accum.add(key)
                    ds_nodes_accum.add(u); ds_nodes_accum.add(v)
        except Exception:
            pass

        ds_graph_nodes_snapshot = [[int(n[0]), int(n[1]), int(n[2])] for n in sorted(ds_nodes_accum)]
        ds_graph_edges_snapshot = [
            [[int(u[0]), int(u[1]), int(u[2])], [int(v[0]), int(v[1]), int(v[2])]]
            for (u, v) in sorted(ds_edges_accum)
        ]

        cand_time_ms = (t_cand_done - t_cand_start) * 1000.0
        # Trim heavy arrays for logging (keep top-N ranked for UI even in minimal mode)
        ranked_for_log = ranked_candidates[: min(32, len(ranked_candidates))]
        selected_for_log = selected_links[: min(16, len(selected_links))]

        # Update dynamic AG history after evaluating this step
        try:
            g0_history.append(float(g0))
        except Exception:
            pass

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
                delta_sp=float(delta_sp),
                delta_sp_min=float(delta_sp_min),
                delta_h=float(delta_h),
                delta_h_min=float(delta_h_min),
                structural_cost=float(structural_cost_val),
                structural_improvement=float(structural_impr_val),
                # profiling
                ring_Rr=int(Rr),
                ring_Rc=int(Rc),
                ring_max_cells=int(ring_max_cells_counter),
                ring_cells=int(ring_cells_counter),
                ring_nodes=int(ring_nodes_counter),
                ring_fallback=bool(ring_fallback_used),
                obs_dist_evals=int(obs_dist_evals_counter),
                mem_dist_evals=int(mem_dist_evals_counter),
                total_dist_evals=int(obs_dist_evals_counter + mem_dist_evals_counter),
                time_ms_candidates=float(cand_time_ms),
                time_ms_eval=float(t_eval_ms),
                sp_sssp_du=int(locals().get('sp_sssp_du_ct', 0)),
                sp_sssp_dv=int(locals().get('sp_sssp_dv_ct', 0)),
                sp_dv_leaf_skips=int(locals().get('sp_dv_leaf_skips_ct', 0)),
                sp_cycle_verifies=int(locals().get('sp_cycle_verifies_ct', 0)),
                linkset_delta_ged=float(linkset_delta_ged),
                linkset_delta_h=float(linkset_delta_h),
                linkset_delta_sp=float(linkset_delta_sp),
                linkset_g=float(linkset_g),
                linkset_entropy_before=float(linkset_entropy_before),
                linkset_entropy_after=float(linkset_entropy_after),
                linkset_pos_w_before=int(linkset_pos_w_before),
                linkset_pos_w_after=int(linkset_pos_w_after),
                linkset_topw_before=list(linkset_topw_before),
                linkset_topw_after=list(linkset_topw_after),
                g0=float(g0),
                gmin=float(gmin),
                best_hop=best_hop,
                is_dead_end=bool(obs.is_dead_end),
                is_dead_end_pre=bool(deadend_pre),
                reward=float(reward),
                done=bool(done),
                possible_moves=list(possible_moves),
                possible_moves_post=list(possible_moves_post),
                candidate_pool=(candidate_pool if not minimal_snap else []),
                selected_links=selected_for_log,
                ranked_candidates=ranked_for_log,
                graph_nodes=graph_nodes_snapshot,
                graph_edges=graph_edges_snapshot,
                graph_nodes_eval=graph_nodes_eval_snapshot,
                graph_edges_eval=graph_edges_eval_snapshot,
                graph_nodes_preselect=graph_nodes_preselect_snapshot,
                graph_edges_preselect=graph_edges_preselect_snapshot,
                graph_nodes_pre=graph_nodes_pre_snapshot,
                graph_edges_pre=graph_edges_pre_snapshot,
                graph_nodes_post=graph_nodes_post_snapshot,
                graph_edges_post=graph_edges_post_snapshot,
                ds_graph_nodes=ds_graph_nodes_snapshot,
                ds_graph_edges=ds_graph_edges_snapshot,
                cand_edges=cand_edges_snapshot,
                new_edge=new_edge,
                episode_vector=current_episode_vec.tolist(),
                query_vector=query_vec_list,
                query_node=[int(query_node_pre[0]), int(query_node_pre[1]), int(query_node_pre[2])],
                query_node_pre=[int(query_node_pre[0]), int(query_node_pre[1]), int(query_node_pre[2])],
                query_node_post=[int(current_query_node[0]), int(current_query_node[1]), int(current_query_node[2])],
                ag_fire=bool(ag_fire),
                dg_fire=bool(dg_fire),
                theta_ag=float(locals().get('theta_ag_used', getattr(config, 'theta_ag', 0.0))),
                ag_auto=bool(getattr(config, 'ag_auto', False)),
                ag_quantile=float(getattr(config, 'ag_quantile', 0.9)),
                g0_history_len=int(len(g0_history)),
                dg_committed_edges=dg_committed_edges_snapshot,
                debug_hop0=debug_hop0,
                hop_series=hop_series,
                timeline_edges=timeline_edges_now,
                committed_only_edges=committed_only_edges_snapshot,
                committed_only_nodes=committed_only_nodes_snapshot,
                committed_only_edges_meta=committed_only_edges_meta_snapshot,
                committed_only_nodes_meta=committed_only_nodes_meta_snapshot,
                forced_edges=forced_edges_snapshot,
                forced_edges_meta=forced_edges_meta_snapshot,
                hop_series_post=hop_series_post,
                ds_nodes_total=int(locals().get('ds_nodes_total_snapshot', 0)),
                ds_edges_total=int(locals().get('ds_edges_total_snapshot', 0)),
                ds_nodes_saved=locals().get('ds_nodes_saved_snapshot', []),
                ds_edges_saved=locals().get('ds_edges_saved_snapshot', []),
                ecand_count=int(len(ecand)),
                ecand_mem_count=int(ecand_mem_count),
                ecand_qpast_count=int(ecand_qpast_count),
                hop_series_len=int(len(hop_series)),
                sp_diagnostics=sp_diagnostics,
                sp_ds_saved=bool(getattr(config, 'sp_ds_sqlite', None)),
                sp_ds_eff_saved=int(sp_ds_eff_saved_ct),
                dbg_gmin_calc=float(gmin_calc),
                dbg_best_hop_calc=int(h_calc),
                gmin_mh=float(gmin_mh_val),
                delta_ged_min_mh=float(ged_mh_val),
                delta_ig_min_mh=float(ig_mh_val),
                delta_sp_min_mh=float(sp_mh_val),
                sp_before=float(hop_series[0].get('sp_before', 0.0) if (hop_series and isinstance(hop_series[0], dict)) else 0.0),
                sp_after=float(hop_series[0].get('sp_after', 0.0) if (hop_series and isinstance(hop_series[0], dict)) else 0.0),
            )
        )

        # Periodic checkpoint: write partial steps JSON to --step-log (atomic replace)
        try:
            ci = int(getattr(config, 'checkpoint_interval', 0) or 0)
            cpath = getattr(config, 'checkpoint_path', None)
            if ci > 0 and cpath:
                if (step + 1) % ci == 0:
                    from pathlib import Path as _Path
                    _p = _Path(str(cpath))
                    _p.parent.mkdir(parents=True, exist_ok=True)
                    # Build a lightweight JSON list of steps so far
                    _steps = []
                    for rec in step_records:
                        _steps.append({
                            "seed": rec.seed,
                            "step": rec.step,
                            "position": list(rec.position),
                            "action": rec.action,
                            "candidate_selection": rec.candidate_selection,
                            "delta_ged": rec.delta_ged,
                            "delta_ig": rec.delta_ig,
                            "delta_ged_min": rec.delta_ged_min,
                            "delta_ig_min": rec.delta_ig_min,
                            "delta_sp": rec.delta_sp,
                            "delta_sp_min": rec.delta_sp_min,
                            "delta_h": rec.delta_h,
                            "delta_h_min": rec.delta_h_min,
                            "g0": rec.g0,
                            "gmin": rec.gmin,
                            "best_hop": rec.best_hop,
                            "is_dead_end": rec.is_dead_end,
                            "reward": rec.reward,
                            "done": rec.done,
                            "possible_moves": list(rec.possible_moves),
                            # perf counters helpful for long runs
                            "time_ms_candidates": getattr(rec, 'time_ms_candidates', 0.0),
                            "time_ms_eval": getattr(rec, 'time_ms_eval', 0.0),
                        })
                    _tmp = _p.with_suffix(_p.suffix + ".part")
                    _tmp.write_text(json.dumps(_steps, indent=2), encoding="utf-8")
                    try:
                        _tmp.replace(_p)
                    except Exception:
                        # Fallback to writing in-place if replace not supported
                        _p.write_text(json.dumps(_steps, indent=2), encoding="utf-8")
        except Exception:
            # Checkpointing is best-effort; ignore failures
            pass

        if done:
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

    # Build per-episode summary (include series used for aggregation)
    episode_summary: MazeSummary = {
        "seed": seed,
        "success": success,
        "steps": len(step_records),
        "edges": graph.number_of_edges(),
        "k_star_series": [rec.candidate_selection["k_star"] for rec in step_records],
        "g0_series": [rec.g0 for rec in step_records],
        "gmin_series": [rec.gmin for rec in step_records],
        "multihop_best_hop": [rec.best_hop for rec in step_records],
        "delta_sp_series": [rec.delta_sp for rec in step_records],
        "delta_sp_min_series": [rec.delta_sp_min for rec in step_records],
        "eval_time_ms_series": [getattr(rec, 'time_ms_eval', 0.0) for rec in step_records],
        # Treat DG fire as acceptance for PSZ-style summaries
        "accepted_series": [bool(getattr(rec, 'dg_fire', False)) for rec in step_records],
        "dead_end_steps": dead_end_steps,
        "dead_end_escape_rate": (dead_end_escape / dead_end_steps) if dead_end_steps else 1.0,
    }

    return EpisodeArtifacts(
        summary=episode_summary,
        steps=step_records,
        maze_snapshot=maze_snapshot,
    )


# --------------------------------------------------------------------------------------
# Aggregation / CLI
# --------------------------------------------------------------------------------------

def aggregate(runs: List[MazeSummary]) -> Dict[str, float]:
    if not runs:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_edges": 0.0,
            "g0_mean": 0.0,
            "gmin_mean": 0.0,
            "avg_k_star": 0.0,
            "avg_delta_sp": 0.0,
            "avg_delta_sp_min": 0.0,
            "best_hop_mean": 0.0,
            "best_hop_hist_0": 0.0,
            "best_hop_hist_1": 0.0,
            "best_hop_hist_2": 0.0,
            "best_hop_hist_3": 0.0,
            "avg_time_ms_eval": 0.0,
            "p95_time_ms_eval": 0.0,
        }

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    success_rate = sum(1.0 if run.get("success") else 0.0 for run in runs) / len(runs)
    avg_steps = _mean([run.get("steps", 0) for run in runs])
    avg_edges = _mean([run.get("edges", 0) for run in runs])

    g0_values: List[float] = []
    gmin_values: List[float] = []
    k_values: List[float] = []
    sp_values: List[float] = []
    sp_min_values: List[float] = []
    best_hops: List[int] = []
    eval_times: List[float] = []
    psz_samples: List[Dict[str, float]] = []
    for run in runs:
        g0_values.extend(float(v) for v in run.get("g0_series", []))
        gmin_values.extend(float(v) for v in run.get("gmin_series", []))
        k_values.extend(float(v) for v in run.get("k_star_series", []))
        sp_values.extend(float(v) for v in run.get("delta_sp_series", []))
        sp_min_values.extend(float(v) for v in run.get("delta_sp_min_series", []))
        best_hops.extend(int(h) for h in run.get("multihop_best_hop", []))
        eval_times.extend(float(t) for t in run.get("eval_time_ms_series", []))
        # Build PSZ samples from accepted flags and eval times (per-step)
        acc_flags = run.get("accepted_series", [])
        lat_values = run.get("eval_time_ms_series", [])
        for a, l in zip(acc_flags, lat_values):
            try:
                psz_samples.append({"accepted": bool(a), "latency_ms": float(l)})
            except Exception:
                continue
    # best hop histogram (0..3; others bucketed to 3+)
    hop_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for h in best_hops:
        if h <= 0:
            hop_hist[0] += 1
        elif h == 1:
            hop_hist[1] += 1
        elif h == 2:
            hop_hist[2] += 1
        else:
            hop_hist[3] += 1

    def _p95(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals_sorted = sorted(vals)
        idx = int(max(0, min(len(vals_sorted) - 1, round(0.95 * (len(vals_sorted) - 1)))))
        return float(vals_sorted[idx])

    def _pctl(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        v = sorted(vals)
        q = max(0.0, min(1.0, float(q)))
        i = int(round(q * (len(v) - 1)))
        return float(v[i])

    # PSZ summary (acceptance/FMR/P50)
    try:
        from insightspike.metrics.psz import summarize_accept_latency as _psz
        psz = _psz(psz_samples) if psz_samples else None
    except Exception:
        psz = None

    out = {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_edges": avg_edges,
        "g0_mean": _mean(g0_values),
        "gmin_mean": _mean(gmin_values),
        "g0_p50": _pctl(g0_values, 0.50),
        "g0_p90": _pctl(g0_values, 0.90),
        "g0_p95": _pctl(g0_values, 0.95),
        "gmin_p50": _pctl(gmin_values, 0.50),
        "gmin_p90": _pctl(gmin_values, 0.90),
        "gmin_p95": _pctl(gmin_values, 0.95),
        "avg_k_star": _mean(k_values),
        "avg_delta_sp": _mean(sp_values),
        "avg_delta_sp_min": _mean(sp_min_values),
        "best_hop_mean": _mean([float(h) for h in best_hops]) if best_hops else 0.0,
        "best_hop_hist_0": float(hop_hist[0]),
        "best_hop_hist_1": float(hop_hist[1]),
        "best_hop_hist_2": float(hop_hist[2]),
        "best_hop_hist_3": float(hop_hist[3]),
        "avg_time_ms_eval": _mean(eval_times),
        "p95_time_ms_eval": _p95(eval_times),
    }
    if psz is not None:
        try:
            out.update({
                "psz_acceptance_rate": float(psz.acceptance_rate),
                "psz_fmr": float(psz.fmr),
                "psz_latency_p50_ms": float(psz.latency_p50_ms),
                "psz_inside": bool(psz.inside_psz),
                # Heuristic recommendations for gating thresholds
                # - θ_DG ≈ gminの95パーセンタイル（gmin < θ_DG を ~95% にする）
                "rec_theta_dg_p95": _pctl(gmin_values, 0.95),
            })
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze Query-Hub geDIG prototype")
    parser.add_argument("--maze-size", type=int, default=15)
    parser.add_argument("--maze-type", type=str, default="dfs")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--theta-cand", type=float, default=1.0)
    parser.add_argument("--theta-link", type=float, default=0.1)
    parser.add_argument("--candidate-cap", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--cand-radius", type=float, default=1.0)
    parser.add_argument("--link-radius", type=float, default=0.05)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--max-hops", type=int, default=10)
    parser.add_argument("--decay-factor", type=float, default=0.7)
    parser.add_argument("--adaptive-hops", action="store_true")
    parser.add_argument("--sp-beta", type=float, default=1.0)
    parser.add_argument("--linkset-mode", action="store_true")
    parser.add_argument("--linkset-base", type=str, default="mem", choices=["link","mem","pool"], help="Base set for linkset IG before: link=S_link, mem=memory candidates, pool=all candidates")
    parser.add_argument("--sp-scope", type=str, default="auto", choices=["auto", "union"], help="SP evaluation scope")
    parser.add_argument("--sp-hop-expand", type=int, default=0, help="Extra hops to expand SP neighborhood")
    parser.add_argument("--sp-boundary", type=str, default="trim", choices=["induced","trim","nodes"], help="SP boundary mode (Core) for subgraph evaluation")
    parser.add_argument("--theta-ag", type=float, default=0.0)
    parser.add_argument("--theta-dg", type=float, default=0.0)
    parser.add_argument("--top-link", type=int, default=1)
    parser.add_argument("--link-autowire-all", dest="link_autowire_all", action="store_true", help="Autowire all S_link edges at hop0 (base)")
    parser.add_argument("--no-link-autowire-all", dest="link_autowire_all", action="store_false", help="Disable auto-wiring all S_link at hop0; use Top-L only")
    parser.add_argument("--commit-budget", type=int, default=1)
    parser.add_argument("--dg-commit-policy", type=str, default="threshold", choices=["threshold","always","never"], help="DG commit gating policy")
    parser.add_argument("--dg-commit-all-linkset", dest="dg_commit_all_linkset", action="store_true", help="On DG fire, commit all S_link (hop0) edges instead of Top-L only")
    parser.add_argument("--skip-mh-on-deadend", dest="skip_mh_on_deadend", action="store_true", help="Skip multi-hop evaluation on dead-end/backtrack steps (use hop0 only)")
    parser.add_argument("--commit-from", type=str, default="cand", choices=["cand", "link"])
    parser.add_argument("--norm-base", type=str, default="link", choices=["cand", "link"])
    parser.add_argument("--action-policy", type=str, default="softmax", choices=["argmax", "softmax"], help="Action selection policy for observation candidates")
    parser.add_argument("--action-temp", type=float, default=0.1, help="Temperature for softmax action selection (if enabled)")
    parser.add_argument("--anti-backtrack", dest="anti_backtrack", action="store_true", help="Avoid immediately reversing previous action if alternatives exist")
    parser.add_argument("--no-anti-backtrack", dest="anti_backtrack", action="store_false")
    parser.set_defaults(anti_backtrack=True)
    parser.add_argument("--anchor-recent-q", type=int, default=12, help="Number of recent Q nodes to include in SP anchors")
    # Dynamic AG (online percentile threshold)
    parser.add_argument("--ag-auto", dest="ag_auto", action="store_true", help="Enable dynamic AG threshold from g0 percentile history")
    parser.add_argument("--ag-window", type=int, default=30, help="Window size (steps) for AG percentile history")
    parser.add_argument("--ag-quantile", type=float, default=0.9, help="Quantile [0,1] for AG threshold (e.g., 0.9=90th percentile)")
    # SP cache options
    parser.add_argument("--sp-cache", action="store_true", help="Enable SP DistanceCache path")
    parser.add_argument("--sp-cache-mode", type=str, default="core", choices=["core", "cached", "cached_incr"], help="SP cache mode: core=Core委譲, cached=端点SSSP推定, cached_incr=端点SSSPの増分合成+検証")
    parser.add_argument("--sp-cand-topk", type=int, default=0, help="Top-K cap for Ecand during greedy (0 = unlimited)")
    parser.add_argument("--sp-allpairs", dest="sp_allpairs", action="store_true", help="Force SP evaluation to use ALL-PAIRS average (no fixed-before-pairs) [diagnostic]")
    parser.add_argument("--sp-allpairs-exact", dest="sp_allpairs_exact", action="store_true", help="Greedy exact ALL-PAIRS on evaluation subgraph with incremental 2-BFS update per candidate")
    parser.add_argument("--sp-exact-stable-nodes", dest="sp_exact_stable_nodes", action="store_true", help="Keep evaluation SA node set monotonically growing across steps to improve APSP reuse (off by default)")
    parser.add_argument("--sp-signed", dest="sp_signed", action="store_true", help="Treat ΔSP as signed (no clamp) in reporting and diagnostics")
    parser.add_argument("--sp-report-best-hop", dest="sp_report_best_hop", action="store_true", help="Report delta_sp as best-hop value instead of hop0")
    parser.add_argument("--sp-pair-samples", type=int, default=400, help="Fixed-pair sampling size for SP (before graph)")
    parser.add_argument("--ig-hop-apply", type=str, default="all", choices=["all","hop0"], help="Apply linkset IG to which hops (all or hop0 only)")
    parser.add_argument("--eval-all-hops", action="store_true", help="Force evaluation to add one candidate per hop (diagnostic)")
    parser.add_argument("--no-ged-hop0-const", dest="ged_hop0_const", action="store_false")
    # Optional: write recommended thresholds next to summary
    parser.add_argument("--write-recommendations", action="store_true", help="Write recommended θ values (DG/AG) as JSON next to output")
    # Optional: automatically rerun once with suggested thresholds (guarded by env INSIGHTSPIKE_AUTORERUN_OK=1)
    parser.add_argument("--auto-rerun-with-suggested", action="store_true", help="Rerun once with suggested θ values (requires env INSIGHTSPIKE_AUTORERUN_OK=1)")
    # Snapshot level
    parser.add_argument("--snapshot-level", type=str, default="standard", choices=["minimal","standard","full"], help="Step log snapshot level")
    # Observation guard ablation
    parser.add_argument("--no-obs-guard", dest="obs_guard", action="store_false", help="Disable observation guard (allow non-passable/wall actions as options)")
    parser.set_defaults(obs_guard=True)
    # Sequence ablation controls
    parser.add_argument("--gh-mode", type=str, default="greedy", choices=["greedy", "radius"], help="Multi-hop evaluation: greedy=add edges per hop, radius=do not add edges (radius-only evaluation)")
    parser.add_argument("--no-pre-eval", dest="pre_eval", action="store_false", help="Disable pre-eval (IG/SP before wiring) diagnostics")
    parser.set_defaults(pre_eval=True)
    parser.add_argument("--snapshot-mode", type=str, default="after_select", choices=["before_select", "after_select"], help="When to snapshot prev_graph")
    # Timeline graph policies
    parser.add_argument("--timeline-to-graph", dest="timeline_to_graph", action="store_true", help="Also add timeline edges (Q_prev→dir→Q_next, Q_prev↔Q_next) to graph. Default: off (visual-only)")
    parser.add_argument("--add-next-q", dest="add_next_q", action="store_true", help="Also add next-step Q node to graph at end of step. Default: off")
    # Paper preset: linkset IG + candidate-base GED + ig-hop-apply=all
    parser.add_argument("--preset", type=str, default=None, choices=["paper"], help="Quick preset: 'paper' = linkset IG + candidate-base GED + ig-hop-apply=all")
    parser.set_defaults(ged_hop0_const=True)
    # Lite path: delegate per-step evaluation to main L3 (query-centric)
    parser.add_argument("--use-main-l3", dest="use_main_l3", action="store_true", help="Use main L3GraphReasoner (query-centric) for per-step eval (hop0 only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-log", type=Path)
    # Optional explicit maze snapshot JSON for paper reproducibility
    parser.add_argument("--maze-snapshot-out", type=Path, default=None, help="If set, write maze layout/start/goal/size to this JSON path at run start")
    # Persistence
    parser.add_argument("--persist-graph-sqlite", type=str, default="", help="If set, persist committed diffs (nodes/edges) to a SQLite DB at this path")
    parser.add_argument("--persist-namespace", type=str, default="maze_query_hub")
    parser.add_argument("--persist-forced-candidates", dest="persist_forced_candidates", action="store_true", help="Persist forced candidate edges (Q↔dir) to SQLite/steps even if not committed")
    parser.add_argument("--persist-timeline-edges", dest="persist_timeline_edges", action="store_true", help="Persist timeline edges (Q_prev→dir→Q_next 等) to SQLite/steps for relaxed/strict再生")
    # Forced-as-base toggle (when S_link is empty, use forced Top‑L as base)
    parser.add_argument("--link-forced-as-base", dest="link_forced_as_base", action="store_true", help="Use forced Top‑L as linkset base when S_link is empty (affects Cmax and pre-eval)")
    parser.add_argument("--no-link-forced-as-base", dest="link_forced_as_base", action="store_false")
    # Periodic checkpointing of step log (for long runs / timeout resilience)
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Write partial steps JSON every N steps to --step-log (0 disables)")
    # Defaults for persistence toggles
    parser.set_defaults(persist_timeline_edges=True)
    # Ensure dead-end skip is OFF by default (explicit)
    parser.set_defaults(skip_mh_on_deadend=False)
    # Layer1 vector prefilter options
    # Layer1 vector prefilter options (weighted radius mode is default)
    parser.add_argument("--layer1-prefilter", dest="layer1_prefilter", action="store_true", default=False, help="Use Layer1 weighted-L2 vector prefilter for mem candidates (disables spatial ring)")
    parser.add_argument("--no-layer1-prefilter", dest="layer1_prefilter", action="store_false")
    parser.add_argument("--l1-cap", type=int, default=16, help="Top-K cap for Layer1 vector prefilter")
    # Spatial prefilter options
    parser.add_argument("--ring-ellipse", dest="ring_ellipse", action="store_true", default=True, help="Use elliptical window for spatial prefilter (default: rectangular)")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Verbose per-step logging (debug)")
    parser.add_argument("--log-minimal", dest="log_minimal", action="store_true", help="Write only minimal fields to steps.json (reduces IO)")
    # DS-backed SP pairsets (default: enabled with shared cache)
    parser.add_argument("--sp-ds-sqlite", type=str, default="results/mq_sp.sqlite", help="SQLite DB path for SP pairset reuse (before/after)")
    parser.add_argument("--sp-ds-namespace", type=str, default="mq_sp", help="Namespace for SP pairsets in DS")
    # Post-step diagnostics
    parser.add_argument("--no-post-sp-diagnostics", dest="post_sp_diagnostics", action="store_false", help="Disable post-step SP diagnostics (hop_series_post) to save time")
    # Ultra-light steps (skip heavy graph snapshots entirely)
    parser.add_argument("--steps-ultra-light", dest="steps_ultra_light", action="store_true", help="Do not generate heavy snapshot arrays in steps.json; prefer DS/HTML light mode")
    parser.add_argument("--force-per-hop", dest="force_per_hop", action="store_true", help="Force per-hop series via evaluator even in L3-only path")
    parser.add_argument("--eval-per-hop-on-ag", dest="eval_per_hop_on_ag", action="store_true", help="Fallback to evaluator (per-hop) only when AG fires in L3-only path")
    parser.add_argument("--dg-bfs-shortcut", dest="dg_bfs_shortcut", action="store_true", help="When DG fires, commit chosen multi-hop edges as a BFS-like shortcut in one step")
    # Defaults
    parser.set_defaults(link_autowire_all=True)
    # Prefer recording per-hop on AG in L3-light path unless explicitly disabled
    parser.set_defaults(eval_per_hop_on_ag=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Apply quick presets
    if getattr(args, 'preset', None) == 'paper':
        args.linkset_mode = True
        args.linkset_base = 'link'
        args.norm_base = 'link'
        try:
            setattr(args, 'ig_hop_apply', 'all')
        except Exception:
            pass
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
    # Enforce lambda=1.0 as fixed per spec
    args.lambda_weight = 1.0
    gedig_params = {
        "lambda_weight": args.lambda_weight,
        "max_hops": args.max_hops,
        "decay_factor": args.decay_factor,
        "adaptive_hops": args.adaptive_hops,
        "sp_beta": args.sp_beta,
        "sp_scope_mode": args.sp_scope,
        "sp_hop_expand": args.sp_hop_expand,
        "sp_boundary_mode": args.sp_boundary,
        "ig_hop_apply": str(args.ig_hop_apply),
    }
    config = QueryHubConfig(
        maze_size=args.maze_size,
        maze_type=args.maze_type,
        max_steps=args.max_steps,
        selector=selector_params,
        gedig=gedig_params,
        linkset_mode=bool(args.linkset_mode),
        linkset_base=str(args.linkset_base),
        theta_ag=float(args.theta_ag),
        theta_dg=float(args.theta_dg),
        top_link=int(args.top_link),
        link_autowire_all=bool(getattr(args, 'link_autowire_all', False)),
        commit_budget=int(args.commit_budget),
        commit_from=str(args.commit_from),
        norm_base=str(args.norm_base),
        action_policy=str(args.action_policy),
        action_temp=float(args.action_temp),
        anti_backtrack=bool(args.anti_backtrack),
        anchor_recent_q=int(args.anchor_recent_q),
        sp_cache=bool(args.sp_cache),
        sp_cache_mode=str(args.sp_cache_mode),
        sp_cand_topk=int(args.sp_cand_topk),
        sp_eval_allpairs=bool(getattr(args, 'sp_allpairs', False)),
        sp_eval_allpairs_exact=bool(getattr(args, 'sp_allpairs_exact', False)),
        sp_exact_stable_nodes=bool(getattr(args, 'sp_exact_stable_nodes', False)),
        sp_signed=bool(getattr(args, 'sp_signed', False)),
        sp_report_best_hop=bool(getattr(args, 'sp_report_best_hop', False)),
        sp_pair_samples=int(args.sp_pair_samples),
        eval_all_hops=bool(args.eval_all_hops),
        ged_hop0_const=bool(args.ged_hop0_const),
        gh_mode=str(args.gh_mode),
        pre_eval=bool(args.pre_eval),
        snapshot_mode=str(args.snapshot_mode),
        timeline_to_graph=bool(getattr(args, 'timeline_to_graph', False)),
        add_next_q=bool(getattr(args, 'add_next_q', False)),
        persist_sqlite_path=(str(args.persist_graph_sqlite).strip() or None),
        persist_namespace=str(args.persist_namespace),
        persist_forced_candidates=bool(getattr(args, 'persist_forced_candidates', False)),
        link_forced_as_base=bool(getattr(args, 'link_forced_as_base', True)),
        persist_timeline_edges=bool(getattr(args, 'persist_timeline_edges', False)),
        dg_commit_policy=str(args.dg_commit_policy),
        dg_commit_all_linkset=bool(getattr(args, 'dg_commit_all_linkset', False)),
        skip_mh_on_deadend=bool(getattr(args, 'skip_mh_on_deadend', False)),
        snapshot_level=str(args.snapshot_level),
        ring_ellipse=bool(getattr(args, 'ring_ellipse', False)),
        layer1_prefilter=bool(getattr(args, 'layer1_prefilter', True)),
        l1_cap=int(getattr(args, 'l1_cap', 16)),
        ag_auto=bool(getattr(args, 'ag_auto', False)),
        ag_window=int(getattr(args, 'ag_window', 30)),
        ag_quantile=float(getattr(args, 'ag_quantile', 0.9)),
        verbose=bool(getattr(args, 'verbose', False)),
        sp_ds_sqlite=(str(getattr(args, 'sp_ds_sqlite', '')).strip() or None),
        sp_ds_namespace=str(getattr(args, 'sp_ds_namespace', 'maze_query_hub_sp')),
        checkpoint_interval=int(getattr(args, 'checkpoint_interval', 0)),
        checkpoint_path=(str(getattr(args, 'step_log', '')).strip() or None),
        post_sp_diagnostics=bool(getattr(args, 'post_sp_diagnostics', True)),
        steps_ultra_light=bool(getattr(args, 'steps_ultra_light', False)),
        maze_snapshot_out=(str(getattr(args, 'maze_snapshot_out', '')).strip() or None),
        force_per_hop=bool(getattr(args, 'force_per_hop', False)),
        eval_per_hop_on_ag=bool(getattr(args, 'eval_per_hop_on_ag', False)),
        dg_bfs_shortcut=bool(getattr(args, 'dg_bfs_shortcut', False)),
    )

    runs: List[MazeSummary] = []
    all_steps: List[StepLog] = []
    maze_data: Dict[str, Any] = {}

    for offset in range(args.seeds):
        seed = args.seed_start + offset
        artifacts = run_episode_query(seed=seed, config=config)
        runs.append(artifacts.summary)
        maze_data[str(seed)] = artifacts.maze_snapshot
        # Append this seed's step records as we go (not only the last seed)
        for record in artifacts.steps:
            minimal = bool(getattr(args, 'log_minimal', False) or getattr(args, 'steps_ultra_light', False))
            if minimal:
                # Minimal logging: keep UIに必要な基本値は残す（g0/gminに加えΔEPC/ΔIGも出力）
                all_steps.append({
                    "seed": record.seed,
                    "step": record.step,
                    "position": list(record.position),
                    "action": record.action,
                    # geDIG core metrics
                    "g0": record.g0,
                    "gmin": record.gmin,
                    "delta_ged": record.delta_ged,
                    "delta_ig": record.delta_ig,
                    "delta_ged_min": record.delta_ged_min,
                    "delta_ig_min": record.delta_ig_min,
                    "delta_sp": record.delta_sp,
                    "delta_sp_min": record.delta_sp_min,
                    "best_hop": record.best_hop,
                    # gating
                    "ag_fire": bool(getattr(record, 'ag_fire', False)),
                    "dg_fire": bool(getattr(record, 'dg_fire', False)),
                    "theta_ag": float(getattr(record, 'theta_ag', 0.0)),
                    # topology context
                    "possible_moves": getattr(record, 'possible_moves', []),
                    "is_dead_end": bool(getattr(record, 'is_dead_end', False)),
                    # perf
                    "time_ms_candidates": getattr(record, 'time_ms_candidates', 0.0),
                    "time_ms_eval": getattr(record, 'time_ms_eval', 0.0),
                })
                continue
            all_steps.append({
                    "seed": record.seed,
                    "step": record.step,
                    "position": list(record.position),
                    "action": record.action,
                    "candidate_selection": record.candidate_selection,
                    "delta_ged": record.delta_ged,
                    "delta_ig": record.delta_ig,
                    "delta_ged_min": record.delta_ged_min,
                    "delta_ig_min": record.delta_ig_min,
                    "delta_sp": record.delta_sp,
                    "delta_sp_min": record.delta_sp_min,
                    "delta_h": record.delta_h,
                    "delta_h_min": record.delta_h_min,
                    # profiling
                    "ring_Rr": getattr(record, 'ring_Rr', 0),
                    "ring_Rc": getattr(record, 'ring_Rc', 0),
                    "ring_max_cells": getattr(record, 'ring_max_cells', 0),
                    "ring_cells": getattr(record, 'ring_cells', 0),
                    "ring_nodes": getattr(record, 'ring_nodes', 0),
                    "ring_fallback": getattr(record, 'ring_fallback', False),
                    "obs_dist_evals": getattr(record, 'obs_dist_evals', 0),
                    "mem_dist_evals": getattr(record, 'mem_dist_evals', 0),
                    "total_dist_evals": getattr(record, 'total_dist_evals', 0),
                    "time_ms_candidates": getattr(record, 'time_ms_candidates', 0.0),
                    "time_ms_eval": getattr(record, 'time_ms_eval', 0.0),
                    # SP perf counters
                    "sp_sssp_du": getattr(record, 'sp_sssp_du', 0),
                    "sp_sssp_dv": getattr(record, 'sp_sssp_dv', 0),
                    "sp_dv_leaf_skips": getattr(record, 'sp_dv_leaf_skips', 0),
                    "sp_cycle_verifies": getattr(record, 'sp_cycle_verifies', 0),
                    "linkset_delta_ged": record.linkset_delta_ged,
                    "linkset_delta_h": record.linkset_delta_h,
                    "linkset_delta_sp": record.linkset_delta_sp,
                    "linkset_g": record.linkset_g,
                    "linkset_entropy_before": getattr(record, 'linkset_entropy_before', None),
                    "linkset_entropy_after": getattr(record, 'linkset_entropy_after', None),
                    "linkset_pos_w_before": getattr(record, 'linkset_pos_w_before', None),
                    "linkset_pos_w_after": getattr(record, 'linkset_pos_w_after', None),
                    "linkset_topw_before": getattr(record, 'linkset_topw_before', []),
                    "linkset_topw_after": getattr(record, 'linkset_topw_after', []),
                    "linkset_entropy_before": getattr(record, 'linkset_entropy_before', None),
                    "linkset_entropy_after": getattr(record, 'linkset_entropy_after', None),
                    "structural_cost": record.structural_cost,
                    "structural_improvement": record.structural_improvement,
                    "g0": record.g0,
                    "gmin": record.gmin,
                    "theta_ag": getattr(record, 'theta_ag', 0.0),
                    "ag_auto": getattr(record, 'ag_auto', False),
                    "ag_quantile": getattr(record, 'ag_quantile', 0.0),
                    "g0_history_len": getattr(record, 'g0_history_len', 0),
                    "ag_fire": getattr(record, 'ag_fire', False),
                    "dg_fire": getattr(record, 'dg_fire', False),
                    "best_hop": record.best_hop,
                    "dg_committed_edges": getattr(record, 'dg_committed_edges', []),
                    "is_dead_end": record.is_dead_end,
                    "reward": record.reward,
                    "done": record.done,
                    "possible_moves": record.possible_moves,
                    "candidate_pool": record.candidate_pool,
                    "selected_links": record.selected_links,
                    "ranked_candidates": record.ranked_candidates,
                    "graph_nodes": record.graph_nodes,
                    "graph_edges": record.graph_edges,
                    "graph_nodes_eval": getattr(record, 'graph_nodes_eval', []),
                    "graph_edges_eval": getattr(record, 'graph_edges_eval', []),
                    "graph_nodes_preselect": getattr(record, 'graph_nodes_preselect', []),
                    "graph_edges_preselect": getattr(record, 'graph_edges_preselect', []),
                    "graph_nodes_pre": getattr(record, 'graph_nodes_pre', []),
                    "graph_edges_pre": getattr(record, 'graph_edges_pre', []),
                    "graph_nodes_post": getattr(record, 'graph_nodes_post', []),
                    "graph_edges_post": getattr(record, 'graph_edges_post', []),
                    "ds_graph_nodes": getattr(record, 'ds_graph_nodes', []),
                    "ds_graph_edges": getattr(record, 'ds_graph_edges', []),
                    "graph_nodes_eval": getattr(record, 'graph_nodes_eval', []),
                    "graph_edges_eval": getattr(record, 'graph_edges_eval', []),
                    "committed_only_edges": record.committed_only_edges,
                    "committed_only_nodes": getattr(record, 'committed_only_nodes', []),
                    "cand_edges": record.cand_edges,
                    "forced_edges": getattr(record, 'forced_edges', []),
                    "new_edge": record.new_edge,
                    "episode_vector": record.episode_vector,
                    "query_vector": record.query_vector,
                    "query_node": record.query_node,
                    "query_node_pre": getattr(record, 'query_node_pre', []),
                    "query_node_post": getattr(record, 'query_node_post', []),
                    "debug_hop0": record.debug_hop0,
                    "hop_series": record.hop_series,
                    "timeline_edges": record.timeline_edges,
                    "hop_series_post": record.hop_series_post,
                    "committed_only_edges": record.committed_only_edges,
                    "ecand_count": getattr(record, 'ecand_count', 0),
                    "ecand_mem_count": getattr(record, 'ecand_mem_count', 0),
                    "ecand_qpast_count": getattr(record, 'ecand_qpast_count', 0),
                    "hop_series_len": getattr(record, 'hop_series_len', 0),
                    "sp_diagnostics": getattr(record, 'sp_diagnostics', []),
                    "dbg_gmin_calc": getattr(record, 'dbg_gmin_calc', 0.0),
                    "dbg_best_hop_calc": getattr(record, 'dbg_best_hop_calc', 0),
                    "gmin_mh": getattr(record, 'gmin_mh', None),
                    "delta_ged_min_mh": getattr(record, 'delta_ged_min_mh', None),
                    "delta_ig_min_mh": getattr(record, 'delta_ig_min_mh', None),
                    "delta_sp_min_mh": getattr(record, 'delta_sp_min_mh', None),
                    "sp_before": getattr(record, 'sp_before', None),
                    "sp_after": getattr(record, 'sp_after', None),
                    "sp_ds_saved": getattr(record, 'sp_ds_saved', False),
                    "sp_ds_eff_saved": getattr(record, 'sp_ds_eff_saved', 0),
                    "ds_nodes_total": getattr(record, 'ds_nodes_total', 0),
                    "ds_edges_total": getattr(record, 'ds_edges_total', 0),
                    "ds_nodes_saved": getattr(record, 'ds_nodes_saved', []),
                    "ds_edges_saved": getattr(record, 'ds_edges_saved', []),
                })

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
            "graph_mode": "query_hub",
            "sequence": {"gh_mode": args.gh_mode, "pre_eval": bool(args.pre_eval), "snapshot_mode": args.snapshot_mode},
            "theta_ag": float(args.theta_ag),
            "theta_dg": float(args.theta_dg),
            "action_policy": str(args.action_policy),
            "action_temp": float(args.action_temp),
            # Also expose whether main L3 lite path was used (query-centric hop0)
            "use_main_l3": bool(getattr(args, 'use_main_l3', False)),
            "sp_cache_mode": str(getattr(args, 'sp_cache_mode', 'core')),
        },
        "summary": summary,
        "runs": runs,
        "maze_data": maze_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    # Optional: write recommended thresholds next to summary
    try:
        if bool(getattr(args, 'write_recommendations', False)):
            rec = {
                "theta_dg_recommended": float(summary.get("rec_theta_dg_p95", 0.0)),
                # Heuristic: use g0_p90 as AG目安（大きいg0の上位10%を弾く）
                "theta_ag_recommended": float(summary.get("g0_p90", 0.0)),
            }
            rec_path = args.output.with_name(args.output.stem + "_recommend.json")
            rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    except Exception:
        pass

    if args.step_log:
        args.step_log.parent.mkdir(parents=True, exist_ok=True)
        args.step_log.write_text(json.dumps(all_steps, indent=2), encoding="utf-8")

    # Optional console hint for DG/AG threshold suggestions (paper-aligned)
    try:
        rec_dg = summary.get("rec_theta_dg_p95")
        if isinstance(rec_dg, (int, float)):
            print(f"[recommend] theta_dg ≈ p95(gmin) = {float(rec_dg):.4f}")
    except Exception:
        pass
    print(json.dumps(summary, indent=2))

    # Optional: auto rerun once with suggested thresholds (guarded by env)
    try:
        import os, sys, shlex, subprocess
        if bool(getattr(args, 'auto_rerun_with_suggested', False)):
            ok = os.getenv('INSIGHTSPIKE_AUTORERUN_OK', '').strip().lower() in ('1','true','yes','on')
            if ok:
                theta_dg_s = summary.get('rec_theta_dg_p95')
                theta_ag_s = summary.get('g0_p90')
                if isinstance(theta_dg_s, (int, float)):
                    new_argv = [sys.executable, sys.argv[0]]
                    # filter out flags that should not carry into rerun
                    skip_next = False
                    for i, tok in enumerate(sys.argv[1:]):
                        if skip_next:
                            skip_next = False
                            continue
                        if tok in ('--auto-rerun-with-suggested','--write-recommendations'):
                            continue
                        if tok in ('--theta-dg','--theta-ag','--output','--step-log'):
                            skip_next = True
                            continue
                        new_argv.append(tok)
                    # add overridden thresholds
                    new_argv += ['--theta-dg', f"{float(theta_dg_s):.6f}"]
                    if isinstance(theta_ag_s, (int, float)):
                        new_argv += ['--theta-ag', f"{float(theta_ag_s):.6f}"]
                    # redirect outputs to *_rerun.json
                    try:
                        out_path = Path(args.output)
                        new_out = out_path.with_name(out_path.stem + '_rerun.json')
                        new_argv += ['--output', str(new_out)]
                    except Exception:
                        pass
                    try:
                        if args.step_log:
                            step_path = Path(args.step_log)
                            new_step = step_path.with_name(step_path.stem + '_rerun.json')
                            new_argv += ['--step-log', str(new_step)]
                    except Exception:
                        pass
                    print('[autorun] Rerunning with suggested thresholds:', ' '.join(shlex.quote(a) for a in new_argv))
                    subprocess.run(new_argv, check=False)
            else:
                print('[autorun] Skipped (set INSIGHTSPIKE_AUTORERUN_OK=1 to enable)')
    except Exception as _e:
        print('[autorun] Failed to schedule rerun:', _e)


if __name__ == "__main__":
    main()
class HopGreedyEvaluator:
    """Greedy staged evaluator for g(h) up to max_h.

    - h=0: base after_graph with Q↔dir (S_link Top‑L)
    - h>=1: greedily add one edge per hop from Ecand (mem-only, prev_graph-existing) maximizing ΔSP (fixed-pair)
    - g(h) = ΔGED(h) − λ·(ΔH + γ·ΔSP(h))  with ΔH from linkset (paper-mode)
    """

    def __init__(self, core: GeDIGCore, config: QueryHubConfig) -> None:
        self.core = core
        self.config = config

    def _allpairs_sp(self, anchors: Set[Tuple[int,int,int]], g_before: nx.Graph, g_after: nx.Graph, hop: int) -> float:
        core = self.core
        sp_g1, nodes_sp1 = core._extract_k_hop_subgraph(g_before, anchors, hop)
        sp_g2, nodes_sp2 = core._extract_k_hop_subgraph(g_after, anchors, hop)
        if str(core.sp_scope_mode).lower() in ("union","merge","superset"):
            all_nodes = set(nodes_sp1) | set(nodes_sp2)
            if all_nodes:
                sp_g1 = g_before.subgraph(all_nodes).copy()
                sp_g2 = g_after.subgraph(all_nodes).copy()
        if str(core.sp_boundary_mode).lower() in ("trim","terminal","nodes"):
            sp_g1 = core._trim_terminal_edges(sp_g1, anchors, hop)
            sp_g2 = core._trim_terminal_edges(sp_g2, anchors, hop)
        try:
            return float(core._compute_sp_gain_norm(sp_g1, sp_g2, mode=core.sp_norm_mode))
        except Exception:
            return 0.0

    def evaluate(
        self,
        *,
        prev_graph: nx.Graph,
        graph_pre: nx.Graph,
        current_query_node: Tuple[int,int,int],
        commit_items: List[Dict[str,Any]],
        base_count: int,
        base_ig: float,
        ecand_items: List[Dict[str,Any]],
        anchor_nodes: Set[Tuple[int,int,int]],
    ) -> Tuple[List[Dict[str,Any]], float, float, float, float, int]:
        core = self.core
        max_h = int(self.config.gedig["max_hops"]) if int(self.config.gedig["max_hops"]) > 0 else 0

        # Build base after_graph (h=0): Q↔dir from commit_items
        def ensure_dir(target_graph: nx.Graph, item: Dict[str,Any]) -> Optional[Tuple[int,int,int]]:
            anchor_source = item.get("anchor_position") or item.get("position")
            if not anchor_source:
                return None
            at = (int(anchor_source[0]), int(anchor_source[1]))
            d = item.get("direction")
            if d is None:
                rd = tuple(item.get("relative_delta") or item.get("meta_delta") or (0, 0))
                d = direction_from_delta(rd)
            if d is None:
                return None
            nid = make_direction_node(at, int(d))
            if nid not in graph_pre:
                graph_pre.add_node(nid)
            return nid

        base_graph = graph_pre.copy()
        stage_anchors = set(anchor_nodes)
        for it in commit_items:
            nid = ensure_dir(base_graph, it)
            if nid is None:
                continue
            if not base_graph.has_edge(current_query_node, nid):
                base_graph.add_edge(current_query_node, nid)
            stage_anchors.add(nid)

        denom_cmax = float(core.node_cost + core.edge_cost * max(1, int(base_count)))
        records: List[Dict[str,Any]] = []

        # h=0 evaluation
        sub_b0 = prev_graph.subgraph(stage_anchors).copy()
        sub_a0 = base_graph.subgraph(stage_anchors).copy()
        ged0 = float(_norm_ged(sub_b0, sub_a0, node_cost=core.node_cost, edge_cost=core.edge_cost,
                               normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                               enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                               norm_override=denom_cmax)["normalized_ged"]) if denom_cmax > 0 else 0.0
        sp0 = 0.0
        ig0 = base_ig + core.sp_beta * sp0
        g0v = float(ged0 - core.lambda_weight * ig0)
        records.append({"hop": 0, "g": g0v, "ged": ged0, "ig": ig0, "h": base_ig, "sp": sp0})
        g_best = g0v; h_best = 0

        # Build Ecand (mem & prev exists)
        def edge_from_item(item: Dict[str,Any]) -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
            anchor_source = item.get("anchor_position") or item.get("position")
            if not anchor_source:
                return None
            at = (int(anchor_source[0]), int(anchor_source[1]))
            d = item.get("direction")
            if d is None:
                rd = tuple(item.get("relative_delta") or item.get("meta_delta") or (0, 0))
                d = direction_from_delta(rd)
            if d is None:
                return None
            nid = make_direction_node(at, int(d))
            if nid not in prev_graph:
                return None
            return canonical_edge_id(current_query_node, nid)

        ecand_edges: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = []
        seen = set()
        for it in ecand_items:
            if (it.get("origin") or "").lower() != "mem":
                continue
            e = edge_from_item(it)
            if e is None or e in seen:
                continue
            seen.add(e)
            ecand_edges.append(e)

        used_edges = set(canonical_edge_id(u,v) for u,v in base_graph.edges())
        h_graph = base_graph.copy()
        for h in range(1, max_h + 1):
            best_delta = 0.0; best_edge = None
            for e_u, e_v in ecand_edges:
                if (e_u, e_v) in used_edges:
                    continue
                g_try = h_graph.copy()
                if not g_try.has_node(e_u): g_try.add_node(e_u)
                if not g_try.has_node(e_v): g_try.add_node(e_v)
                if not g_try.has_edge(e_u, e_v): g_try.add_edge(e_u, e_v)
                eff_hop = h + int(max(0, int(self.config.gedig.get("sp_hop_expand", 0))))
                de = self._allpairs_sp(stage_anchors, h_graph, g_try, max(1, eff_hop))
                if de > best_delta:
                    best_delta = de; best_edge = (e_u, e_v)
            prev_stage = h_graph.copy()
            if best_edge is not None and best_delta > 0.0:
                e_u, e_v = best_edge
                h_graph.add_edge(e_u, e_v)
                used_edges.add(best_edge)
                stage_anchors.update([e_u, e_v])
            # 機械的に hop を埋める: 候補が無くても評価して記録
            sub_b = prev_graph.subgraph(stage_anchors).copy()
            sub_a = h_graph.subgraph(stage_anchors).copy()
            ged_h = float(_norm_ged(sub_b, sub_a, node_cost=core.node_cost, edge_cost=core.edge_cost,
                                    normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                                    enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                                    norm_override=denom_cmax)["normalized_ged"]) if denom_cmax > 0 else 0.0
            eff_hop_eval = h + int(max(0, int(self.config.gedig.get("sp_hop_expand", 0))))
            sp_h = self._allpairs_sp(stage_anchors, prev_stage, h_graph, max(1, eff_hop_eval))
            ig_h = base_ig + core.sp_beta * sp_h
            g_h = float(ged_h - core.lambda_weight * ig_h)
            records.append({"hop": h, "g": g_h, "ged": ged_h, "ig": ig_h, "h": base_ig, "sp": sp_h})
            if g_h < g_best:
                g_best = g_h; h_best = h

        # Outputs
        g0 = records[0]["g"]
        gmin = g_best
        delta_ged = records[0]["ged"]
        delta_ig = records[0]["ig"]
        t_eval_end = time.perf_counter()
        return records, g0, gmin, delta_ged, delta_ig, int(h_best), (t_eval_end - t_eval_start) * 1000.0
