"""Data models for query-hub maze experiments.

Contains:
- StepRecord: Per-step diagnostic data
- QueryHubConfig: Experiment configuration
- EpisodeArtifacts: Episode results container
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Type alias for episode summary
MazeSummary = Dict[str, Any]


@dataclass
class StepRecord:
    """Per-step diagnostic record for maze experiments."""
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
    ged_min_proxy: float = 0.0
    lambda_weight: float = 1.0
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
    # DS snapshot at episode start (persisted memory baseline)
    ds_edges_baseline: List[List[List[int]]] = field(default_factory=list)
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
    # DG staged (proposed) edges snapshot: chosen_edges_by_hop[:best_hop] (before commit budget / policy)
    dg_staged_edges: List[List[List[int]]] = field(default_factory=list)
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
    # Sleep guidance diagnostics (curriculum / path compression)
    sleep_plan_action: int = -1  # action id (0-3) suggested by Sleep plan, -1 if none
    sleep_guided: bool = False  # whether executed action followed Sleep plan
    sleep_plan_beta: float = 0.0  # soft bias strength used in prefer mode (0 if disabled)
    # Sleep Q guidance (curriculum / value propagation)
    sleep_q_applied: bool = False
    sleep_q_beta: float = 0.0
    sleep_q_value: float = 0.0
    sleep_q_max: float = 0.0
    sleep_q_adv: float = 0.0
    # Sleep edge-weight prior diagnostics
    sleep_edge_applied: bool = False
    sleep_edge_weight: float = 0.0
    sleep_edge_mode: str = ""
    # Event-based prior diagnostics
    event_bias: float = 0.0
    event_bias_applied: bool = False
    # Affordance (graph memory) bias diagnostics
    affordance_bias: float = 0.0
    affordance_bias_applied: bool = False
    affordance_bias_update: float = 0.0
    affordance_bias_post: float = 0.0
    affordance_bias_map: Dict[str, float] = field(default_factory=dict)
    # Cortisol (stress/adaptation) diagnostics: label "wasted" exploration as negative examples
    cortisol_level: float = 0.0
    cortisol_fire: bool = False
    cortisol_reason: str = ""
    cortisol_ag_streak: int = 0
    cortisol_stuck_streak: int = 0
    # β₁ (first Betti number) diagnostics
    graph_node_count: int = 0
    graph_edge_count: int = 0
    betti_1: int = 0
    # Three-layer search diagnostics
    search_layer_used: int = -1          # 0, 1, 2 (-1 = legacy)
    search_is_revisit: bool = False
    search_revisit_similarity: float = 0.0
    search_time_ms: float = 0.0
    search_l1_candidates: int = 0


@dataclass
class QueryHubConfig:
    """Configuration for query-hub maze experiments."""
    maze_size: int
    maze_type: str
    max_steps: int
    selector: Dict[str, Any]
    gedig: Dict[str, Any]
    linkset_mode: bool = True  # use linkset-based entropy calculation (paper mode)
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
    action_source: str = "obs"  # 'obs' (default) or 'mix' (obs+mem)
    # Sleep Q (policy prior) controls: used when sleep_guide == 'prefer'
    sleep_q_beta: float = 4.0
    # Sleep plan soft bias controls: used when sleep_guide == 'prefer'
    sleep_plan_beta: float = 0.0
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
    timeline_to_graph: bool = True  # add timeline edges to graph (required for SP gain calculation)
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
    # Cortisol (stress/adaptation): log-only negative-example signal from stuckness
    cortisol_mode: str = "off"  # 'off' | 'log'
    cortisol_ag_streak: int = 30
    cortisol_stuck_streak: int = 10
    cortisol_repeat_visits: int = 2
    # Sleep Q learning (value propagation from warmup logs)
    sleep_q_gamma: float = 0.99
    sleep_q_alpha: float = 0.4
    sleep_q_iters: int = 50
    sleep_q_step_penalty: float = -0.01
    sleep_q_goal_reward: float = 1.0
    sleep_q_revisit_penalty: float = -0.2
    sleep_q_deadend_penalty: float = 0.0
    sleep_q_blocked_penalty: float = 0.0
    # Sleep edge-weight prior (geDIG-style)
    sleep_edge_enabled: bool = False
    sleep_edge_beta: float = 1.0
    sleep_edge_alpha: float = 0.4
    sleep_edge_gamma: float = 0.95
    sleep_edge_alpha_explore: float = 0.05
    sleep_edge_revisit_penalty: float = 0.2
    sleep_edge_deadend_penalty: float = 0.2
    sleep_edge_blocked_penalty: float = 0.2
    sleep_edge_mode: str = "mul"  # 'mul' (similarity * (1+tanh)) or 'add' (logit add)
    # Event-based prior (semantic-space PoC): action bias from event weights
    event_weights: Dict[str, float] = field(default_factory=dict)
    event_beta: float = 1.0
    # Affordance bias (graph memory): stored on direction nodes
    affordance_bias: bool = False
    affordance_beta: float = 1.0
    affordance_lr: float = 0.2
    affordance_clamp: float = 3.0
    # Force per-hop series via evaluator fallback even in L3-only mode
    force_per_hop: bool = False
    # Per-hop evaluator fallback only when AG fires (L3-only)
    eval_per_hop_on_ag: bool = False
    # DG fire: commit chosen multi-hop edges as a BFS-like shortcut in one step
    dg_bfs_shortcut: bool = False
    # Force SP gain evaluation at hop0 (diagnostic)
    force_sp_gain_eval: bool = False
    # Graph-persistent DG: extended vector mode and Sleep propagation
    vector_mode: str = "standard"  # 'standard' (8D) or 'extended' (10D)
    propagated_alpha: float = 1.0
    sleep_propagate_gamma: float = 0.95
    sleep_propagate_iters: int = 50
    # SP definition mode: asp (default), betti1, both (parallel recording)
    sp_mode: str = "asp"
    # Three-layer search mode
    search_mode: str = "legacy"            # 'legacy' or 'threelayer'
    theta_attention: float = 0.3           # attention threshold for L1
    attention_decay: float = 0.95          # per-step attention decay rate
    attention_boost: float = 0.1           # traverse boost
    attention_alpha: float = 0.5           # attention exponent for effective_score
    min_layer1_candidates: int = 2         # min L1 candidates to skip L2


@dataclass
class EpisodeArtifacts:
    """Container for episode results."""
    summary: MazeSummary
    steps: List[StepRecord]
    maze_snapshot: Dict[str, Any]
    graph: Optional[Any] = None
