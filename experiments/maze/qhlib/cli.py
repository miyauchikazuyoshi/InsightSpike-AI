"""CLI argument parsing for maze experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .models import QueryHubConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for maze query-hub experiments."""
    parser = argparse.ArgumentParser(description="Maze Query-Hub geDIG prototype")
    parser.add_argument("--maze-size", type=int, default=15)
    parser.add_argument("--maze-type", type=str, default="dfs")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for seed execution (default: 1 = sequential)")
    parser.add_argument("--theta-cand", type=float, default=1.0)
    parser.add_argument("--theta-link", type=float, default=0.1)
    parser.add_argument("--candidate-cap", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--cand-radius", type=float, default=1.0)
    parser.add_argument("--link-radius", type=float, default=0.05)
    parser.add_argument("--lambda-weight", type=float, default=1.0)
    parser.add_argument("--lambda-sweep", type=float, nargs="+", default=None, help="Optional list of lambda values to sweep; overrides --lambda-weight when set")
    parser.add_argument("--max-hops", type=int, default=10)
    parser.add_argument("--decay-factor", type=float, default=0.7)
    parser.add_argument("--adaptive-hops", action="store_true")
    parser.add_argument("--sp-beta", type=float, default=1.0)
    parser.add_argument("--sp-mode", type=str, default="asp", choices=["asp", "betti1", "both"],
                        help="SP definition: asp (default), betti1, or both (parallel recording)")
    parser.add_argument("--two-graph", action="store_true", default=False,
                        help="Two-graph mode: skip hop loop, compare prev vs full graph (β₁ only, O(V+E))")
    parser.add_argument("--betti-scale-invariant", action="store_true", default=False,
                        help="Use scale-invariant β₁ normalization (SP-parity: normalize by max possible β₁)")
    parser.add_argument("--linkset-mode", action="store_true", default=True, help="Use linkset-based entropy calculation (paper mode, default: on)")
    parser.add_argument("--no-linkset-mode", dest="linkset_mode", action="store_false", help="Disable linkset-based entropy calculation")
    parser.add_argument("--linkset-base", type=str, default="mem", choices=["link","mem","pool"], help="Base set for linkset IG before: link=S_link, mem=memory candidates, pool=all candidates")
    parser.add_argument("--sp-scope", type=str, default="auto", choices=["auto", "union"], help="SP evaluation scope")
    parser.add_argument("--sp-hop-expand", type=int, default=0, help="Extra hops to expand SP neighborhood")
    parser.add_argument("--sp-boundary", type=str, default="trim", choices=["induced","trim","nodes"], help="SP boundary mode (Core) for subgraph evaluation")
    parser.add_argument("--theta-ag", type=float, default=-1.0, help="AG threshold for multi-hop evaluation (default: -1.0 to always evaluate)")
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
    parser.add_argument("--action-source", type=str, default="obs", choices=["obs", "mix"], help="Candidate source for action selection: obs=visual only, mix=obs+mem (still respects possible_moves).")
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
    parser.add_argument("--force-sp-gain-eval", action="store_true", help="Force SP gain evaluation at hop0 (diagnostic; combines with GED_min diag)")
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
    parser.add_argument("--timeline-to-graph", dest="timeline_to_graph", action="store_true", default=True, help="Add timeline edges (Q_prev→dir→Q_next) to graph (required for SP gain calculation)")
    parser.add_argument("--no-timeline-to-graph", dest="timeline_to_graph", action="store_false", help="Disable timeline edges in graph")
    parser.add_argument("--add-next-q", dest="add_next_q", action="store_true", help="Also add next-step Q node to graph at end of step. Default: off")
    # Paper preset: linkset IG + candidate-base GED + ig-hop-apply=all
    parser.add_argument("--preset", type=str, default=None, choices=["paper"], help="Quick preset: 'paper' = linkset IG + candidate-base GED + ig-hop-apply=all")
    parser.set_defaults(ged_hop0_const=True)
    # Lite path: delegate per-step evaluation to main L3 (query-centric)
    parser.add_argument("--use-main-l3", dest="use_main_l3", action="store_true", help="Use main L3GraphReasoner (query-centric) for per-step eval (hop0 only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-log", type=Path)
    parser.add_argument(
        "--dg-ledger-log",
        type=Path,
        default=None,
        help="If set, write DG ledger events as JSONL (commit/reject for multi-hop proposals)",
    )
    parser.add_argument(
        "--dg-ledger-mode",
        type=str,
        default="dg",
        choices=["commit", "dg", "all"],
        help="DG ledger emission mode: commit=only committed, dg=only DG-eligible (best_hop>=1), all=every step",
    )
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
    # Cortisol (stress/adaptation): define negative examples from "stuckness" without illegal-move probing (no GABA).
    parser.add_argument(
        "--cortisol-mode",
        type=str,
        choices=["off", "log"],
        default="off",
        help="Enable cortisol stress signal (log-only) to label negative steps when AG stays open too long or the agent gets stuck.",
    )
    parser.add_argument("--cortisol-ag-streak", type=int, default=30, help="Trigger cortisol after N consecutive AG-open steps (0-hop ambiguity).")
    parser.add_argument("--cortisol-stuck-streak", type=int, default=10, help="Trigger cortisol after N consecutive stuck steps (no-move/deadend/revisit).")
    parser.add_argument("--cortisol-repeat-visits", type=int, default=2, help="Treat a position as 'revisited' after this many visits (negative label).")
    # Sleep Q (value propagation) controls: used when sleep guide is 'prefer'
    parser.add_argument("--sleep-plan-beta", type=float, default=0.0, help="Bias strength for BFS Sleep plan action when --sleep-guide prefer (logit bonus).")
    parser.add_argument("--sleep-q-beta", type=float, default=0.0, help="DEPRECATED: Q bias now controlled by --action-temp. Ignored if set.")
    parser.add_argument("--sleep-q-gamma", type=float, default=0.99, help="Discount factor for Sleep Q-learning replay.")
    parser.add_argument("--sleep-q-alpha", type=float, default=0.4, help="Learning rate for Sleep Q-learning replay.")
    parser.add_argument("--sleep-q-iters", type=int, default=50, help="Replay iterations for Sleep Q-learning.")
    parser.add_argument("--sleep-q-step-penalty", type=float, default=-0.01, help="Per-step penalty used in Sleep Q-learning.")
    parser.add_argument("--sleep-q-goal-reward", type=float, default=1.0, help="Goal reward used in Sleep Q-learning (added when reaching goal).")
    parser.add_argument("--sleep-q-revisit-penalty", type=float, default=-0.2, help="Penalty used in Sleep Q-learning when a step is labeled revisit.")
    parser.add_argument("--sleep-q-deadend-penalty", type=float, default=0.0, help="Penalty used in Sleep Q-learning when landing in a dead-end.")
    parser.add_argument("--sleep-q-blocked-penalty", type=float, default=0.0, help="Penalty used in Sleep Q-learning when an action is blocked (no move).")
    parser.add_argument("--sleep-edge-weight", dest="sleep_edge_weight", action="store_true", help="Enable Sleep edge-weight prior derived from warmup transitions.")
    parser.add_argument("--sleep-edge-beta", type=float, default=1.0, help="Bias strength for Sleep edge-weight prior.")
    parser.add_argument("--sleep-edge-alpha", type=float, default=0.4, help="Goal-path reinforcement magnitude for Sleep edge weights.")
    parser.add_argument("--sleep-edge-gamma", type=float, default=0.95, help="Goal-path discount for Sleep edge weights.")
    parser.add_argument("--sleep-edge-alpha-explore", type=float, default=0.05, help="Positive update for novel moves in Sleep edge weights.")
    parser.add_argument("--sleep-edge-revisit-penalty", type=float, default=0.2, help="Penalty for revisits in Sleep edge weights.")
    parser.add_argument("--sleep-edge-deadend-penalty", type=float, default=0.2, help="Penalty for dead-end moves in Sleep edge weights.")
    parser.add_argument("--sleep-edge-blocked-penalty", type=float, default=0.2, help="Penalty for blocked moves in Sleep edge weights.")
    parser.add_argument("--sleep-edge-mode", type=str, default="mul", choices=["mul", "add"], help="How to apply edge weights: mul=scale similarity, add=logit bonus.")
    # Event-based prior (semantic space PoC)
    parser.add_argument("--event-weights", type=Path, default=None, help="Path to event_weights.json produced by export_event_weights.py")
    parser.add_argument("--event-beta", type=float, default=1.0, help="Bias strength for event-weighted prior (logit multiplier).")
    # Affordance bias (graph memory)
    parser.add_argument("--affordance-bias", action="store_true", help="Enable affordance bias stored on direction nodes (graph memory).")
    parser.add_argument("--affordance-beta", type=float, default=1.0, help="Bias strength for affordance memory (logit multiplier).")
    parser.add_argument("--affordance-lr", type=float, default=0.2, help="Update rate for affordance memory (per-step additive).")
    parser.add_argument("--affordance-clamp", type=float, default=3.0, help="Clamp range for affordance bias magnitude (abs max).")
    # Graph-persistent DG: vector mode and propagation parameters
    parser.add_argument("--vector-mode", type=str, default="standard", choices=["standard", "extended"],
                        help="Vector mode: standard (8D) or extended (10D with reward/propagated dims).")
    parser.add_argument("--propagated-alpha", type=float, default=1.0,
                        help="Scalar bonus weight for propagated values in action selection.")
    parser.add_argument("--propagated-mode", type=str, default="abs", choices=["abs", "gradient"],
                        help="How to use propagated values: abs (raw value) or gradient (prop(next) - prop(here)).")
    parser.add_argument("--wsw-cycles", type=int, default=1,
                        help="Number of Wake-Sleep cycles before final eval (1=W-S-W, 2=W-S-W-S-W). Warmup budget split evenly.")
    parser.add_argument("--advantage-commit", type=float, default=0.0,
                        help="Advantage-gated selection: if best_weight/second_weight > threshold, use argmax. 0=disabled.")
    parser.add_argument("--sleep-propagate-gamma", type=float, default=0.95,
                        help="Discount factor for graph reward propagation in Sleep phase.")
    parser.add_argument("--sleep-propagate-iters", type=int, default=50,
                        help="Max iterations for graph reward propagation in Sleep phase.")
    parser.add_argument("--sleep-propagate", type=str, choices=["on", "off", "replay"], default="on",
                        help="Sleep graph optimization. on = undirected max-propagation (saturates; "
                             "kept for v6_perseed comparability). replay = trajectory-based Q backup "
                             "(sleep_q table written onto nodes; directed, bounded, negatives survive). "
                             "off = inherit the raw Wake1 graph unchanged (ablation control).")
    # Curriculum (Wake→Sleep→Wake): warmup run to ensure goal experience, then eval guided by Sleep path plan
    parser.add_argument(
        "--curriculum-warmup-steps",
        type=int,
        default=0,
        help="If >0, run a warmup episode with this step cap (ensure goal experience), derive a Sleep path plan from experienced transitions, then run an eval episode with --max-steps guided by the plan.",
    )
    parser.add_argument(
        "--sleep-guide",
        type=str,
        choices=["off", "prefer", "override"],
        default="override",
        help="How to apply Sleep guidance during the eval episode: override=follow 1-step plan, prefer=soft bias using Sleep Q(s,a), off=disable.",
    )
    parser.add_argument("--force-per-hop", dest="force_per_hop", action="store_true", help="Force per-hop series via evaluator even in L3-only path")
    parser.add_argument("--eval-per-hop-on-ag", dest="eval_per_hop_on_ag", action="store_true", help="Fallback to evaluator (per-hop) only when AG fires in L3-only path")
    parser.add_argument("--dg-bfs-shortcut", dest="dg_bfs_shortcut", action="store_true", help="When DG fires, commit chosen multi-hop edges as a BFS-like shortcut in one step")
    # Three-layer search
    parser.add_argument("--search-mode", type=str, default="legacy", choices=["legacy", "threelayer"],
                        help="Search mode: legacy (default) or threelayer (L0 hash → L1 attention walk → L2 full sort)")
    parser.add_argument("--theta-attention", type=float, default=0.3, help="Attention threshold for L1 graph walk")
    parser.add_argument("--attention-decay", type=float, default=0.95, help="Per-step attention decay rate")
    parser.add_argument("--attention-boost", type=float, default=0.1, help="Attention boost on edge traversal")
    parser.add_argument("--attention-alpha", type=float, default=0.5, help="Attention exponent for effective_score")
    parser.add_argument("--min-layer1-candidates", type=int, default=2, help="Min L1 candidates to skip L2 fallback")
    parser.add_argument("--dg-gate-tau", type=float, default=1.0, help="DG gate temperature: σ(propagated/τ) modulates L1 effective_score")
    parser.add_argument("--l1-tau-dg", type=float, default=0.3, help="Temperature for σ(-dg_attention/τ) in 3-attention L1 scoring")
    parser.add_argument("--l1-tau-reward", type=float, default=0.3, help="Temperature for σ(reward_attention/τ) in 3-attention L1 scoring")
    parser.add_argument("--l1-score-mode", type=str, default="legacy", choices=["legacy", "3att"], help="L1 scoring: legacy (att^α×sim×gate) or 3att (ag×σ(-dg/τ)×σ(rw/τ))")
    parser.add_argument("--l1-score-switch-step", type=int, default=0, help="Step at which to switch from legacy to l1-score-mode (0=no switch, use l1-score-mode from start)")
    # Defaults
    parser.set_defaults(link_autowire_all=True)
    # Prefer recording per-hop on AG in L3-light path unless explicitly disabled
    parser.set_defaults(eval_per_hop_on_ag=True)
    return parser.parse_args()


def build_selector_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Build selector parameter dictionary from parsed args."""
    return {
        "theta_cand": args.theta_cand,
        "theta_link": args.theta_link,
        "candidate_cap": args.candidate_cap,
        "top_m": args.top_m,
        "cand_radius": args.cand_radius,
        "link_radius": args.link_radius,
    }


def build_gedig_params(args: argparse.Namespace, lambda_weight: float) -> Dict[str, Any]:
    """Build geDIG parameter dictionary from parsed args."""
    return {
        "lambda_weight": float(lambda_weight),
        "max_hops": args.max_hops,
        "decay_factor": args.decay_factor,
        "adaptive_hops": args.adaptive_hops,
        "sp_beta": args.sp_beta,
        "sp_scope_mode": args.sp_scope,
        "sp_hop_expand": args.sp_hop_expand,
        "sp_boundary_mode": args.sp_boundary,
        "ig_hop_apply": str(args.ig_hop_apply),
    }


def build_config(
    args: argparse.Namespace,
    *,
    lambda_weight: float,
    step_path: Optional[Path] = None,
    event_weights: Optional[Dict[str, float]] = None,
) -> QueryHubConfig:
    """Build QueryHubConfig from parsed arguments.

    Args:
        args: Parsed command-line arguments
        lambda_weight: Lambda weight value for this config
        step_path: Optional step log path (for checkpoint_path)
        event_weights: Optional event weight dictionary

    Returns:
        Configured QueryHubConfig instance
    """
    selector_params = build_selector_params(args)
    gedig_params = build_gedig_params(args, lambda_weight)

    return QueryHubConfig(
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
        action_source=str(getattr(args, "action_source", "obs")),
        anti_backtrack=bool(args.anti_backtrack),
        sleep_q_beta=0.0,  # deprecated: Q bias now controlled by action_temp
        sleep_plan_beta=float(getattr(args, "sleep_plan_beta", 0.0)),
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
        timeline_to_graph=bool(getattr(args, 'timeline_to_graph', True)),
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
        checkpoint_path=(str(step_path) if step_path else None),
        post_sp_diagnostics=bool(getattr(args, 'post_sp_diagnostics', True)),
        steps_ultra_light=bool(getattr(args, 'steps_ultra_light', False)),
        maze_snapshot_out=(str(getattr(args, 'maze_snapshot_out', '')).strip() or None),
        cortisol_mode=str(getattr(args, "cortisol_mode", "off")),
        cortisol_ag_streak=int(getattr(args, "cortisol_ag_streak", 30)),
        cortisol_stuck_streak=int(getattr(args, "cortisol_stuck_streak", 10)),
        cortisol_repeat_visits=int(getattr(args, "cortisol_repeat_visits", 2)),
        sleep_q_gamma=float(getattr(args, "sleep_q_gamma", 0.99)),
        sleep_q_alpha=float(getattr(args, "sleep_q_alpha", 0.4)),
        sleep_q_iters=int(getattr(args, "sleep_q_iters", 50)),
        sleep_q_step_penalty=float(getattr(args, "sleep_q_step_penalty", -0.01)),
        sleep_q_goal_reward=float(getattr(args, "sleep_q_goal_reward", 1.0)),
        sleep_q_revisit_penalty=float(getattr(args, "sleep_q_revisit_penalty", -0.2)),
        sleep_q_deadend_penalty=float(getattr(args, "sleep_q_deadend_penalty", 0.0)),
        sleep_q_blocked_penalty=float(getattr(args, "sleep_q_blocked_penalty", 0.0)),
        sleep_edge_enabled=bool(getattr(args, "sleep_edge_weight", False)),
        sleep_edge_beta=float(getattr(args, "sleep_edge_beta", 1.0)),
        sleep_edge_alpha=float(getattr(args, "sleep_edge_alpha", 0.4)),
        sleep_edge_gamma=float(getattr(args, "sleep_edge_gamma", 0.95)),
        sleep_edge_alpha_explore=float(getattr(args, "sleep_edge_alpha_explore", 0.05)),
        sleep_edge_revisit_penalty=float(getattr(args, "sleep_edge_revisit_penalty", 0.2)),
        sleep_edge_deadend_penalty=float(getattr(args, "sleep_edge_deadend_penalty", 0.2)),
        sleep_edge_blocked_penalty=float(getattr(args, "sleep_edge_blocked_penalty", 0.2)),
        sleep_edge_mode=str(getattr(args, "sleep_edge_mode", "mul")),
        vector_mode=str(getattr(args, "vector_mode", "standard")),
        propagated_alpha=float(getattr(args, "propagated_alpha", 1.0)),
        propagated_mode=str(getattr(args, "propagated_mode", "abs")),
        advantage_commit=float(getattr(args, "advantage_commit", 0.0)),
        sleep_propagate_gamma=float(getattr(args, "sleep_propagate_gamma", 0.95)),
        sleep_propagate_iters=int(getattr(args, "sleep_propagate_iters", 50)),
        sleep_propagate=str(getattr(args, "sleep_propagate", "on")),
        event_weights=dict(event_weights or {}),
        event_beta=float(getattr(args, "event_beta", 1.0)),
        affordance_bias=bool(getattr(args, "affordance_bias", False)),
        affordance_beta=float(getattr(args, "affordance_beta", 1.0)),
        affordance_lr=float(getattr(args, "affordance_lr", 0.2)),
        affordance_clamp=float(getattr(args, "affordance_clamp", 3.0)),
        force_per_hop=bool(getattr(args, 'force_per_hop', False)),
        eval_per_hop_on_ag=bool(getattr(args, 'eval_per_hop_on_ag', False)),
        dg_bfs_shortcut=bool(getattr(args, 'dg_bfs_shortcut', False)),
        force_sp_gain_eval=bool(getattr(args, 'force_sp_gain_eval', False)),
        search_mode=str(getattr(args, "search_mode", "legacy")),
        theta_attention=float(getattr(args, "theta_attention", 0.3)),
        attention_decay=float(getattr(args, "attention_decay", 0.95)),
        attention_boost=float(getattr(args, "attention_boost", 0.1)),
        attention_alpha=float(getattr(args, "attention_alpha", 0.5)),
        min_layer1_candidates=int(getattr(args, "min_layer1_candidates", 2)),
        dg_gate_tau=float(getattr(args, "dg_gate_tau", 1.0)),
        l1_tau_dg=float(getattr(args, "l1_tau_dg", 0.3)),
        l1_tau_reward=float(getattr(args, "l1_tau_reward", 0.3)),
        l1_score_mode=str(getattr(args, "l1_score_mode", "legacy")),
        l1_score_switch_step=int(getattr(args, "l1_score_switch_step", 0)),
        sp_mode=str(getattr(args, "sp_mode", "asp")),
        two_graph_mode=bool(getattr(args, "two_graph", False)),
        betti_scale_invariant=bool(getattr(args, "betti_scale_invariant", False)),
    )
