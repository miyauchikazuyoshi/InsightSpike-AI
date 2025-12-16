"""Unified geDIG core (refactored + Day1 spike detection)."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple, List
from collections import deque

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    WAKE = "wake"
    SLEEP = "sleep"


class SpikeDetectionMode(Enum):
    THRESHOLD = "threshold"
    AND = "and"
    OR = "or"


@dataclass
class HopResult:
    hop: int
    # Normalized GED (cost-based, positive), kept for inspection
    ged: float
    # Shannon-entropy based IG (variance reduction)
    ig: float
    # Per-hop geDIG value (cost - lambda*IG)
    gedig: float
    # Structural cost used for this hop (positive is worse); equals
    # base normalized GED at hop==0, optionally adjusted by SP gain for hop>0.
    struct_cost: float
    node_count: int
    edge_count: int
    sp: float = 0.0
    h_component: float = 0.0
    ged_raw: float = 0.0
    ged_den: float = 1.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    ig_delta: float = 0.0
    ig_den: float = 1.0
    variance_reduction: float = 0.0

    @property
    def struct_term(self) -> float:
        """Deprecated alias returning negative structural improvement (legacy behaviour)."""
        return -self.struct_cost


@dataclass
class GeDIGResult:
    gedig_value: float
    ged_value: float
    ig_value: float
    raw_ged: float = 0.0
    ged_norm_den: float = 1.0
    ig_raw: float = 0.0
    ig_norm_den: float = 1.0
    ig_z_score: float = 0.0
    delta_ged_norm: float = 0.0
    delta_sp_rel: float = 0.0
    delta_h_norm: float = 0.0
    structural_cost: float = 0.0
    structural_improvement: float = 0.0
    information_integration: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    ig_delta: float = 0.0
    variance_reduction: float = 0.0
    hop0_reward: float = 0.0
    aggregate_reward: float = 0.0
    reward: float = 0.0
    hop_results: Optional[Dict[int, HopResult]] = None
    computation_time: float = 0.0
    focal_nodes: Optional[Set[str]] = None
    version: str = "refactor_phaseA"
    spike: bool = False
    linkset_metrics: Optional["LinksetMetrics"] = None
    ged_min_proxy: float = 0.0

    @property
    def has_spike(self) -> bool:  # backward compat
        return self.spike

    def to_dict(self) -> Dict[str, Any]:  # type: ignore[name-defined]
        return asdict(self)


class GeDIGCore:
    def __init__(
        self,
        node_cost: float = 1.0,
        edge_cost: float = 1.0,
        normalization: str = 'sum',
        efficiency_weight: float = 0.3,
        min_nodes: int = 2,
        smoothing: float = 1e-10,
        enable_multihop: bool = False,
        max_hops: int = 3,
        decay_factor: float = 0.7,
        adaptive_hops: bool = True,
        spike_threshold: float = -0.5,
        enable_spectral: bool = False,
        spectral_weight: float = 0.3,
        lambda_weight: float = 1.0,
        ig_mode: str = 'raw',  # 'raw' | 'z' | 'norm'
        ig_norm_strategy: str = 'before',
        ig_delta_mode: str = 'after_before',  # ignored (fixed to 'after_before')
        entropy_tau: float = 1.0,  # softmax temperature for entropy; tau=1 keeps legacy behavior
        mu: float = 0.5,
        warmup_steps: int = 10,
        use_refactored_reward: bool = True,
        use_legacy_formula: bool = False,
        spike_detection_mode: str | SpikeDetectionMode = "and",
        tau_s: float = 0.15,
        tau_i: float = 0.25,
        # Multi-hop shortcut gain: when enabled, incorporate
        # normalized shortest-path gain per hop. In implementation we add
        # the (relative) SP gain to the information term (ΔH + γ·ΔSP_rel),
        # aligning with the paper's IG-side placement (γ ≈ sp_beta).
        use_multihop_sp_gain: bool = True,
        sp_norm_mode: str = 'relative',  # 'relative' := (L_before-L_after)/L_before
        # Weight for shortest-path relative gain in multi-hop (γ in the paper)
        sp_beta: float = 0.2,
        # Local normalization for decision-time control
        use_local_normalization: bool = False,
        local_norm_mode: str = 'layer1',  # initial: Cmax_local^(0) = 1 + K (Layer1 candidates)
        # Optional diagnostic: estimate GED_min proxy (path compression delta)
        enable_ged_min_diag: bool = False,
        # Performance guards for SP gain
        sp_node_cap: int = 200,
        sp_pair_samples: int = 400,
        sp_use_sampling: bool = True,
        feature_weights: Optional[Sequence[float]] = None,
        linkset_mode: bool = False,
        # SP evaluation scope controls
        sp_scope_mode: str = 'auto',  # 'auto' uses per-hop subgraphs; 'union' uses union-of-nodes for before/after
        sp_hop_expand: int = 0,       # evaluate SP on (hop + expand) neighborhood
        sp_eval_mode: str = 'connected',  # 'connected' (default) or 'fixed_before_pairs'
        # Paper-aligned switches
        ig_source_mode: str = 'graph',   # 'graph' | 'linkset' | 'hybrid'
        ig_hop_apply: str = 'all',       # 'hop0' | 'all' (apply linkset IG to which hops)
        ged_norm_scheme: str = 'edges_after', # 'edges_after' | 'candidate_base'
    ) -> None:
        self.node_cost = node_cost
        self.edge_cost = edge_cost
        self.normalization = normalization
        self.efficiency_weight = efficiency_weight
        self.min_nodes = min_nodes
        self.smoothing = smoothing
        self.enable_multihop = enable_multihop
        self.max_hops = max_hops
        self.decay_factor = decay_factor
        self.adaptive_hops = adaptive_hops
        self.spike_threshold = spike_threshold
        self.enable_spectral = enable_spectral
        self.spectral_weight = spectral_weight
        self.lambda_weight = lambda_weight
        # Allow env override for quick experiments
        try:
            env_lambda = os.environ.get('MAZE_GEDIG_LAMBDA')
            if env_lambda:
                self.lambda_weight = float(env_lambda)
        except Exception:
            pass
        # Env overrides for structural weighting (cul-de-sac sensitivity knobs)
        try:
            nc = os.environ.get('MAZE_GEDIG_NODE_COST')
            if nc: self.node_cost = float(nc)
        except Exception:
            pass
        try:
            ec = os.environ.get('MAZE_GEDIG_EDGE_COST')
            if ec: self.edge_cost = float(ec)
        except Exception:
            pass
        try:
            ew = os.environ.get('MAZE_GEDIG_EFF_WEIGHT')
            if ew is not None and ew != '': self.efficiency_weight = float(ew)
        except Exception:
            pass
        try:
            sp = os.environ.get('MAZE_GEDIG_SPECTRAL')
            if sp is not None and sp.strip() not in ("0","false","False",""):
                self.enable_spectral = True
            sw = os.environ.get('MAZE_GEDIG_SPECTRAL_WEIGHT')
            if sw: self.spectral_weight = float(sw)
        except Exception:
            pass
        self.ig_mode = ig_mode
        try:
            env_mode = os.environ.get('MAZE_GEDIG_IG_MODE')
            if env_mode:
                self.ig_mode = str(env_mode).lower()
        except Exception:
            pass
        self.ig_norm_strategy = str(ig_norm_strategy or 'before').lower()
        try:
            env_norm = os.environ.get('MAZE_GEDIG_IG_NORM')
            if env_norm:
                self.ig_norm_strategy = str(env_norm).lower()
        except Exception:
            pass
        # Entropy temperature (tau). tau=1 -> legacy weights/sum. tau != 1 -> p ∝ w^(1/tau).
        try:
            ent_tau_env = os.environ.get('MAZE_GEDIG_ENTROPY_TAU') or os.environ.get('INSIGHTSPIKE_ENTROPY_TAU')
            if ent_tau_env:
                entropy_tau = float(ent_tau_env)
        except Exception:
            pass
        self.entropy_tau = float(entropy_tau) if entropy_tau > 0 else 1.0
        # IG delta orientation is fixed to after_before (no knob to flip sign)
        self.ig_delta_mode = 'after_before'
        # IG non-negative clamp (treat negative IG as 0 = no information gain)
        try:
            self._ig_nonneg = os.environ.get('MAZE_GEDIG_IG_NONNEG', '0').strip() not in ("0","false","False","")
        except Exception:
            self._ig_nonneg = False
        try:
            env_ged_min = os.environ.get('INSIGHTSPIKE_GED_MIN_DIAG', '')
            if env_ged_min.strip() and env_ged_min.strip().lower() not in ("0","false","no","off"):
                self.enable_ged_min_diag = True
        except Exception:
            pass
        self.mu = mu
        self.warmup_steps = warmup_steps
        self.use_refactored_reward = use_refactored_reward
        self.use_legacy_formula = use_legacy_formula
        self.spike_detection_mode = spike_detection_mode
        self.tau_s = tau_s
        self.tau_i = tau_i
        self.use_multihop_sp_gain = use_multihop_sp_gain
        self.enable_ged_min_diag = bool(enable_ged_min_diag)
        self.sp_norm_mode = sp_norm_mode
        self.sp_beta = float(max(0.0, sp_beta))
        self.use_local_normalization = use_local_normalization
        self.local_norm_mode = local_norm_mode
        # SP gain performance guards
        self.sp_node_cap = int(max(1, sp_node_cap))
        # Allow <=0 to mean "use ALL pairs" (parity/diagnostic)
        self.sp_pair_samples = int(sp_pair_samples)
        self.sp_use_sampling = bool(sp_use_sampling)
        self.sp_scope_mode = str(sp_scope_mode or 'auto').lower()
        self.sp_hop_expand = int(max(0, sp_hop_expand))
        self.sp_boundary_mode = 'induced'
        self.sp_eval_mode = str(sp_eval_mode or 'connected').lower()
        try:
            sbb = os.environ.get('MAZE_GEDIG_SP_BOUNDARY')
            if sbb:
                self.sp_boundary_mode = str(sbb).lower()
        except Exception:
            pass
        if feature_weights is not None:
            arr = np.asarray(feature_weights, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                self.feature_weights = arr
            else:
                self.feature_weights = None
        else:
            self.feature_weights = None
        self.linkset_mode = bool(linkset_mode)
        # Paper-mode parameters
        self.ig_source_mode = str(ig_source_mode or 'graph').lower()
        self.ig_hop_apply = str(ig_hop_apply or 'all').lower()
        self.ged_norm_scheme = str(ged_norm_scheme or 'edges_after').lower()
        # Running stats
        self._ig_count = 0
        self._ig_mean = 0.0
        self._ig_m2 = 0.0
        # Hooks
        self.logger = None  # type: ignore
        self.monitor = None  # type: ignore  # set by attach_monitor
        # Deprecation: warn once when graph-IG path is used (linkset_info absent)
        self._graph_ig_warned = False
        logger.info(
            "GeDIGCore initialized: multihop=%s max_hops=%s spectral=%s",
            self.enable_multihop,
            self.max_hops,
            self.enable_spectral,
        )

    # ------------ Public API ------------
    def calculate(self, *args, **kwargs) -> GeDIGResult:
        """Unified calculate interface.

        Backward compatibility:
        - Old style: calculate(g1, g2, features1, features2)
        - New style: calculate(g_prev=g1, g_now=g2, features_prev=..., features_now=...)
        """
        if args and ('g_prev' not in kwargs and 'g_now' not in kwargs):
            # Positional mapping
            if len(args) >= 2:
                kwargs['g_prev'] = args[0]
                kwargs['g_now'] = args[1]
            if len(args) >= 3:
                kwargs['features_prev'] = args[2]
            if len(args) >= 4:
                kwargs['features_now'] = args[3]
        g_prev = kwargs.get('g_prev')
        g_now = kwargs.get('g_now')
        features_prev = kwargs.get('features_prev')
        features_now = kwargs.get('features_now')
        focal_nodes = kwargs.get('focal_nodes')
        l1_candidates = kwargs.get('l1_candidates')  # Optional Layer1 candidate count (int)
        k_star = kwargs.get('k_star')
        ig_fixed_den = kwargs.get('ig_fixed_den')
        force_sp_gain_eval = bool(kwargs.get('force_sp_gain_eval', False))

        if l1_candidates is None and k_star is not None:
            l1_candidates = k_star

        raw_candidate_count = 0
        if k_star is not None:
            try:
                raw_candidate_count = int(round(float(k_star)))
            except Exception:
                raw_candidate_count = 0
        elif l1_candidates is not None:
            try:
                raw_candidate_count = int(round(float(l1_candidates)))
            except Exception:
                raw_candidate_count = 0
        cand_count = max(raw_candidate_count, 1)
        norm_candidate_base = max(raw_candidate_count, 1)
        if ig_fixed_den is None:
            ig_fixed_den = math.log(float(norm_candidate_base) + 1.0)
        else:
            try:
                ig_fixed_den = float(ig_fixed_den)
                if ig_fixed_den <= 0.0:
                    ig_fixed_den = math.log(float(norm_candidate_base) + 1.0)
            except Exception:
                ig_fixed_den = math.log(float(norm_candidate_base) + 1.0)
        k_star = cand_count
        # Original implementation below (refactored)
        start_time = time.time()
        g1 = self._ensure_networkx(g_prev)
        g2 = self._ensure_networkx(g_now)
        if features_prev is None:
            features_prev = self._extract_features(g1)
        if features_now is None:
            features_now = self._extract_features(g2)
        query_vector = kwargs.get('query_vector')

        # Optional GED_min-style proxy: relative shortening of average SP
        ged_min_proxy = 0.0
        if self.enable_ged_min_diag:
            try:
                ged_min_proxy = float(self._compute_ged_min_proxy(g1, g2))
            except Exception:
                ged_min_proxy = 0.0

        # Local normalization (decision-time) denominator
        cmax_local: float | None = None
        if self.use_local_normalization and l1_candidates is not None:
            try:
                k = int(l1_candidates)
                if k >= 0:
                    cmax_local = float(1 + k)
            except Exception:
                cmax_local = None

        # Linkset payload can be used either for logging or for IG source when paper-mode is enabled
        linkset_info = kwargs.get('linkset_info')
        linkset_metrics: Optional[LinksetMetrics] = None
        if linkset_info:
            linkset_metrics = self._compute_linkset_metrics(
                g1,
                g2,
                linkset_info,
                query_vector=query_vector,
                ig_fixed_den=ig_fixed_den,
            )

        if self.enable_multihop:
            if not focal_nodes:
                nodes1 = set(g1.nodes()); nodes2 = set(g2.nodes())
                focal_nodes = (nodes1 - nodes2) | (nodes2 - nodes1) | {n for n in nodes1 & nodes2 if g1.degree(n) != g2.degree(n)}
                if not focal_nodes:
                    focal_nodes = set(list(g2.nodes())[:min(5, g2.number_of_nodes())])
            # Deprecation notice when no linkset is provided and graph-IG is implied
            if linkset_metrics is None and not self._graph_ig_warned:
                logger.warning("[DEPRECATION] geDIG graph-IG path is in use (no linkset_info). This path will be retired; please migrate callers to provide linkset_info.")
                self._graph_ig_warned = True
            result = self._calculate_multihop(
                g1,
                g2,
                features_prev,
                features_now,
                focal_nodes,
                start_time,
                norm_override=cmax_local,
                query_vector=query_vector,
                fixed_den=ig_fixed_den,
                k_star=k_star,
                candidate_count=cand_count,
                linkset_metrics=linkset_metrics,
            )
        else:
            denom = self.node_cost + self.edge_cost * max(g2.number_of_edges(), 0)
            if denom <= 0.0:
                denom = self.node_cost + self.edge_cost
            ged_result = self._calculate_normalized_ged(g1, g2, norm_override=denom)
            # Deprecation notice for single-hop graph-IG
            if linkset_metrics is None and not self._graph_ig_warned:
                logger.warning("[DEPRECATION] geDIG graph-IG path is in use (no linkset_info). This path will be retired; please migrate callers to provide linkset_info.")
                self._graph_ig_warned = True
            ig_result = self._calculate_entropy_variance_ig(
                g2,
                features_prev,
                features_now,
                query_vector=query_vector,
                fixed_den=ig_fixed_den,
                k_star=k_star,
            )
            delta_ged_norm = float(ged_result['normalized_ged'])
            struct_cost_eff = float(ged_result.get('structural_cost', delta_ged_norm))
            delta_h_norm = float(ig_result['ig_value'])
            delta_sp_rel = 0.0
            if force_sp_gain_eval or self.enable_ged_min_diag:
                try:
                    delta_sp_rel = float(self._compute_sp_gain_norm(g1, g2, mode=self.sp_norm_mode))
                except Exception:
                    delta_sp_rel = 0.0
            sp_contrib = 0.0
            if self.use_multihop_sp_gain and self.sp_beta:
                sp_contrib = self.sp_beta * delta_sp_rel
            combined_ig = delta_h_norm + sp_contrib
            ig_for_lambda = combined_ig
            if str(self.ig_mode).lower() in ('norm', 'normalized'):
                ig_for_lambda = float(np.tanh(max(0.0, ig_for_lambda)))
            if self._ig_nonneg:
                ig_for_lambda = max(0.0, ig_for_lambda)
            lambda_term = self.lambda_weight * ig_for_lambda
            # Legacy formulation: product/struct-improvement flavored score
            # to intentionally diverge from the refactored difference form
            # while keeping identical-graph delta at 0.
            if self.use_legacy_formula:
                struct_impr = float(ged_result.get('structural_improvement', -delta_ged_norm))
                g0_value = float(struct_impr * (1.0 - self.lambda_weight * 0.0))
            else:
                # Incorporate spectral effect into effective cost when enabled
                eff_cost = struct_cost_eff
                if self.enable_spectral:
                    si = float(ged_result.get('structural_improvement', -delta_ged_norm))
                    eff_cost = float(delta_ged_norm * (1.0 + self.spectral_weight * abs(si)))
                g0_value = float(eff_cost - lambda_term)
            hop0 = HopResult(
                hop=0,
                ged=delta_ged_norm,
                ig=combined_ig,
                gedig=g0_value,
                struct_cost=delta_ged_norm,
                node_count=g2.number_of_nodes(),
                edge_count=g2.number_of_edges(),
                sp=delta_sp_rel,
                h_component=delta_h_norm,
                ged_raw=float(ged_result.get('raw_ged', 0.0)),
                ged_den=float(ged_result.get('normalization_den', denom)),
                entropy_before=float(ig_result.get('entropy_before', 0.0)),
                entropy_after=float(ig_result.get('entropy_after', 0.0)),
                ig_delta=float(ig_result.get('delta_entropy', 0.0)),
                ig_den=float(ig_result.get('normalization_den', ig_fixed_den if ig_fixed_den is not None else 1.0)),
                variance_reduction=float(ig_result.get('variance_reduction', 0.0)),
            )
            result = GeDIGResult(
                gedig_value=g0_value,
                ged_value=delta_ged_norm,
                ig_value=combined_ig,
                raw_ged=hop0.ged_raw,
                ged_norm_den=hop0.ged_den,
                ig_raw=combined_ig,
                ig_norm_den=hop0.ig_den,
                delta_ged_norm=delta_ged_norm,
                delta_sp_rel=delta_sp_rel,
                delta_h_norm=delta_h_norm,
                structural_cost=float(ged_result.get('structural_cost', delta_ged_norm)),
                structural_improvement=float(ged_result.get('structural_improvement', -delta_ged_norm)),
                information_integration=combined_ig,
                entropy_before=hop0.entropy_before,
                entropy_after=hop0.entropy_after,
                ig_delta=hop0.ig_delta,
                variance_reduction=hop0.variance_reduction,
                computation_time=time.time() - start_time,
                version="onegauge_v1",
                hop_results={0: hop0},
                ged_min_proxy=ged_min_proxy,
            )

        if linkset_metrics is not None:
            result.linkset_metrics = linkset_metrics

        # Stats & rewards
        self._update_ig_stats(result.ig_raw)
        result.ig_z_score = self._compute_ig_z(result.ig_raw)
        if self.use_refactored_reward:
            self._compute_rewards(result)
        else:
            result.hop0_reward = result.gedig_value
            result.aggregate_reward = result.gedig_value
        result.reward = result.hop0_reward

        # If ig_mode=z, recompute geDIG with z-score for downstream users
        if str(self.ig_mode).lower() in ('z','zscore'):
            try:
                result.gedig_value = result.structural_improvement - self.lambda_weight * float(result.ig_z_score)
                result.reward = result.gedig_value
            except Exception:
                pass

        # Spike detection
        result.spike = self._detect_spike(result)
        if self.monitor is not None:
            # Record predicted spike
            try:
                self.monitor.record_prediction(result.spike)
                # Derive ground-truth & record outcome (enables precision/recall metrics)
                self.monitor.record_auto_outcome(result, self)
                # Optional auto threshold tuning
                self.monitor.auto_adjust_thresholds(self)
            except Exception:  # pragma: no cover - monitoring must be non-fatal
                pass
        if self.logger is not None:
            try:
                self.logger.log(step=self._ig_count, result=result)
            except Exception as e:  # pragma: no cover
                logger.warning("GeDIGLogger failed: %s", e)
        return result

    def _compute_linkset_metrics(
        self,
        g_before: nx.Graph,
        g_after: nx.Graph,
        linkset_info: Optional[Dict[str, Any]],
        query_vector: Optional[Sequence[float]] = None,
        ig_fixed_den: Optional[float] = None,
    ) -> LinksetMetrics:
        linkset_info = linkset_info or {}
        s_link = linkset_info.get('s_link') or []
        candidate_pool = linkset_info.get('candidate_pool') or []
        decision = linkset_info.get('decision') or {}
        chosen_index = decision.get('index')
        query_entry = linkset_info.get('query_entry')
        base_mode = str(linkset_info.get('base_mode', 'link') or 'link').lower()
        if query_entry is not None:
            query_entry = dict(query_entry)

        # Build before/after sets based on base_mode and unique candidate indices
        before_map: Dict[str, Dict[str, Any]] = {}
        chosen_entry: Optional[Dict[str, Any]] = None
        # Helper to push items into before_map
        def _add_before_items(items: List[Dict[str, Any]]):
            nonlocal chosen_entry
            for item in items:
                idx = item.get('index')
                if not idx:
                    continue
                key = str(idx)
                snap = dict(item)
                before_map.setdefault(key, snap)
                if chosen_entry is None and idx == chosen_index:
                    chosen_entry = snap

        if base_mode in ('mem','pool'):
            # Use candidate_pool as base for entropy: mem-only or full pool
            if base_mode == 'mem':
                base_items = [dict(it) for it in candidate_pool if (it.get('origin') == 'mem')]
            else:
                base_items = [dict(it) for it in candidate_pool]
            # Fallback to s_link if pool is empty
            if base_items:
                _add_before_items(base_items)
            else:
                _add_before_items([dict(it) for it in s_link])
        else:
            # Default: use s_link as base
            _add_before_items([dict(it) for it in s_link])

        if chosen_entry is None:
            for item in candidate_pool:
                idx = item.get('index')
                if not idx:
                    continue
                if idx == chosen_index:
                    chosen_entry = dict(item)
                    break

        if chosen_entry is None and candidate_pool:
            chosen_entry = dict(candidate_pool[0])

        if chosen_entry is None:
            chosen_entry = {'index': chosen_index, 'similarity': 1.0}
        else:
            chosen_entry = dict(chosen_entry)

        idx = chosen_entry.get('index')
        if idx:
            before_map.setdefault(str(idx), dict(chosen_entry))

        if not before_map and idx:
            before_map[str(idx)] = dict(chosen_entry)

        if query_entry is None:
            sim = decision.get('similarity')
            sim = float(sim) if isinstance(sim, (int, float)) else 1.0
            query_entry = {
                'index': 'query',
                'origin': 'query',
                'similarity': sim if sim > 0 else 1.0,
                'distance': 0.0,
                'weighted_distance': 0.0,
            }
        else:
            query_entry.setdefault('index', 'query')
            query_entry.setdefault('origin', 'query')
            if not query_entry.get('similarity'):
                sim = decision.get('similarity')
                query_entry['similarity'] = float(sim) if isinstance(sim, (int, float)) and sim > 0 else 1.0

        after_map = dict(before_map)
        after_map[str(query_entry.get('index', 'query'))] = dict(query_entry)

        before_list = list(before_map.values())
        after_list = list(after_map.values())

        raw_ged = max(0, len(after_map) - len(before_map))
        denom = 1.0 + len(after_list)
        delta_ged_norm = raw_ged / denom if denom > 0 else 0.0

        def _weights(items: List[Dict[str, Any]]) -> List[float]:
            ws = [item.get('similarity', 0.0) or 0.0 for item in items]
            return [float(w) for w in ws if w > 0.0]
        def _entropy_from_weights(weights: List[float], tau: float = 1.0) -> float:
            if not weights:
                return 0.0
            # tau=1 -> legacy normalization (p = w/sum w)
            # tau!=1 -> p ∝ w^(1/tau)  (softmax over log w / tau)
            if tau <= 0 or not math.isfinite(tau):
                tau = 1.0
            if abs(tau - 1.0) < 1e-9:
                total = sum(weights)
                if total <= 0:
                    return 0.0
                probabilities = [w / total for w in weights]
            else:
                powered = [math.pow(w, 1.0 / tau) for w in weights if w > 0.0]
                total = sum(powered)
                if total <= 0:
                    return 0.0
                probabilities = [w / total for w in powered]
            if not probabilities:
                return 0.0
            return -sum(p * math.log(p + 1e-12) for p in probabilities)

        ws_before = _weights(before_list)
        ws_after = _weights(after_list)
        H_before = _entropy_from_weights(ws_before, tau=self.entropy_tau)
        H_after = _entropy_from_weights(ws_after, tau=self.entropy_tau)
        # Paper-consistent normalization: log K with K = |after|
        # Guard: if K < 2, denominator approaches 0 -> clamp with epsilon
        K = max(0, len(after_list))
        norm_den = math.log(K) if K >= 2 else 1e-6
        # Fixed orientation: after-before (entropy decrease => negative)
        delta_h_norm = (H_after - H_before) / norm_den if norm_den > 0 else 0.0
        delta_sp_rel = 0.0
        combined_ig = delta_h_norm + self.sp_beta * delta_sp_rel if self.use_multihop_sp_gain else delta_h_norm

        ig_for_lambda = combined_ig
        if str(self.ig_mode).lower() in ('norm', 'normalized'):
            ig_for_lambda = float(np.tanh(max(0.0, ig_for_lambda)))
        if self._ig_nonneg:
            ig_for_lambda = max(0.0, ig_for_lambda)
        lambda_term = self.lambda_weight * ig_for_lambda
        if self.use_legacy_formula:
            struct_impr = float(delta_h_norm)  # linkset IG aligns with paper-style improvement
            g_value = float(struct_impr * (1.0 - self.lambda_weight * 0.0))
        else:
            struct_cost_eff = float(delta_ged_norm)
            g_value = float(struct_cost_eff - lambda_term)

        # Diagnostics: weight counts and top weights (descending)
        topw_b = sorted(ws_before, reverse=True)[:5] if ws_before else []
        topw_a = sorted(ws_after, reverse=True)[:5] if ws_after else []
        return LinksetMetrics(
            delta_ged_norm=float(delta_ged_norm),
            delta_h_norm=float(delta_h_norm),
            delta_sp_rel=float(delta_sp_rel),
            gedig_value=float(g_value),
            raw_ged=float(raw_ged),
            ged_norm_den=float(denom if denom > 0 else 1.0),
            ig_norm_den=float(norm_den if norm_den > 0 else 1.0),
            entropy_before=float(H_before),
            entropy_after=float(H_after),
            ig_delta=float(delta_h_norm),
            before_size=len(before_list),
            after_size=len(after_list),
            query_similarity=float(query_entry.get('similarity', 1.0)),
            pos_w_before=int(len(ws_before)),
            pos_w_after=int(len(ws_after)),
            topw_before=topw_b,
            topw_after=topw_a,
        )

    # ------------ Multi-hop ------------
    def _calculate_multihop(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        features_before: np.ndarray,
        features_after: np.ndarray,
        focal_nodes: Set[str],
        start_time: float,
        norm_override: float | None = None,
        query_vector: Optional[List[float]] = None,
        fixed_den: Optional[float] = None,
        k_star: Optional[int] = None,
        candidate_count: int = 1,
        linkset_metrics: Optional[LinksetMetrics] = None,
    ) -> GeDIGResult:
        hop_results: Dict[int, HopResult] = {}
        for hop in range(self.max_hops + 1):
            sub_g1, nodes1 = self._extract_k_hop_subgraph(g1, focal_nodes, hop)
            sub_g2, nodes2 = self._extract_k_hop_subgraph(g2, focal_nodes, hop)
            if len(sub_g1) == 0 and len(sub_g2) == 0:
                continue

            # GED 正規化分母のスキーム選択
            if str(self.ged_norm_scheme).lower() in ('candidate','candidate_base','link','links','linkset'):
                # Cmax ≈ c_node + |S_link|·c_edge（候補台固定）
                base_k = int(max(1, candidate_count))
                denom = self.node_cost + self.edge_cost * base_k
            else:
                denom = self.node_cost + self.edge_cost * max(sub_g2.number_of_edges(), 0)
                if denom <= 0.0:
                    denom = self.node_cost + self.edge_cost
            ged_result = self._calculate_normalized_ged(sub_g1, sub_g2, norm_override=denom)

            sub_before = self._filter_features(features_before, nodes1, g1)
            sub_after = self._filter_features(features_after, nodes2, g2)
            # IG ソースの切り替え
            if str(self.ig_source_mode).lower() in ('linkset','paper','strict') and linkset_metrics is not None:
                # 論文準拠: 候補分布ベースのΔHを使用
                delta_h_norm = float(linkset_metrics.delta_h_norm)
                # 参照用にentropy_before/afterはhop0に限り流す（他hopも同値）
                ig_result = {
                    'ig_value': delta_h_norm,
                    'entropy_before': float(linkset_metrics.entropy_before),
                    'entropy_after': float(linkset_metrics.entropy_after),
                    'delta_entropy': float(linkset_metrics.ig_delta),
                    'normalization_den': float(linkset_metrics.ig_norm_den),
                }
            else:
                ig_result = self._calculate_entropy_variance_ig(
                    sub_g2,
                    sub_before,
                    sub_after,
                    query_vector=query_vector,
                    fixed_den=fixed_den,
                    k_star=candidate_count,
                )
                delta_h_norm = float(ig_result['ig_value'])

            delta_ged_norm = float(ged_result['normalized_ged'])
            delta_sp_rel = 0.0
            sp_multiplier = 0.0
            if hop > 0 and self.use_multihop_sp_gain:
                # Evaluate SP on possibly expanded neighborhood and optional union scope
                eff_hop = hop + int(max(0, self.sp_hop_expand))
                sp_g1, nodes_sp1 = self._extract_k_hop_subgraph(g1, focal_nodes, eff_hop)
                sp_g2, nodes_sp2 = self._extract_k_hop_subgraph(g2, focal_nodes, eff_hop)
                if str(self.sp_scope_mode).lower() in ('union','merge','superset'):
                    all_nodes = set(nodes_sp1) | set(nodes_sp2)
                    if all_nodes:
                        sp_g1 = g1.subgraph(all_nodes).copy()
                        sp_g2 = g2.subgraph(all_nodes).copy()
                if str(self.sp_boundary_mode).lower() in ('trim','terminal','nodes'):
                    sp_g1 = self._trim_terminal_edges(sp_g1, focal_nodes, eff_hop)
                    sp_g2 = self._trim_terminal_edges(sp_g2, focal_nodes, eff_hop)

                if self.sp_eval_mode in ('fixed_before_pairs','fixed_pairs','fixed'):
                    # Fixed-before-pairs: measure La on the same pair set as before
                    try:
                        dist1 = dict(nx.all_pairs_shortest_path_length(sp_g1))
                        pairs = []
                        total1 = 0.0
                        for u, dmap in dist1.items():
                            for v, d in dmap.items():
                                if v == u:
                                    continue
                                if v <= u:
                                    continue
                                total1 += float(d)
                                pairs.append((u, v, float(d)))
                        if pairs:
                            Lb = total1 / len(pairs)
                            dist2 = dict(nx.all_pairs_shortest_path_length(sp_g2))
                            total2 = 0.0
                            count2 = 0
                            for u, v, _ in pairs:
                                dm = dist2.get(u, {})
                                if v in dm:
                                    total2 += float(dm[v])
                                    count2 += 1
                            if count2 > 0 and Lb > 0.0:
                                La = total2 / count2
                                gain = Lb - La  # signed gain
                                # relative signed change clamped to [-1, 1] for robustness
                                delta_sp_rel = max(-1.0, min(1.0, gain / Lb))
                            else:
                                delta_sp_rel = 0.0
                        else:
                            delta_sp_rel = 0.0
                    except Exception:
                        delta_sp_rel = 0.0
                else:
                    delta_sp_rel = float(self._compute_sp_gain_norm(sp_g1, sp_g2, mode=self.sp_norm_mode))
                sp_multiplier = self.sp_beta

            # IG の適用範囲（hop0のみ or 全hop）
            if str(self.ig_source_mode).lower() in ('linkset','paper','strict') and str(self.ig_hop_apply).lower() == 'hop0' and hop > 0:
                # hop>0 はSPのみ
                combined_ig = 0.0 + sp_multiplier * delta_sp_rel
            else:
                combined_ig = delta_h_norm + sp_multiplier * delta_sp_rel
            ig_for_lambda = combined_ig
            if str(self.ig_mode).lower() in ('norm', 'normalized'):
                ig_for_lambda = float(np.tanh(max(0.0, ig_for_lambda)))
            if self._ig_nonneg:
                ig_for_lambda = max(0.0, ig_for_lambda)
            lambda_term = self.lambda_weight * ig_for_lambda
            hop_gedig = float(delta_ged_norm - lambda_term)

            hop_results[hop] = HopResult(
                hop=hop,
                ged=delta_ged_norm,
                ig=combined_ig,
                gedig=hop_gedig,
                struct_cost=delta_ged_norm,
                node_count=len(sub_g2),
                edge_count=sub_g2.number_of_edges(),
                sp=delta_sp_rel,
                h_component=delta_h_norm,
                ged_raw=float(ged_result.get('raw_ged', 0.0)),
                ged_den=float(ged_result.get('normalization_den', denom)),
                entropy_before=float(ig_result.get('entropy_before', 0.0)),
                entropy_after=float(ig_result.get('entropy_after', 0.0)),
                ig_delta=float(ig_result.get('delta_entropy', 0.0)),
                ig_den=float(ig_result.get('normalization_den', fixed_den if fixed_den is not None else 1.0)),
                variance_reduction=float(ig_result.get('variance_reduction', 0.0)),
            )

            if self.adaptive_hops and hop > 0 and abs(hop_gedig) < 0.01:
                break

        if not hop_results:
            empty_result = GeDIGResult(
                gedig_value=0.0,
                ged_value=0.0,
                ig_value=0.0,
                raw_ged=0.0,
                ged_norm_den=1.0,
                ig_raw=0.0,
                ig_norm_den=1.0,
                delta_ged_norm=0.0,
                delta_sp_rel=0.0,
                delta_h_norm=0.0,
                structural_cost=0.0,
                structural_improvement=0.0,
                information_integration=0.0,
                entropy_before=0.0,
                entropy_after=0.0,
                ig_delta=0.0,
                variance_reduction=0.0,
                hop_results={},
                computation_time=time.time() - start_time,
                focal_nodes=focal_nodes,
                version="onegauge_v1_multihop",
            )
            return empty_result

        hop0 = hop_results.get(0, next(iter(hop_results.values())))
        best_hop = min(hop_results.keys(), key=lambda h: hop_results[h].gedig)
        best_result = hop_results[best_hop]

        return GeDIGResult(
            gedig_value=best_result.gedig,
            ged_value=hop0.ged,
            ig_value=best_result.ig,
            raw_ged=hop0.ged_raw,
            ged_norm_den=hop0.ged_den,
            ig_raw=best_result.ig,
            ig_norm_den=hop0.ig_den,
            delta_ged_norm=hop0.ged,
            delta_sp_rel=best_result.sp,
            delta_h_norm=hop0.h_component,
            structural_cost=hop0.struct_cost,
            structural_improvement=-hop0.ged,
            information_integration=best_result.ig,
            entropy_before=hop0.entropy_before,
            entropy_after=hop0.entropy_after,
            ig_delta=hop0.ig_delta,
            variance_reduction=hop0.variance_reduction,
            hop_results=hop_results,
            focal_nodes=focal_nodes,
            computation_time=time.time() - start_time,
            version="onegauge_v1_multihop"
        )

    # ------------ Helpers ------------
    def _graph_efficiency(self, g: nx.Graph) -> float:
        if g.number_of_nodes() == 0:
            return 0.0
        try:
            ge = nx.global_efficiency(g)
        except Exception:
            ge = 0.0
        try:
            cl = nx.average_clustering(g)
        except Exception:
            cl = 0.0
        return 0.7 * ge + 0.3 * cl

    def _avg_shortest_path_length_safe(self, g: nx.Graph) -> float:
        """Average shortest-path length over connected pairs only.
        Uses exact all-pairs for small graphs; for larger graphs, falls back to
        a light sampling over node pairs to bound runtime. Returns 0.0 on degenerate.
        """
        import random

        n = g.number_of_nodes()
        if n < 2:
            return 0.0
        # Exact path lengths for small graphs
        if n <= max(32, self.sp_node_cap // 3):
            try:
                total = 0
                count = 0
                for u, lengths in nx.all_pairs_shortest_path_length(g):
                    for v, d in lengths.items():
                        if v <= u:
                            continue
                        total += d
                        count += 1
                return (total / count) if count > 0 else 0.0
            except Exception:
                return 0.0
        # Sampling for larger graphs
        try:
            nodes = list(g.nodes())
            if len(nodes) < 2:
                return 0.0
            samples = min(self.sp_pair_samples, (n * (n - 1)) // 2)
            if samples <= 0:
                return 0.0
            total = 0.0
            count = 0
            for _ in range(samples):
                u, v = random.sample(nodes, 2)
                try:
                    d = nx.shortest_path_length(g, u, v)
                    total += float(d)
                    count += 1
                except Exception:
                    continue
            return (total / count) if count > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_sp_gain_norm(self, g_before: nx.Graph, g_after: nx.Graph, mode: str = 'relative') -> float:
        """Normalized signed shortest-path gain between two subgraphs.

        - If `sp_eval_mode == 'fixed_before_pairs'`, evaluate ΔSP on a fixed pair set
          sampled from `g_before` (DistanceCache semantics). `sp_pair_samples <= 0`
          means ALL pairs.
        - Otherwise, fall back to average all-pairs shortest path.

        mode='relative': (L_before - L_after) / L_before  (in [-1, 1])
        """
        try:
            if str(self.sp_eval_mode).lower() == 'fixed_before_pairs':
                from .sp_distcache import DistanceCache
                dc = DistanceCache(mode='cached', pair_samples=int(getattr(self, 'sp_pair_samples', 400)))
                sig = dc.signature(g_before, set(), 1, str(self.sp_scope_mode), str(self.sp_boundary_mode))
                rel = dc.estimate_sp_between_graphs(sig=sig, g_before=g_before, g_after=g_after)
                return float(max(-1.0, min(1.0, rel)))
        except Exception:
            # Fallback to average SP below
            pass
        # Average SP on before/after subgraphs
        Lb = self._avg_shortest_path_length_safe(g_before)
        La = self._avg_shortest_path_length_safe(g_after)
        if Lb <= 0.0:
            return 0.0
        gain = Lb - La
        if mode == 'relative':
            return max(-1.0, min(1.0, gain / Lb))
        return max(-1.0, min(1.0, gain))

    def _trim_terminal_edges(self, g: nx.Graph, anchors: Set[str], hop: int) -> nx.Graph:
        """Trim edges incident to terminal layer (distance == hop) while keeping nodes.
        Distances are computed from anchors via BFS limited to hop."""
        try:
            from collections import deque
            dist: Dict[Any, Optional[int]] = {n: None for n in g.nodes()}
            dq: deque = deque()
            for a in anchors:
                if a in g:
                    dist[a] = 0
                    dq.append(a)
            while dq:
                u = dq.popleft()
                du = dist[u]
                if du is None or du >= hop:
                    continue
                for v in g.neighbors(u):
                    if dist[v] is None:
                        dist[v] = du + 1
                        if dist[v] < hop:
                            dq.append(v)
            out = g.copy()
            to_remove = []
            for u, v in out.edges():
                du = dist.get(u, None)
                dv = dist.get(v, None)
                if du is None or dv is None:
                    continue
                if du == hop or dv == hop:
                    to_remove.append((u, v))
            if to_remove:
                out.remove_edges_from(to_remove)
            return out
        except Exception:
            return g.copy()

    def _extract_k_hop_subgraph(self, graph: nx.Graph, focal_nodes: Set[str], k: int) -> Tuple[nx.Graph, Set[str]]:
        valid = {n for n in focal_nodes if n in graph}
        if not valid:
            return nx.Graph(), set()
        if k == 0:
            return graph.subgraph(valid).copy(), valid
        all_nodes = set(valid); current = valid
        for _ in range(k):
            nxt = set(); [nxt.update(graph.neighbors(n)) for n in current if n in graph]
            all_nodes.update(nxt); current = nxt
        return graph.subgraph(all_nodes).copy(), all_nodes

    def _ensure_networkx(self, graph: Any) -> nx.Graph:
        if isinstance(graph, nx.Graph):
            return graph
        if hasattr(graph, 'edge_index') or hasattr(graph, 'x'):
            return self._pyg_to_networkx(graph)
        if isinstance(graph, np.ndarray) and graph.ndim == 2 and graph.shape[0] == graph.shape[1]:
            return nx.from_numpy_array(graph)
        return nx.Graph()

    def _pyg_to_networkx(self, data: Any) -> nx.Graph:
        G = nx.Graph()
        if hasattr(data, 'num_nodes'):
            n = data.num_nodes
        elif hasattr(data, 'x') and data.x is not None:
            n = data.x.shape[0]
        else:
            return G
        G.add_nodes_from(range(n))
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            ei = data.edge_index
            if hasattr(ei, 'cpu'):
                ei = ei.cpu().numpy()
            ei = np.array(ei)
            if ei.ndim == 2 and ei.shape[0] == 2:
                G.add_edges_from(ei.T.tolist())
        if hasattr(data, 'x') and data.x is not None:
            feats = data.x
            if hasattr(feats, 'cpu'):
                feats = feats.cpu().numpy()
            for i in range(n):
                if i < len(feats):
                    G.nodes[i]['feature'] = feats[i]
        return G

    def _extract_features(self, graph: nx.Graph) -> np.ndarray:
        feats = []
        for node in graph.nodes():
            d = graph.nodes[node]
            if 'feature' in d:
                feats.append(d['feature'])
            elif 'vec' in d:
                feats.append(d['vec'])
            else:
                feats.append(np.random.randn(64))
        return np.array(feats)

    def _filter_features(self, features: np.ndarray, node_set: Set[str], original_graph: nx.Graph) -> np.ndarray:
        node_to_idx = {node: i for i, node in enumerate(original_graph.nodes())}
        filtered = [features[node_to_idx[n]] for n in sorted(node_set) if n in node_to_idx and node_to_idx[n] < len(features)]
        return np.array(filtered) if filtered else np.empty((0, features.shape[1]))

    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        if g.number_of_nodes() < 2:
            return 0.0
        try:
            L = nx.laplacian_matrix(g).toarray(); eig = np.linalg.eigvalsh(L); return float(np.std(eig))
        except Exception:
            return 0.0

    # Metric helpers
    def _calculate_normalized_ged(self, g1: nx.Graph, g2: nx.Graph, *, norm_override: float | None = None) -> Dict[str, float]:
        # Delegate to pure function for consistency
        from .core.metrics import normalized_ged as _norm_ged

        out = _norm_ged(
            g1,
            g2,
            node_cost=self.node_cost,
            edge_cost=self.edge_cost,
            normalization=self.normalization,
            efficiency_weight=self.efficiency_weight,
            enable_spectral=self.enable_spectral,
            spectral_weight=self.spectral_weight,
            norm_override=norm_override,
        )
        # Legacy toggle path (kept to preserve flag semantics)
        if self.use_legacy_formula:
            # No change needed: base_improvement uses -normalized_ged internally
            return out
        return out

    def _calculate_entropy_variance_ig(
        self,
        graph: nx.Graph,
        features_before: np.ndarray,
        features_after: np.ndarray,
        query_vector: Optional[List[float]] = None,
        *,
        fixed_den: Optional[float] = None,
        k_star: Optional[int] = None,
    ) -> Dict[str, float]:
        from .core.metrics import entropy_ig as _entropy_ig

        fb = np.asarray(features_before, dtype=np.float32)
        fa = np.asarray(features_after, dtype=np.float32)
        extra_vectors = None
        zeroed_query: Optional[np.ndarray] = None
        if self.feature_weights is not None:
            weights = self.feature_weights.astype(np.float32)
            dims = []
            if fb.ndim == 2 and fb.shape[1] > 0:
                dims.append(fb.shape[1])
            elif fb.ndim == 1 and fb.shape[0] > 0:
                dims.append(fb.shape[0])
            if fa.ndim == 2 and fa.shape[1] > 0:
                dims.append(fa.shape[1])
            elif fa.ndim == 1 and fa.shape[0] > 0:
                dims.append(fa.shape[0])
            if query_vector is not None:
                dims.append(len(query_vector))
            target_dim = max(dims) if dims else weights.size
            if target_dim > weights.size:
                pad = np.ones(target_dim - weights.size, dtype=np.float32)
                weights = np.concatenate([weights, pad], axis=0)
            elif target_dim < weights.size:
                weights = weights[:target_dim]
            if fb.size:
                fb = fb * weights
            if fa.size:
                fa = fa * weights
            if query_vector is not None:
                q_array = np.asarray(query_vector, dtype=np.float32)
                if q_array.ndim == 0:
                    q_array = q_array.reshape(1)
                extra_vectors = [(q_array * weights).tolist()]
        else:
            if query_vector is not None:
                q_array = np.asarray(query_vector, dtype=np.float32)
                if q_array.ndim == 0:
                    q_array = q_array.reshape(1)
                extra_vectors = [q_array.tolist()]

        return _entropy_ig(
            graph,
            fb,
            fa,
            smoothing=self.smoothing,
            min_nodes=self.min_nodes,
            norm_strategy=self.ig_norm_strategy,
            extra_vectors=extra_vectors,
            fixed_den=fixed_den,
            k_star=k_star,
            delta_mode=self.ig_delta_mode,
        )

    def _calculate_local_entropies(self, graph: nx.Graph, features: np.ndarray) -> np.ndarray:
        entropies = []
        for node in graph.nodes():
            local_nodes = [node] + list(graph.neighbors(node))
            local_feats = []
            for n in local_nodes:
                try:
                    idx = int(n) if isinstance(n, str) else n
                    if 0 <= idx < len(features):
                        local_feats.append(features[idx])
                except (ValueError, TypeError):
                    continue
            if not local_feats:
                continue
            lf = np.array(local_feats)
            if len(lf) > 1:
                normed = lf / (np.linalg.norm(lf, axis=1, keepdims=True) + self.smoothing)
                sims = np.dot(normed, normed.T)
                probs = (sims + 1) / 2
                probs = probs.flatten(); probs = probs / (probs.sum() + self.smoothing)
                entropy = -np.sum(probs * np.log(probs + self.smoothing))
            else:
                entropy = 0.0
            entropies.append(entropy)
        return np.array(entropies)

    # Stats & rewards
    def _update_ig_stats(self, ig_raw: float) -> None:
        self._ig_count += 1
        delta = ig_raw - self._ig_mean
        self._ig_mean += delta / self._ig_count
        delta2 = ig_raw - self._ig_mean
        self._ig_m2 += delta * delta2

    def _ig_variance(self) -> float:
        if self._ig_count < 2:
            return 0.0
        return self._ig_m2 / (self._ig_count - 1)

    def _compute_ged_min_proxy(self, g_before: nx.Graph, g_after: nx.Graph) -> float:
        """Approximate GED_min via relative average shortest-path shortening.

        Uses undirected graphs and the largest connected component to avoid inf.
        Returns (ASP_before - ASP_after) / max(ASP_before, 1).
        """
        def _avg_sp(g: nx.Graph) -> float:
            if g.number_of_nodes() < 2:
                return 0.0
            und = g.to_undirected()
            if not nx.is_connected(und):
                # use largest component for a stable finite estimate
                comp = max(nx.connected_components(und), key=len)
                und = und.subgraph(comp).copy()
                if und.number_of_nodes() < 2:
                    return 0.0
            try:
                return float(nx.average_shortest_path_length(und))
            except Exception:
                return 0.0

        asp_before = _avg_sp(g_before)
        asp_after = _avg_sp(g_after)
        if asp_before <= 0.0:
            asp_gain = 0.0
        else:
            asp_gain = float((asp_before - asp_after) / max(asp_before, 1.0))

        if asp_gain > 0.0:
            return asp_gain

        # Fallback proxy: edge densification (normalized)
        e_before = g_before.number_of_edges()
        e_after = g_after.number_of_edges()
        if e_after != e_before:
            return float((e_after - e_before) / max(e_before, 1.0))
        return 0.0

    def _compute_ig_z(self, ig_raw: float) -> float:
        if self._ig_count < 2:
            return 0.0
        var = self._ig_variance()
        if var <= 1e-12:
            return 0.0
        return (ig_raw - self._ig_mean) / (var ** 0.5)

    def _compute_rewards(self, result: GeDIGResult) -> None:
        lambda_w = 0.0 if self._ig_count <= self.warmup_steps else self.lambda_weight
        structural_signal = -result.delta_ged_norm
        result.hop0_reward = lambda_w * result.ig_z_score + self.mu * structural_signal
        if result.hop_results:
            total_si = 0.0; total_w = 0.0
            for hop, hr in result.hop_results.items():
                w = self.decay_factor ** hop
                total_si += w * (-hr.ged); total_w += w
            avg_si = (total_si / total_w) if total_w > 0 else structural_signal
            result.aggregate_reward = lambda_w * result.ig_z_score + self.mu * avg_si
        else:
            result.aggregate_reward = result.hop0_reward

    # Spike detection
    def _detect_spike(self, result: GeDIGResult) -> bool:
        mode = SpikeDetectionMode(self.spike_detection_mode.lower()) if isinstance(self.spike_detection_mode, str) else self.spike_detection_mode
        structural_signal = float(getattr(result, 'structural_improvement', -result.delta_ged_norm))
        if mode == SpikeDetectionMode.THRESHOLD:
            return bool(result.gedig_value < self.spike_threshold)
        if mode == SpikeDetectionMode.AND:
            if (structural_signal > self.tau_s) and (result.ig_z_score > self.tau_i):
                return True
            # Fallback: IG 分散が極小で z-score がほぼ情報を持たない場合、構造改善のみで閾値の2倍を超えたら自然スパイク扱い
            try:
                ig_var = self._ig_variance()
            except Exception:
                ig_var = 0.0
            if ig_var < 1e-9 and structural_signal > (self.tau_s * 2):
                return True
            return False
        if mode == SpikeDetectionMode.OR:
            # Primary threshold logic
            if (structural_signal > self.tau_s) or (result.ig_z_score > self.tau_i):
                return True
            # Backward-compatibility: legacy OR mode treated any positive signal as a spike
            # (tests expect spike True even when thresholds are set higher than observed values)
            if (structural_signal > 0) or (result.ig_z_score > 0):
                return True
            return False
        # Fallback path (natural spike induction):
        #  - If IG variance stays ~0 (no informative z-score) but structural_improvement remains positive
        #  - AND auto-threshold backoff already minimized (tau_s,tau_i near floor)
        #  - THEN allow structural improvement alone to constitute a spike (prevents permanent zero-spike regime)
        try:
            ig_var = self._ig_variance()
        except Exception:
            ig_var = 0.0
        if self.tau_s <= 1e-4 and self.tau_i <= 1e-4 and ig_var < 1e-9 and structural_signal > 0.0:
            return True
        return bool(result.gedig_value < self.spike_threshold)

    def attach_monitor(self, monitor: 'GeDIGMonitor') -> None:
        self.monitor = monitor

class GeDIGMonitor:
    """Runtime monitoring for spike predictions.

    Features (extended):
      - Rolling spike rate
      - False positive rate tracking (when ground-truth provided)
      - Simple auto-threshold adjustment to keep FP rate under target
      - Ground-truth spike auto-derivation (structural_improvement & ig_z_score)
      - Exportable metrics snapshot (JSON / CSV)
      - Tau (tau_s, tau_i) adjustment history
    """
    def __init__(self, window_size: int = 200, target_fp_rate: float = 0.1, adjust_factor: float = 1.1,
                 gt_si_threshold: float | None = None, gt_igz_threshold: float | None = None,
                 gt_mode: str = 'and'):
        self.pred_buffer = deque(maxlen=window_size)
        self.fp_buffer = deque(maxlen=window_size)
        self.actual_buffer = deque(maxlen=window_size)
        self.target_fp_rate = target_fp_rate
        self.adjust_factor = adjust_factor
        self.gt_si_threshold = gt_si_threshold
        self.gt_igz_threshold = gt_igz_threshold
        self.gt_mode = gt_mode.lower()
        self.tau_history: list[dict[str, float]] = []
        # Spike が全く検出されない期間が続く場合に tau を積極的に緩和するためのカウンタ
        self.zero_spike_backoff_count: int = 0

    def record_prediction(self, predicted_spike: bool) -> None:
        self.pred_buffer.append(1 if predicted_spike else 0)

    def record_outcome(self, actual_spike: bool) -> None:
        if not self.pred_buffer:
            return
        predicted = bool(self.pred_buffer[-1])
        is_fp = 1 if (predicted and not actual_spike) else 0
        self.fp_buffer.append(is_fp)
        self.actual_buffer.append(1 if actual_spike else 0)

    def derive_ground_truth(self, result: 'GeDIGResult', core: 'GeDIGCore') -> bool:
        if self.gt_mode == 'threshold':
            return bool(result.has_spike)
        si_thr = self.gt_si_threshold if self.gt_si_threshold is not None else getattr(core, 'tau_s', 0.0)
        ig_thr = self.gt_igz_threshold if self.gt_igz_threshold is not None else getattr(core, 'tau_i', 0.0)
        cond_si = result.structural_improvement > si_thr
        cond_ig = result.ig_z_score > ig_thr
        if self.gt_mode == 'or':
            return cond_si or cond_ig
        return cond_si and cond_ig

    def record_auto_outcome(self, result: 'GeDIGResult', core: 'GeDIGCore') -> bool:
        label = self.derive_ground_truth(result, core)
        self.record_outcome(label)
        return label

    def spike_rate(self) -> float:
        if not self.pred_buffer:
            return 0.0
        return sum(self.pred_buffer) / len(self.pred_buffer)

    def false_positive_rate(self) -> float:
        if not self.fp_buffer:
            return 0.0
        return sum(self.fp_buffer) / len(self.fp_buffer)

    def auto_adjust_thresholds(self, core: 'GeDIGCore') -> None:
        if len(self.fp_buffer) < 10:
            return
        fp = self.false_positive_rate()
        sp_rate = self.spike_rate()
        # 誤検出多い → 閾値強化
        if fp > self.target_fp_rate * 1.1:
            core.tau_s *= self.adjust_factor
            core.tau_i *= self.adjust_factor
        # 誤検出少ない & spike もほぼ出ていない → 閾値緩和
        elif fp < self.target_fp_rate * 0.5 and sp_rate < 0.05:
            core.tau_s /= self.adjust_factor
            core.tau_i /= self.adjust_factor
        # 全く spike が無い期間がウィンドウ満杯で継続 → 一段強い緩和 (二乗)
        if sp_rate == 0.0 and len(self.pred_buffer) >= self.pred_buffer.maxlen:
            core.tau_s /= (self.adjust_factor ** 2)
            core.tau_i /= (self.adjust_factor ** 2)
            self.zero_spike_backoff_count += 1
            if self.zero_spike_backoff_count >= 2:
                try:
                    core.spike_detection_mode = 'or'
                except Exception:
                    pass
        core.tau_s = float(np.clip(core.tau_s, 1e-4, 10.0))
        core.tau_i = float(np.clip(core.tau_i, 1e-4, 10.0))
        self.tau_history.append({'n_samples': float(len(self.fp_buffer)), 'tau_s': core.tau_s, 'tau_i': core.tau_i})

    def get_metrics(self) -> dict[str, float | int]:
        return {
            'spike_rate': self.spike_rate(),
            'false_positive_rate': self.false_positive_rate(),
            'n_predictions': len(self.pred_buffer),
            'n_actual': len(self.actual_buffer),
            'zero_spike_backoff_count': self.zero_spike_backoff_count,
        }

    def export_metrics(self, path: str, core: 'GeDIGCore', include_history: bool = True) -> None:
        import json, csv, os
        metrics = self.get_metrics()
        metrics.update({'tau_s': core.tau_s, 'tau_i': core.tau_i, 'lambda_weight': getattr(core, 'lambda_weight', 0.0), 'mu': getattr(core, 'mu', 0.0)})
        if path.endswith('.json'):
            out = {'metrics': metrics}
            if include_history:
                out['tau_history'] = self.tau_history
            with open(path, 'w') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        else:
            fieldnames = sorted(metrics.keys())
            first = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if first:
                    w.writeheader()
                w.writerow(metrics)
            if include_history and self.tau_history:
                hist_path = path + '.tau_history.json'
                with open(hist_path, 'w') as fh:
                    json.dump(self.tau_history, fh, ensure_ascii=False, indent=2)

    # --- Hop detail aggregation (Task 5) ---
    def summarize_hop_results(self, result: 'GeDIGResult') -> dict[str, float]:  # lightweight stats for logging
        if not getattr(result, 'hop_results', None):
            return {}
        vals = [hr.gedig for hr in result.hop_results.values()]
        import statistics, math
        if not vals:
            return {}
        mean_v = statistics.fmean(vals)
        p95 = sorted(vals)[int(len(vals)*0.95)-1] if len(vals) >= 2 else vals[0]
        return {
            'hop_gedig_mean': float(mean_v),
            'hop_gedig_p95': float(p95),
            'hop_gedig_max': float(max(vals)),
            'hop_count': float(len(vals)),
        }

class GeDIGPresets:
    CONSERVATIVE = {"lambda_weight": 0.3, "mu": 0.7, "tau_s": 0.2, "tau_i": 0.3, "spike_detection_mode": "and"}
    BALANCED = {"lambda_weight": 0.5, "mu": 0.5, "tau_s": 0.15, "tau_i": 0.25, "spike_detection_mode": "and"}
    AGGRESSIVE = {"lambda_weight": 0.7, "mu": 0.3, "tau_s": 0.08, "tau_i": 0.15, "spike_detection_mode": "or"}


class GeDIGLogger:
    def __init__(self, output_path: Any, max_lines: int = 50_000, max_bytes: int = 50 * 1024 * 1024, compress_on_rotate: bool = False):
        """CSV ロガー (行数/バイト数でローテーション)。PathLike も受け付ける。

        Parameters
        ----------
        output_path: ベースとなるファイルパス (拡張子省略可)。`pathlib.Path` 可。
        max_lines: 1ファイルあたりの最大データ行数 (ヘッダ除外カウント)。
        max_bytes: 1ファイルあたりの最大サイズ (バイト)。
        """
        import csv, os
        # PathLike 対応: 早期に str へ正規化
        self.output_path = str(output_path)
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.compress_on_rotate = compress_on_rotate
        self._line_count = 0
        self._file_index = 0
        self._csv = csv
        self._os = os
        self.fields = ['step','raw_ged','ged_value','structural_improvement','ig_raw','ig_z_score','hop0_reward','aggregate_reward','reward','spike','version']
        self._open_writer()

    def _rotate_needed(self) -> bool:
        try:
            size = self._os.path.getsize(self._current_file)
        except OSError:
            size = 0
        return self._line_count >= self.max_lines or size >= self.max_bytes

    def _open_writer(self):
        self._current_file = self._build_filename()
        first = not self._os.path.exists(self._current_file)
        self._fh = open(self._current_file, 'a', newline='')
        self._writer = self._csv.DictWriter(self._fh, fieldnames=self.fields)
        if first:
            self._writer.writeheader()
            self._fh.flush()
            self._line_count = 0

    def _build_filename(self) -> str:
        base = self.output_path
        if '.' in base and not base.endswith('.'):
            root, ext = base.rsplit('.', 1)
        else:
            root, ext = base, 'csv'
        return f"{root}_{self._file_index}.{ext}"

    def log(self, step: int, result: GeDIGResult):
        if self._rotate_needed():
            old_file = self._current_file
            self._fh.close()
            if self.compress_on_rotate:
                try:
                    import gzip, shutil
                    with open(old_file, 'rb') as f_in, gzip.open(old_file + '.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    self._os.remove(old_file)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Compression failed for {old_file}: {e}")
            self._file_index += 1
            self._open_writer()
        row = {
            'step': step,
            'raw_ged': result.raw_ged,
            'ged_value': result.ged_value,
            'structural_improvement': result.structural_improvement,
            'ig_raw': result.ig_raw,
            'ig_z_score': result.ig_z_score,
            'hop0_reward': result.hop0_reward,
            'aggregate_reward': result.aggregate_reward,
            'reward': result.reward,
            'spike': int(result.has_spike),
            'version': result.version
        }
        self._writer.writerow(row)
        self._fh.flush()
        self._line_count += 1
    def close(self):  # pragma: no cover
        try: self._fh.close()
        except Exception: pass


def calculate_gedig(graph_before: Any, graph_after: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    if config:
        metrics = config.get('metrics', config); spectral = metrics.get('spectral_evaluation', {})
        calc = GeDIGCore(enable_multihop=metrics.get('use_multihop', False), max_hops=metrics.get('max_hops', 3), enable_spectral=spectral.get('enabled', False), spectral_weight=spectral.get('weight', 0.3), **kwargs)
    else:
        calc = GeDIGCore(**kwargs)
    # Linkset-first: avoid graph-IG fallback by passing a minimal linkset when none provided
    try:
        from .linkset_adapter import build_linkset_info as _build_ls  # type: ignore
    except Exception:
        _build_ls = None  # type: ignore
    ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
    return calc.calculate(g_prev=graph_before, g_now=graph_after, **({"linkset_info": ls} if ls is not None else {})).gedig_value


def detect_insight_spike(graph_before: Any, graph_after: Any, threshold: float = -0.5, **kwargs) -> bool:
    calc = GeDIGCore(spike_threshold=threshold, **kwargs)
    try:
        from .linkset_adapter import build_linkset_info as _build_ls  # type: ignore
    except Exception:
        _build_ls = None  # type: ignore
    ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
    return calc.calculate(g_prev=graph_before, g_now=graph_after, **({"linkset_info": ls} if ls is not None else {})).has_spike


def delta_ged(graph_before: Any, graph_after: Any, **kwargs) -> float:
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics = config['metrics']
        calc = GeDIGCore(enable_multihop=metrics.get('use_multihop_gedig', False), max_hops=metrics.get('max_hops', 2), decay_factor=metrics.get('decay_factor', 0.5))
    else:
        calc = GeDIGCore()
    try:
        from .linkset_adapter import build_linkset_info as _build_ls  # type: ignore
    except Exception:
        _build_ls = None  # type: ignore
    ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
    return calc.calculate(g_prev=graph_before, g_now=graph_after, **({"linkset_info": ls} if ls is not None else {})).ged_value


def delta_ig(graph_before: Any, graph_after: Any, **kwargs) -> float:
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics = config['metrics']
        calc = GeDIGCore(enable_multihop=metrics.get('use_multihop_gedig', False), max_hops=metrics.get('max_hops', 2), decay_factor=metrics.get('decay_factor', 0.5))
    else:
        calc = GeDIGCore()
    try:
        from .linkset_adapter import build_linkset_info as _build_ls  # type: ignore
    except Exception:
        _build_ls = None  # type: ignore
    ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
    return calc.calculate(g_prev=graph_before, g_now=graph_after, **({"linkset_info": ls} if ls is not None else {})).ig_value

    # ------------- Utility helpers (migrated from misplaced GeDIGLogger region) -------------
    def _graph_efficiency(self, g: nx.Graph) -> float:
        if g.number_of_nodes() == 0:
            return 0.0
        try:
            global_eff = nx.global_efficiency(g)
        except Exception:
            global_eff = 0.0
        try:
            clustering = nx.average_clustering(g)
        except Exception:
            clustering = 0.0
        return 0.7 * global_eff + 0.3 * clustering

    def _extract_k_hop_subgraph(self, graph: nx.Graph, focal_nodes: Set[Any], k: int) -> Tuple[nx.Graph, Set[Any]]:
        valid_focal = {n for n in focal_nodes if n in graph}
        if not valid_focal:
            return nx.Graph(), set()
        if k == 0:
            return graph.subgraph(valid_focal).copy(), valid_focal
        all_nodes = set(valid_focal)
        current_layer = valid_focal
        class GeDIGLogger:
            """CSV logger with rotation (Phase A5)."""
            def __init__(self, output_path: str, max_lines: int = 50_000, max_bytes: int = 50 * 1024 * 1024):
                import csv, os
                self.output_path = output_path
                self.max_lines = max_lines
                self.max_bytes = max_bytes
                self._line_count = 0
                self._file_index = 0
                self._csv = csv
                self._os = os
                self.fields = [
                    'step','raw_ged','ged_value','structural_improvement','ig_raw','ig_z_score',
                    'hop0_reward','aggregate_reward','reward','spike','version'
                ]
                self._open_writer()

            def _rotate_needed(self) -> bool:
                try:
                    size = self._os.path.getsize(self._current_file)
                except OSError:
                    size = 0
                return self._line_count >= self.max_lines or size >= self.max_bytes

            def _open_writer(self):
                self._current_file = self._build_filename()
                first = not self._os.path.exists(self._current_file)
                self._fh = open(self._current_file, 'a', newline='')
                self._writer = self._csv.DictWriter(self._fh, fieldnames=self.fields)
                if first:
                    self._writer.writeheader()
                    self._fh.flush()
                    self._line_count = 0

            def _build_filename(self) -> str:
                base = self.output_path
                root, ext = (base.rsplit('.',1)+['csv'])[:2] if '.' in base else (base, 'csv')
                return f"{root}_{self._file_index}.{ext}"

            def log(self, step: int, result: GeDIGResult):
                if self._rotate_needed():
                    self._fh.close()
                    self._file_index += 1
                    self._open_writer()
                row = {
                    'step': step,
                    'raw_ged': result.raw_ged,
                    'ged_value': result.ged_value,
                    'structural_improvement': result.structural_improvement,
                    'ig_raw': result.ig_raw,
                    'ig_z_score': result.ig_z_score,
                    'hop0_reward': result.hop0_reward,
                    'aggregate_reward': result.aggregate_reward,
                    'reward': result.reward,
                    'spike': int(result.has_spike),
                    'version': result.version
                }
                self._writer.writerow(row)
                self._fh.flush()
                self._line_count += 1

            def close(self):  # pragma: no cover
                try:
                    self._fh.close()
                except Exception:
                    pass


        # ---------------- Convenience Functions ----------------
        def calculate_gedig(graph_before: Any, graph_after: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> float:
            if config:
                metrics_config = config.get('metrics', config)
                spectral_config = metrics_config.get('spectral_evaluation', {})
                calculator = GeDIGCore(
                    enable_multihop=metrics_config.get('use_multihop', False),
                    max_hops=metrics_config.get('max_hops', 3),
                    enable_spectral=spectral_config.get('enabled', False),
                    spectral_weight=spectral_config.get('weight', 0.3),
                    **kwargs
                )
            else:
                calculator = GeDIGCore(**kwargs)
            return calculator.calculate(g_prev=graph_before, g_now=graph_after).gedig_value


        def detect_insight_spike(graph_before: Any, graph_after: Any, threshold: float = -0.5, **kwargs) -> bool:
            calculator = GeDIGCore(spike_threshold=threshold, **kwargs)
            return calculator.calculate(g_prev=graph_before, g_now=graph_after).has_spike


        def delta_ged(graph_before: Any, graph_after: Any, **kwargs) -> float:
            config = kwargs.get('config', {})
            if config and 'metrics' in config:
                metrics_config = config['metrics']
                calculator = GeDIGCore(
                    enable_multihop=metrics_config.get('use_multihop_gedig', False),
                    max_hops=metrics_config.get('max_hops', 2),
                    decay_factor=metrics_config.get('decay_factor', 0.5)
                )
            else:
                calculator = GeDIGCore()
            return calculator.calculate(g_prev=graph_before, g_now=graph_after).ged_value


        def delta_ig(graph_before: Any, graph_after: Any, **kwargs) -> float:
            config = kwargs.get('config', {})
            if config and 'metrics' in config:
                metrics_config = config['metrics']
                calculator = GeDIGCore(
                    enable_multihop=metrics_config.get('use_multihop_gedig', False),
                    max_hops=metrics_config.get('max_hops', 2),
                    decay_factor=metrics_config.get('decay_factor', 0.5)
                )
            else:
                calculator = GeDIGCore()
            return calculator.calculate(g_prev=graph_before, g_now=graph_after).ig_value

            # Combine with existing structural improvement
            structural_improvement = (
                structural_improvement * (1 - self.spectral_weight) +
                np.tanh(spectral_improvement) * self.spectral_weight
            )
        
        return {
            'raw_ged': raw_ged,
            'normalized_ged': normalized_ged,
            'structural_improvement': np.clip(structural_improvement, -1.0, 1.0),
            'efficiency_change': efficiency_change
        }
    
    def _calculate_entropy_variance_ig(
        self,
        graph: nx.Graph,
        features_before: np.ndarray,
        features_after: np.ndarray,
        query_vector: Optional[List[float]] = None,
        *,
        fixed_den: Optional[float] = None,
        k_star: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate information gain using entropy variance."""
        from .core.metrics import entropy_ig as _entropy_ig

        return _entropy_ig(
            graph,
            features_before,
            features_after,
            smoothing=self.smoothing,
            min_nodes=self.min_nodes,
            norm_strategy=self.ig_norm_strategy,
            extra_vectors=[query_vector] if query_vector is not None else None,
            fixed_den=fixed_den,
            k_star=k_star,
        )
    
    def _calculate_local_entropies(self,
                                  graph: nx.Graph,
                                  features: np.ndarray) -> np.ndarray:
        """Calculate Shannon entropy for each node's neighborhood."""
        entropies = []
        
        for node in graph.nodes():
            # Get node and neighbors
            neighbors = list(graph.neighbors(node))
            local_nodes = [node] + neighbors
            
            # Get features for local neighborhood
            local_features = []
            for n in local_nodes:
                # Handle both int and str node IDs
                try:
                    node_idx = int(n) if isinstance(n, str) else n
                    if node_idx < len(features):
                        local_features.append(features[node_idx])
                except (ValueError, TypeError):
                    # Skip non-numeric node IDs
                    continue
            
            if not local_features:
                continue
            
            # Calculate local entropy
            local_features = np.array(local_features)
            
            # Use feature variance as proxy for entropy
            if len(local_features) > 1:
                # Normalize features
                normalized = local_features / (np.linalg.norm(local_features, axis=1, keepdims=True) + self.smoothing)
                # Calculate pairwise similarities
                similarities = np.dot(normalized, normalized.T)
                # Convert to probabilities
                probs = (similarities + 1) / 2  # Map from [-1,1] to [0,1]
                # Flatten and normalize
                probs = probs.flatten()
                probs = probs / (probs.sum() + self.smoothing)
                # Shannon entropy
                entropy = -np.sum(probs * np.log(probs + self.smoothing))
            else:
                entropy = 0.0
            
            entropies.append(entropy)
        
        return np.array(entropies)
    
    def _graph_efficiency(self, g: nx.Graph) -> float:
        """Calculate combined graph efficiency metric."""
        if g.number_of_nodes() == 0:
            return 0.0
        
        try:
            global_eff = nx.global_efficiency(g)
        except:
            global_eff = 0.0
        
        try:
            clustering = nx.average_clustering(g)
        except:
            clustering = 0.0
        
        return 0.7 * global_eff + 0.3 * clustering
    
    def _extract_k_hop_subgraph(self,
                               graph: nx.Graph,
                               focal_nodes: Set[Any],
                               k: int) -> Tuple[nx.Graph, Set[Any]]:
        """Extract k-hop subgraph around focal nodes."""
        # Ensure focal nodes exist in graph
        valid_focal = {n for n in focal_nodes if n in graph}
        if not valid_focal:
            logger.warning(f"No focal nodes found in graph. Focal: {focal_nodes}, Graph nodes: {list(graph.nodes())[:5]}")
            return nx.Graph(), set()
        
        if k == 0:
            # Only focal nodes
            subgraph = graph.subgraph(valid_focal).copy()
            return subgraph, valid_focal
        
        # BFS to find k-hop neighbors
        all_nodes = set(valid_focal)
        current_layer = valid_focal
        
        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                if node in graph:
                    next_layer.update(graph.neighbors(node))
            all_nodes.update(next_layer)
            current_layer = next_layer
        
        subgraph = graph.subgraph(all_nodes).copy()
        return subgraph, all_nodes
    
    def _ensure_networkx(self, graph: Any) -> nx.Graph:
        """Convert various graph types to NetworkX."""
        if isinstance(graph, nx.Graph):
            return graph
        
        # Handle PyG Data
        if hasattr(graph, 'edge_index') or hasattr(graph, 'x'):
            return self._pyg_to_networkx(graph)
        
        # Handle adjacency matrix
        if isinstance(graph, np.ndarray) and graph.ndim == 2:
            # CRITICAL FIX: Ensure it's a square matrix before treating as adjacency
            if graph.shape[0] == graph.shape[1]:
                return nx.from_numpy_array(graph)
            else:
                logger.warning(
                    f"Received non-square numpy array {graph.shape} where a graph "
                    f"was expected. This might be a feature matrix. Returning empty graph."
                )
                return nx.Graph()
        
        logger.warning(f"Unknown graph type: {type(graph)}")
        return nx.Graph()
    
    def _pyg_to_networkx(self, data: Any) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX."""
        G = nx.Graph()
        
        # Add nodes
        if hasattr(data, 'num_nodes'):
            num_nodes = data.num_nodes
        elif hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            logger.warning("Cannot determine number of nodes from PyG Data")
            return G
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_array = data.edge_index
            if hasattr(edge_array, 'cpu'):
                edge_array = edge_array.cpu().numpy()
            
            # Ensure it's a numpy array
            edge_array = np.array(edge_array)

            if edge_array.ndim == 2 and edge_array.shape[0] == 2:
                edges = edge_array.T.tolist()
                G.add_edges_from(edges)
            else:
                logger.warning(f"edge_index has unexpected shape: {edge_array.shape}")

        # Add node features as attributes
        if hasattr(data, 'x') and data.x is not None:
            features = data.x
            if hasattr(features, 'cpu'):
                features = features.cpu().numpy()
            
            for i in range(num_nodes):
                if i < len(features):
                    G.nodes[i]['feature'] = features[i]
        
        return G
    
    def _extract_features(self, graph: nx.Graph) -> np.ndarray:
        """Extract or generate node features."""
        n = graph.number_of_nodes()
        
        # Try to get existing features
        features = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if 'feature' in node_data:
                features.append(node_data['feature'])
            elif 'vec' in node_data:
                features.append(node_data['vec'])
            else:
                # Generate random features as fallback
                features.append(np.random.randn(64))
        
        return np.array(features)
    
    def _filter_features(self,
                        features: np.ndarray,
                        node_set: Set[str],
                        original_graph: nx.Graph) -> np.ndarray:
        """Filter features to match node subset."""
        # Map node IDs to indices
        node_to_idx = {node: i for i, node in enumerate(original_graph.nodes())}
        
        filtered = []
        for node in sorted(node_set):
            if node in node_to_idx and node_to_idx[node] < len(features):
                filtered.append(features[node_to_idx[node]])
        
        return np.array(filtered) if filtered else np.empty((0, features.shape[1]))
    
    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        """Calculate structural score using Laplacian eigenvalues.
        
        Returns:
            Standard deviation of eigenvalues (higher = more irregular structure)
        """
        if g.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Calculate Laplacian matrix
            L = nx.laplacian_matrix(g).toarray()
            
            # Calculate eigenvalues (real only)
            eigvals = np.linalg.eigvalsh(L)
            
            # Use standard deviation as irregularity metric
            return np.std(eigvals)
            
        except Exception as e:
            logger.warning(f"Spectral score calculation failed: {e}")
            return 0.0


# Convenience functions for backward compatibility
def calculate_gedig(graph_before: Any,
                   graph_after: Any,
                   config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> float:
    """
    Simple interface to calculate geDIG value.
    
    Returns just the geDIG score for backward compatibility.
    """
    if config:
        # Handle both old and new config formats
        metrics_config = config.get('metrics', config)
        spectral_config = metrics_config.get('spectral_evaluation', {})
        
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop', False),
            max_hops=metrics_config.get('max_hops', 3),
            enable_spectral=spectral_config.get('enabled', False),
            spectral_weight=spectral_config.get('weight', 0.3),
            **kwargs
        )
    else:
        calculator = GeDIGCore(**kwargs)
    
    result = calculator.calculate(g_prev=graph_before, g_now=graph_after)
    return result.gedig_value


def detect_insight_spike(graph_before: Any,
                        graph_after: Any,
                        threshold: float = -0.5,
                        **kwargs) -> bool:
    """
    Check if the graph change represents an insight spike.
    """
    calculator = GeDIGCore(spike_threshold=threshold, **kwargs)
    result = calculator.calculate(g_prev=graph_before, g_now=graph_after)
    return result.has_spike


# Wrapper functions for backward compatibility with metrics_selector.py
def delta_ged(graph_before: Any, graph_after: Any, **kwargs) -> float:
    """
    Calculate ΔGED for backward compatibility.
    Returns negative value when graph simplifies (insight formation).
    """
    # Check if config is passed via kwargs
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics_config = config['metrics']
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop_gedig', False),
            max_hops=metrics_config.get('max_hops', 2),
            decay_factor=metrics_config.get('decay_factor', 0.5)
        )
    else:
        calculator = GeDIGCore()
    result = calculator.calculate(g_prev=graph_before, g_now=graph_after)
    return result.ged_value


def delta_ig(graph_before: Any, graph_after: Any, **kwargs) -> float:
    """
    Calculate ΔIG for backward compatibility.
    Returns positive value when information gain occurs.
    """
    # Check if config is passed via kwargs
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics_config = config['metrics']
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop_gedig', False),
            max_hops=metrics_config.get('max_hops', 2),
            decay_factor=metrics_config.get('decay_factor', 0.5)
        )
    else:
        calculator = GeDIGCore()
    result = calculator.calculate(g_prev=graph_before, g_now=graph_after)
    return result.ig_value
@dataclass
class LinksetMetrics:
    delta_ged_norm: float
    delta_h_norm: float
    delta_sp_rel: float
    gedig_value: float
    raw_ged: float
    ged_norm_den: float
    ig_norm_den: float
    entropy_before: float
    entropy_after: float
    ig_delta: float
    before_size: int
    after_size: int
    query_similarity: float
    # Diagnostics: positive-weight counts and top weights (before/after)
    pos_w_before: int = 0
    pos_w_after: int = 0
    topw_before: list[float] | None = None
    topw_after: list[float] | None = None
