"""Unified geDIG core (refactored + Day1 spike detection)."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

# Import types from dedicated module
from .gedig.types import (
    ProcessingMode,
    SpikeDetectionMode,
    HopResult,
    GeDIGResult,
    LinksetMetrics,
)
from .gedig.config import GeDIGConfig, GeDIGPresets
from .gedig.spike import detect_spike, compute_rewards
from .gedig.graph_utils import (
    graph_efficiency,
    spectral_score,
    avg_shortest_path_length_safe,
    compute_sp_gain_norm,
    extract_k_hop_subgraph,
    trim_terminal_edges,
    ensure_networkx,
    pyg_to_networkx,
    extract_features,
    filter_features,
    compute_ged_min_proxy,
)
from .gedig.monitor import GeDIGMonitor
from .gedig.logger import GeDIGLogger

logger = logging.getLogger(__name__)


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
        # Structural similarity for analogy detection
        structural_similarity_config: Optional[Dict[str, Any]] = None,
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

        # Structural similarity for analogy detection
        self._ss_evaluator = None
        self._ss_config = structural_similarity_config or {}
        if self._ss_config.get('enabled', False):
            try:
                from .structural_similarity import StructuralSimilarityEvaluator
                from ..config.models import StructuralSimilarityConfig
                ss_cfg = StructuralSimilarityConfig(**self._ss_config)
                self._ss_evaluator = StructuralSimilarityEvaluator(ss_cfg)
                logger.info("Structural similarity evaluator enabled: method=%s", ss_cfg.method)
            except Exception as e:
                logger.warning("Failed to initialize structural similarity evaluator: %s", e)
                self._ss_evaluator = None

        logger.info(
            "GeDIGCore initialized: multihop=%s max_hops=%s spectral=%s structural_sim=%s",
            self.enable_multihop,
            self.max_hops,
            self.enable_spectral,
            self._ss_evaluator is not None,
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

            # Structural similarity bonus for analogy detection
            analogy_bonus = 0.0
            if self._ss_evaluator is not None:
                try:
                    center_node = list(focal_nodes)[0] if focal_nodes else None
                    analogy_bonus = self._ss_evaluator.compute_analogy_bonus(
                        sub_g1, sub_g2,
                        center1=center_node,
                        center2=center_node,
                    )
                    if analogy_bonus > 0:
                        combined_ig += analogy_bonus
                        logger.debug(
                            "[ANALOGY] hop=%d bonus=%.4f combined_ig=%.4f",
                            hop, analogy_bonus, combined_ig
                        )
                except Exception as e:
                    logger.warning("Structural similarity evaluation failed: %s", e)

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
        return graph_efficiency(g)

    def _avg_shortest_path_length_safe(self, g: nx.Graph) -> float:
        """Average shortest-path length over connected pairs only."""
        return avg_shortest_path_length_safe(g, self.sp_node_cap, self.sp_pair_samples)

    def _compute_sp_gain_norm(self, g_before: nx.Graph, g_after: nx.Graph, mode: str = 'relative') -> float:
        """Normalized signed shortest-path gain between two subgraphs."""
        # Handle fixed_before_pairs mode specially (requires DistanceCache)
        try:
            if str(self.sp_eval_mode).lower() == 'fixed_before_pairs':
                from .sp_distcache import DistanceCache
                dc = DistanceCache(mode='cached', pair_samples=int(getattr(self, 'sp_pair_samples', 400)))
                sig = dc.signature(g_before, set(), 1, str(self.sp_scope_mode), str(self.sp_boundary_mode))
                rel = dc.estimate_sp_between_graphs(sig=sig, g_before=g_before, g_after=g_after)
                return float(max(-1.0, min(1.0, rel)))
        except Exception:
            pass
        return compute_sp_gain_norm(g_before, g_after, mode, self.sp_node_cap, self.sp_pair_samples)

    def _trim_terminal_edges(self, g: nx.Graph, anchors: Set[str], hop: int) -> nx.Graph:
        """Trim edges incident to terminal layer."""
        return trim_terminal_edges(g, anchors, hop)

    def _extract_k_hop_subgraph(self, graph: nx.Graph, focal_nodes: Set[str], k: int) -> Tuple[nx.Graph, Set[str]]:
        return extract_k_hop_subgraph(graph, focal_nodes, k)

    def _ensure_networkx(self, graph: Any) -> nx.Graph:
        return ensure_networkx(graph)

    def _pyg_to_networkx(self, data: Any) -> nx.Graph:
        return pyg_to_networkx(data)

    def _extract_features(self, graph: nx.Graph) -> np.ndarray:
        return extract_features(graph)

    def _filter_features(self, features: np.ndarray, node_set: Set[str], original_graph: nx.Graph) -> np.ndarray:
        return filter_features(features, node_set, original_graph)

    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        return spectral_score(g)

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
        """Approximate GED_min via relative average shortest-path shortening."""
        return compute_ged_min_proxy(g_before, g_after)

    def _compute_ig_z(self, ig_raw: float) -> float:
        if self._ig_count < 2:
            return 0.0
        var = self._ig_variance()
        if var <= 1e-12:
            return 0.0
        return (ig_raw - self._ig_mean) / (var ** 0.5)

    def _compute_rewards(self, result: GeDIGResult) -> None:
        compute_rewards(
            result,
            lambda_weight=self.lambda_weight,
            mu=self.mu,
            decay_factor=self.decay_factor,
            warmup_steps=self.warmup_steps,
            ig_count=self._ig_count,
        )

    # Spike detection
    def _detect_spike(self, result: GeDIGResult) -> bool:
        try:
            ig_var = self._ig_variance()
        except Exception:
            ig_var = 0.0
        return detect_spike(
            result,
            mode=self.spike_detection_mode,
            spike_threshold=self.spike_threshold,
            tau_s=self.tau_s,
            tau_i=self.tau_i,
            ig_variance=ig_var,
        )

    def attach_monitor(self, monitor: 'GeDIGMonitor') -> None:
        self.monitor = monitor


# ------------ Convenience Functions ------------


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

