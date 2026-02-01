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
from .gedig.linkset import compute_linkset_metrics
from .gedig.multihop import calculate_multihop

logger = logging.getLogger(__name__)


class GeDIGCore:
    """Unified geDIG calculator with configurable parameters.

    Can be initialized with individual kwargs or a GeDIGConfig object.
    Environment variables are automatically applied via GeDIGConfig.from_kwargs().
    """

    def __init__(
        self,
        config: Optional[GeDIGConfig] = None,
        *,
        # All parameters below are for backward compatibility.
        # If config is provided, these are ignored.
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
        ig_mode: str = 'raw',
        ig_norm_strategy: str = 'before',
        ig_delta_mode: str = 'after_before',
        entropy_tau: float = 1.0,
        mu: float = 0.5,
        warmup_steps: int = 10,
        use_refactored_reward: bool = True,
        use_legacy_formula: bool = False,
        spike_detection_mode: str | SpikeDetectionMode = "and",
        tau_s: float = 0.15,
        tau_i: float = 0.25,
        use_multihop_sp_gain: bool = True,
        sp_norm_mode: str = 'relative',
        sp_beta: float = 0.2,
        use_local_normalization: bool = False,
        local_norm_mode: str = 'layer1',
        enable_ged_min_diag: bool = False,
        sp_node_cap: int = 200,
        sp_pair_samples: int = 400,
        sp_use_sampling: bool = True,
        feature_weights: Optional[Sequence[float]] = None,
        linkset_mode: bool = False,
        sp_scope_mode: str = 'auto',
        sp_hop_expand: int = 0,
        sp_eval_mode: str = 'connected',
        ig_source_mode: str = 'graph',
        ig_hop_apply: str = 'all',
        ged_norm_scheme: str = 'edges_after',
        structural_similarity_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Build config from kwargs if not provided (with env overrides)
        if config is None:
            config = GeDIGConfig.from_kwargs(
                node_cost=node_cost,
                edge_cost=edge_cost,
                normalization=normalization,
                efficiency_weight=efficiency_weight,
                min_nodes=min_nodes,
                smoothing=smoothing,
                enable_multihop=enable_multihop,
                max_hops=max_hops,
                decay_factor=decay_factor,
                adaptive_hops=adaptive_hops,
                spike_threshold=spike_threshold,
                enable_spectral=enable_spectral,
                spectral_weight=spectral_weight,
                lambda_weight=lambda_weight,
                ig_mode=ig_mode,
                ig_norm_strategy=ig_norm_strategy,
                entropy_tau=entropy_tau,
                mu=mu,
                warmup_steps=warmup_steps,
                use_refactored_reward=use_refactored_reward,
                use_legacy_formula=use_legacy_formula,
                spike_detection_mode=spike_detection_mode,
                tau_s=tau_s,
                tau_i=tau_i,
                use_multihop_sp_gain=use_multihop_sp_gain,
                sp_norm_mode=sp_norm_mode,
                sp_beta=sp_beta,
                use_local_normalization=use_local_normalization,
                local_norm_mode=local_norm_mode,
                enable_ged_min_diag=enable_ged_min_diag,
                sp_node_cap=sp_node_cap,
                sp_pair_samples=sp_pair_samples,
                sp_use_sampling=sp_use_sampling,
                linkset_mode=linkset_mode,
                sp_scope_mode=sp_scope_mode,
                sp_hop_expand=sp_hop_expand,
                sp_eval_mode=sp_eval_mode,
                ig_source_mode=ig_source_mode,
                ig_hop_apply=ig_hop_apply,
                ged_norm_scheme=ged_norm_scheme,
                feature_weights=feature_weights,
                structural_similarity_config=structural_similarity_config,
            )

        # Apply config to instance attributes
        self._apply_config(config)

        # Initialize runtime state
        self._ig_count = 0
        self._ig_mean = 0.0
        self._ig_m2 = 0.0
        self.logger = None
        self.monitor = None
        self._graph_ig_warned = False

        # Initialize structural similarity evaluator if enabled
        self._ss_evaluator = None
        self._ss_config = structural_similarity_config or {}
        if self._ss_config.get('enabled', False):
            self._init_structural_similarity()

        logger.info(
            "GeDIGCore initialized: multihop=%s max_hops=%s spectral=%s structural_sim=%s",
            self.enable_multihop, self.max_hops, self.enable_spectral,
            self._ss_evaluator is not None,
        )

    def _apply_config(self, config: GeDIGConfig) -> None:
        """Apply GeDIGConfig values to instance attributes."""
        self.node_cost = config.node_cost
        self.edge_cost = config.edge_cost
        self.normalization = config.normalization
        self.efficiency_weight = config.efficiency_weight
        self.min_nodes = config.min_nodes
        self.smoothing = config.smoothing
        self.enable_multihop = config.enable_multihop
        self.max_hops = config.max_hops
        self.decay_factor = config.decay_factor
        self.adaptive_hops = config.adaptive_hops
        self.spike_threshold = config.spike_threshold
        self.enable_spectral = config.enable_spectral
        self.spectral_weight = config.spectral_weight
        self.lambda_weight = config.lambda_weight
        self.ig_mode = config.ig_mode
        self.ig_norm_strategy = str(config.ig_norm_strategy or 'before').lower()
        self.ig_delta_mode = 'after_before'  # fixed
        self.entropy_tau = float(config.entropy_tau) if config.entropy_tau > 0 else 1.0
        self._ig_nonneg = config.ig_nonneg
        self.mu = config.mu
        self.warmup_steps = config.warmup_steps
        self.use_refactored_reward = config.use_refactored_reward
        self.use_legacy_formula = config.use_legacy_formula
        self.spike_detection_mode = config.spike_detection_mode
        self.tau_s = config.tau_s
        self.tau_i = config.tau_i
        self.use_multihop_sp_gain = config.use_multihop_sp_gain
        self.enable_ged_min_diag = config.enable_ged_min_diag
        self.sp_norm_mode = config.sp_norm_mode
        self.sp_beta = float(max(0.0, config.sp_beta))
        self.use_local_normalization = config.use_local_normalization
        self.local_norm_mode = config.local_norm_mode
        self.sp_node_cap = int(max(1, config.sp_node_cap))
        self.sp_pair_samples = int(config.sp_pair_samples)
        self.sp_use_sampling = config.sp_use_sampling
        self.sp_scope_mode = str(config.sp_scope_mode or 'auto').lower()
        self.sp_hop_expand = int(max(0, config.sp_hop_expand))
        self.sp_boundary_mode = config.sp_boundary_mode
        self.sp_eval_mode = str(config.sp_eval_mode or 'connected').lower()
        self.feature_weights = config.feature_weights
        self.linkset_mode = config.linkset_mode
        self.ig_source_mode = str(config.ig_source_mode or 'graph').lower()
        self.ig_hop_apply = str(config.ig_hop_apply or 'all').lower()
        self.ged_norm_scheme = str(config.ged_norm_scheme or 'edges_after').lower()

    def _init_structural_similarity(self) -> None:
        """Initialize structural similarity evaluator."""
        try:
            from .structural_similarity import StructuralSimilarityEvaluator
            from ..config.models import StructuralSimilarityConfig
            ss_cfg = StructuralSimilarityConfig(**self._ss_config)
            self._ss_evaluator = StructuralSimilarityEvaluator(ss_cfg)
            logger.info("Structural similarity evaluator enabled: method=%s", ss_cfg.method)
        except Exception as e:
            logger.warning("Failed to initialize structural similarity evaluator: %s", e)
            self._ss_evaluator = None

    @classmethod
    def from_config(cls, config: GeDIGConfig) -> "GeDIGCore":
        """Create GeDIGCore from a GeDIGConfig object.

        Args:
            config: GeDIGConfig object with all parameters.

        Returns:
            GeDIGCore instance configured from the config object.
        """
        return cls(config=config)

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
        g1 = ensure_networkx(g_prev)
        g2 = ensure_networkx(g_now)
        if features_prev is None:
            features_prev = extract_features(g1)
        if features_now is None:
            features_now = extract_features(g2)
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
            linkset_metrics = compute_linkset_metrics(
                g1,
                g2,
                linkset_info,
                entropy_tau=self.entropy_tau,
                sp_beta=self.sp_beta,
                use_multihop_sp_gain=self.use_multihop_sp_gain,
                ig_mode=self.ig_mode,
                ig_nonneg=self._ig_nonneg,
                lambda_weight=self.lambda_weight,
                use_legacy_formula=self.use_legacy_formula,
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
            result = calculate_multihop(
                g1,
                g2,
                features_prev,
                features_now,
                focal_nodes,
                start_time,
                # GED parameters
                node_cost=self.node_cost,
                edge_cost=self.edge_cost,
                ged_norm_scheme=self.ged_norm_scheme,
                candidate_count=cand_count,
                # IG parameters
                ig_source_mode=self.ig_source_mode,
                ig_hop_apply=self.ig_hop_apply,
                ig_mode=self.ig_mode,
                ig_nonneg=self._ig_nonneg,
                ig_norm_strategy=self.ig_norm_strategy,
                ig_delta_mode=self.ig_delta_mode,
                smoothing=self.smoothing,
                min_nodes=self.min_nodes,
                # Multi-hop parameters
                max_hops=self.max_hops,
                adaptive_hops=self.adaptive_hops,
                lambda_weight=self.lambda_weight,
                # SP gain parameters
                use_multihop_sp_gain=self.use_multihop_sp_gain,
                sp_beta=self.sp_beta,
                sp_scope_mode=self.sp_scope_mode,
                sp_hop_expand=self.sp_hop_expand,
                sp_boundary_mode=self.sp_boundary_mode,
                sp_eval_mode=self.sp_eval_mode,
                sp_node_cap=self.sp_node_cap,
                sp_pair_samples=self.sp_pair_samples,
                # Optional inputs
                norm_override=cmax_local,
                query_vector=query_vector,
                fixed_den=ig_fixed_den,
                k_star=k_star,
                linkset_metrics=linkset_metrics,
                # Feature weights
                feature_weights=self.feature_weights,
                # Structural similarity evaluator
                ss_evaluator=self._ss_evaluator,
                # Callbacks for GED and IG computation
                ged_calculator=self._calculate_normalized_ged,
                ig_calculator=self._calculate_entropy_variance_ig,
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

    # ------------ Helpers ------------
    def _compute_sp_gain_norm(self, g_before: nx.Graph, g_after: nx.Graph, mode: str = 'relative') -> float:
        """Normalized signed shortest-path gain between two subgraphs."""
        # Check if before graph has edges - if not, skip DistanceCache path
        # since fixed_before_pairs requires connectivity in the before graph.
        has_edges_before = g_before.number_of_edges() > 0

        # Handle fixed_before_pairs mode specially (requires DistanceCache)
        try:
            if has_edges_before and str(self.sp_eval_mode).lower() == 'fixed_before_pairs':
                from .sp_distcache import DistanceCache
                dc = DistanceCache(mode='cached', pair_samples=int(getattr(self, 'sp_pair_samples', 400)))
                sig = dc.signature(g_before, set(), 1, str(self.sp_scope_mode), str(self.sp_boundary_mode))
                rel = dc.estimate_sp_between_graphs(sig=sig, g_before=g_before, g_after=g_after)
                return float(max(-1.0, min(1.0, rel)))
        except Exception:
            pass
        return compute_sp_gain_norm(g_before, g_after, mode, self.sp_node_cap, self.sp_pair_samples)

    # Backward compatibility methods for tests
    def _graph_efficiency(self, g: nx.Graph) -> float:
        return graph_efficiency(g)

    def _extract_k_hop_subgraph(self, graph: nx.Graph, focal_nodes: Set[str], k: int) -> Tuple[nx.Graph, Set[str]]:
        return extract_k_hop_subgraph(graph, focal_nodes, k)

    def _trim_terminal_edges(self, graph: nx.Graph, focal_nodes: Set[str], max_dist: int) -> nx.Graph:
        return trim_terminal_edges(graph, focal_nodes, max_dist)

    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        return spectral_score(g)

    def _compute_linkset_metrics(
        self,
        g_before: nx.Graph,
        g_after: nx.Graph,
        linkset_info: Optional[Dict[str, Any]],
        *,
        query_vector: Optional[List[float]] = None,
        ig_fixed_den: Optional[float] = None,
    ) -> LinksetMetrics:
        """Backward compatibility wrapper for compute_linkset_metrics."""
        return compute_linkset_metrics(
            g_before,
            g_after,
            linkset_info,
            entropy_tau=self.entropy_tau,
            sp_beta=self.sp_beta,
            use_multihop_sp_gain=self.use_multihop_sp_gain,
            ig_mode=self.ig_mode,
            ig_nonneg=self._ig_nonneg,
            lambda_weight=self.lambda_weight,
            use_legacy_formula=self.use_legacy_formula,
            query_vector=query_vector,
            ig_fixed_den=ig_fixed_den,
        )

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

