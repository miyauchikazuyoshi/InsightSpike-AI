"""geDIG configuration management.

This module provides centralized configuration for geDIG calculations,
including environment variable overrides and presets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable with fallback."""
    val = os.environ.get(key)
    if val is not None and val.strip():
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable with fallback."""
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off", ""):
        return False
    return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment variable with fallback."""
    val = os.environ.get(key)
    if val is not None and val.strip():
        return val.strip().lower()
    return default


@dataclass
class GeDIGConfig:
    """Configuration for geDIG calculation.

    All geDIG parameters are centralized here. Use `from_env()` to create
    a config that respects environment variable overrides, or use `preset()`
    to get predefined configurations.

    Environment Variables:
        MAZE_GEDIG_LAMBDA: Override lambda_weight
        MAZE_GEDIG_NODE_COST: Override node_cost
        MAZE_GEDIG_EDGE_COST: Override edge_cost
        MAZE_GEDIG_EFF_WEIGHT: Override efficiency_weight
        MAZE_GEDIG_SPECTRAL: Enable spectral evaluation
        MAZE_GEDIG_SPECTRAL_WEIGHT: Override spectral_weight
        MAZE_GEDIG_IG_MODE: Override ig_mode ('raw', 'z', 'norm')
        MAZE_GEDIG_IG_NORM: Override ig_norm_strategy
        MAZE_GEDIG_ENTROPY_TAU: Override entropy_tau
        INSIGHTSPIKE_ENTROPY_TAU: Alternative for entropy_tau
        MAZE_GEDIG_IG_NONNEG: Enable non-negative IG clamping
        INSIGHTSPIKE_GED_MIN_DIAG: Enable GED min diagnostic
        MAZE_GEDIG_SP_BOUNDARY: Override sp_boundary_mode
    """

    # === GED Parameters ===
    node_cost: float = 1.0
    edge_cost: float = 1.0
    normalization: str = "sum"
    efficiency_weight: float = 0.3

    # === IG Parameters ===
    lambda_weight: float = 1.0
    ig_mode: str = "raw"  # 'raw' | 'z' | 'norm'
    ig_norm_strategy: str = "before"
    ig_delta_mode: str = "after_before"  # fixed
    entropy_tau: float = 1.0
    ig_nonneg: bool = False
    min_nodes: int = 2
    smoothing: float = 1e-10

    # === Spike Detection ===
    spike_threshold: float = -0.5
    spike_detection_mode: str = "and"  # 'threshold' | 'and' | 'or'
    tau_s: float = 0.15
    tau_i: float = 0.25

    # === Multi-hop ===
    enable_multihop: bool = False
    max_hops: int = 3
    decay_factor: float = 0.7
    adaptive_hops: bool = True

    # === Shortest Path Gain ===
    use_multihop_sp_gain: bool = True
    sp_norm_mode: str = "relative"
    sp_beta: float = 0.2
    sp_node_cap: int = 200
    sp_pair_samples: int = 400
    sp_use_sampling: bool = True
    sp_scope_mode: str = "auto"
    sp_hop_expand: int = 0
    sp_boundary_mode: str = "induced"
    sp_eval_mode: str = "connected"

    # === Spectral ===
    enable_spectral: bool = False
    spectral_weight: float = 0.3

    # === Reward ===
    mu: float = 0.5
    warmup_steps: int = 10
    use_refactored_reward: bool = True
    use_legacy_formula: bool = False

    # === Local Normalization ===
    use_local_normalization: bool = False
    local_norm_mode: str = "layer1"

    # === Diagnostic ===
    enable_ged_min_diag: bool = False

    # === Paper-mode ===
    ig_source_mode: str = "graph"  # 'graph' | 'linkset' | 'hybrid'
    ig_hop_apply: str = "all"  # 'hop0' | 'all'
    ged_norm_scheme: str = "edges_after"  # 'edges_after' | 'candidate_base'

    # === Linkset ===
    linkset_mode: bool = False

    # === Feature weights (optional) ===
    feature_weights: Optional[np.ndarray] = field(default=None, repr=False)

    # === Structural Similarity ===
    structural_similarity_config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @classmethod
    def from_env(cls, **overrides) -> "GeDIGConfig":
        """Create config with environment variable overrides.

        Args:
            **overrides: Additional overrides to apply after env vars.

        Returns:
            GeDIGConfig with environment overrides applied.
        """
        # Start with defaults
        config = cls()

        # Apply environment overrides
        config.lambda_weight = _env_float("MAZE_GEDIG_LAMBDA", config.lambda_weight)
        config.node_cost = _env_float("MAZE_GEDIG_NODE_COST", config.node_cost)
        config.edge_cost = _env_float("MAZE_GEDIG_EDGE_COST", config.edge_cost)
        config.efficiency_weight = _env_float("MAZE_GEDIG_EFF_WEIGHT", config.efficiency_weight)
        config.spectral_weight = _env_float("MAZE_GEDIG_SPECTRAL_WEIGHT", config.spectral_weight)
        config.enable_spectral = _env_bool("MAZE_GEDIG_SPECTRAL", config.enable_spectral)
        config.ig_mode = _env_str("MAZE_GEDIG_IG_MODE", config.ig_mode)
        config.ig_norm_strategy = _env_str("MAZE_GEDIG_IG_NORM", config.ig_norm_strategy)

        # Entropy tau has two possible env vars
        entropy_tau_env = os.environ.get("MAZE_GEDIG_ENTROPY_TAU") or os.environ.get("INSIGHTSPIKE_ENTROPY_TAU")
        if entropy_tau_env:
            try:
                config.entropy_tau = float(entropy_tau_env)
            except ValueError:
                pass

        config.ig_nonneg = _env_bool("MAZE_GEDIG_IG_NONNEG", config.ig_nonneg)
        config.enable_ged_min_diag = _env_bool("INSIGHTSPIKE_GED_MIN_DIAG", config.enable_ged_min_diag)
        config.sp_boundary_mode = _env_str("MAZE_GEDIG_SP_BOUNDARY", config.sp_boundary_mode)

        # Apply explicit overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def preset(cls, name: str) -> "GeDIGConfig":
        """Get a predefined configuration preset.

        Args:
            name: Preset name ('conservative', 'balanced', 'aggressive',
                  'maze', 'transformer', 'rag')

        Returns:
            GeDIGConfig with preset values.
        """
        presets = {
            "conservative": {
                "lambda_weight": 0.3,
                "mu": 0.7,
                "tau_s": 0.2,
                "tau_i": 0.3,
                "spike_detection_mode": "and",
            },
            "balanced": {
                "lambda_weight": 0.5,
                "mu": 0.5,
                "tau_s": 0.15,
                "tau_i": 0.25,
                "spike_detection_mode": "and",
            },
            "aggressive": {
                "lambda_weight": 0.7,
                "mu": 0.3,
                "tau_s": 0.08,
                "tau_i": 0.15,
                "spike_detection_mode": "or",
            },
            "maze": {
                "lambda_weight": 0.5,
                "max_hops": 2,
                "enable_multihop": True,
            },
            "transformer": {
                "lambda_weight": 0.3,
                "max_hops": 1,
                "enable_multihop": False,
            },
            "rag": {
                "lambda_weight": 0.7,
                "max_hops": 3,
                "enable_multihop": True,
                "linkset_mode": True,
            },
        }

        name_lower = name.lower()
        if name_lower not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

        return cls(**presets[name_lower])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for GeDIGCore kwargs."""
        result = {
            "node_cost": self.node_cost,
            "edge_cost": self.edge_cost,
            "normalization": self.normalization,
            "efficiency_weight": self.efficiency_weight,
            "min_nodes": self.min_nodes,
            "smoothing": self.smoothing,
            "enable_multihop": self.enable_multihop,
            "max_hops": self.max_hops,
            "decay_factor": self.decay_factor,
            "adaptive_hops": self.adaptive_hops,
            "spike_threshold": self.spike_threshold,
            "enable_spectral": self.enable_spectral,
            "spectral_weight": self.spectral_weight,
            "lambda_weight": self.lambda_weight,
            "ig_mode": self.ig_mode,
            "ig_norm_strategy": self.ig_norm_strategy,
            "entropy_tau": self.entropy_tau,
            "mu": self.mu,
            "warmup_steps": self.warmup_steps,
            "use_refactored_reward": self.use_refactored_reward,
            "use_legacy_formula": self.use_legacy_formula,
            "spike_detection_mode": self.spike_detection_mode,
            "tau_s": self.tau_s,
            "tau_i": self.tau_i,
            "use_multihop_sp_gain": self.use_multihop_sp_gain,
            "sp_norm_mode": self.sp_norm_mode,
            "sp_beta": self.sp_beta,
            "use_local_normalization": self.use_local_normalization,
            "local_norm_mode": self.local_norm_mode,
            "enable_ged_min_diag": self.enable_ged_min_diag,
            "sp_node_cap": self.sp_node_cap,
            "sp_pair_samples": self.sp_pair_samples,
            "sp_use_sampling": self.sp_use_sampling,
            "sp_scope_mode": self.sp_scope_mode,
            "sp_hop_expand": self.sp_hop_expand,
            "sp_eval_mode": self.sp_eval_mode,
            "ig_source_mode": self.ig_source_mode,
            "ig_hop_apply": self.ig_hop_apply,
            "ged_norm_scheme": self.ged_norm_scheme,
            "linkset_mode": self.linkset_mode,
        }

        if self.feature_weights is not None:
            result["feature_weights"] = self.feature_weights

        if self.structural_similarity_config is not None:
            result["structural_similarity_config"] = self.structural_similarity_config

        return result


# Legacy compatibility: GeDIGPresets as class with class attributes
class GeDIGPresets:
    """Legacy preset class for backward compatibility.

    Prefer using GeDIGConfig.preset() instead.
    """
    CONSERVATIVE = {"lambda_weight": 0.3, "mu": 0.7, "tau_s": 0.2, "tau_i": 0.3, "spike_detection_mode": "and"}
    BALANCED = {"lambda_weight": 0.5, "mu": 0.5, "tau_s": 0.15, "tau_i": 0.25, "spike_detection_mode": "and"}
    AGGRESSIVE = {"lambda_weight": 0.7, "mu": 0.3, "tau_s": 0.08, "tau_i": 0.15, "spike_detection_mode": "or"}


__all__ = [
    "GeDIGConfig",
    "GeDIGPresets",
]
