"""Graph analysis and reward calculators for Layer3 (self-contained copy)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # optional PyG import for typing/attr access
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    class Data:  # minimal fallback sentinel
        def __init__(self, x=None, edge_index=None):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = getattr(x, "shape", [0])[0] if x is not None else 0


class GraphAnalyzer:
    """Analyzes graph structures and calculates metrics."""

    def __init__(self, config=None):
        self.config = config or {}

    def _count_nodes(self, g):
        try:
            if hasattr(g, "num_nodes"):
                return g.num_nodes  # PyG style
            if hasattr(g, "number_of_nodes"):
                return g.number_of_nodes()
            return len(g) if g is not None else 0
        except Exception:
            return 0

    def calculate_metrics(
        self,
        current_graph: Any,  # Accept NetworkX or PyG Data
        previous_graph: Optional[Any],
        delta_ged_func,
        delta_ig_func,
    ) -> Dict[str, float]:
        """Calculate ΔGED and ΔIG metrics between graphs."""
        if previous_graph is None:
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "delta_h": 0.0,
                "delta_ged_norm": 0.0,
                "delta_sp": 0.0,
                "g0": 0.0,
                "gmin": 0.0,
                "graph_size_current": self._count_nodes(current_graph),
                "graph_size_previous": 0,
            }

        try:
            kwargs = {}
            if self.config and "graph" in self.config:
                graph_config = self.config["graph"]
                if "metrics" in graph_config:
                    kwargs["config"] = {"metrics": graph_config["metrics"]}

            # ΔGED (fall back to NX conversion if needed)
            try:
                ged = delta_ged_func(previous_graph, current_graph, **kwargs)
            except Exception:
                try:
                    import networkx as nx

                    def _to_nx(data: Any):
                        g = nx.Graph()
                        for i in range(getattr(data, "num_nodes", 0)):
                            g.add_node(i)
                        if hasattr(data, "edge_index") and data.edge_index is not None:
                            edges = data.edge_index.t().tolist()
                            g.add_edges_from(edges)
                        return g

                    g_prev = _to_nx(previous_graph)
                    g_curr = _to_nx(current_graph)
                    ged = delta_ged_func(g_prev, g_curr)
                except Exception as conv_e:
                    logger.debug("delta_ged_func fallback conversion failed: %s", conv_e)
                    ged = 0.0

            # ΔIG (prefer full graphs, fall back to embeddings)
            try:
                ig = delta_ig_func(previous_graph, current_graph, **kwargs)
                if isinstance(ig, (list, tuple)):
                    ig = float(ig[0])
            except Exception as ig_e:
                try:
                    import numpy as _np

                    def _extract_vecs(g):
                        if hasattr(g, "x") and g.x is not None:
                            arr = g.x
                            return arr.cpu().numpy() if hasattr(arr, "cpu") else _np.asarray(arr)
                        if hasattr(g, "nodes") and len(getattr(g, "nodes")()) > 0:
                            feats = []
                            for _, d in g.nodes(data=True):
                                if "feature" in d:
                                    feats.append(d["feature"])
                            if feats:
                                return _np.asarray(feats)
                        return _np.zeros((0,))

                    prev_vecs = _extract_vecs(previous_graph)
                    curr_vecs = _extract_vecs(current_graph)
                    ig = delta_ig_func(prev_vecs, curr_vecs)
                except Exception:
                    logger.debug("delta_ig_func failed: %s", ig_e)
                    ig = 0.0

            ig_val = float(ig)
            ged_val = float(ged)
            return {
                "delta_ged": ged_val,
                "delta_ig": ig_val,
                "delta_h": ig_val,
                "delta_ged_norm": float(abs(ged_val)),
                "delta_sp": 0.0,
                "g0": float(ged_val),
                "gmin": float(ged_val),
                "graph_size_current": self._count_nodes(current_graph),
                "graph_size_previous": self._count_nodes(previous_graph),
            }

        except Exception as e:
            logger.error("Metrics calculation failed: %s", e)
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "delta_h": 0.0,
                "delta_ged_norm": 0.0,
                "delta_sp": 0.0,
                "g0": 0.0,
                "gmin": 0.0,
                "graph_size_current": self._count_nodes(current_graph),
                "graph_size_previous": self._count_nodes(previous_graph),
            }

    def detect_spike(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> bool:
        """Detect if current state represents an insight spike."""
        ged_val = metrics.get("delta_ged", 0.0)
        ig_val = metrics.get("delta_ig", 0.0)
        ged_thr = thresholds.get("ged", -0.5)
        ig_thr = thresholds.get("ig", 0.2)
        conflict_thr = thresholds.get("conflict", 0.5)

        ged_margin = 0.05
        high_ged = (ged_val < ged_thr) or (
            abs(ged_val - ged_thr) <= ged_margin and ged_val <= ged_thr + ged_margin / 2
        )
        high_ig = ig_val > ig_thr
        if "total" in conflicts:
            low_conflict = conflicts.get("total", 0.0) < conflict_thr
        else:
            low_conflict = False
        low_conflict_growth = conflicts.get("total", 0.0) < conflict_thr

        improvement_spike = high_ged and high_ig and low_conflict
        growth_factor = 200.0
        structural_growth = ged_val > abs(ged_thr) * growth_factor and (ig_val > -0.01) and low_conflict_growth
        spike = improvement_spike or structural_growth
        try:  # pragma: no cover
            import logging as _lg

            _lg.getLogger(__name__).debug(
                f"SpikeCheck ged={ged_val:.3f} thr={ged_thr} ig={ig_val:.3f} thr={ig_thr} conflict={conflicts.get('total',0):.3f} -> {spike}"
            )
        except Exception:
            pass
        return spike

    def assess_quality(self, metrics: Dict[str, float], conflicts: Dict[str, float]) -> float:
        """Assess overall quality of reasoning process."""
        ged_score = abs(metrics.get("delta_ged", 0))
        ig_score = metrics.get("delta_ig", 0)
        metric_score = (ged_score + ig_score) / 2
        conflict_penalty = conflicts.get("total", 0)
        quality = max(0.0, min(1.0, metric_score - conflict_penalty))
        return float(quality)


class RewardCalculator:
    """Calculates reward signals for memory updates."""

    def __init__(self, config=None):
        self.config = config or {}
        from ....config.constants import Defaults

        if hasattr(config, "graph"):
            self.weights = {
                "ged": getattr(config.graph, "weight_ged", Defaults.REWARD_WEIGHT_GED),
                "ig": getattr(config.graph, "weight_ig", Defaults.REWARD_WEIGHT_IG),
            }
            self.temperature = getattr(config.graph, "temperature", Defaults.REWARD_TEMPERATURE)
        else:
            self.weights = {
                "ged": Defaults.REWARD_WEIGHT_GED,
                "ig": Defaults.REWARD_WEIGHT_IG,
            }
            self.temperature = Defaults.REWARD_TEMPERATURE

        self.optimal_graph_size = (
            getattr(config.graph, "optimal_graph_size", Defaults.OPTIMAL_GRAPH_SIZE)
            if hasattr(config, "graph")
            else Defaults.OPTIMAL_GRAPH_SIZE
        )

    def calculate_reward(self, metrics: Dict[str, float], conflicts: Dict[str, float]) -> Dict[str, float]:
        """Calculate multi-component reward signal using geDIG formula."""
        base_reward = self.weights["ged"] * metrics.get("delta_ged", 0) - self.weights["ig"] * self.temperature * metrics.get(
            "delta_ig", 0
        )
        structure_reward = self._calculate_structure_reward(metrics)
        novelty_reward = self._calculate_novelty_reward(metrics, conflicts)

        return {
            "base": float(base_reward),
            "structure": float(structure_reward),
            "novelty": float(novelty_reward),
            "total": float(base_reward + structure_reward + novelty_reward),
        }

    def _calculate_structure_reward(self, metrics: Dict[str, float]) -> float:
        current_size = metrics.get("graph_size_current", 0)
        if current_size == 0:
            return 0.0
        deviation = abs(current_size - self.optimal_graph_size) / max(self.optimal_graph_size, 1)
        return max(0.0, 1.0 - deviation)

    def _calculate_novelty_reward(self, metrics: Dict[str, float], conflicts: Dict[str, float]) -> float:
        conflict_penalty = conflicts.get("total", 0)
        return max(0.0, metrics.get("delta_ig", 0) - conflict_penalty)


__all__ = ["GraphAnalyzer", "RewardCalculator"]
