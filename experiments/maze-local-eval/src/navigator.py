from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.util
import sys

import networkx as nx
import math
import math

from .local_adapter import LocalizedGeDIGAdapter

_BASE_SRC = Path(__file__).resolve().parents[2] / "maze-online-phase1-querylog" / "src"


def _load_base_navigator():
    module_name = "maze_online_phase1_querylog.src.navigator"
    if module_name in sys.modules:
        return sys.modules[module_name]
    import types
    base_package = "maze_online_phase1_querylog"
    if base_package not in sys.modules:
        pkg = types.ModuleType(base_package)
        pkg.__path__ = [str(_BASE_SRC.parent)]  # type: ignore[attr-defined]
        sys.modules[base_package] = pkg
    src_package = f"{base_package}.src"
    if src_package not in sys.modules:
        subpkg = types.ModuleType(src_package)
        subpkg.__path__ = [str(_BASE_SRC)]  # type: ignore[attr-defined]
        sys.modules[src_package] = subpkg
    spec = importlib.util.spec_from_file_location(module_name, _BASE_SRC / "navigator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_base_nav_mod = _load_base_navigator()
BaseNavigator = _base_nav_mod.GeDIGNavigator  # type: ignore
BaseConfig = _base_nav_mod.NavigatorConfig  # type: ignore


@dataclass
class NavigatorConfig(BaseConfig):
    local_radius: int = 1


class LocalizedGeDIGNavigator(BaseNavigator):
    """Navigator that evaluates geDIG within a localized neighborhood."""

    def __init__(self, env, config: Optional[NavigatorConfig] = None) -> None:
        if config is None:
            config = NavigatorConfig()
        super().__init__(env, config)
        self._localized_adapter = LocalizedGeDIGAdapter(
            lambda_weight=config.lambda_weight,
            max_hops=config.max_hops,
            decay_factor=config.decay_factor,
            sp_beta=config.sp_beta,
            adaptive_hops=config.adaptive_hops,
            ig_norm_strategy=config.ig_norm_strategy,
            radius=config.local_radius,
        )

    def _evaluate(self, prev_graph, curr_graph, query_vector=None):
        center = self._node_id(self.current_pos, self.current_incoming)
        before = self._build_virtual_graph(curr_graph, connect_observed=False)
        after = self._build_virtual_graph(curr_graph, connect_observed=True)
        return self._localized_adapter.evaluate(
            before,
            after,
            center_node=center,
            query_vector=query_vector,
        )

    def _build_virtual_graph(self, base_graph: nx.Graph, *, connect_observed: bool) -> nx.Graph:
        graph = base_graph.copy()
        if not connect_observed:
            virtual_edges = [
                (u, v)
                for u, v, data in graph.edges(data=True)
                if data.get("virtual") or data.get("observed_candidate")
            ]
            if virtual_edges:
                graph.remove_edges_from(virtual_edges)
            virtual_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if data.get("virtual") or data.get("observed_candidate")
            ]
            if virtual_nodes:
                for node in virtual_nodes:
                    if graph.has_node(node) and graph.degree(node) == 0:
                        graph.remove_node(node)
            return graph

        center_node = self._node_id(self.current_pos, self.current_incoming)
        if not graph.has_node(center_node):
            graph.add_node(center_node, virtual=False, observed_candidate=False)

        query_vector = getattr(self, "_last_query_vector", None)
        if query_vector is None:
            return graph
        base_weights = list(self.vector_proc.weights)
        if self.config.dir_weight and self.config.dir_weight > 0:
            base_weights[2] = float(self.config.dir_weight)
            base_weights[3] = float(self.config.dir_weight)
        weighted_query = self.vector_proc.apply_weights(query_vector, base_weights)

        for (pos, direction), info in self._episodes.items():
            if info.get("is_wall"):
                continue
            vector = info.get("vector")
            if not vector:
                continue
            edge_traversals = int(info.get("edge_traversals", 0) or 0)
            if edge_traversals > 0:
                continue
            try:
                weighted_ep = self.vector_proc.apply_weights(vector, base_weights)
                distance = self.vector_proc.euclidean_distance(weighted_query, weighted_ep)
            except Exception:
                continue
            threshold = float(self.config.observed_wiring_tau_unvisited)
            if threshold <= 0 or distance > threshold:
                continue
            target = self._target_position(pos, direction)
            if not self.env.is_open(target):
                continue
            node_attrs = {
                "virtual": True,
                "observed_candidate": True,
                "position": list(target),
                "origin": list(pos),
                "is_wall": False,
                "unvisited": True,
                "observations": int(info.get("observations", 0) or 0),
                "visits": int(info.get("visits", 0) or 0),
                "feature": list(vector),
            }
            node_id = self._virtual_node_id(target)
            if not graph.has_node(node_id):
                graph.add_node(node_id, **node_attrs)
            else:
                graph.nodes[node_id].update(node_attrs)

            edge_attrs = {
                "virtual": True,
                "observed_candidate": True,
                "distance": distance,
                "direction": direction,
                "is_wall": False,
                "unvisited": True,
            }
            if not graph.has_edge(center_node, node_id):
                graph.add_edge(center_node, node_id, **edge_attrs)
            else:
                graph.edges[center_node, node_id].update(edge_attrs)

        return graph
