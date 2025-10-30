from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional, Set, List
import importlib.util
import sys

import networkx as nx

_BASE_SRC = Path(__file__).resolve().parents[2] / "maze-online-phase1-querylog" / "src"


def _load_base_adapter():
    module_name = "maze_local_base_gedig_adapter"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _BASE_SRC / "gedig_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_base_adapter_mod = _load_base_adapter()
GeDIGAdapter = _base_adapter_mod.GeDIGAdapter  # type: ignore
GeDIGPayload = _base_adapter_mod.GeDIGPayload  # type: ignore


class LocalizedGeDIGAdapter(GeDIGAdapter):
    """GeDIGAdapter that restricts evaluation to a local neighborhood."""

    def __init__(
        self,
        *,
        radius: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.radius = max(0, int(radius))

    def evaluate(
        self,
        prev_graph: nx.Graph,
        curr_graph: nx.Graph,
        *,
        center_node: Optional[Any] = None,
        focal_nodes: Optional[Set[Any]] = None,
        query_vector: Optional[List[float]] = None,
    ) -> GeDIGPayload:
        if center_node is None or self.radius <= 0:
            return super().evaluate(prev_graph, curr_graph, focal_nodes=focal_nodes, query_vector=query_vector)

        sub_prev = self._induced_neighborhood(prev_graph, center_node)
        sub_curr = self._induced_neighborhood(curr_graph, center_node)
        effective_focal: Optional[Set[Any]] = None
        if center_node in sub_curr:
            effective_focal = {center_node}

        return super().evaluate(
            sub_prev,
            sub_curr,
            focal_nodes=effective_focal,
            query_vector=query_vector,
        )

    # ------------------------------------------------------------------
    def _induced_neighborhood(self, graph: nx.Graph, center: Any) -> nx.Graph:
        if center not in graph:
            return graph.copy()
        nodes = self._collect_nodes_within_radius(graph, center, self.radius)
        return graph.subgraph(nodes).copy()

    @staticmethod
    def _collect_nodes_within_radius(graph: nx.Graph, center: Any, radius: int) -> Set[Any]:
        visited: Set[Any] = {center}
        queue: deque[tuple[Any, int]] = deque([(center, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= radius:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
        return visited
