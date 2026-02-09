"""Edge attention lifecycle management.

Handles creation, decay, boost, and reactivation of edge attention weights.
"""

from typing import List, Tuple

import networkx as nx

Node = Tuple[int, int, int]


class AttentionManager:
    """Manage attention weights on graph edges."""

    def __init__(
        self,
        decay_rate: float = 0.95,
        use_boost: float = 0.1,
        theta: float = 0.3,
    ):
        self.decay_rate = decay_rate
        self.use_boost = use_boost
        self.theta = theta

    def on_new_edge(
        self,
        G: nx.Graph,
        u: Node,
        v: Node,
        edge_type: str = "explore",
    ) -> None:
        """Set attention=1.0 on a newly created edge."""
        if not G.has_edge(u, v):
            G.add_edge(u, v, attention=1.0, edge_type=edge_type, use_count=0)

    def on_step(self, G: nx.Graph) -> None:
        """Decay all edge attention values by decay_rate (called every step)."""
        for _u, _v, d in G.edges(data=True):
            d["attention"] = d.get("attention", 0.0) * self.decay_rate

    def on_traverse(self, G: nx.Graph, u: Node, v: Node) -> None:
        """Boost attention on a traversed edge."""
        if G.has_edge(u, v):
            d = G[u][v]
            d["attention"] = min(1.0, d.get("attention", 0.0) + self.use_boost)
            d["use_count"] = d.get("use_count", 0) + 1

    def on_ag_fire(self, G: nx.Graph, node: Node) -> List[Tuple[Node, float]]:
        """Reactivate sub-threshold edges around a node (AG firing)."""
        reactivated = []
        if node not in G:
            return reactivated
        for nb in G.neighbors(node):
            att = G[node][nb].get("attention", 0.0)
            if att < self.theta:
                new_att = self.theta + 0.1
                G[node][nb]["attention"] = new_att
                reactivated.append((nb, new_att))
        return reactivated

    def beta1(self, G: nx.Graph, theta: float | None = None) -> int:
        """Compute β₁ of the subgraph with attention > theta."""
        theta = theta if theta is not None else self.theta
        edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("attention", 0) > theta
        ]
        sub = nx.Graph()
        sub.add_nodes_from(G.nodes())
        sub.add_edges_from(edges)
        V = sub.number_of_nodes()
        E = sub.number_of_edges()
        if V == 0:
            return 0
        C = nx.number_connected_components(sub)
        return E - V + C
