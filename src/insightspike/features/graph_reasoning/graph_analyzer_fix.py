"""
Graph Analyzer Fix - Pass graph structures instead of features

This fix ensures that the full graph structure is passed to delta_ig_func,
not just the feature vectors.
"""

import logging

logger = logging.getLogger(__name__)


def apply_graph_analyzer_fix():
    """Apply fix to GraphAnalyzer to pass graphs instead of features."""
    try:
        from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer
        
        # Store original method
        original_calculate_metrics = GraphAnalyzer.calculate_metrics
        
        def fixed_calculate_metrics(
            self,
            current_graph,
            previous_graph,
            delta_ged_func,
            delta_ig_func,
        ):
            """Fixed calculate_metrics that passes graphs to delta_ig."""
            if previous_graph is None:
                return {
                    "delta_ged": 0.0,
                    "delta_ig": 0.0,
                    "graph_size_current": current_graph.num_nodes if current_graph else 0,
                    "graph_size_previous": 0,
                }

            try:
                # For GED, still convert to NetworkX
                import networkx as nx

                # Create NetworkX graphs from PyTorch Geometric data
                g_prev = nx.Graph()
                g_curr = nx.Graph()

                # Add nodes
                for i in range(previous_graph.num_nodes):
                    g_prev.add_node(i)
                for i in range(current_graph.num_nodes):
                    g_curr.add_node(i)

                # Add edges if available
                if (
                    hasattr(previous_graph, "edge_index")
                    and previous_graph.edge_index is not None
                ):
                    edges = previous_graph.edge_index.t().tolist()
                    g_prev.add_edges_from(edges)
                if (
                    hasattr(current_graph, "edge_index")
                    and current_graph.edge_index is not None
                ):
                    edges = current_graph.edge_index.t().tolist()
                    g_curr.add_edges_from(edges)

                ged = delta_ged_func(g_prev, g_curr)
                
                logger.debug(
                    f"Graph structure: prev={g_prev.number_of_nodes()} nodes, {g_prev.number_of_edges()} edges; "
                    f"curr={g_curr.number_of_nodes()} nodes, {g_curr.number_of_edges()} edges"
                )

                # FIXED: Pass the full graph structures to delta_ig, not just features!
                # The delta_ig function should handle graph conversion internally
                ig = delta_ig_func(previous_graph, current_graph)
                
                logger.debug(f"Calculated metrics: ΔGED={ged:.3f}, ΔIG={ig:.3f}")

                return {
                    "delta_ged": float(ged),
                    "delta_ig": float(ig),
                    "graph_size_current": current_graph.num_nodes,
                    "graph_size_previous": previous_graph.num_nodes,
                }

            except Exception as e:
                logger.error(f"Failed to calculate graph metrics: {e}")
                return {
                    "delta_ged": 0.0,
                    "delta_ig": 0.0,
                    "graph_size_current": current_graph.num_nodes if current_graph else 0,
                    "graph_size_previous": previous_graph.num_nodes if previous_graph else 0,
                }
        
        # Apply patch
        GraphAnalyzer.calculate_metrics = fixed_calculate_metrics
        
        logger.info("GraphAnalyzer fix applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply GraphAnalyzer fix: {e}")
        return False


# Auto-apply fix when imported
if __name__ != "__main__":
    apply_graph_analyzer_fix()