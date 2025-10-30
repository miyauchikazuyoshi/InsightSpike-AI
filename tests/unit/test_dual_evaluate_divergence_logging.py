import os, tempfile, networkx as nx
from insightspike.algorithms.gedig_factory import GeDIGFactory, dual_evaluate

def test_dual_evaluate_divergence_logging_csv():
    legacy = GeDIGFactory.create({'use_refactored_gedig': False})
    ref = GeDIGFactory.create({'use_refactored_gedig': True})
    g_prev = nx.path_graph(5)
    g_now = nx.path_graph(8)
    with tempfile.TemporaryDirectory() as td:
        log_path = os.path.join(td, 'divergence.csv')
        _, delta = dual_evaluate(legacy, ref, g_prev=g_prev, g_now=g_now, delta_threshold=0.0, divergence_logger=log_path)
        assert delta > 0
        with open(log_path) as fh:
            lines = fh.read().strip().splitlines()
        assert len(lines) == 2  # header + one row
        assert lines[0].startswith('ts,legacy,ref,delta')
