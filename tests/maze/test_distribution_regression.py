import os, sys, json, tempfile
import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
pytest.importorskip(
    "scripts.distribution_regression",
    reason="Utility script not available in current packaging; skip in cloud runs.",
)
from scripts.distribution_regression import extract_distribution, compare_dists, main  # type: ignore


def _make_graph_linear(n):
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
        if i>0:
            g.add_edge(i-1, i)
    return g


def test_extract_and_compare_basic():
    g1 = _make_graph_linear(10)
    g2 = _make_graph_linear(12)
    d1 = extract_distribution(g1)
    d2 = extract_distribution(g2)
    comp = compare_dists(d1, d2)
    assert 'degree' in comp
    assert comp['degree']['ks'] >= 0.0


def test_main_baseline_and_compare(tmp_path):
    # Create temp graph directory with edge list
    g = _make_graph_linear(8)
    target_dir = tmp_path / 'g'
    target_dir.mkdir()
    el_path = target_dir / 'graph.edgelist'
    with open(el_path, 'w') as f:
        for u, v in g.edges():
            f.write(f"{u} {v}\n")
    baseline_dir = tmp_path / 'baseline'
    # Write baseline
    res1 = main(str(baseline_dir), str(target_dir), update=True)
    assert res1['mode'] == 'baseline_written'
    # Second run compares
    res2 = main(str(baseline_dir), str(target_dir), update=False)
    assert res2['mode'] == 'compared'
    assert 'degree' in res2['comparison']
