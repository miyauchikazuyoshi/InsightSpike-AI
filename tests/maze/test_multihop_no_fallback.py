import os, sys
import networkx as nx
import pytest

# Add project root and experiments path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_PATH = os.path.abspath(os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src'))
if EXP_PATH not in sys.path:
    sys.path.insert(0, EXP_PATH)

if not os.path.exists(EXP_PATH):
    pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)

from core.gedig_evaluator import GeDIGEvaluator  # type: ignore


def build_chain(n: int):
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
        if i>0:
            g.add_edge(i-1, i)
    return g

def test_core_raw_hop1_approx_removed():
    g1 = build_chain(3)
    g2 = build_chain(4)  # add one node
    ev = GeDIGEvaluator(always_multihop=True, mode='core_raw', max_hops=2)
    res = ev.evaluate_escalating(g1, g2, escalation_threshold=-1.0)  # force escalate by low threshold
    assert res.get('hop1_approx_fallback') is False


def test_details_flag_deprecated_false_removed():
    g1 = build_chain(2)
    g2 = build_chain(3)
    ev = GeDIGEvaluator(always_multihop=True, mode='core_raw', max_hops=2)
    res = ev.evaluate_escalating(g1, g2, escalation_threshold=-1.0)
    assert res.get('hop1_approx_fallback') is False
    assert res['details']['hop1_approx_deprecated'] is False
