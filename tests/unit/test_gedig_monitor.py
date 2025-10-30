import networkx as nx
import numpy as np
from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGMonitor


def build_pair(modify=False):
    g1 = nx.Graph(); g1.add_nodes_from(range(6)); g1.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5)])
    g2 = g1.copy()
    if modify:
        g2.add_edge(0,5); g2.add_node(6); g2.add_edge(2,6)
    return g1, g2

def test_monitor_false_positive_and_adjust():
    core = GeDIGCore(spike_detection_mode="or", tau_s=0.2, tau_i=5.0)
    monitor = GeDIGMonitor(window_size=50, target_fp_rate=0.1)
    core.attach_monitor(monitor)

    # Simulate stream: alternate changes (should spike sometimes) and no-change (potential FP)
    g_base1, g_base2 = build_pair(modify=True)
    g_same1, g_same2 = build_pair(modify=False)

    for i in range(40):
        if i % 4 == 0:  # structural change
            res = core.calculate(g_prev=g_base1, g_now=g_base2)
            monitor.record_outcome(actual_spike=True)  # treat as true spike
        else:
            res = core.calculate(g_prev=g_same1, g_now=g_same2)
            monitor.record_outcome(actual_spike=False)
        # Periodically adjust
        monitor.auto_adjust_thresholds(core)

    fp_rate = monitor.false_positive_rate()
    assert 0.0 <= fp_rate <= 1.0
    # thresholds stay finite and positive
    assert 1e-4 <= core.tau_s <= 10.0
    assert 1e-4 <= core.tau_i <= 10.0
    # After adjustments, FP rate should not explode
    assert fp_rate < 0.5
