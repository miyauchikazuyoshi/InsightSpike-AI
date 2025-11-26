import types

from insightspike.algorithms.gedig_calculator import GeDIGCalculator
from insightspike.config.presets import ConfigPresets


class _DummyResult:
    """Minimal stand-in for GeDIGResult used by the calculator wrapper."""

    def __init__(self):
        self.gedig_value = 0.0
        self.ged_value = 0.0
        self.ig_value = 0.0
        self.structural_improvement = 0.0
        self.information_integration = 0.0
        self.hop_results = {}
        self.spike = False

    @property
    def has_spike(self):
        return self.spike


def test_paper_path_supplies_linkset_info(monkeypatch):
    """Paper preset should drive the linkset-first path (no graph-IG fallback)."""

    calc = GeDIGCalculator(config=ConfigPresets.paper())

    calls = {}

    def _fake_calculate(**kwargs):
        calls["linkset_info"] = kwargs.get("linkset_info")
        return _DummyResult()

    monkeypatch.setattr(calc, "_gedig_core", types.SimpleNamespace(calculate=_fake_calculate))

    calc.calculate(graph_before=None, graph_after=None)

    assert calls["linkset_info"] is not None, "paper preset must provide linkset_info to gedig_core"
