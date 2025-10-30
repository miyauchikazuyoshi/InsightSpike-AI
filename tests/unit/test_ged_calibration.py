"""Tests for GED calibration normalization (M3).

If torch / torch_geometric が未インストール (ローカル軽量環境) の場合は
テスト全体を skip して収集段階エラーを回避する。CI (Linux) では依存が
導入され本テストが有効化される想定。
"""
import os
import pytest

_torch_available = True
try:  # 軽量環境ではtorch未導入前提
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception:  # noqa: BLE001
    _torch_available = False

pytestmark = pytest.mark.skipif(not _torch_available, reason="torch / torch_geometric not available")

from insightspike.algorithms.metrics_selector import MetricsSelector

@pytest.fixture
def small_graphs():
    g1 = Data(x=torch.randn(3, 4))
    g1.num_nodes = 3
    g2 = Data(x=torch.randn(5, 4))
    g2.num_nodes = 5
    return g1, g2

class TestGEDCalibration:
    def test_alpha_normalization(self, small_graphs, monkeypatch):
        g1, g2 = small_graphs
        sel = MetricsSelector(config=None)
        monkeypatch.setenv('GED_ALPHA', '2.0')
        raw = sel._ged_method(g1, g2)
        norm = sel.delta_ged(g1, g2)
        # When alpha=2.0 normalized should be ~ raw/2 (allowing float noise)
        assert norm == pytest.approx(raw / 2.0, rel=1e-3, abs=1e-3)

    def test_metrics_stored_with_debug(self, small_graphs, monkeypatch):
        g1, g2 = small_graphs
        sel = MetricsSelector(config=None)
        monkeypatch.setenv('INSIGHT_DEBUG_METRICS', '1')
        monkeypatch.setenv('GED_ALPHA', '1.5')
        _ = sel.delta_ged(g1, g2)
        m = sel.get_last_ged_metrics()
        assert m is not None
        assert {'raw','possible','efficiency','alpha','normalized'} <= set(m.keys())
        assert m['alpha'] == 1.5
    
    def test_efficiency_ema(self, small_graphs, monkeypatch):
        g1, g2 = small_graphs
        sel = MetricsSelector(config=None)
        monkeypatch.setenv('INSIGHT_DEBUG_METRICS', '1')
        monkeypatch.setenv('GEDIG_EFF_EWMA', '5')
        first = sel.delta_ged(g1, g2)
        m1 = sel.get_last_ged_metrics()
        second = sel.delta_ged(g1, g2)  # second call updates EMA
        m2 = sel.get_last_ged_metrics()
        assert 'efficiency_ema' in m2
        # EMA should lie between first efficiency and second efficiency
        assert min(m1['efficiency'], m2['efficiency']) <= m2['efficiency_ema'] <= max(m1['efficiency'], m2['efficiency'])

if __name__ == '__main__':
    pytest.main([__file__, '-q'])
