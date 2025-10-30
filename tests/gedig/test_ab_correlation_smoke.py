import math
from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger

def test_ab_correlation_increases_and_stays_high():
    logger = GeDIGABLogger(window=20, threshold=0.85, min_pairs=2, flush_every=100)
    # pure/full をほぼ同一系列 (微小ノイズ) にして高相関を誘導
    for i in range(10):
        base = 1.0 + i * 0.05
        ged = base * 0.8
        ig = base * 0.2 + 0.01
        gedig = ged - 0.5 * ig
        # full に微小ノイズ
        noise = 1e-4 * (i % 3)
        pure = {'gedig': gedig, 'ged': ged, 'ig': ig}
        full = {'gedig': gedig + noise, 'ged': ged + noise, 'ig': ig}
        logger.record(f"q{i}", pure, full)
    m = logger.current_metrics()
    assert m['count'] == 10
    corr = m['gedig_corr']
    assert isinstance(corr, float) and not math.isnan(corr)
    # 高相関確認
    assert corr > 0.99
