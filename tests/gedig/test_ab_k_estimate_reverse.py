import math
from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger

def test_k_estimate_reverse_engineering_precision():
    logger = GeDIGABLogger(window=10, min_pairs=1)
    k_true = 0.37
    for i in range(8):
        ged = 1.5 + i * 0.2
        ig = 0.3 + i * 0.07
        gedig = ged - k_true * ig
        pure = {'gedig': gedig, 'ged': ged, 'ig': ig}
        full = {'gedig': gedig, 'ged': ged, 'ig': ig}
        logger.record(f"q{i}", pure, full)
    # export and inspect first row k_estimate
    import tempfile, os, csv
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ab.csv')
        logger.export_csv(path)
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            idx = header.index('k_estimate')
            row = next(r)
            val = float(row[idx])
            assert abs(val - k_true) < 1e-6


def test_k_estimate_none_when_insufficient_values():
    logger = GeDIGABLogger(window=5, min_pairs=1)
    # ig 0 のケース -> None
    pure = {'gedig': 1.0, 'ged': 1.0, 'ig': 0.0}
    full = {'gedig': 0.9, 'ged': 0.9, 'ig': 0.0}
    logger.record('q0', pure, full)
    import tempfile, os, csv
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ab.csv')
        logger.export_csv(path)
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            idx = header.index('k_estimate')
            reason_idx = header.index('k_missing_reason')
            row = next(r)
            cell = row[idx]
            assert cell in ('', 'None')
            assert row[reason_idx] == 'ig_zero'
