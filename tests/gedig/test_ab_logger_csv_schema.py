import os
import csv
import tempfile
import networkx as nx

from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger


def test_ab_logger_csv_schema_and_k_estimate():
    logger = GeDIGABLogger(window=10, threshold=0.9, min_pairs=2, flush_every=5)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ab.csv')
        logger.set_auto_csv_path(path)
        # 合成データ: gedig = ged - k*ig になるように設計 (k=0.5)
        k_true = 0.5
        for i in range(5):
            ged = 1.0 + i * 0.1
            ig = 0.2 + i * 0.05
            gedig = ged - k_true * ig
            pure = {'gedig': gedig, 'ged': ged, 'ig': ig}
            full = {'gedig': gedig * 1.0, 'ged': ged, 'ig': ig}  # 同一値で高相関
            logger.record(f"q{i}", pure, full)
        logger.export_csv(path)
        assert os.path.exists(path)
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            expected = ['query_id','pure_gedig','full_gedig','pure_ged','full_ged','pure_ig','full_ig','k_estimate','k_missing_reason','window_corr_at_record','timestamp']
            assert header == expected
            row = next(r)
            # k_estimate 列は 0.5 に近い
            k_idx = header.index('k_estimate')
            k_val = float(row[k_idx]) if row[k_idx] not in ('', 'None') else None
            assert k_val is not None and abs(k_val - 0.5) < 1e-6


def test_ab_logger_handles_missing_values():
    logger = GeDIGABLogger(window=5, min_pairs=1)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ab.csv')
        logger.set_auto_csv_path(path)
        # ig = 0 のケース → k_estimate None
        pure = {'gedig': 1.0, 'ged': 1.0, 'ig': 0.0}
        full = {'gedig': 0.9, 'ged': 0.9, 'ig': 0.0}
        logger.record('q0', pure, full)
        logger.export_csv(path)
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            row = next(r)
            k_idx = header.index('k_estimate')
            reason_idx = header.index('k_missing_reason')
            # ig=0 のため k_estimate 空、理由コード ig_zero
            assert row[k_idx] in ('', 'None')
            assert row[reason_idx] == 'ig_zero'
