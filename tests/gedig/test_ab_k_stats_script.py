import csv, tempfile, os, importlib.util


def _load_module(path):
    spec = importlib.util.spec_from_file_location('ab_k_stats', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_ab_k_stats_basic():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'gedig_ab_log_test.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['query_id','pure_gedig','full_gedig','pure_ged','full_ged','pure_ig','full_ig','k_estimate','k_missing_reason','window_corr_at_record','timestamp'])
            w.writerow(['q0',1.0,1.0,1.2,1.2,0.4,0.4,0.5,'',0.99,'123.0'])
            w.writerow(['q1',1.1,1.1,1.3,1.3,0.5,0.5,'','ig_zero',0.98,'124.0'])  # missing k
            w.writerow(['q2',1.2,1.2,1.4,1.4,0.6,0.6,0.7,'',0.97,'125.0'])
        mod = _load_module('scripts/analyze_ab_k_stats.py')
        stats = mod.compute_k_stats([csv_path])
        assert stats['total_rows'] == 3
        assert stats['rows_with_k'] == 2
        assert abs(stats['missing_rate'] - (1/3)) < 1e-9
        assert stats['k_min'] == 0.5 and stats['k_max'] == 0.7
        assert stats['window_corr_last'] == 0.97
